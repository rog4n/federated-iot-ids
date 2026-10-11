import io
import socket
import struct
import numpy as np

FEATURE_D = 10
COV_ELEMS = FEATURE_D * (FEATURE_D + 1) // 2
# n + normalization mean + normalization std + normalized model mean + lower cov + threshold
MODEL_SIZE = 1 + FEATURE_D + FEATURE_D + FEATURE_D + COV_ELEMS + 1


def rec_exact(sock, num_bytes):
    data = bytearray()
    while len(data) < num_bytes:
        chunk = sock.recv(num_bytes - len(data))
        if not chunk:
            raise ConnectionResetError(
                f"connection dropped: expected {num_bytes}, got {len(data)}"
            )
        data.extend(chunk)
    return bytes(data)


def send_exact(sock, data):
    sock.sendall(data)


def recv_model(sock):
    raw_size = rec_exact(sock, 8)
    payload_size = struct.unpack("!Q", raw_size)[0]

    if payload_size <= 0 or payload_size > 1024 * 1024:
        raise ValueError(f"invalid payload size: {payload_size}")

    raw_payload = rec_exact(sock, payload_size)
    model = np.load(io.BytesIO(raw_payload), allow_pickle=False)
    model = np.asarray(model, dtype=np.float32).reshape(-1)

    if model.size != MODEL_SIZE:
        raise ValueError(
            f"invalid anomaly model size: expected {MODEL_SIZE}, got {model.size}"
        )

    if not np.all(np.isfinite(model)):
        raise ValueError("anomaly model contains NaN/Inf")

    return model


def make_npy_payload(model):
    model = np.asarray(model, dtype=np.float32).reshape(-1)
    buf = io.BytesIO()
    np.save(buf, model, allow_pickle=False)
    return buf.getvalue()


def send_model(sock, model):
    payload = make_npy_payload(model)
    send_exact(sock, struct.pack("!Q", len(payload)))
    send_exact(sock, payload)


def unpack_model(model):
    """Return n, normalization mean/std, normalized mean, covariance, threshold."""
    p = 0
    n = int(round(float(model[p])))
    p += 1

    norm_mean = model[p:p + FEATURE_D].astype(np.float64)
    p += FEATURE_D

    norm_std = model[p:p + FEATURE_D].astype(np.float64)
    p += FEATURE_D
    if np.any(norm_std <= 0.0):
        raise ValueError("client sent non-positive normalization std")

    mean = model[p:p + FEATURE_D].astype(np.float64)
    p += FEATURE_D

    cov = np.zeros((FEATURE_D, FEATURE_D), dtype=np.float64)
    for i in range(FEATURE_D):
        count = i + 1
        vals = model[p:p + count].astype(np.float64)
        cov[i, :count] = vals
        cov[:count, i] = vals
        p += count

    threshold = float(model[p])
    return n, norm_mean, norm_std, mean, cov, threshold


def pack_model(n, norm_mean, norm_std, mean, cov, threshold):
    out = np.empty(MODEL_SIZE, dtype=np.float32)
    p = 0

    out[p] = float(n)
    p += 1

    out[p:p + FEATURE_D] = np.asarray(norm_mean, dtype=np.float32)
    p += FEATURE_D

    out[p:p + FEATURE_D] = np.asarray(norm_std, dtype=np.float32)
    p += FEATURE_D

    out[p:p + FEATURE_D] = np.asarray(mean, dtype=np.float32)
    p += FEATURE_D

    for i in range(FEATURE_D):
        count = i + 1
        out[p:p + count] = np.asarray(cov[i, :count], dtype=np.float32)
        p += count

    out[p] = np.float32(threshold)
    return out


def project_psd(cov, floor=1e-3):
    cov = (cov + cov.T) * 0.5
    eigvals, eigvecs = np.linalg.eigh(cov)
    eigvals = np.maximum(eigvals, floor)
    return (eigvecs * eigvals) @ eigvecs.T


def normalized_to_raw(n, norm_mean, norm_std, mean_norm, cov_norm):
    """Convert a client's normalized Gaussian back to raw feature units."""
    raw_mean = norm_mean + norm_std * mean_norm
    raw_cov = cov_norm * np.outer(norm_std, norm_std)
    return raw_mean, raw_cov


def raw_to_normalized(raw_mean, raw_cov):
    """Choose the pooled raw mean/std as the new common normalization space."""
    raw_mean = np.asarray(raw_mean, dtype=np.float64)
    raw_cov = project_psd(raw_cov, floor=1e-9)

    raw_var = np.maximum(np.diag(raw_cov), 0.0)
    raw_std = np.sqrt(raw_var)
    # A constant feature gets a neutral scale instead of an artificial tiny scale.
    raw_std = np.where(raw_std < 1e-6, 1.0, raw_std)
    # Normalize around the pooled raw mean. The normalized global mean is zero.
    mean_norm = np.zeros(FEATURE_D, dtype=np.float64)
    cov_norm = raw_cov / np.outer(raw_std, raw_std)
    cov_norm = project_psd(cov_norm)

    return raw_mean, raw_std, mean_norm, cov_norm


def aggregate_models(models):
    """Aggregate locally normalized client models in common raw feature space.

Each client model may have been automatically adapted from its recent-normal FIFO.

    This is important because each edge now learns its normalization parameters
    from its own verified-safe traffic. Directly averaging normalized-space
    models would be invalid when those normalizations differ.
    """
    if not models:
        raise ValueError("no client models")

    parsed = [unpack_model(m) for m in models]

    for n, _, _, _, _, _ in parsed:
        if n < 2:
            raise ValueError(f"invalid client sample count: {n}; need n >= 2")

    counts = np.array([n for n, _, _, _, _, _ in parsed], dtype=np.float64)
    total_n = int(np.sum(counts))
    weights = counts / float(total_n)

    raw_means = []
    raw_covs = []
    thresholds = []

    for n, norm_mean, norm_std, mean_norm, cov_norm, threshold in parsed:
        raw_mean, raw_cov = normalized_to_raw(
            n, norm_mean, norm_std, mean_norm, cov_norm
        )
        raw_means.append(raw_mean)
        raw_covs.append(raw_cov)
        thresholds.append(threshold)

    raw_means = np.asarray(raw_means, dtype=np.float64)
    raw_covs = np.asarray(raw_covs, dtype=np.float64)
    thresholds = np.asarray(thresholds, dtype=np.float64)

    global_raw_mean = np.sum(weights[:, None] * raw_means, axis=0)

    # Pool client sample covariances correctly in raw space.
    pooled_M2 = np.zeros((FEATURE_D, FEATURE_D), dtype=np.float64)
    for weightless_n, mean_i, cov_i in zip(counts.astype(np.int64), raw_means, raw_covs):
        delta = (mean_i - global_raw_mean).reshape(-1, 1)
        pooled_M2 += (weightless_n - 1) * cov_i
        pooled_M2 += weightless_n * (delta @ delta.T)

    if total_n <= 1:
        raise ValueError("not enough samples to build global covariance")

    global_raw_cov = pooled_M2 / float(total_n - 1)
    global_raw_cov = project_psd(global_raw_cov)

    # Create a single common normalization space for the returned global model.
    global_norm_mean, global_norm_std, global_mean_norm, global_cov_norm = \
        raw_to_normalized(global_raw_mean, global_raw_cov)

    global_threshold = float(np.sum(weights * thresholds))
    return pack_model(
        total_n,
        global_norm_mean,
        global_norm_std,
        global_mean_norm,
        global_cov_norm,
        global_threshold,
    )


def federate_round(n_of_clients):
    if n_of_clients < 1:
        raise ValueError("n_of_clients must be >= 1")

    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server.bind(("127.0.0.1", 8080))
    server.listen(n_of_clients)
    print(f"Server listening on 127.0.0.1:8080 for {n_of_clients} clients ...")

    clients = []
    models = []

    try:
        while len(models) < n_of_clients:
            cli_sock, cli_addr = server.accept()
            print(f"[+] Connected to {cli_addr}")

            try:
                model = recv_model(cli_sock)
                clients.append(cli_sock)
                models.append(model)
                print(f"    received local adaptive model ({model.size} float32 values)")
            except (ConnectionResetError, ValueError, OSError) as exc:
                print(f"[!] Rejected client {cli_addr}: {exc}")
                cli_sock.close()

        global_model = aggregate_models(models)
        print(f"[+] Aggregated {len(models)} local models in raw feature space")
        print(f"    global model size: {global_model.size} float32 values")

        for sock in clients:
            try:
                send_model(sock, global_model)
            finally:
                sock.close()

        return global_model

    finally:
        server.close()


if __name__ == "__main__":
    global_model = federate_round(2)
    print("Global anomaly model produced successfully.")
