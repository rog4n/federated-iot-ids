#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <arpa/inet.h>
#include <stdint.h>
#include <math.h>
#include <errno.h>
#include <sys/time.h>

#ifdef PC_LIVE_CAPTURE
#include <pcap.h>
#endif

#define FORWARD_IP   "127.0.0.1"
#define FORWARD_PORT 8080

#define FEATURE_D                10
#define COV_ELEMS               (FEATURE_D * (FEATURE_D + 1) / 2)
#define MODEL_PAYLOAD_FLOATS   (1 + FEATURE_D + FEATURE_D + FEATURE_D + COV_ELEMS + 1)

/* PC live-traffic prototype. Tune these after benchmarking. */
#define MAX_FLOWS               64
#define OBS_MAX_PACKETS         20
#define OBS_MAX_US              100000ULL   /* 100 ms */
#define FLOW_IDLE_US            1000000ULL  /* 1 s */

/* Initial model is learned only from operator-verified safe live windows. */
#define BOOTSTRAP_TRAIN_WINDOWS 256
#define BOOTSTRAP_VALID_WINDOWS 128
#define BOOTSTRAP_THRESHOLD_Q   0.99f

#define TRAIN_FIFO_SIZE         128
#define ADAPT_MIN_SAMPLES       64
#define ADAPT_UPDATE_INTERVAL   32

#define DEFAULT_MODEL_FILE      "local_anomaly_model.bin"
#define MODEL_FILE_VERSION      1u
#define MODEL_MAGIC             "AIDSLM01"

/* -------------------- MODEL -------------------- */

typedef struct {
    uint32_t n;
    float norm_mean[FEATURE_D];
    float norm_std[FEATURE_D];
    float mean[FEATURE_D];
    float M2[FEATURE_D][FEATURE_D];
    float cov[FEATURE_D][FEATURE_D];
    float L[FEATURE_D][FEATURE_D];
    float threshold;
} AnomalyModel;

static void model_reset_stats(AnomalyModel *m)
{
    float keep_threshold = m->threshold;

    memset(m->mean, 0, sizeof(m->mean));
    memset(m->M2, 0, sizeof(m->M2));
    memset(m->cov, 0, sizeof(m->cov));
    memset(m->L, 0, sizeof(m->L));
    m->n = 0;
    m->threshold = keep_threshold;
}

static void model_update(AnomalyModel *m, const float *x)
{
    m->n++;

    float delta[FEATURE_D];
    float delta2[FEATURE_D];

    for (uint32_t i = 0; i < FEATURE_D; i++) {
        delta[i] = x[i] - m->mean[i];
        m->mean[i] += delta[i] / (float)m->n;
        delta2[i] = x[i] - m->mean[i];
    }

    for (uint32_t i = 0; i < FEATURE_D; i++) {
        for (uint32_t j = 0; j < FEATURE_D; j++)
            m->M2[i][j] += delta[i] * delta2[j];
    }
}

static int model_finalize(AnomalyModel *m)
{
    if (m->n < 2)
        return -1;

    memset(m->cov, 0, sizeof(m->cov));
    memset(m->L, 0, sizeof(m->L));

    for (uint32_t i = 0; i < FEATURE_D; i++) {
        for (uint32_t j = 0; j < FEATURE_D; j++) {
            m->cov[i][j] = m->M2[i][j] / (float)(m->n - 1);
            if (i == j)
                m->cov[i][j] += 1e-3f;
        }
    }

    for (uint32_t i = 0; i < FEATURE_D; i++) {
        for (uint32_t j = 0; j <= i; j++) {
            float sum = m->cov[i][j];
            for (uint32_t k = 0; k < j; k++)
                sum -= m->L[i][k] * m->L[j][k];

            if (i == j) {
                if (sum <= 0.0f)
                    return -1;
                m->L[i][j] = sqrtf(sum);
            } else {
                if (m->L[j][j] == 0.0f)
                    return -1;
                m->L[i][j] = sum / m->L[j][j];
            }
        }
    }

    return 0;
}

static float mahalanobis_score(const AnomalyModel *m, const float *x)
{
    float diff[FEATURE_D];
    float z[FEATURE_D];

    for (uint32_t i = 0; i < FEATURE_D; i++)
        diff[i] = x[i] - m->mean[i];

    for (uint32_t i = 0; i < FEATURE_D; i++) {
        float sum = diff[i];
        for (uint32_t j = 0; j < i; j++)
            sum -= m->L[i][j] * z[j];
        z[i] = sum / m->L[i][i];
    }

    float score = 0.0f;
    for (uint32_t i = 0; i < FEATURE_D; i++)
        score += z[i] * z[i];

    return score;
}

/* -------------------- LOCAL FEATURE TRANSFORMS + SAFE BOOTSTRAP -------------------- */

/* These are feature transforms, not dataset-specific normalizers. The mean/std
 * used for z-scoring are learned from verified-safe traffic on the target network. */
static float feature_log(float x)
{
    if (x < 0.0f)
        x = 0.0f;
    return log1pf(x);
}

static void normalize_features(const AnomalyModel *model,
                               const float *raw,
                               float *x)
{
    for (uint32_t i = 0; i < FEATURE_D; i++) {
        float sd = model->norm_std[i];
        if (sd < 1e-6f)
            sd = 1.0f;
        x[i] = (raw[i] - model->norm_mean[i]) / sd;
    }
}

static void pack_model_payload(const AnomalyModel *m, float *payload)
{
    uint32_t p = 0;
    payload[p++] = (float)m->n;

    for (uint32_t i = 0; i < FEATURE_D; i++)
        payload[p++] = m->norm_mean[i];

    for (uint32_t i = 0; i < FEATURE_D; i++)
        payload[p++] = m->norm_std[i];

    for (uint32_t i = 0; i < FEATURE_D; i++)
        payload[p++] = m->mean[i];

    for (uint32_t i = 0; i < FEATURE_D; i++)
        for (uint32_t j = 0; j <= i; j++)
            payload[p++] = m->cov[i][j];

    payload[p++] = m->threshold;
}

static int unpack_model_payload(const float *payload, size_t count,
                                AnomalyModel *m)
{
    if (count != MODEL_PAYLOAD_FLOATS)
        return -1;

    uint32_t p = 0;
    m->n = (uint32_t)(payload[p++] + 0.5f);

    for (uint32_t i = 0; i < FEATURE_D; i++)
        m->norm_mean[i] = payload[p++];

    for (uint32_t i = 0; i < FEATURE_D; i++) {
        m->norm_std[i] = payload[p++];
        if (!isfinite(m->norm_std[i]) || m->norm_std[i] <= 0.0f)
            return -1;
    }

    for (uint32_t i = 0; i < FEATURE_D; i++)
        m->mean[i] = payload[p++];

    memset(m->cov, 0, sizeof(m->cov));
    for (uint32_t i = 0; i < FEATURE_D; i++) {
        for (uint32_t j = 0; j <= i; j++) {
            float v = payload[p++];
            m->cov[i][j] = v;
            m->cov[j][i] = v;
        }
    }

    m->threshold = payload[p++];

    memset(m->L, 0, sizeof(m->L));
    for (uint32_t i = 0; i < FEATURE_D; i++) {
        for (uint32_t j = 0; j <= i; j++) {
            float sum = m->cov[i][j];
            for (uint32_t k = 0; k < j; k++)
                sum -= m->L[i][k] * m->L[j][k];

            if (i == j) {
                if (sum <= 0.0f)
                    return -1;
                m->L[i][j] = sqrtf(sum);
            } else {
                if (m->L[j][j] == 0.0f)
                    return -1;
                m->L[i][j] = sum / m->L[j][j];
            }
        }
    }

    return 0;
}

static int save_local_model(const char *path, const AnomalyModel *m)
{
    float payload[MODEL_PAYLOAD_FLOATS];
    pack_model_payload(m, payload);

    FILE *fp = fopen(path, "wb");
    if (!fp) {
        perror("Failed to save local model");
        return -1;
    }

    uint32_t version = MODEL_FILE_VERSION;
    uint32_t count = MODEL_PAYLOAD_FLOATS;

    int ok = 0;
    if (fwrite(MODEL_MAGIC, 1, 8, fp) != 8 ||
        fwrite(&version, sizeof(version), 1, fp) != 1 ||
        fwrite(&count, sizeof(count), 1, fp) != 1 ||
        fwrite(payload, sizeof(float), MODEL_PAYLOAD_FLOATS, fp) != MODEL_PAYLOAD_FLOATS)
        ok = -1;

    fclose(fp);
    return ok;
}

static int load_local_model(const char *path, AnomalyModel *m)
{
    FILE *fp = fopen(path, "rb");
    if (!fp) {
        perror("Failed to open local model");
        return -1;
    }

    char magic[8];
    uint32_t version = 0;
    uint32_t count = 0;
    float payload[MODEL_PAYLOAD_FLOATS];

    int ok = 0;
    if (fread(magic, 1, 8, fp) != 8 ||
        fread(&version, sizeof(version), 1, fp) != 1 ||
        fread(&count, sizeof(count), 1, fp) != 1 ||
        count != MODEL_PAYLOAD_FLOATS ||
        fread(payload, sizeof(float), MODEL_PAYLOAD_FLOATS, fp) != MODEL_PAYLOAD_FLOATS) {
        ok = -1;
    }

    fclose(fp);

    if (ok != 0 || memcmp(magic, MODEL_MAGIC, 8) != 0 ||
        version != MODEL_FILE_VERSION)
        return -1;

    for (uint32_t i = 0; i < MODEL_PAYLOAD_FLOATS; i++) {
        if (!isfinite(payload[i]))
            return -1;
    }

    return unpack_model_payload(payload, MODEL_PAYLOAD_FLOATS, m);
}

typedef struct {
    float raw[BOOTSTRAP_TRAIN_WINDOWS][FEATURE_D];
    uint32_t count;
    float validation_scores[BOOTSTRAP_VALID_WINDOWS];
    uint32_t validation_count;
    char model_path[256];
    int phase; /* 0=train, 1=validate, 2=done */
} SafeBootstrap;

static void safe_bootstrap_init(SafeBootstrap *b, const char *model_path)
{
    memset(b, 0, sizeof(*b));
    b->phase = 0;
    snprintf(b->model_path, sizeof(b->model_path), "%s", model_path);
}

static int build_model_from_safe_windows(AnomalyModel *model,
                                         const SafeBootstrap *b)
{
    if (b->count < 2)
        return -1;

    double mean[FEATURE_D] = {0};
    double M2[FEATURE_D][FEATURE_D] = {{0}};

    uint32_t n = 0;
    for (uint32_t s = 0; s < b->count; s++) {
        n++;
        double delta[FEATURE_D];
        double delta2[FEATURE_D];

        for (uint32_t i = 0; i < FEATURE_D; i++) {
            delta[i] = (double)b->raw[s][i] - mean[i];
            mean[i] += delta[i] / (double)n;
            delta2[i] = (double)b->raw[s][i] - mean[i];
        }

        for (uint32_t i = 0; i < FEATURE_D; i++)
            for (uint32_t j = 0; j < FEATURE_D; j++)
                M2[i][j] += delta[i] * delta2[j];
    }

    memset(model, 0, sizeof(*model));
    model->threshold = 0.0f;

    for (uint32_t i = 0; i < FEATURE_D; i++) {
        model->norm_mean[i] = (float)mean[i];
        double var = M2[i][i] / (double)(n - 1);
        if (var < 1e-8)
            var = 1.0; /* constant safe feature: give it a neutral scale */
        model->norm_std[i] = (float)sqrt(var);
    }

    /* Rebuild the Gaussian model in the new local normalized space. */
    for (uint32_t s = 0; s < b->count; s++) {
        float x[FEATURE_D];
        normalize_features(model, b->raw[s], x);
        model_update(model, x);
    }

    return model_finalize(model);
}

static int float_compare(const void *a, const void *b)
{
    const float fa = *(const float *)a;
    const float fb = *(const float *)b;
    return (fa > fb) - (fa < fb);
}

static float percentile99(const float *scores, uint32_t n)
{
    if (n == 0)
        return 0.0f;

    /* Small fixed-size temporary buffer: bootstrap uses 128 samples and
     * continual adaptation uses 64 calibration samples. */
    float tmp[BOOTSTRAP_VALID_WINDOWS];
    if (n > BOOTSTRAP_VALID_WINDOWS)
        n = BOOTSTRAP_VALID_WINDOWS;

    memcpy(tmp, scores, n * sizeof(float));
    qsort(tmp, n, sizeof(float), float_compare);

    uint32_t idx = (uint32_t)ceilf(BOOTSTRAP_THRESHOLD_Q * (float)(n - 1));
    if (idx >= n)
        idx = n - 1;
    return tmp[idx];
}

/* -------------------- CONTINUAL-LEARNING FIFO -------------------- */

/*
 * Adaptation policy:
 *   - Keep the most recent 128 NORMAL windows in a rolling FIFO.
 *   - Use the oldest 64 windows in that FIFO to rebuild the Gaussian
 *     (mean + covariance).
 *   - Use the newest 64 windows as a held-out normal calibration set.
 *   - Score those 64 windows with the NEW Gaussian and set the threshold
 *     to their P99 score.
 *   - The first adaptation happens when 128 windows are available.
 *   - After that, adaptation happens every 32 newly accepted normal windows.
 */
typedef struct {
    float sample[TRAIN_FIFO_SIZE][FEATURE_D];
    uint32_t head;
    uint32_t count;
    uint32_t since_rebuild;
} TrainingFIFO;

static void fifo_push(TrainingFIFO *f, const float *x)
{
    memcpy(f->sample[f->head], x, FEATURE_D * sizeof(float));
    f->head = (f->head + 1) % TRAIN_FIFO_SIZE;
    if (f->count < TRAIN_FIFO_SIZE)
        f->count++;
    f->since_rebuild++;
}

static const float *training_fifo_ptr(const TrainingFIFO *f, uint32_t logical_index)
{
    uint32_t oldest =
        (f->head + TRAIN_FIFO_SIZE - f->count) % TRAIN_FIFO_SIZE;
    uint32_t idx = (oldest + logical_index) % TRAIN_FIFO_SIZE;
    return f->sample[idx];
}

static int adapt_from_fifo(AnomalyModel *model, TrainingFIFO *fifo)
{
    if (fifo->count < TRAIN_FIFO_SIZE ||
        fifo->since_rebuild < ADAPT_UPDATE_INTERVAL)
        return 0;

    AnomalyModel updated = *model;
    float old_threshold = model->threshold;

    /* Train the NEW Gaussian using the oldest 64 windows. */
    model_reset_stats(&updated);
    updated.threshold = old_threshold;

    for (uint32_t i = 0; i < ADAPT_MIN_SAMPLES; i++)
        model_update(&updated, training_fifo_ptr(fifo, i));

    if (model_finalize(&updated) != 0)
        return -1;

    /* Calibrate the NEW threshold using a separate held-out half of the FIFO. */
    float calibration_scores[ADAPT_MIN_SAMPLES];
    for (uint32_t i = 0; i < ADAPT_MIN_SAMPLES; i++) {
        const float *x = training_fifo_ptr(fifo, ADAPT_MIN_SAMPLES + i);
        calibration_scores[i] = mahalanobis_score(&updated, x);
    }

    float new_threshold = percentile99(calibration_scores, ADAPT_MIN_SAMPLES);
    if (new_threshold < 1e-6f)
        new_threshold = 1e-6f;

    updated.threshold = new_threshold;
    *model = updated;
    fifo->since_rebuild = 0;

    printf("[ADAPT] Gaussian rebuilt from %u normal windows; "
           "threshold calibrated from %u held-out windows: %.6f -> %.6f\n",
           ADAPT_MIN_SAMPLES, ADAPT_MIN_SAMPLES,
           old_threshold, new_threshold);
    return 1;
}

/* -------------------- FEDERATION SOCKET -------------------- */

static int write_exact(int sock, const void *buf, size_t len)
{
    size_t sent = 0;
    while (sent < len) {
        ssize_t n = write(sock, (const char *)buf + sent, len - sent);
        if (n <= 0)
            return -1;
        sent += (size_t)n;
    }
    return 0;
}

static int read_exact(int sock, void *buf, size_t len)
{
    size_t got = 0;
    while (got < len) {
        ssize_t n = read(sock, (char *)buf + got, len - got);
        if (n <= 0)
            return -1;
        got += (size_t)n;
    }
    return 0;
}

static uint64_t my_htonll(uint64_t x)
{
    uint32_t hi = htonl((uint32_t)(x >> 32));
    uint32_t lo = htonl((uint32_t)(x & 0xffffffffULL));
    return ((uint64_t)lo << 32) | hi;
}

static int receive_global_model(int sock, AnomalyModel *m)
{
    uint64_t net_size;
    if (read_exact(sock, &net_size, sizeof(net_size)) < 0)
        return -1;

    uint64_t payload_size = __builtin_bswap64(net_size);
    if (payload_size < 10 || payload_size > 1024 * 1024)
        return -1;

    unsigned char *payload = malloc((size_t)payload_size);
    if (!payload)
        return -1;

    if (read_exact(sock, payload, (size_t)payload_size) < 0) {
        free(payload);
        return -1;
    }

    if (payload[0] != 0x93 || payload[1] != 'N' || payload[2] != 'U' ||
        payload[3] != 'M' || payload[4] != 'P' || payload[5] != 'Y' ||
        payload[6] != 1 || payload[7] != 0) {
        free(payload);
        return -1;
    }

    uint16_t header_len = (uint16_t)payload[8] |
                          ((uint16_t)payload[9] << 8);
    uint64_t offset = 10 + header_len;
    uint64_t expected = (uint64_t)MODEL_PAYLOAD_FLOATS * sizeof(float);

    if (offset + expected != payload_size) {
        free(payload);
        return -1;
    }

    const float *data = (const float *)(payload + offset);
    int result = unpack_model_payload(data, MODEL_PAYLOAD_FLOATS, m);
    free(payload);
    return result;
}

static int forward_model(AnomalyModel *m)
{
    float payload[MODEL_PAYLOAD_FLOATS];
    pack_model_payload(m, payload);

    int sock = socket(AF_INET, SOCK_STREAM, 0);
    if (sock < 0) {
        perror("socket");
        return -1;
    }

    struct sockaddr_in dest = {
        .sin_family = AF_INET,
        .sin_port = htons(FORWARD_PORT)
    };

    if (inet_pton(AF_INET, FORWARD_IP, &dest.sin_addr) != 1 ||
        connect(sock, (struct sockaddr *)&dest, sizeof(dest)) < 0) {
        perror("federator connect");
        close(sock);
        return -1;
    }

    char header[128];
    int dict_len = snprintf(
        header, sizeof(header),
        "{'descr': '<f4', 'fortran_order': False, 'shape': (%u,), }",
        MODEL_PAYLOAD_FLOATS
    );
    if (dict_len < 0 || dict_len >= (int)sizeof(header)) {
        close(sock);
        return -1;
    }

    int padding = 16 - ((10 + dict_len + 1) % 16);
    if (padding == 16)
        padding = 0;
    memset(header + dict_len, ' ', (size_t)padding);
    header[dict_len + padding] = '\n';
    uint16_t header_total = (uint16_t)(dict_len + padding + 1);

    uint64_t payload_size = 10 + header_total +
                            (uint64_t)sizeof(payload);
    uint64_t net_size = my_htonll(payload_size);
    const unsigned char magic[6] = {0x93, 'N', 'U', 'M', 'P', 'Y'};
    const unsigned char version[2] = {1, 0};

    if (write_exact(sock, &net_size, sizeof(net_size)) < 0 ||
        write_exact(sock, magic, 6) < 0 ||
        write_exact(sock, version, 2) < 0 ||
        write_exact(sock, &header_total, sizeof(header_total)) < 0 ||
        write_exact(sock, header, header_total) < 0 ||
        write_exact(sock, payload, sizeof(payload)) < 0) {
        close(sock);
        return -1;
    }

    /* Same TCP connection is used by the server to return the global model. */
    int result = receive_global_model(sock, m);
    close(sock);
    return result;
}

/* -------------------- PC LIVE TRAFFIC -------------------- */

#ifdef PC_LIVE_CAPTURE

typedef struct {
    int used;
    int blocked;

    uint32_t src_ip;
    uint32_t dst_ip;
    uint16_t src_port;
    uint16_t dst_port;
    uint8_t proto;

    uint64_t window_start_us;
    uint64_t last_us;
    uint64_t last_src_us;
    uint64_t last_dst_us;

    uint32_t window_packets;
    uint64_t spkts;
    uint64_t dpkts;
    uint64_t sbytes;
    uint64_t dbytes;
    uint64_t src_gap_sum_us;
    uint64_t dst_gap_sum_us;
    uint32_t src_gap_n;
    uint32_t dst_gap_n;

    uint8_t sttl;
    uint8_t dttl;

    uint8_t syn_seen;
    uint8_t synack_seen;
    uint8_t ack_seen;
    uint8_t fin_seen;
    uint8_t fin_src_seen;
    uint8_t fin_dst_seen;
    uint8_t rst_seen;
} Flow;

typedef struct {
    uint32_t src_ip;
    uint32_t dst_ip;
    uint16_t src_port;
    uint16_t dst_port;
    uint8_t proto;
    uint8_t ttl;
    uint8_t tcp_flags;
    uint16_t ip_len;
    uint64_t ts_us;
} ParsedPacket;

static uint64_t packet_time_us(const struct pcap_pkthdr *h)
{
    return (uint64_t)h->ts.tv_sec * 1000000ULL +
           (uint64_t)h->ts.tv_usec;
}

static int parse_ipv4_packet(const unsigned char *p, size_t caplen,
                             uint64_t ts_us, int linktype,
                             ParsedPacket *out)
{
    size_t off = 0;

    if (linktype == DLT_NULL) {
        if (caplen < 4)
            return -1;
        off = 4;              // BSD loopback header
    }
    else if (linktype == DLT_EN10MB) {
        if (caplen < 14)
            return -1;

        uint16_t ether = ((uint16_t)p[12] << 8) | p[13];
        off = 14;

        if (ether == 0x8100 || ether == 0x88a8) {
            if (caplen < 18)
                return -1;
            ether = ((uint16_t)p[16] << 8) | p[17];
            off = 18;
        }

        if (ether != 0x0800)
            return 0;
    }
    else if (linktype == DLT_RAW) {
        off = 0;
    }
    else {
        return -1;
    }

    /* existing IPv4 parsing continues here */

    uint8_t version = p[off] >> 4;
    uint8_t ihl = (p[off] & 0x0f) * 4;
    if (version != 4 || ihl < 20 || caplen < off + ihl)
        return -1;

    uint16_t frag = ntohs(*(const uint16_t *)(p + off + 6));
    if ((frag & 0x1fff) != 0)
        return 0; /* ignore non-first fragments */

    uint16_t total_len = ntohs(*(const uint16_t *)(p + off + 2));
    if (total_len < ihl)
        return -1;
    if (off + total_len > caplen)
        total_len = (uint16_t)(caplen - off);

    out->src_ip = *(const uint32_t *)(p + off + 12);
    out->dst_ip = *(const uint32_t *)(p + off + 16);
    out->ttl = p[off + 8];
    out->proto = p[off + 9];
    out->ip_len = total_len;
    out->ts_us = ts_us;
    out->tcp_flags = 0;

    const unsigned char *l4 = p + off + ihl;
    size_t l4_len = total_len - ihl;

    if (out->proto == IPPROTO_TCP) {
        if (l4_len < 20)
            return -1;
        out->src_port = ntohs(*(const uint16_t *)(l4 + 0));
        out->dst_port = ntohs(*(const uint16_t *)(l4 + 2));
        out->tcp_flags = l4[13];
    } else if (out->proto == IPPROTO_UDP) {
        if (l4_len < 8)
            return -1;
        out->src_port = ntohs(*(const uint16_t *)(l4 + 0));
        out->dst_port = ntohs(*(const uint16_t *)(l4 + 2));
    } else {
        return 0; /* prototype tracks TCP/UDP flows */
    }
    
    return 1;
}

static int same_flow_direction(const Flow *f, const ParsedPacket *p)
{
    return f->src_ip == p->src_ip && f->dst_ip == p->dst_ip &&
           f->src_port == p->src_port && f->dst_port == p->dst_port &&
           f->proto == p->proto;
}

static int reverse_flow_direction(const Flow *f, const ParsedPacket *p)
{
    return f->src_ip == p->dst_ip && f->dst_ip == p->src_ip &&
           f->src_port == p->dst_port && f->dst_port == p->src_port &&
           f->proto == p->proto;
}

static void flow_reset_window(Flow *f, uint64_t now_us)
{
    /* Reset only per-observation-window statistics.
     * TCP connection state is intentionally preserved across windows. */
    f->window_start_us = now_us;
    f->last_src_us = 0;
    f->last_dst_us = 0;
    f->window_packets = 0;
    f->spkts = 0;
    f->dpkts = 0;
    f->sbytes = 0;
    f->dbytes = 0;
    f->src_gap_sum_us = 0;
    f->dst_gap_sum_us = 0;
    f->src_gap_n = 0;
    f->dst_gap_n = 0;
    /* sttl/dttl are endpoint properties and persist across observation windows. */
}

/* Return non-zero when the first observed packet appears to be
 * server -> client and the stored flow should therefore be reversed
 * into client -> server canonical orientation. */
static int is_likely_server_port(uint16_t port)
{
    switch (port) {
    case 20: case 21: case 22: case 23: case 25:
    case 53: case 67: case 68: case 80: case 110:
    case 123: case 143: case 443: case 465: case 587:
    case 993: case 995: case 1812: case 1813:
        return 1;
    default:
        return 0;
    }
}

static int packet_is_server_to_client(const ParsedPacket *p)
{
    if (p->proto == IPPROTO_TCP) {
        const uint8_t syn = p->tcp_flags & 0x02;
        const uint8_t ack = p->tcp_flags & 0x10;
        if (syn && ack)
            return 1; /* SYN-ACK: server -> client */
        if (syn && !ack)
            return 0; /* SYN: client -> server */
    }

    int src_is_server = is_likely_server_port(p->src_port);
    int dst_is_server = is_likely_server_port(p->dst_port);
    if (src_is_server && !dst_is_server) return 1;
    if (dst_is_server && !src_is_server) return 0;

    if (p->src_port < 1024 && p->dst_port >= 1024) return 1;
    if (p->dst_port < 1024 && p->src_port >= 1024) return 0;

    return 0; /* ambiguous: first-seen direction */
}

static void flow_init(Flow *f, const ParsedPacket *p, int *src_direction)
{
    memset(f, 0, sizeof(*f));
    f->used = 1;
    f->proto = p->proto;
    f->last_us = p->ts_us;

    if (packet_is_server_to_client(p)) {
        /* Store every flow canonically as client -> server. */
        f->src_ip = p->dst_ip;
        f->dst_ip = p->src_ip;
        f->src_port = p->dst_port;
        f->dst_port = p->src_port;
        *src_direction = 0;
    } else {
        f->src_ip = p->src_ip;
        f->dst_ip = p->dst_ip;
        f->src_port = p->src_port;
        f->dst_port = p->dst_port;
        *src_direction = 1;
    }

    flow_reset_window(f, p->ts_us);
}

enum {
    FLOW_STATE_FIN = 0,
    FLOW_STATE_INT = 1,
    FLOW_STATE_CON = 2,
    FLOW_STATE_REQ = 3,
    FLOW_STATE_ACC = 4,
    FLOW_STATE_CLO = 5,
    FLOW_STATE_RST = 6
};

static uint8_t flow_state_index(const Flow *f)
{
    if (f->proto != IPPROTO_TCP)
        return FLOW_STATE_INT;

    if (f->rst_seen)
        return FLOW_STATE_RST;

    /* Both directions have sent FIN: treat the connection as closed. */
    if (f->fin_src_seen && f->fin_dst_seen)
        return FLOW_STATE_CLO;

    if (f->fin_seen)
        return FLOW_STATE_FIN;

    if (f->syn_seen && !f->synack_seen)
        return FLOW_STATE_REQ;

    if (f->synack_seen && f->ack_seen)
        return FLOW_STATE_CON;

    if (f->ack_seen)
        return FLOW_STATE_ACC;

    return FLOW_STATE_INT;
}

/*
 * UNSW-NB15 ct_state_ttl is a deterministic feature generated from the
 * current connection's state, source TTL and destination TTL. It is NOT a
 * history/count of previously accepted normal flows. The mapping below is
 * the published feature-generation rule used for UNSW-NB15.
 */
static uint32_t ct_state_ttl_value(uint8_t state, uint8_t sttl, uint8_t dttl)
{
    if (state == FLOW_STATE_FIN &&
        (sttl == 62 || sttl == 63 || sttl == 254 || sttl == 255) &&
        (dttl == 252 || dttl == 253))
        return 1;

    if (state == FLOW_STATE_INT &&
        (sttl == 0 || sttl == 62 || sttl == 254) &&
        dttl == 0)
        return 2;

    if (state == FLOW_STATE_CON &&
        (sttl == 62 || sttl == 254) &&
        (dttl == 60 || dttl == 252 || dttl == 253))
        return 3;

    if (state == FLOW_STATE_ACC &&
        sttl == 254 && dttl == 252)
        return 4;

    if (state == FLOW_STATE_CLO &&
        sttl == 254 && dttl == 252)
        return 5;

    if (state == FLOW_STATE_REQ &&
        sttl == 254 && dttl == 0)
        return 6;

    return 0;
}

static void flow_build_raw_features(const Flow *f, uint32_t ct_state_ttl,
                                    float *raw)
{
    memset(raw, 0, FEATURE_D * sizeof(float));

    uint64_t duration_us = f->last_us > f->window_start_us ?
                           f->last_us - f->window_start_us : 0;
    float duration_s = (float)duration_us / 1000000.0f;
    if (duration_s < 1e-6f)
        duration_s = 1e-6f;

    float smean = f->spkts ? (float)f->sbytes / (float)f->spkts : 0.0f;
    float dmean = f->dpkts ? (float)f->dbytes / (float)f->dpkts : 0.0f;
    float sload = ((float)f->sbytes * 8.0f) / duration_s;
    float rate = ((float)(f->spkts + f->dpkts)) / duration_s;
    float dinpkt_ms = f->dst_gap_n ?
        ((float)f->dst_gap_sum_us / (float)f->dst_gap_n) / 1000.0f : 0.0f;

    raw[0] = feature_log((float)f->sbytes);
    raw[1] = smean;
    raw[2] = feature_log(sload);
    raw[3] = feature_log((float)f->dbytes);
    raw[4] = (float)ct_state_ttl;
    raw[5] = feature_log(rate);
    raw[6] = feature_log(dmean);
    raw[7] = (float)f->dttl;
    raw[8] = feature_log(dinpkt_ms);
    raw[9] = (f->proto == IPPROTO_TCP) ? 1.0f : 0.0f;
}

static void print_normalization(const AnomalyModel *model)
{
    printf("\nLOCAL NORMALIZATION (learned from verified-safe traffic):\n");
    for (uint32_t i = 0; i < FEATURE_D; i++)
        printf("feature[%u]: mean=%.6f std=%.6f\n",
               i, model->norm_mean[i], model->norm_std[i]);
}

static int evaluate_flow_window(Flow *f, AnomalyModel *model,
                                 TrainingFIFO *fifo,
                                 SafeBootstrap *bootstrap,
                                 int learning_enabled,
                                 const char *model_path)
{
    if (!f->used || f->window_packets < 2)
        return 0;

    uint64_t duration_us = f->last_us - f->window_start_us;
    if (duration_us == 0)
        return 0;

    /* ct_state_ttl is a deterministic mapping of this flow's current
     * state + source TTL + destination TTL. No history is involved. */
    uint8_t state = flow_state_index(f);
    uint32_t ct_state_ttl = ct_state_ttl_value(state, f->sttl, f->dttl);

    float raw[FEATURE_D];
    flow_build_raw_features(f, ct_state_ttl, raw);

    /* ---------------------------------------------------------
     * PHASE 1: initial model from operator-verified safe windows
     * --------------------------------------------------------- */
    if (bootstrap && bootstrap->phase == 0) {
        if (bootstrap->count < BOOTSTRAP_TRAIN_WINDOWS) {
            memcpy(bootstrap->raw[bootstrap->count], raw, sizeof(raw));
            bootstrap->count++;
        }

        printf("[BOOTSTRAP TRAIN] safe window %u/%u\n",
               bootstrap->count, BOOTSTRAP_TRAIN_WINDOWS);

        if (bootstrap->count == BOOTSTRAP_TRAIN_WINDOWS) {
            if (build_model_from_safe_windows(model, bootstrap) != 0) {
                fprintf(stderr, "[BOOTSTRAP] local model construction failed\n");
                return -1;
            }

            bootstrap->phase = 1;
            bootstrap->validation_count = 0;
            print_normalization(model);
            printf("[BOOTSTRAP] training complete; collecting %u safe validation windows\n",
                   BOOTSTRAP_VALID_WINDOWS);
        }
        return 0;
    }

    /* ---------------------------------------------------------
     * PHASE 2: threshold calibration on separate safe windows
     * --------------------------------------------------------- */
    if (bootstrap && bootstrap->phase == 1) {
        float x[FEATURE_D];
        normalize_features(model, raw, x);
        float score = mahalanobis_score(model, x);

        if (bootstrap->validation_count < BOOTSTRAP_VALID_WINDOWS) {
            bootstrap->validation_scores[bootstrap->validation_count++] = score;
        }

        printf("[BOOTSTRAP VALIDATE] safe window %u/%u score=%.6f\n",
               bootstrap->validation_count, BOOTSTRAP_VALID_WINDOWS, score);

        if (bootstrap->validation_count == BOOTSTRAP_VALID_WINDOWS) {
            model->threshold = percentile99(bootstrap->validation_scores,
                                             BOOTSTRAP_VALID_WINDOWS);
            if (model->threshold < 1e-6f)
                model->threshold = 1e-6f;

            if (save_local_model(bootstrap->model_path, model) != 0) {
                fprintf(stderr, "[BOOTSTRAP] WARNING: failed to save model to %s\n",
                        bootstrap->model_path);
            } else {
                printf("[BOOTSTRAP] model saved to %s\n", bootstrap->model_path);
            }

            bootstrap->phase = 2;
            printf("[BOOTSTRAP] initial local model ready. Threshold (P%.1f safe score): %.6f\n",
                   BOOTSTRAP_THRESHOLD_Q * 100.0f, model->threshold);
            print_normalization(model);
        }
        return 0;
    }

    /* ---------------------------------------------------------
     * NORMAL DETECTION PHASE
     * --------------------------------------------------------- */
    float x[FEATURE_D];
    normalize_features(model, raw, x);

    float score_all = mahalanobis_score(model, x);

    /* Diagnostic: replace the current ct_state_ttl coordinate with the
     * model's current mean in normalized space. This removes that feature's
     * deviation from the Mahalanobis calculation. */
    float x_without_ct[FEATURE_D];
    memcpy(x_without_ct, x, sizeof(x_without_ct));
    x_without_ct[4] = model->mean[4];
    float score_without_ct = mahalanobis_score(model, x_without_ct);

    static const char *feature_names[FEATURE_D] = {
        "sbytes", "smean", "sload", "dbytes", "ct_state_ttl",
        "rate", "dmean", "dttl", "dinpkt", "proto"
    };

    printf("\n========== FEATURE ABLATION ==========\n");
    printf("score_all         = %.6f\n", score_all);
    printf("score_without_ct  = %.6f\n", score_without_ct);
    printf("ct_state_ttl effect= %.6f\n", score_all - score_without_ct);
    printf("\n%-16s %-14s %-18s\n",
           "feature", "live x", "model mean");
    printf("-----------------------------------------------\n");
    for (int i = 0; i < FEATURE_D; i++)
        printf("%-16s %-14.6f %-18.6f\n",
               feature_names[i], x[i], model->mean[i]);
    printf("=======================================\n");

    float score = score_all;
    int anomaly = (score > model->threshold);

    struct in_addr src_addr = {f->src_ip};
    struct in_addr dst_addr = {f->dst_ip};
    char src_buf[INET_ADDRSTRLEN];
    char dst_buf[INET_ADDRSTRLEN];

    inet_ntop(AF_INET, &src_addr, src_buf, sizeof(src_buf));
    inet_ntop(AF_INET, &dst_addr, dst_buf, sizeof(dst_buf));

    if (anomaly) {
        f->blocked = 1;
        printf("[DROP DECISION] %s:%u -> %s:%u proto=%u score=%.3f threshold=%.3f\n",
               src_buf, f->src_port, dst_buf, f->dst_port,
               f->proto, score, model->threshold);
        return 1;
    }

    printf("[ALLOW] %s:%u -> %s:%u proto=%u score=%.3f threshold=%.3f\n",
           src_buf, f->src_port, dst_buf, f->dst_port,
           f->proto, score, model->threshold);

    /* Continual learning/adaptation is enabled by capture mode.
     * Only windows classified as NORMAL are admitted to the recent-normal FIFO.
     * Anomalous windows are never added. Rebuilds occur periodically from
     * the FIFO contents. */
    if (learning_enabled) {
        fifo_push(fifo, x);

        int adapted = adapt_from_fifo(model, fifo);
        if (adapted < 0)
            fprintf(stderr, "[ADAPT] model rebuild failed\n");
        else if (adapted > 0) {
            printf("[ADAPT] updated model from operator-approved safe windows\n");
            if (model_path && save_local_model(model_path, model) != 0)
                fprintf(stderr, "[ADAPT] warning: failed to save updated model\n");
        }
    }

    return 0;
}

static Flow *find_flow(Flow *flows, const ParsedPacket *p, int *src_direction)
{
    for (uint32_t i = 0; i < MAX_FLOWS; i++) {
        if (!flows[i].used)
            continue;
        if (same_flow_direction(&flows[i], p)) {
            *src_direction = 1;
            return &flows[i];
        }
        if (reverse_flow_direction(&flows[i], p)) {
            *src_direction = 0;
            return &flows[i];
        }
    }
    return NULL;
}

static Flow *allocate_flow(Flow *flows, const ParsedPacket *p, int *src_direction,
                           AnomalyModel *model, TrainingFIFO *fifo,
                           SafeBootstrap *bootstrap, int learning_enabled,
                           const char *model_path)
{
    for (uint32_t i = 0; i < MAX_FLOWS; i++) {
        if (!flows[i].used) {
            flow_init(&flows[i], p, src_direction);
            return &flows[i];
        }
    }

    /* Flow table full: evict the least-recently-seen flow. */
    uint32_t oldest = 0;
    for (uint32_t i = 1; i < MAX_FLOWS; i++) {
        if (flows[i].last_us < flows[oldest].last_us)
            oldest = i;
    }

    if (!flows[oldest].blocked)
        (void)evaluate_flow_window(&flows[oldest], model, fifo, bootstrap, learning_enabled, model_path);

    flow_init(&flows[oldest], p, src_direction);
    return &flows[oldest];
}

static void update_flow(Flow *f, const ParsedPacket *p, int src_direction,
                        AnomalyModel *model, TrainingFIFO *fifo,
                        SafeBootstrap *bootstrap, int learning_enabled,
                        const char *model_path)
{
    f->last_us = p->ts_us;
    f->window_packets++;

    if (src_direction) {
        f->spkts++;
        f->sbytes += p->ip_len;
        if (f->sttl == 0)
            f->sttl = p->ttl;
        if (f->last_src_us != 0 && p->ts_us >= f->last_src_us) {
            f->src_gap_sum_us += p->ts_us - f->last_src_us;
            f->src_gap_n++;
        }
        f->last_src_us = p->ts_us;
    } else {
        f->dpkts++;
        f->dbytes += p->ip_len;
        if (f->dttl == 0)
            f->dttl = p->ttl;
        if (f->last_dst_us != 0 && p->ts_us >= f->last_dst_us) {
            f->dst_gap_sum_us += p->ts_us - f->last_dst_us;
            f->dst_gap_n++;
        }
        f->last_dst_us = p->ts_us;
    }

    if (p->proto == IPPROTO_TCP) {
        if (p->tcp_flags & 0x02) f->syn_seen = 1;
        if ((p->tcp_flags & 0x12) == 0x12) f->synack_seen = 1;
        if (p->tcp_flags & 0x10) f->ack_seen = 1;
        if (p->tcp_flags & 0x01) {
            f->fin_seen = 1;
            if (src_direction)
                f->fin_src_seen = 1;
            else
                f->fin_dst_seen = 1;
        }
        if (p->tcp_flags & 0x04) f->rst_seen = 1;
    }

    if (f->blocked)
        return;

    if (f->window_packets >= OBS_MAX_PACKETS ||
        p->ts_us - f->window_start_us >= OBS_MAX_US) {
        int anomaly = evaluate_flow_window(f, model, fifo, bootstrap, learning_enabled, model_path);
        if (!anomaly)
            flow_reset_window(f, p->ts_us);
    }
}

static void expire_flows(Flow *flows, uint64_t now_us,
                         AnomalyModel *model, TrainingFIFO *fifo,
                         SafeBootstrap *bootstrap, int learning_enabled,
                         const char *model_path)
{
    for (uint32_t i = 0; i < MAX_FLOWS; i++) {
        Flow *f = &flows[i];
        if (!f->used)
            continue;
        if (now_us - f->last_us < FLOW_IDLE_US)
            continue;

        if (!f->blocked)
            (void)evaluate_flow_window(f, model, fifo, bootstrap, learning_enabled, model_path);

        f->used = 0;
    }
}

static int capture_live(const char *iface,
                        AnomalyModel *model,
                        const char *model_path,
                        int bootstrap_mode,
                        int learning_enabled)
{
    char errbuf[PCAP_ERRBUF_SIZE] = {0};
    pcap_t *pc = pcap_open_live(iface, 65535, 1, 50, errbuf);
int linktype = pcap_datalink(pc);
printf("pcap link type: %d\n", linktype);
    if (!pc) {
        fprintf(stderr, "pcap_open_live: %s\n", errbuf);
        return -1;
    }

    struct bpf_program fp;
    if (pcap_compile(pc, &fp, "ip", 1, PCAP_NETMASK_UNKNOWN) < 0 ||
        pcap_setfilter(pc, &fp) < 0) {
        fprintf(stderr, "failed to install BPF filter: %s\n", pcap_geterr(pc));
        pcap_freecode(&fp);
        pcap_close(pc);
        return -1;
    }
    pcap_freecode(&fp);

    Flow flows[MAX_FLOWS] = {0};
    TrainingFIFO fifo = {0};
    SafeBootstrap bootstrap_state;
    SafeBootstrap *bootstrap = NULL;

    if (bootstrap_mode) {
        safe_bootstrap_init(&bootstrap_state, model_path);
        bootstrap = &bootstrap_state;
        memset(model, 0, sizeof(*model));
        printf("[BOOTSTRAP] WARNING: this mode assumes the observed traffic is verified-safe.\n");
        printf("[BOOTSTRAP] Do not run this mode while untrusted/attack traffic is present.\n");
        printf("[BOOTSTRAP] Collecting %u safe training windows, then %u safe validation windows.\n",
               BOOTSTRAP_TRAIN_WINDOWS, BOOTSTRAP_VALID_WINDOWS);
    }

    printf("Live capture started on %s\n", iface);
    printf("Observation window: %u packets OR %llu us\n",
           OBS_MAX_PACKETS, OBS_MAX_US);
    printf("Training FIFO: %u recent normal windows\n", TRAIN_FIFO_SIZE);
    printf("Adaptive threshold: TRAIN=%u, CALIBRATE=%u, P%.1f, every %u new normal windows\n",
           ADAPT_MIN_SAMPLES, ADAPT_MIN_SAMPLES,
           BOOTSTRAP_THRESHOLD_Q * 100.0f, ADAPT_UPDATE_INTERVAL);
    if (learning_enabled)
        printf("Continual learning: ENABLED (normal -> FIFO -> periodic rebuild)\n");
    else if (bootstrap_mode)
        printf("Continual learning: DISABLED (bootstrap training/validation)\n");
    else
        printf("Continual learning: DISABLED (detection-only mode)\n");
    printf("NOTE: DROP is a detection decision only; pcap does not remove the packet from the OS network path.\n");

    while (1) {
        struct pcap_pkthdr *hdr = NULL;
        const unsigned char *data = NULL;
        int rc = pcap_next_ex(pc, &hdr, &data);

        if (rc == 0)
            continue;
        if (rc == -1) {
            fprintf(stderr, "pcap_next_ex: %s\n", pcap_geterr(pc));
            break;
        }
        if (rc == -2)
            break;

        ParsedPacket pkt;
        int parsed = parse_ipv4_packet(data, hdr->caplen,
                               packet_time_us(hdr), linktype, &pkt);
        if (parsed <= 0)
            continue;

        expire_flows(flows, pkt.ts_us, model, &fifo, bootstrap, learning_enabled, model_path);

        int src_direction = 1;
        Flow *f = find_flow(flows, &pkt, &src_direction);
        if (!f)
            f = allocate_flow(flows, &pkt, &src_direction,
                              model, &fifo, bootstrap, learning_enabled, model_path);
        if (!f)
            continue;

        if (f->blocked) {
            printf("[DROP DECISION - BLOCKED FLOW] packet observed\n");
            continue;
        }

        update_flow(f, &pkt, src_direction,
                    model, &fifo, bootstrap, learning_enabled, model_path);

        /* Bootstrap is a one-shot training command.
 * Once the model is built, validated, and saved, exit.
 * Normal detection must be started separately with `capture`.
 */
        if (bootstrap && bootstrap->phase == 2) {
            printf("[BOOTSTRAP] complete. Initial model saved to %s.\n",
                model_path);
            printf("[BOOTSTRAP] exiting. Run `capture %s %s` for normal detection.\n",
                iface, model_path);

            pcap_close(pc);
            return 0;
        }
    }

    pcap_close(pc);
    return 0;
}

#endif /* PC_LIVE_CAPTURE */

/* -------------------- MAIN -------------------- */

int main(int argc, char **argv)
{
    AnomalyModel model = {0};

#ifdef PC_LIVE_CAPTURE
    if (argc >= 3 && strcmp(argv[1], "bootstrap") == 0) {
        const char *iface = argv[2];
        const char *model_path = argc >= 4 ? argv[3] : DEFAULT_MODEL_FILE;
        return capture_live(iface, &model, model_path, 1, 0);
    }

    if (argc >= 3 && strcmp(argv[1], "capture") == 0) {
        const char *iface = argv[2];
        const char *model_path = argc >= 4 ? argv[3] : DEFAULT_MODEL_FILE;

        if (load_local_model(model_path, &model) != 0) {
            fprintf(stderr,
                    "No valid local model at %s. Run bootstrap first.\n",
                    model_path);
            return 1;
        }

        printf("Local model loaded from %s\n", model_path);
        printf("Current model sample count: %u\n", model.n);
        printf("Local anomaly threshold: %.6f\n", model.threshold);
        printf("Score at model mean: %.6f\n",
               mahalanobis_score(&model, model.mean));
        print_normalization(&model);
        /* capture = detection + automatic continual adaptation. */
        printf("Capture mode: detection + automatic continual learning enabled.\n");
        return capture_live(iface, &model, model_path, 0, 1);
    }
#endif

    if (argc >= 2 && strcmp(argv[1], "federate") == 0) {
        const char *model_path = argc >= 3 ? argv[2] : DEFAULT_MODEL_FILE;

        if (load_local_model(model_path, &model) != 0) {
            fprintf(stderr,
                    "No valid local model at %s. Run bootstrap first.\n",
                    model_path);
            return 1;
        }

        printf("Local model loaded from %s\n", model_path);
        if (forward_model(&model) == 0) {
            printf("Global anomaly model received successfully\n");
            printf("Global threshold: %.6f\n", model.threshold);
            print_normalization(&model);
            if (save_local_model(model_path, &model) != 0) {
                fprintf(stderr, "Warning: failed to save received global model\n");
                return 1;
            }
            printf("Updated global model saved to %s\n", model_path);
        } else {
            printf("Federated model exchange failed\n");
            return 1;
        }
        return 0;
    }

    printf("Usage:\n");
#ifdef PC_LIVE_CAPTURE
    printf("  %s bootstrap <interface> [model_file]\n", argv[0]);
    printf("  %s capture   <interface> [model_file]\n", argv[0]);
#endif
    printf("  %s federate  [model_file]\n", argv[0]);
    return 0;
}

/* skeleton */
