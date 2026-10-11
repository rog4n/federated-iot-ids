# Federated IoT Intrusion Detection System

A prototype federated anomaly detection system for network-edge environments, combining C-based traffic analysis with Python-based model aggregation.

## Overview

This project explores distributed network anomaly detection in which edge clients build local statistical models from network traffic and exchange model parameters with a central aggregation server.

The goal is to investigate continual adaptation to changing network behavior while avoiding the routine transfer of raw traffic to the aggregation server.

## Architecture

- **Edge client (C):** Captures IPv4 TCP/UDP traffic using libpcap, extracts flow-level features, and evaluates observations using Mahalanobis distance.
- **Local model training:** Learns normalization parameters and a Gaussian anomaly model from operator-verified safe traffic.
- **Continual adaptation:** Rebuilds local model statistics from a rolling FIFO of observations classified as normal and recalibrates the detection threshold.
- **Federation server (Python):** Receives client model parameters, converts locally normalized models into a common feature space, aggregates their statistical parameters, and returns a global model.
- **Model persistence:** Saves local models for subsequent detection and federation sessions.

## Technical Stack

- C
- Python
- NumPy
- TCP sockets
- libpcap
- Multivariate statistics and linear algebra

## Current Capabilities

- IPv4 TCP/UDP flow observation
- Ten-dimensional flow feature representation
- Mahalanobis-distance anomaly scoring
- Safe-traffic bootstrap and threshold calibration
- Continual local model adaptation
- Statistical model aggregation across clients
- Binary model persistence and parameter exchange

## Current Status

This project is an experimental prototype. Detection quality, false-positive rates, robustness to adversarial traffic, and performance under realistic network conditions require systematic evaluation.

The current server configuration binds to the loopback interface and expects two clients. Multi-host deployment requires additional networking configuration and testing.

## Security and Privacy Considerations

The federation protocol exchanges model parameters rather than routinely uploading raw packet captures. However, model exchange alone does not guarantee privacy or prevent information leakage.

The current prototype should not be considered production-ready network protection. Authentication, transport security, malicious-client defenses, and deployment hardening require further work.

## Planned Evaluation

- Detection performance on representative normal and malicious traffic
- False-positive and false-negative rates
- Adaptation stability under changing normal traffic
- Communication overhead per federation round
- CPU and memory consumption
- Comparison with non-federated baselines

## Disclaimer

This repository is intended for research, learning, and authorized network-security testing. It is not a substitute for a validated production intrusion detection or prevention system.
