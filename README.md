# sandboxLLM

A single `docker compose up` to serve a local LLM via [vLLM](https://github.com/vllm-project/vllm) with automatic VRAM validation, a security-hardened container, and a live monitoring dashboard. Built for agentic coding tools like [Cline](https://github.com/cline/cline).

---

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Quick Start](#quick-start)
3. [Configuration](#configuration)
4. [Model Recommendations for Cline](#model-recommendations-for-cline)
5. [Using with Cline (Remote PC)](#using-with-cline-remote-pc)
6. [Dashboard](#dashboard)
7. [Appendix A: Security Features](#appendix-a-security-features)
8. [Appendix B: Host Firewall (Optional)](#appendix-b-host-firewall-optional)

---

## Prerequisites

- Linux host with NVIDIA GPU(s)
- [Docker](https://docs.docker.com/engine/install/) with the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html)

That's it. No Python, no pip, no model downloads on the host. The container handles everything.

Install the NVIDIA Container Toolkit if you haven't already:

```bash
# Ubuntu / Debian
sudo apt-get install -y nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
```

---

## Quick Start

```bash
# 1. Clone the repo
git clone <this-repo-url> && cd sandboxLLM

# 2. Edit config.yaml to pick your model (see section below)
nano config.yaml

# 3. Launch (starts both the vLLM server and the monitoring dashboard)
docker compose up --build
```

On startup, the vLLM container will:

1. **Download the model** (first run only) — pulls the model from HuggingFace into a persistent Docker volume. Subsequent runs skip this step.
2. **Enable offline mode** — sets `HF_HUB_OFFLINE=1` so no further network calls are made.
3. **Run a VRAM preflight check** — estimates model weights + KV cache memory and verifies it fits within your GPU budget. If it doesn't fit, it prints exactly how much you're over and suggests fixes.
4. **Start the vLLM OpenAI-compatible API server** on the configured port.

After startup:

| Service   | Default URL                    |
|-----------|--------------------------------|
| vLLM API  | `http://<host-ip>:7171`        |
| Dashboard | `http://<host-ip>:7172`        |

---

## Configuration

All settings live in a single **`config.yaml`**:

```yaml
model:
  name: "Qwen/Qwen2.5-Coder-32B-Instruct"
  quantization: none            # none | awq | gptq | fp8 | squeezellm
  active_params_billion: null   # null = auto-detect; set manually for MoE models

serving:
  max_context_length: 32768     # max tokens per request
  max_num_seqs: 64              # max concurrent sequences
  port: 7171                    # container port for the API
  api_key: ""                   # set a key to require authentication
  tensor_parallel: "auto"       # "auto" = use all GPUs, or an integer

gpu:
  max_utilization: 0.85         # fraction of VRAM vLLM may use (0.0–1.0)
  kv_cache_gpu_only: true       # keep all KV cache on GPU (no CPU swap)
  vram_per_gpu_gib: null        # null = auto-detect via nvidia-smi
```

### Key settings to adjust

| Setting | What it does | When to change |
|---|---|---|
| `model.name` | HuggingFace model ID or local path | Always — pick the model you want |
| `model.quantization` | Weight precision | Use `awq`/`gptq` to fit larger models in less VRAM |
| `serving.max_context_length` | Maximum tokens per sequence | Lower it to reduce KV cache VRAM; raise it for long-file edits |
| `serving.max_num_seqs` | Concurrent requests | Lower if VRAM is tight; Cline typically sends 1–2 at a time |
| `serving.api_key` | Authentication token | **Set this** if exposing on a LAN (see [Cline section](#using-with-cline-remote-pc)) |
| `serving.tensor_parallel` | Number of GPUs to shard across | `"auto"` uses all GPUs; set to `1` to use a single GPU |
| `gpu.max_utilization` | VRAM budget per GPU | 0.85 is conservative; push to 0.92 if you know your GPU's limits |

### Environment variable overrides

You can override the host-side binding without editing `config.yaml`:

```bash
# Change the published port on the host
SANDBOXLLM_PORT=9000 docker compose up --build

# Change the dashboard port
DASHBOARD_PORT=9001 docker compose up --build

# Bind to a specific interface (default 0.0.0.0 = all interfaces)
SANDBOXLLM_HOST=192.168.1.100 docker compose up --build

# Restrict to specific GPUs
CUDA_VISIBLE_DEVICES=0,1 docker compose up --build
```

---

## Model Recommendations for Cline

Cline needs a model that excels at **instruction following**, **code generation**, **tool/function calling**, and **long-context reasoning**. The tables below show exact VRAM breakdowns so you can pick the right model for your hardware.

### How to read the tables

Every entry shows three numbers:

- **Model Weights** — the raw size of the model parameters in VRAM (depends on precision).
- **KV Cache** — the memory reserved for key/value attention cache at the given context length and concurrency. Formula: `2 × layers × kv_heads × head_dim × 2 bytes × context_len × max_num_seqs`.
- **Total** — weights + KV cache. This must fit within `per_gpu_vram × num_gpus × max_utilization`.

> All estimates assume fp16 KV cache (vLLM default). Actual usage includes ~1–2 GiB overhead for activations and CUDA kernels — the preflight script accounts for this via the utilization headroom.

---

### Setup A: 2x 48 GB GPUs (96 GB total, 85% = 81.6 GiB budget)

These configurations are designed for a dual-GPU workstation (e.g. 2x RTX A6000, 2x L40S, 2x RTX 6000 Ada).

| Model | Quant | Params | Weights | Ctx | Seqs | KV Cache | **Total** | Fits? |
|---|---|---|---|---|---|---|---|---|
| `Qwen/Qwen2.5-Coder-32B-Instruct` | fp16 | 32.5B | 60.5 GiB | 32k | 4 | 16.0 GiB | **76.5 GiB** | **Yes** (5.1 GiB headroom) |
| `Qwen/Qwen2.5-Coder-32B-Instruct` | fp16 | 32.5B | 60.5 GiB | 32k | 8 | 32.0 GiB | **92.5 GiB** | No (10.9 over) |
| `Qwen/Qwen2.5-Coder-32B-Instruct` | fp16 | 32.5B | 60.5 GiB | 16k | 8 | 16.0 GiB | **76.5 GiB** | **Yes** (5.1 GiB headroom) |
| `mistralai/Codestral-22B-v0.1` | fp16 | 22.2B | 41.3 GiB | 32k | 8 | 16.0 GiB | **57.3 GiB** | **Yes** (24.3 GiB headroom) |
| `Qwen/Qwen2.5-Coder-14B-Instruct` | fp16 | 14.8B | 27.5 GiB | 32k | 8 | 14.0 GiB | **41.5 GiB** | **Yes** (40.1 GiB headroom) |

> **Architecture details used:**
> - Qwen2.5-Coder-32B: 64 layers, 8 KV heads, head_dim 128
> - Codestral-22B: 56 layers, 8 KV heads, head_dim 128
> - Qwen2.5-Coder-14B: 48 layers, 8 KV heads, head_dim 128

**Recommended config — Qwen 32B fp16 on 2x 48 GB GPUs:**

```yaml
model:
  name: "Qwen/Qwen2.5-Coder-32B-Instruct"
  quantization: none

serving:
  max_context_length: 32768
  max_num_seqs: 4              # Cline sends 1-2 at a time; 4 gives headroom
  port: 7171
  api_key: "change-me"
  tensor_parallel: 2

gpu:
  max_utilization: 0.85
  kv_cache_gpu_only: true
```

**Alternative — if you want more concurrent sequences**, drop context to 16k:

```yaml
serving:
  max_context_length: 16384
  max_num_seqs: 8
```

---

### Setup B: 1x 16 GB GPU (RTX 5080, 85% = 13.6 GiB budget)

For testing on a single consumer GPU. Quality won't match the 32B models, but these are functional for validating the setup and simple coding tasks.

| Model | Quant | Params | Weights | Ctx | Seqs | KV Cache | **Total** | Fits? |
|---|---|---|---|---|---|---|---|---|
| `Qwen/Qwen2.5-Coder-7B-Instruct` | fp16 | 7.6B | 14.2 GiB | 32k | 2 | 3.5 GiB | **17.7 GiB** | No (4.1 over) |
| `Qwen/Qwen2.5-Coder-7B-Instruct` | fp16 | 7.6B | 14.2 GiB | 8k | 2 | 0.9 GiB | **15.1 GiB** | No (1.5 over) |
| `Qwen/Qwen2.5-Coder-7B-Instruct-AWQ` | awq | 7.6B | 3.5 GiB | 32k | 2 | 3.5 GiB | **7.0 GiB** | **Yes** (6.6 GiB headroom) |
| `Qwen/Qwen2.5-Coder-7B-Instruct-AWQ` | awq | 7.6B | 3.5 GiB | 32k | 4 | 7.0 GiB | **10.5 GiB** | **Yes** (3.1 GiB headroom) |
| `Qwen/Qwen2.5-Coder-3B-Instruct` | fp16 | 3.1B | 5.8 GiB | 32k | 2 | 2.3 GiB | **8.1 GiB** | **Yes** (5.5 GiB headroom) |
| `Qwen/Qwen2.5-Coder-3B-Instruct` | fp16 | 3.1B | 5.8 GiB | 32k | 4 | 4.5 GiB | **10.3 GiB** | **Yes** (3.3 GiB headroom) |
| `Qwen/Qwen2.5-Coder-1.5B-Instruct` | fp16 | 1.5B | 2.9 GiB | 32k | 4 | 1.8 GiB | **4.7 GiB** | **Yes** (8.9 GiB headroom) |

> **Architecture details used:**
> - Qwen2.5-Coder-7B: 28 layers, 4 KV heads, head_dim 128
> - Qwen2.5-Coder-3B: 36 layers, 2 KV heads, head_dim 128
> - Qwen2.5-Coder-1.5B: 28 layers, 2 KV heads, head_dim 64

**Recommended config — Qwen 7B AWQ on RTX 5080 (testing):**

```yaml
model:
  name: "Qwen/Qwen2.5-Coder-7B-Instruct-AWQ"
  quantization: awq

serving:
  max_context_length: 32768
  max_num_seqs: 2
  port: 7171
  api_key: ""
  tensor_parallel: 1

gpu:
  max_utilization: 0.85
  kv_cache_gpu_only: true
```

**Tiny model for quick smoke-testing:**

```yaml
model:
  name: "Qwen/Qwen2.5-Coder-1.5B-Instruct"
  quantization: none

serving:
  max_context_length: 32768
  max_num_seqs: 4
  port: 7171
  api_key: ""
  tensor_parallel: 1

gpu:
  max_utilization: 0.85
  kv_cache_gpu_only: true
```

This loads in under 5 GiB and is useful for testing the Docker pipeline, networking, and Cline connectivity end-to-end before switching to a heavier model.

---

### Which to pick?

| Your hardware | Best model | Why |
|---|---|---|
| 2x 48 GB GPUs (work) | Qwen2.5-Coder-32B fp16, TP=2 | Best coding quality available at this size. Full precision, no quality loss. |
| 1x 24 GB GPU | Qwen2.5-Coder-32B-Instruct-AWQ | 4-bit quantization with minimal quality loss. |
| 1x 16 GB GPU (home) | Qwen2.5-Coder-7B-Instruct-AWQ | Best quality that fits. Good for single-file edits. |
| Quick test (any GPU) | Qwen2.5-Coder-1.5B-Instruct fp16 | Loads fast, tiny footprint. Validate your setup before committing to a big download. |

---

## Using with Cline (Remote PC)

Since sandboxLLM serves an **OpenAI-compatible API**, Cline can connect to it directly from another machine on the same LAN. Your GPU server runs the model; your workstation runs VS Code + Cline.

```
 ┌─────────────────────┐          LAN          ┌────────────────────────┐
 │   Workstation (PC)  │ ───── HTTP :7171 ────▸ │  GPU Server             │
 │   VS Code + Cline   │                        │  sandboxLLM (Docker)   │
 └─────────────────────┘                        └────────────────────────┘
```

### 1. Set an API key

Edit `config.yaml` on the server:

```yaml
serving:
  api_key: "my-secret-key-change-me"
```

This ensures that only clients with the key can use the API. Without it, anyone on your LAN who knows the IP and port can send requests freely. The API key is passed as a Bearer token in the `Authorization` header — the same way cloud LLM APIs work.

### 2. Find the server's LAN IP

On the GPU server:

```bash
hostname -I | awk '{print $1}'
# Example output: 192.168.1.50
```

### 3. Start sandboxLLM

```bash
docker compose up --build
```

Verify it's working:

```bash
# From the server itself
curl http://localhost:7171/v1/models

# From your VS Code machine
curl http://192.168.1.50:7171/v1/models

# With API key
curl -H "Authorization: Bearer my-secret-key-change-me" http://192.168.1.50:7171/v1/models
```

### 4. Configure Cline in VS Code

1. Open VS Code on your workstation.
2. Open the Cline sidebar (click the Cline icon or `Ctrl+Shift+P` -> "Cline: Focus on View").
3. Click the **settings gear** in the Cline panel.
4. Set the **API Provider** to **OpenAI Compatible**.
5. Fill in:

| Field | Value |
|---|---|
| **Base URL** | `http://192.168.1.50:7171/v1` |
| **API Key** | `my-secret-key-change-me` |
| **Model ID** | `Qwen/Qwen2.5-Coder-32B-Instruct` |

   (Replace the IP, key, and model name with your actual values.)

6. Click **Save** / **Done**.

Cline will now route all LLM requests to your self-hosted sandboxLLM server.

### Tips for Cline + sandboxLLM

- **Context length matters.** Cline sends entire files as context. If you work on large files, set `max_context_length` to 32768 or higher.
- **Concurrency is low.** Cline usually sends 1 request at a time (occasionally 2 for diff previews). You can set `max_num_seqs: 2–4` to save VRAM.
- **Streaming works out of the box.** vLLM supports SSE streaming, which Cline uses for real-time token display.
- **If requests fail with 401**, double-check that the `api_key` in `config.yaml` matches what you entered in Cline.
- **Latency:** On a local LAN (Gigabit), network overhead is negligible (<1ms). All latency comes from model inference.

---

## Dashboard

sandboxLLM includes a lightweight monitoring dashboard that runs alongside the vLLM server. Open it from **any device on the LAN** to monitor your GPU server in real time.

**URL:** `http://<server-ip>:7172`

### Monitoring tab

- **GPU utilization** — per-GPU compute usage percentage with color-coded bars
- **VRAM usage** — used vs. total memory per GPU
- **GPU temperature** — per-GPU temperature in Celsius
- **GPU power draw** — current vs. limit per GPU
- **CPU utilization** — overall busy percentage
- **CPU temperature** — if thermal zone data is available
- **System load average** — 1m / 5m / 15m
- **RAM usage** — used / total / available with percentage bar
- **Served models** — which models are currently loaded and ready in vLLM

### Logs tab

- **Live vLLM logs** — shows the full entrypoint output: model download progress, preflight VRAM check results (pass/fail with exact numbers), and ongoing vLLM server logs
- Syntax highlighting for PASS/FAIL/WARNING/ERROR lines
- Auto-scroll toggle to follow new output

The dashboard auto-refreshes every 2–3 seconds. It has no dependencies beyond Flask and makes zero network calls outside the Docker network — it queries `nvidia-smi` for GPU data, reads `/proc` for CPU/RAM, reads the shared log volume, and hits the vLLM `/v1/models` endpoint for model status.

### Changing the dashboard port

```bash
DASHBOARD_PORT=9090 docker compose up --build
```

---

## Appendix A: Security Features

sandboxLLM is designed to run on machines that handle sensitive code and data. The container is locked down in multiple layers:

### Network isolation

The container has **two phases** with different network postures:

| Phase | Network | Why |
|---|---|---|
| **Model download** (first run) | Internet access needed | Downloads model weights from HuggingFace into a persistent Docker volume |
| **Serving** (after download) | `HF_HUB_OFFLINE=1` enforced | Entrypoint sets offline mode before launching vLLM — no outbound calls |

| Measure | Effect |
|---|---|
| `HF_HUB_OFFLINE=1` (set by entrypoint) | HuggingFace libraries refuse to make network calls during serving |
| `TRANSFORMERS_OFFLINE=1` (set by entrypoint) | Same as above for the `transformers` library |
| `VLLM_USAGE_STATS_SERVER=""` | Disables vLLM's opt-in telemetry |
| `DO_NOT_TRACK=1` | Respects the standard do-not-track signal |

The container can **receive** inbound connections on the published port (needed for Cline and the dashboard). After the first-run download, all outbound download/telemetry paths are disabled at the application level. For hard network-level blocking, see [Appendix B: Host Firewall](#appendix-b-host-firewall-optional).

### Filesystem

| Measure | Effect |
|---|---|
| `read_only: true` | Root filesystem is immutable — nothing can be written to the container image layers |
| HF cache in named volume | Model weights persist across restarts but are isolated from the host filesystem |
| `config.yaml` mounted `:ro` | Configuration cannot be tampered with at runtime |
| `tmpfs /tmp` with `noexec` | Temp files allowed but **cannot be executed** |
| `tmpfs /run` with `noexec` | Same for runtime directory |

### Privilege reduction

| Measure | Effect |
|---|---|
| `cap_drop: ALL` | Drops **every** Linux capability (NET_RAW, SYS_ADMIN, etc.) |
| `no-new-privileges: true` | Prevents setuid/setgid escalation — even if a binary in the container has the suid bit, it won't gain elevated privileges |
| `privileged: false` | Explicitly disabled (defense in depth) |

### Resource limits

| Measure | Effect |
|---|---|
| `pids_limit: 512` | Prevents fork bombs |
| `shm_size: 16gb` | Enough for NCCL multi-GPU communication, but bounded |
| `ulimits.nofile: 65536` | Prevents file-descriptor exhaustion attacks |

### What this means in practice

- During serving, the container **cannot upload your data** — offline mode blocks all HuggingFace/telemetry calls, and for hard guarantees use the host firewall (Appendix B).
- The container **cannot modify your config** (`config.yaml` is read-only).
- The container **cannot escalate privileges** to access the host beyond the explicitly mounted volumes.
- The container **cannot execute arbitrary downloaded binaries** because the filesystem is read-only and tmpfs is `noexec`.

### LAN exposure and the API key

When the API is exposed on `0.0.0.0`, any device on the LAN can reach it. The `api_key` setting adds a Bearer token requirement — requests without a valid `Authorization: Bearer <key>` header receive a `401 Unauthorized`. This is not a substitute for a proper firewall, but it prevents casual or accidental use by other devices on the network.

If you need stronger LAN-level isolation, bind to a specific interface:

```bash
SANDBOXLLM_HOST=192.168.1.50 docker compose up --build
```

Or use host-level firewall rules (see below).

---

## Appendix B: Host Firewall (Optional)

For an additional layer of security, you can use `iptables` rules on the **host** to block all outbound traffic from the Docker container while still allowing LAN clients to connect inbound.

```bash
# Get the container's virtual interface (run after `docker compose up`)
CONTAINER_ID=$(docker compose ps -q vllm)
VETH=$(ip link | grep -A1 "$(docker exec "$CONTAINER_ID" cat /sys/class/net/eth0/iflink | tr -d '\r')" \
       | head -1 | awk -F: '{print $2}' | tr -d ' ')

# Block all outbound traffic FROM the container
sudo iptables -I FORWARD -o eth0 -i "$VETH" -j DROP

# Allow established connections (so inbound request *responses* can get out)
sudo iptables -I FORWARD -o eth0 -i "$VETH" -m state --state ESTABLISHED,RELATED -j ACCEPT
```

This ensures the container can respond to API requests but cannot initiate any outbound connection — even if DNS somehow worked. To make these rules persist across reboots, use `iptables-save` / `iptables-persistent`.

To remove the rules:

```bash
sudo iptables -D FORWARD -o eth0 -i "$VETH" -j DROP
sudo iptables -D FORWARD -o eth0 -i "$VETH" -m state --state ESTABLISHED,RELATED -j ACCEPT
```
