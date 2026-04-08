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

All settings live in a single **`config.yaml`**. Here are two ready-to-use configs:

**96 GB VRAM (2x 48 GB GPUs) — production/work:**

```yaml
model:
  name: "Qwen/Qwen3-Coder-30B-A3B-Instruct"
  quantization: none
  active_params_billion: null

serving:
  max_context_length: 32768
  max_num_seqs: 4
  port: 7171
  api_key: "change-me-to-something-secret"
  tensor_parallel: 2
  tool_call_parser: "qwen3_coder"     # required for Qwen3-Coder function calling

gpu:
  max_utilization: 0.85
  kv_cache_gpu_only: true
  vram_per_gpu_gib: null
```

**16 GB VRAM (1x RTX 5080) — testing/home:**

```yaml
model:
  name: "Qwen/Qwen3.5-9B"
  quantization: fp8
  active_params_billion: null

serving:
  max_context_length: 32768
  max_num_seqs: 2
  port: 7171
  api_key: ""
  tensor_parallel: 1
  tool_call_parser: "hermes"

gpu:
  max_utilization: 0.85
  kv_cache_gpu_only: true
  vram_per_gpu_gib: null
```

### Key settings to adjust

| Setting | What it does | When to change |
|---|---|---|
| `model.name` | HuggingFace model ID or local path | Always — pick the model you want |
| `model.quantization` | Weight precision | Use `awq`/`gptq`/`fp8` to fit larger models in less VRAM |
| `serving.max_context_length` | Maximum tokens per sequence | Lower it to reduce KV cache VRAM; raise it for long-file edits |
| `serving.max_num_seqs` | Concurrent requests | Lower if VRAM is tight; Cline typically sends 1–2 at a time |
| `serving.api_key` | Authentication token | **Set this** if exposing on a LAN (see [Cline section](#using-with-cline-remote-pc)) |
| `serving.tensor_parallel` | Number of GPUs to shard across | `"auto"` uses all GPUs; set to `1` to use a single GPU |
| `serving.tool_call_parser` | Function calling format | `"qwen3_coder"` for Qwen3-Coder, `"hermes"` for Qwen2.5/most others, `"none"` to disable |
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

### Agentic suitability for Cline

Not every model is equally suited for agentic coding. Cline relies heavily on **function/tool calling** (to read/write files, run commands, etc.) and **instruction following** (to execute multi-step plans). Here's how the model families compare:

| Rating | Family | Why |
|---|---|---|
| **Excellent** | Qwen3-Coder | Purpose-built for agentic coding tools (Cline, Claude Code, Roo Code). Best-in-class tool calling. Trained specifically on agentic coding trajectories. |
| **Excellent** | Llama 4 Scout/Maverick | Optimized for tool calling and agentic systems. Parallel tool calls supported. 10M context window. |
| **Very Good** | Qwen3.5 (9B+) | Officially optimized for agentic coding via Cline. Strong tool calling, 262k context. The 9B model punches above many prior 13–30B models on reasoning/coding benchmarks. |
| **Good** | Qwen2.5-Coder | Proven with Cline, dedicated coding model, reliable tool calling. Older generation but battle-tested. |
| **Good** | Llama 3.1 8B | Decent tool calling (llama3_json parser), 128k context. Good for its size but not coding-specific. |
| **Fair** | Llama 3.2 3B | Matches Llama 3.1 8B on tool use benchmarks (BFCL v2), but small size limits complex reasoning. |
| **Fair** | Qwen3.5-4B / Qwen2.5-Coder-3B | Functional for simple edits. Struggle with complex multi-file refactors or long agentic chains. |
| **Smoke test only** | Any 1.5B model | Too small for real agentic work. Fine for validating your setup. |

### How to read the VRAM tables

Every entry shows three numbers:

- **Model Weights** — the raw size of all model parameters in VRAM. For MoE models, *all* experts must be loaded even though only a few are active per token.
- **KV Cache** — the memory reserved for key/value attention cache at the given context length and concurrency. Formula: `2 × attn_layers × kv_heads × head_dim × 2 bytes × context_len × max_num_seqs`. For hybrid models (Qwen3.5) that mix DeltaNet with standard attention, only the attention layers contribute KV cache — a significant saving.
- **Total** — weights + KV cache. This must fit within `per_gpu_vram × num_gpus × max_utilization`.

> All estimates assume fp16 KV cache (vLLM default). Actual usage includes ~1–2 GiB overhead for activations and CUDA kernels — the preflight script accounts for this via the utilization headroom.

### Model generations at a glance

| Family | Released | Strengths | Tool parser | Notes |
|---|---|---|---|---|
| **Qwen3-Coder** | 2025–2026 | Purpose-built for agentic coding (Cline, Claude Code). Best tool calling. | `qwen3_coder` | MoE — big total params, small active params. Requires vLLM >= 0.15. |
| **Qwen3.5** | Mar 2026 | General-purpose with strong coding, 262k native context. | `hermes` | Hybrid architecture (DeltaNet + Attention) — very efficient KV cache. |
| **Qwen2.5-Coder** | 2024–2025 | Proven, widely tested with Cline. Dedicated coding model. | `hermes` | Dense architecture. Stable and well-supported. |
| **Llama 4 Scout** | Apr 2025 | Strong general-purpose + coding. 10M context. Parallel tool calls. | `llama4_pythonic` | MoE (109B total, 17B active). Large even quantized. |
| **Llama 3.1/3.2** | 2024 | Solid general-purpose. 3.2-3B matches 3.1-8B on tool use. | `llama3_json` | Dense. Smaller sizes available for 16 GB GPUs. |

---

### Setup A: 2x 48 GB GPUs (96 GB total, 85% = 81.6 GiB budget)

These configurations are designed for a dual-GPU workstation (e.g. 2x RTX A6000, 2x L40S, 2x RTX 6000 Ada).

| Model | Quant | Total Params | Weights | Ctx | Seqs | KV Cache | **Total** | Agentic | Fits? |
|---|---|---|---|---|---|---|---|---|---|
| **`Qwen/Qwen3-Coder-30B-A3B-Instruct`** | fp16 | 30.5B (3.3B active) | 56.8 GiB | 32k | 4 | 12.0 GiB | **68.8 GiB** | Excellent | **Yes** (12.8 GiB headroom) |
| `Qwen/Qwen3-Coder-30B-A3B-Instruct` | fp16 | 30.5B | 56.8 GiB | 32k | 8 | 24.0 GiB | **80.8 GiB** | Excellent | **Yes** (0.8 GiB headroom) |
| `meta-llama/Llama-4-Scout-17B-16E-Instruct` | 4-bit | 109B (17B active) | 54.5 GiB | 32k | 2 | 20.0 GiB | **74.5 GiB** | Excellent | **Yes** (7.1 GiB headroom) |
| `Qwen/Qwen2.5-Coder-32B-Instruct` | fp16 | 32.5B | 60.5 GiB | 32k | 4 | 16.0 GiB | **76.5 GiB** | Good | **Yes** (5.1 GiB headroom) |
| `Qwen/Qwen3.5-27B` | fp16 | 27B | 50.3 GiB | 32k | 4 | 8.0 GiB | **58.3 GiB** | Very Good | **Yes** (23.3 GiB headroom) |
| `Qwen/Qwen3.5-27B` | fp16 | 27B | 50.3 GiB | 32k | 12 | 24.0 GiB | **74.3 GiB** | Very Good | **Yes** (7.3 GiB headroom) |

> **Architecture details used:**
> - Qwen3-Coder-30B-A3B: 48 layers, 4 KV heads, head_dim 128, MoE (128 experts, 8 active)
> - Llama 4 Scout: 80 layers, 8 KV heads, head_dim 128, MoE (16 experts, 2 active). 4-bit via `RedHatAI/Llama-4-Scout-17B-16E-Instruct-quantized.w4a16` or on-the-fly quantization.
> - Qwen2.5-Coder-32B: 64 layers, 8 KV heads, head_dim 128, dense
> - Qwen3.5-27B: 64 total layers but only 16 attention layers (hybrid DeltaNet), 4 KV heads, head_dim 256

**Recommended: Qwen3-Coder-30B-A3B** — purpose-built for agentic coding (Cline, Claude Code). Best tool calling of any open model. 12.8 GiB headroom at 32k. Set `tool_call_parser: "qwen3_coder"`.

**Meta alternative: Llama 4 Scout (4-bit)** — Meta's flagship open model. Excellent tool calling with parallel tool call support. 109B total params (17B active MoE) squeezed into 96 GB via 4-bit quantization. Limited to 2 concurrent seqs at 32k due to heavy KV cache (80 dense attention layers). Set `tool_call_parser: "llama4_pythonic"`.

**High-concurrency: Qwen3.5-27B** — hybrid architecture gives it the smallest KV cache in this table. You can run 12 concurrent seqs at 32k and still fit. Great for multi-user or parallel Cline sessions. Set `tool_call_parser: "hermes"`.

**Proven fallback: Qwen2.5-Coder-32B** — battle-tested with Cline, dense architecture (simplest to deploy), but tighter on VRAM. Set `tool_call_parser: "hermes"`.

---

### Setup B: 1x 16 GB GPU (RTX 5080, 85% = 13.6 GiB budget)

For testing at home on a single consumer GPU. The Qwen3.5 hybrid architecture shines here — tiny KV cache means you can run a surprisingly capable 9B model in fp8, or a 4B model in full fp16, with 32k context.

| Model | Quant | Total Params | Weights | Ctx | Seqs | KV Cache | **Total** | Agentic | Fits? |
|---|---|---|---|---|---|---|---|---|---|
| **`Qwen/Qwen3.5-9B`** | fp8 | 9B | 8.4 GiB | 32k | 2 | 2.0 GiB | **10.4 GiB** | Very Good | **Yes** (3.2 GiB headroom) |
| `Qwen/Qwen3.5-4B` | fp16 | 4B | 7.5 GiB | 32k | 2 | 2.0 GiB | **9.5 GiB** | Fair | **Yes** (4.1 GiB headroom) |
| `Qwen/Qwen3.5-4B` | fp16 | 4B | 7.5 GiB | 32k | 4 | 4.0 GiB | **11.5 GiB** | Fair | **Yes** (2.1 GiB headroom) |
| `Qwen/Qwen2.5-Coder-7B-Instruct-AWQ` | awq | 7.6B | 3.5 GiB | 32k | 2 | 3.5 GiB | **7.0 GiB** | Good | **Yes** (6.6 GiB headroom) |
| `meta-llama/Llama-3.1-8B-Instruct-AWQ` | awq | 8B | 3.7 GiB | 16k | 2 | 4.0 GiB | **7.7 GiB** | Good | **Yes** (5.9 GiB headroom) |
| `meta-llama/Llama-3.1-8B-Instruct-AWQ` | awq | 8B | 3.7 GiB | 32k | 2 | 8.0 GiB | **11.7 GiB** | Good | **Yes** (1.9 GiB headroom) |
| `meta-llama/Llama-3.2-3B-Instruct` | fp16 | 3.2B | 6.0 GiB | 16k | 2 | 3.5 GiB | **9.5 GiB** | Fair | **Yes** (4.1 GiB headroom) |
| `Qwen/Qwen2.5-Coder-1.5B-Instruct` | fp16 | 1.5B | 2.9 GiB | 32k | 4 | 1.8 GiB | **4.7 GiB** | Smoke test | **Yes** (8.9 GiB headroom) |

> **Architecture details used:**
> - Qwen3.5-9B: 32 total layers, **8 attention layers** (hybrid DeltaNet), 4 KV heads, head_dim 256
> - Qwen3.5-4B: 32 total layers, **8 attention layers** (hybrid DeltaNet), 4 KV heads, head_dim 256
> - Qwen2.5-Coder-7B: 28 layers (all attention), 4 KV heads, head_dim 128
> - Llama 3.1-8B: 32 layers (all attention), 8 KV heads, head_dim 128. Note: heavy KV cache (8 KV heads, all layers)
> - Llama 3.2-3B: 28 layers (all attention), 8 KV heads, head_dim 128. Same KV head count as the 8B!
> - Qwen2.5-Coder-1.5B: 28 layers (all attention), 2 KV heads, head_dim 64

Notice the KV cache differences: Qwen3.5 models use only **2.0 GiB** at 32k context (8 attention layers out of 32), while the similarly-sized Llama 3.1-8B needs **8.0 GiB** (all 32 layers, 8 KV heads). The hybrid DeltaNet architecture is a game-changer for VRAM-constrained setups.

**Recommended: Qwen3.5-9B (fp8)** — latest generation, 9B params with only 2 GiB KV cache at 32k. The 9B model punches above many prior 13–30B models on reasoning/coding benchmarks. Officially optimized for Cline. Set `tool_call_parser: "hermes"`.

**Meta alternative: Llama 3.1-8B-Instruct-AWQ** — Meta's best option for 16 GB. Solid tool calling and 128k native context. However, the dense architecture with 8 KV heads means KV cache is expensive — use 16k context to keep headroom. Set `tool_call_parser: "llama3_json"`.

**Coding-specific: Qwen2.5-Coder-7B-AWQ** — dedicated coding model, proven with Cline. Smallest total footprint (7.0 GiB) thanks to aggressive AWQ quantization. Set `tool_call_parser: "hermes"`.

**Smoke test: Qwen2.5-Coder-1.5B** — loads in under 5 GiB. Too small for real agentic work, but validates Docker, networking, and Cline connectivity end-to-end.

---

### Which to pick?

| Your hardware | Best model | Tool parser | Agentic | Why |
|---|---|---|---|---|
| 2x 48 GB GPUs (work) | Qwen3-Coder-30B-A3B fp16, TP=2 | `qwen3_coder` | Excellent | Purpose-built for Cline. Best tool calling of any open model. |
| 2x 48 GB GPUs (Meta) | Llama 4 Scout 4-bit, TP=2 | `llama4_pythonic` | Excellent | Meta flagship. Parallel tool calls. Heavy KV cache limits concurrency. |
| 2x 48 GB GPUs (alt) | Qwen3.5-27B fp16, TP=2 | `hermes` | Very Good | Tiny KV cache. Best for high concurrency or parallel Cline sessions. |
| 1x 16 GB GPU (home) | Qwen3.5-9B fp8 | `hermes` | Very Good | Latest gen, hybrid arch, punches above its weight on coding. |
| 1x 16 GB GPU (Meta) | Llama 3.1-8B-Instruct-AWQ | `llama3_json` | Good | Solid Meta option. Use 16k context to manage KV cache. |
| 1x 16 GB GPU (coding) | Qwen2.5-Coder-7B-Instruct-AWQ | `hermes` | Good | Dedicated coding model. Proven Cline compatibility. |
| Quick test (any GPU) | Qwen2.5-Coder-1.5B-Instruct fp16 | `hermes` | Smoke test | Tiny footprint. Validates setup before big download. |

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
