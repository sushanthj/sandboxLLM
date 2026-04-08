#!/usr/bin/env bash
set -euo pipefail

CONFIG="/workspace/config.yaml"
LOG_DIR="/var/log/sandboxllm"
LOG_FILE="${LOG_DIR}/vllm.log"

# Ensure log directory exists (shared volume)
mkdir -p "$LOG_DIR"

# Tee all stdout/stderr to the log file AND the console
exec > >(tee -a "$LOG_FILE") 2>&1

echo "============================================================"
echo "  sandboxLLM — Secure vLLM Entrypoint"
echo "============================================================"
echo "  Started at: $(date -u '+%Y-%m-%d %H:%M:%S UTC')"

# Helper: read a yaml value (simple — works for flat scalar values)
yq() {
    python3 -c "
import yaml, sys
cfg = yaml.safe_load(open('$CONFIG'))
keys = '$1'.split('.')
v = cfg
for k in keys:
    v = v[k]
print('' if v is None else v)
"
}

MODEL=$(yq model.name)

# --- Phase 1: Download model if not already cached ---
echo ""
echo "[1/3] Checking model cache..."

# Check if the model is already downloaded by looking for config.json in the cache
MODEL_CACHED=false
if python3 -c "
from huggingface_hub import try_to_load_from_cache
import sys
result = try_to_load_from_cache('$MODEL', 'config.json')
sys.exit(0 if result is not None else 1)
" 2>/dev/null; then
    MODEL_CACHED=true
fi

if [ "$MODEL_CACHED" = true ]; then
    echo "  Model '$MODEL' found in cache. Skipping download."
else
    echo "  Model '$MODEL' not in cache. Downloading..."
    echo "  (This is a one-time download — cached in a Docker volume for future runs)"
    echo ""
    huggingface-cli download "$MODEL"
    echo ""
    echo "  Download complete."
fi

# --- Lock down: enable offline mode for the rest of the process ---
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
echo "[1/3] Offline mode enabled. No further network calls."

# --- Phase 2: Preflight VRAM check ---
echo ""
echo "[2/3] Running VRAM preflight check..."
python3 /workspace/preflight.py
echo "[2/3] Preflight passed."

# --- Phase 3: Parse config and build vLLM args ---
echo "[3/3] Starting vLLM server..."
QUANT=$(yq model.quantization)
MAX_CTX=$(yq serving.max_context_length)
MAX_SEQS=$(yq serving.max_num_seqs)
PORT=$(yq serving.port)
API_KEY=$(yq serving.api_key)
TP=$(yq serving.tensor_parallel)
MAX_UTIL=$(yq gpu.max_utilization)
KV_GPU_ONLY=$(yq gpu.kv_cache_gpu_only)

# Build argument list
ARGS=(
    --model "$MODEL"
    --max-model-len "$MAX_CTX"
    --max-num-seqs "$MAX_SEQS"
    --port "$PORT"
    --gpu-memory-utilization "$MAX_UTIL"
    --trust-remote-code
)

# Tensor parallel
if [ "$TP" = "auto" ]; then
    NUM_GPUS=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
    ARGS+=(--tensor-parallel-size "$NUM_GPUS")
else
    ARGS+=(--tensor-parallel-size "$TP")
fi

# Quantization
if [ "$QUANT" != "none" ] && [ -n "$QUANT" ]; then
    ARGS+=(--quantization "$QUANT")
fi

# API key
if [ -n "$API_KEY" ]; then
    ARGS+=(--api-key "$API_KEY")
fi

# Force KV cache on GPU
if [ "$KV_GPU_ONLY" = "True" ] || [ "$KV_GPU_ONLY" = "true" ]; then
    ARGS+=(--swap-space 0)
fi

echo "  Command: python -m vllm.entrypoints.openai.api_server ${ARGS[*]}"
echo ""

exec python3 -m vllm.entrypoints.openai.api_server "${ARGS[@]}"
