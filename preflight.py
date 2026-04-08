#!/usr/bin/env python3
"""
VRAM Preflight Check
====================
Estimates total GPU memory required (model weights + KV cache) and validates
that it fits within the user-specified GPU utilization budget *before* vLLM
starts serving.

Exit codes:
  0  — check passed
  1  — VRAM budget exceeded or runtime error
"""

import json
import math
import subprocess
import sys
from pathlib import Path

import yaml

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

BYTES_PER_GIB = 1 << 30

# Bytes per parameter for each quantization method
QUANT_BYTES = {
    "none": 2.0,    # fp16 / bf16
    "fp8": 1.0,
    "awq": 0.5,     # 4-bit
    "gptq": 0.5,    # 4-bit
    "squeezellm": 0.5,
}

# KV cache is always stored in fp16 (2 bytes per element) in vLLM
KV_DTYPE_BYTES = 2


def load_config(path: str = "/workspace/config.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def detect_gpus() -> list[dict]:
    """Return list of dicts with 'name' and 'vram_gib' for each GPU."""
    try:
        raw = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.total",
                "--format=csv,noheader,nounits",
            ],
            text=True,
        )
    except FileNotFoundError:
        print("ERROR: nvidia-smi not found. Are NVIDIA drivers installed?", file=sys.stderr)
        sys.exit(1)

    gpus = []
    for line in raw.strip().splitlines():
        name, mem_mib = [s.strip() for s in line.split(",")]
        gpus.append({"name": name, "vram_gib": float(mem_mib) / 1024.0})
    return gpus


def resolve_model_params(config: dict) -> tuple[float, int, int]:
    """
    Return (total_params_billion, num_hidden_layers, num_kv_heads, head_dim)
    by reading the HF model config.json.

    Falls back to active_params_billion from user config if set.
    """
    model_name = config["model"]["name"]

    # Try to read config.json from HF cache or local path
    hf_config = _load_hf_config(model_name)

    total_params_b = config["model"].get("active_params_billion")
    num_layers = hf_config.get("num_hidden_layers", 0)
    num_kv_heads = hf_config.get("num_key_value_heads") or hf_config.get("num_attention_heads", 0)
    head_dim = hf_config.get("head_dim") or (
        hf_config.get("hidden_size", 0) // hf_config.get("num_attention_heads", 1)
    )

    if total_params_b is None:
        # Estimate from HF config: very rough but serviceable
        # Better: user should set active_params_billion explicitly
        hidden = hf_config.get("hidden_size", 0)
        intermediate = hf_config.get("intermediate_size", 0)
        vocab = hf_config.get("vocab_size", 0)
        # Rough param estimate: embedding + transformer blocks
        est_params = vocab * hidden + num_layers * (
            4 * hidden * hidden  # attn projections (approx)
            + 2 * hidden * intermediate  # FFN up/down
        )
        total_params_b = est_params / 1e9
        print(f"  Auto-estimated model params: {total_params_b:.1f}B")

    return total_params_b, num_layers, num_kv_heads, head_dim


def _load_hf_config(model_name: str) -> dict:
    """Load config.json from HF cache or local path."""
    # Local path
    local = Path(model_name) / "config.json"
    if local.exists():
        return json.loads(local.read_text())

    # Search HF cache
    cache_root = Path.home() / ".cache" / "huggingface" / "hub"
    if cache_root.exists():
        # models--Org--Name format
        folder_name = "models--" + model_name.replace("/", "--")
        model_dir = cache_root / folder_name
        if model_dir.exists():
            # Find the latest snapshot
            refs = model_dir / "refs" / "main"
            if refs.exists():
                sha = refs.read_text().strip()
                cfg = model_dir / "snapshots" / sha / "config.json"
                if cfg.exists():
                    return json.loads(cfg.read_text())

    # Try huggingface_hub if available
    try:
        from huggingface_hub import hf_hub_download
        path = hf_hub_download(repo_id=model_name, filename="config.json")
        return json.loads(Path(path).read_text())
    except Exception:
        pass

    print(f"  WARNING: Could not load config.json for '{model_name}'.", file=sys.stderr)
    print("  Set 'active_params_billion' in config.yaml for accurate estimation.", file=sys.stderr)
    return {}


# ---------------------------------------------------------------------------
# VRAM estimation
# ---------------------------------------------------------------------------

def estimate_model_weight_vram_gib(total_params_b: float, quant: str) -> float:
    """Estimate VRAM for model weights in GiB."""
    bytes_per_param = QUANT_BYTES.get(quant, 2.0)
    total_bytes = total_params_b * 1e9 * bytes_per_param
    return total_bytes / BYTES_PER_GIB


def estimate_kv_cache_vram_gib(
    num_layers: int,
    num_kv_heads: int,
    head_dim: int,
    max_context_length: int,
    max_num_seqs: int,
) -> float:
    """
    Estimate peak KV cache VRAM in GiB.

    Per token per layer: 2 (K+V) * num_kv_heads * head_dim * dtype_bytes
    Total: above * num_layers * max_context_length * max_num_seqs
    """
    per_token_per_layer = 2 * num_kv_heads * head_dim * KV_DTYPE_BYTES
    total_bytes = per_token_per_layer * num_layers * max_context_length * max_num_seqs
    return total_bytes / BYTES_PER_GIB


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 60)
    print("  sandboxLLM — VRAM Preflight Check")
    print("=" * 60)

    config = load_config()

    # GPU detection
    gpus = detect_gpus()
    if not gpus:
        print("ERROR: No GPUs detected.", file=sys.stderr)
        sys.exit(1)

    tp = config["serving"].get("tensor_parallel", "auto")
    if tp == "auto":
        tp = len(gpus)
    else:
        tp = int(tp)

    if tp > len(gpus):
        print(f"ERROR: tensor_parallel={tp} but only {len(gpus)} GPU(s) detected.", file=sys.stderr)
        sys.exit(1)

    vram_override = config["gpu"].get("vram_per_gpu_gib")
    if vram_override:
        per_gpu_vram = float(vram_override)
    else:
        per_gpu_vram = min(g["vram_gib"] for g in gpus[:tp])

    max_util = config["gpu"]["max_utilization"]
    usable_per_gpu = per_gpu_vram * max_util
    total_usable = usable_per_gpu * tp

    print(f"\n  GPUs detected       : {len(gpus)}")
    for i, g in enumerate(gpus):
        print(f"    [{i}] {g['name']}  — {g['vram_gib']:.1f} GiB")
    print(f"  Tensor parallel     : {tp}")
    print(f"  VRAM per GPU        : {per_gpu_vram:.1f} GiB")
    print(f"  Max utilization     : {max_util:.0%}")
    print(f"  Usable VRAM (total) : {total_usable:.1f} GiB")

    # Model params
    quant = config["model"].get("quantization", "none")
    total_params_b, num_layers, num_kv_heads, head_dim = resolve_model_params(config)

    if num_layers == 0 or num_kv_heads == 0:
        print("\n  WARNING: Could not determine model architecture details.")
        print("  Skipping KV cache estimation — relying on vLLM's own checks.\n")
        return

    # Weight VRAM
    weight_vram = estimate_model_weight_vram_gib(total_params_b, quant)

    # KV cache VRAM
    max_ctx = config["serving"]["max_context_length"]
    max_seqs = config["serving"]["max_num_seqs"]
    kv_vram = estimate_kv_cache_vram_gib(num_layers, num_kv_heads, head_dim, max_ctx, max_seqs)

    total_required = weight_vram + kv_vram

    print(f"\n  Model               : {config['model']['name']}")
    print(f"  Quantization        : {quant}")
    print(f"  Active params       : {total_params_b:.1f}B")
    print(f"  Layers              : {num_layers}")
    print(f"  KV heads            : {num_kv_heads}")
    print(f"  Head dim            : {head_dim}")
    print(f"  Context length      : {max_ctx:,}")
    print(f"  Max concurrent seqs : {max_seqs}")
    print()
    print(f"  Model weights VRAM  : {weight_vram:.2f} GiB")
    print(f"  KV cache VRAM (peak): {kv_vram:.2f} GiB")
    print(f"  ─────────────────────────────────")
    print(f"  Total required      : {total_required:.2f} GiB")
    print(f"  Total usable budget : {total_usable:.2f} GiB")
    print()

    if total_required > total_usable:
        overshoot = total_required - total_usable
        print(f"  FAIL: Exceeds budget by {overshoot:.2f} GiB")
        print()
        print("  Suggestions:")
        print("    - Reduce max_context_length or max_num_seqs")
        print("    - Use quantization (awq, gptq, fp8)")
        print("    - Increase tensor_parallel (use more GPUs)")
        print("    - Raise max_utilization (at your own risk)")
        print()
        sys.exit(1)
    else:
        headroom = total_usable - total_required
        print(f"  PASS: {headroom:.2f} GiB headroom remaining")
        print()


if __name__ == "__main__":
    main()
