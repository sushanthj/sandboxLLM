#!/usr/bin/env python3
"""
sandboxLLM Dashboard
====================
Lightweight Flask server that exposes GPU, CPU, RAM, and temperature metrics
plus the list of models currently being served by vLLM.

Designed to run as a sidecar container alongside the vLLM service.
"""

import json
import os
import subprocess
import time

import yaml
from flask import Flask, jsonify, Response

app = Flask(__name__)

VLLM_INTERNAL_URL = os.environ.get("VLLM_URL", "http://vllm:8000")

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

def _load_config() -> dict:
    try:
        with open("/workspace/config.yaml") as f:
            return yaml.safe_load(f) or {}
    except Exception:
        return {}

_CONFIG = _load_config()
VLLM_API_KEY = _CONFIG.get("serving", {}).get("api_key", "") or ""
VLLM_MODEL = _CONFIG.get("model", {}).get("name", "")
VLLM_PORT = _CONFIG.get("serving", {}).get("port", 7171)


def _lan_ip() -> str:
    """Return the host's LAN IP.

    Priority: HOST_LAN_IP env var > auto-detect from host's PID 1 network
    namespace (accessible via bind-mounted /host/proc/1/net/route + fib_trie).
    """
    env_ip = os.environ.get("HOST_LAN_IP", "")
    if env_ip:
        return env_ip

    import struct

    def _hex_to_ip(h: str) -> str:
        return ".".join(str(b) for b in struct.pack("<I", int(h, 16)))

    try:
        # Find the default-route interface from the HOST's routing table
        default_iface = ""
        with open("/host/proc/1/net/route") as f:
            for line in f:
                parts = line.split()
                if len(parts) >= 2 and parts[1] == "00000000":
                    default_iface = parts[0]
                    break
        if not default_iface:
            return ""

        # Find the subnet for that interface
        with open("/host/proc/1/net/route") as f:
            next(f)
            for line in f:
                parts = line.split()
                if (len(parts) >= 8 and parts[0] == default_iface
                        and parts[1] != "00000000"):
                    subnet_ip = _hex_to_ip(parts[1])
                    break
            else:
                return ""

        # Find the LOCAL IP in that subnet from host's fib_trie
        prefix = subnet_ip.rsplit(".", 1)[0] + "."
        with open("/host/proc/1/net/fib_trie") as f:
            lines = f.readlines()
        for i, line in enumerate(lines):
            stripped = line.strip()
            if not (stripped.startswith("|--") or stripped.startswith("+--")):
                continue
            ip_str = stripped.split()[-1]
            if ip_str.startswith(prefix) and i + 1 < len(lines) and "/32 host LOCAL" in lines[i + 1]:
                return ip_str
        return ""
    except Exception:
        return ""

# ---------------------------------------------------------------------------
# Data collectors
# ---------------------------------------------------------------------------

def gpu_stats() -> list[dict]:
    """Query nvidia-smi for per-GPU metrics."""
    try:
        raw = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu="
                "index,name,temperature.gpu,utilization.gpu,utilization.memory,"
                "memory.used,memory.total,power.draw,power.limit",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            timeout=5,
        )
    except Exception:
        return []

    gpus = []
    for line in raw.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 9:
            continue
        gpus.append({
            "index": int(parts[0]),
            "name": parts[1],
            "temp_c": _safe_float(parts[2]),
            "gpu_util_pct": _safe_float(parts[3]),
            "mem_util_pct": _safe_float(parts[4]),
            "mem_used_mib": _safe_float(parts[5]),
            "mem_total_mib": _safe_float(parts[6]),
            "power_w": _safe_float(parts[7]),
            "power_limit_w": _safe_float(parts[8]),
        })
    return gpus


def cpu_stats() -> dict:
    """Read /proc/stat and /proc/loadavg for CPU metrics."""
    try:
        with open("/host/proc/stat") as f:
            first_line = f.readline()  # cpu  user nice system idle ...
        parts = first_line.split()
        user, nice, system, idle, iowait = (int(parts[i]) for i in range(1, 6))
        total = user + nice + system + idle + iowait
        busy = total - idle
        utilization = round(busy / total * 100, 1) if total else 0

        with open("/host/proc/loadavg") as f:
            load_parts = f.read().split()
        load_1, load_5, load_15 = load_parts[0], load_parts[1], load_parts[2]
    except Exception:
        return {"error": "could not read /proc"}

    return {
        "utilization_pct": utilization,
        "load_avg_1m": float(load_1),
        "load_avg_5m": float(load_5),
        "load_avg_15m": float(load_15),
    }


def ram_stats() -> dict:
    """Read /proc/meminfo for RAM metrics."""
    try:
        info = {}
        with open("/host/proc/meminfo") as f:
            for line in f:
                key, rest = line.split(":", 1)
                val_kb = int(rest.strip().split()[0])
                info[key] = val_kb
        total = info.get("MemTotal", 0)
        available = info.get("MemAvailable", 0)
        used = total - available
        return {
            "total_gib": round(total / 1048576, 1),
            "used_gib": round(used / 1048576, 1),
            "available_gib": round(available / 1048576, 1),
            "used_pct": round(used / total * 100, 1) if total else 0,
        }
    except Exception:
        return {"error": "could not read /proc/meminfo"}


def cpu_temp() -> float | None:
    """Try to read CPU temperature from thermal zones."""
    try:
        # Try host sysfs first
        for base in ["/host/sys/class/thermal", "/sys/class/thermal"]:
            try:
                entries = os.listdir(base)
            except FileNotFoundError:
                continue
            for entry in sorted(entries):
                if entry.startswith("thermal_zone"):
                    temp_path = os.path.join(base, entry, "temp")
                    type_path = os.path.join(base, entry, "type")
                    try:
                        with open(type_path) as f:
                            zone_type = f.read().strip()
                        if "cpu" in zone_type.lower() or "x86" in zone_type.lower() or "coretemp" in zone_type.lower():
                            with open(temp_path) as f:
                                return round(int(f.read().strip()) / 1000, 1)
                    except (FileNotFoundError, ValueError):
                        continue
            # Fallback: return first thermal zone
            for entry in sorted(entries):
                if entry.startswith("thermal_zone"):
                    temp_path = os.path.join(base, entry, "temp")
                    try:
                        with open(temp_path) as f:
                            return round(int(f.read().strip()) / 1000, 1)
                    except (FileNotFoundError, ValueError):
                        continue
    except Exception:
        pass
    return None


def vllm_models() -> list[dict]:
    """Query the vLLM sidecar for served models."""
    import urllib.request
    try:
        req = urllib.request.Request(f"{VLLM_INTERNAL_URL}/v1/models", method="GET")
        if VLLM_API_KEY:
            req.add_header("Authorization", f"Bearer {VLLM_API_KEY}")
        with urllib.request.urlopen(req, timeout=3) as resp:
            data = json.loads(resp.read())
        return data.get("data", [])
    except Exception:
        return []


def _safe_float(s: str) -> float | None:
    try:
        return float(s)
    except (ValueError, TypeError):
        return None


LOG_FILE = "/var/log/sandboxllm/vllm.log"
LOG_MAX_BYTES = 10 * 1024 * 1024  # 10 MB — truncate to last 5 MB when exceeded


def read_logs(tail: int = 200) -> str:
    """Read the last N lines from the shared vLLM log file."""
    try:
        # Truncate if log file is too large (keep last half)
        try:
            size = os.path.getsize(LOG_FILE)
            if size > LOG_MAX_BYTES:
                with open(LOG_FILE, "r+") as f:
                    f.seek(size - LOG_MAX_BYTES // 2)
                    f.readline()  # skip partial line
                    keep = f.read()
                    f.seek(0)
                    f.write(keep)
                    f.truncate()
        except (OSError, IOError):
            pass

        with open(LOG_FILE, "r") as f:
            lines = f.readlines()
        return "".join(lines[-tail:])
    except FileNotFoundError:
        return "(waiting for vLLM to start...)\n"
    except Exception as e:
        return f"(error reading logs: {e})\n"


# ---------------------------------------------------------------------------
# API endpoints
# ---------------------------------------------------------------------------

@app.route("/api/stats")
def api_stats():
    """JSON endpoint returning all metrics."""
    return jsonify({
        "timestamp": time.time(),
        "gpus": gpu_stats(),
        "cpu": cpu_stats(),
        "cpu_temp_c": cpu_temp(),
        "ram": ram_stats(),
        "served_models": vllm_models(),
    })


@app.route("/api/connection")
def api_connection():
    """Return connection details for Cline / API clients."""
    return jsonify({
        "model": VLLM_MODEL,
        "port": VLLM_PORT,
        "has_api_key": bool(VLLM_API_KEY),
        "lan_ip": _lan_ip(),
    })


@app.route("/api/logs")
def api_logs():
    """Return recent vLLM logs (preflight + server output)."""
    from flask import request
    tail = min(int(request.args.get("tail", 200)), 2000)
    return jsonify({"logs": read_logs(tail)})


@app.route("/")
def index():
    """Serve the single-page dashboard."""
    return Response(DASHBOARD_HTML, mimetype="text/html")


# ---------------------------------------------------------------------------
# Inline HTML/JS dashboard (no build step, no npm)
# ---------------------------------------------------------------------------

DASHBOARD_HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>sandboxLLM Dashboard</title>
<style>
  :root {
    --bg: #0f1117; --card: #1a1d27; --border: #2a2d3a;
    --text: #e1e4ed; --muted: #8b8fa3; --accent: #6c72ff;
    --green: #34d399; --yellow: #fbbf24; --red: #f87171;
  }
  * { margin: 0; padding: 0; box-sizing: border-box; }
  body { background: var(--bg); color: var(--text); font-family: 'Inter', system-ui, -apple-system, sans-serif; padding: 24px; }
  h1 { font-size: 1.5rem; font-weight: 600; margin-bottom: 4px; }
  .subtitle { color: var(--muted); font-size: 0.85rem; margin-bottom: 20px; }

  /* ---- Tabs ---- */
  .tabs { display: flex; gap: 0; margin-bottom: 20px; border-bottom: 2px solid var(--border); }
  .tab { padding: 10px 24px; cursor: pointer; color: var(--muted); font-size: 0.9rem; font-weight: 500;
         border-bottom: 2px solid transparent; margin-bottom: -2px; transition: all 0.2s; user-select: none; }
  .tab:hover { color: var(--text); }
  .tab.active { color: var(--accent); border-bottom-color: var(--accent); }
  .tab-content { display: none; }
  .tab-content.active { display: block; }

  /* ---- Cards grid ---- */
  .grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(340px, 1fr)); gap: 16px; margin-bottom: 16px; }
  .card { background: var(--card); border: 1px solid var(--border); border-radius: 12px; padding: 20px; }
  .card h2 { font-size: 0.95rem; font-weight: 500; color: var(--muted); margin-bottom: 14px; text-transform: uppercase; letter-spacing: 0.05em; }
  .metric { display: flex; justify-content: space-between; align-items: center; padding: 8px 0; border-bottom: 1px solid var(--border); }
  .metric:last-child { border-bottom: none; }
  .metric-label { font-size: 0.85rem; color: var(--muted); }
  .metric-value { font-size: 1.1rem; font-weight: 600; font-variant-numeric: tabular-nums; }
  .bar-wrap { width: 120px; height: 8px; background: var(--border); border-radius: 4px; overflow: hidden; margin-left: 12px; flex-shrink: 0; }
  .bar { height: 100%; border-radius: 4px; transition: width 0.6s ease; }
  .bar.green { background: var(--green); }
  .bar.yellow { background: var(--yellow); }
  .bar.red { background: var(--red); }
  .val-group { display: flex; align-items: center; }
  .gpu-card { margin-bottom: 12px; padding: 14px; background: rgba(108,114,255,0.06); border-radius: 8px; border: 1px solid var(--border); }
  .gpu-name { font-weight: 600; font-size: 0.9rem; margin-bottom: 10px; color: var(--accent); }
  .model-chip { display: inline-block; background: rgba(108,114,255,0.15); color: var(--accent); padding: 6px 14px; border-radius: 20px; font-size: 0.85rem; font-weight: 500; margin: 4px 4px 4px 0; }
  .status-dot { width: 8px; height: 8px; border-radius: 50%; display: inline-block; margin-right: 8px; }
  .status-dot.on { background: var(--green); }
  .status-dot.off { background: var(--red); }

  /* ---- Logs ---- */
  .log-container { background: var(--card); border: 1px solid var(--border); border-radius: 12px; padding: 20px; }
  .log-header { display: flex; justify-content: space-between; align-items: center; margin-bottom: 14px; }
  .log-header h2 { margin-bottom: 0; font-size: 0.95rem; font-weight: 500; color: var(--muted); text-transform: uppercase; letter-spacing: 0.05em; }
  .log-controls { display: flex; gap: 8px; align-items: center; }
  .log-controls label { font-size: 0.8rem; color: var(--muted); cursor: pointer; }
  .log-controls input[type="checkbox"] { accent-color: var(--accent); }
  .log-pre { background: #0a0c10; border: 1px solid var(--border); border-radius: 8px; padding: 16px;
             font-family: 'JetBrains Mono', 'Fira Code', 'Cascadia Code', monospace; font-size: 0.8rem;
             line-height: 1.6; color: var(--text); overflow-x: auto; max-height: 70vh; overflow-y: auto;
             white-space: pre-wrap; word-break: break-word; }
  .log-pre .log-pass { color: var(--green); font-weight: 600; }
  .log-pre .log-fail { color: var(--red); font-weight: 600; }
  .log-pre .log-warn { color: var(--yellow); }
  .log-pre .log-header-line { color: var(--accent); font-weight: 600; }
  .log-pre .log-dim { color: var(--muted); }

  /* ---- Connect tab ---- */
  .connect-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(400px, 1fr)); gap: 16px; }
  .setting-row { display: flex; justify-content: space-between; align-items: center; padding: 10px 0; border-bottom: 1px solid var(--border); }
  .setting-row:last-child { border-bottom: none; }
  .setting-label { font-size: 0.85rem; color: var(--muted); flex-shrink: 0; margin-right: 16px; }
  .setting-value { font-family: 'JetBrains Mono', 'Fira Code', monospace; font-size: 0.9rem; color: var(--text);
                   background: #0a0c10; padding: 6px 12px; border-radius: 6px; border: 1px solid var(--border);
                   user-select: all; word-break: break-all; }
  .setting-value.secret { filter: blur(4px); transition: filter 0.2s; cursor: pointer; }
  .setting-value.secret:hover, .setting-value.secret.revealed { filter: none; }
  .connect-hint { font-size: 0.8rem; color: var(--muted); margin-top: 12px; line-height: 1.5; }

  .footer { text-align: center; color: var(--muted); font-size: 0.75rem; margin-top: 24px; }
  #error-banner { display: none; background: rgba(248,113,113,0.15); color: var(--red); padding: 10px 16px; border-radius: 8px; margin-bottom: 16px; font-size: 0.85rem; }
</style>
</head>
<body>

<h1>sandboxLLM Dashboard</h1>
<p class="subtitle">Auto-refreshes every 2 seconds &bull; <span id="last-update">--</span></p>
<div id="error-banner"></div>

<!-- Tabs -->
<div class="tabs">
  <div class="tab active" data-tab="monitoring">Monitoring</div>
  <div class="tab" data-tab="logs">Logs</div>
  <div class="tab" data-tab="connect">Connect</div>
</div>

<!-- Monitoring tab -->
<div class="tab-content active" id="tab-monitoring">
  <div class="grid">
    <div class="card" id="gpu-section">
      <h2>GPUs</h2>
      <div id="gpu-container"><span style="color:var(--muted)">Loading...</span></div>
    </div>
    <div class="card">
      <h2>CPU</h2>
      <div id="cpu-container"><span style="color:var(--muted)">Loading...</span></div>
    </div>
    <div class="card">
      <h2>System RAM</h2>
      <div id="ram-container"><span style="color:var(--muted)">Loading...</span></div>
    </div>
    <div class="card">
      <h2>Served Models</h2>
      <div id="model-container"><span style="color:var(--muted)">Loading...</span></div>
    </div>
  </div>
</div>

<!-- Logs tab -->
<div class="tab-content" id="tab-logs">
  <div class="log-container">
    <div class="log-header">
      <h2>vLLM Server Logs</h2>
      <div class="log-controls">
        <label><input type="checkbox" id="log-autoscroll" checked> Auto-scroll</label>
      </div>
    </div>
    <pre class="log-pre" id="log-output">Loading...</pre>
  </div>
</div>

<!-- Connect tab -->
<div class="tab-content" id="tab-connect">
  <div class="connect-grid">
    <div class="card">
      <h2>Cline Settings</h2>
      <p class="connect-hint" style="margin-bottom:14px">Copy these into Cline &gt; Settings (gear icon) &gt; API Provider: <strong>OpenAI Compatible</strong></p>
      <div class="setting-row">
        <span class="setting-label">API Provider</span>
        <span class="setting-value">OpenAI Compatible</span>
      </div>
      <div class="setting-row">
        <span class="setting-label">Base URL</span>
        <span class="setting-value" id="conn-base-url">--</span>
      </div>
      <div class="setting-row">
        <span class="setting-label">API Key</span>
        <span class="setting-value secret" id="conn-api-key" title="Hover to reveal, click to keep visible">--</span>
      </div>
      <div class="setting-row">
        <span class="setting-label">Model ID</span>
        <span class="setting-value" id="conn-model-id">--</span>
      </div>
    </div>
    <div class="card">
      <h2>Quick Test</h2>
      <p class="connect-hint" style="margin-bottom:14px">Run this from your other PC to verify connectivity:</p>
      <pre class="log-pre" id="conn-curl-cmd" style="max-height:none;font-size:0.8rem;">Loading...</pre>
      <p class="connect-hint" style="margin-top:14px">If you get a JSON response listing the model, the connection is working.</p>
    </div>
  </div>
</div>

<div class="footer">sandboxLLM &mdash; secure local LLM serving</div>

<script>
/* ---- Tab switching ---- */
document.querySelectorAll('.tab').forEach(tab => {
  tab.addEventListener('click', () => {
    document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
    document.querySelectorAll('.tab-content').forEach(c => c.classList.remove('active'));
    tab.classList.add('active');
    document.getElementById('tab-' + tab.dataset.tab).classList.add('active');
    // Fetch logs immediately when switching to logs tab
    if (tab.dataset.tab === 'logs') refreshLogs();
  });
});

/* ---- Monitoring helpers ---- */
function barClass(pct) {
  if (pct < 60) return 'green';
  if (pct < 85) return 'yellow';
  return 'red';
}
function bar(pct) {
  const p = pct ?? 0;
  return `<div class="bar-wrap"><div class="bar ${barClass(p)}" style="width:${p}%"></div></div>`;
}
function fmt(v, unit) {
  if (v === null || v === undefined) return '--';
  return v + (unit || '');
}

function renderGPUs(gpus) {
  if (!gpus.length) return '<span style="color:var(--muted)">No GPUs detected</span>';
  return gpus.map(g => `
    <div class="gpu-card">
      <div class="gpu-name">GPU ${g.index}: ${g.name}</div>
      <div class="metric">
        <span class="metric-label">Utilization</span>
        <div class="val-group"><span class="metric-value">${fmt(g.gpu_util_pct, '%')}</span>${bar(g.gpu_util_pct)}</div>
      </div>
      <div class="metric">
        <span class="metric-label">VRAM</span>
        <div class="val-group"><span class="metric-value">${fmt(g.mem_used_mib,'M')} / ${fmt(g.mem_total_mib,'M')}</span>${bar(g.mem_total_mib ? Math.round(g.mem_used_mib / g.mem_total_mib * 100) : 0)}</div>
      </div>
      <div class="metric">
        <span class="metric-label">Temperature</span>
        <span class="metric-value">${fmt(g.temp_c, '\u00b0C')}</span>
      </div>
      <div class="metric">
        <span class="metric-label">Power</span>
        <span class="metric-value">${fmt(g.power_w,'W')} / ${fmt(g.power_limit_w,'W')}</span>
      </div>
    </div>
  `).join('');
}

function renderCPU(cpu, temp) {
  if (cpu.error) return `<span style="color:var(--red)">${cpu.error}</span>`;
  return `
    <div class="metric">
      <span class="metric-label">Utilization</span>
      <div class="val-group"><span class="metric-value">${fmt(cpu.utilization_pct, '%')}</span>${bar(cpu.utilization_pct)}</div>
    </div>
    <div class="metric">
      <span class="metric-label">Temperature</span>
      <span class="metric-value">${temp !== null ? fmt(temp, '\u00b0C') : '--'}</span>
    </div>
    <div class="metric">
      <span class="metric-label">Load (1 / 5 / 15m)</span>
      <span class="metric-value">${cpu.load_avg_1m} / ${cpu.load_avg_5m} / ${cpu.load_avg_15m}</span>
    </div>
  `;
}

function renderRAM(ram) {
  if (ram.error) return `<span style="color:var(--red)">${ram.error}</span>`;
  return `
    <div class="metric">
      <span class="metric-label">Used</span>
      <div class="val-group"><span class="metric-value">${ram.used_gib} / ${ram.total_gib} GiB</span>${bar(ram.used_pct)}</div>
    </div>
    <div class="metric">
      <span class="metric-label">Available</span>
      <span class="metric-value">${ram.available_gib} GiB</span>
    </div>
  `;
}

function renderModels(models) {
  if (!models.length) return '<span class="status-dot off"></span><span style="color:var(--muted)">No models loaded (vLLM may still be starting)</span>';
  return models.map(m => `<span class="model-chip"><span class="status-dot on"></span>${m.id}</span>`).join('');
}

/* ---- Log highlighting ---- */
function highlightLogs(raw) {
  return raw
    .replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;')
    .replace(/^(=+.*=+)$/gm, '<span class="log-header-line">$1</span>')
    .replace(/^(.*PASS.*)$/gm, '<span class="log-pass">$1</span>')
    .replace(/^(.*FAIL.*)$/gm, '<span class="log-fail">$1</span>')
    .replace(/^(.*WARNING.*)$/gm, '<span class="log-warn">$1</span>')
    .replace(/^(.*ERROR.*)$/gm, '<span class="log-fail">$1</span>')
    .replace(/^(  Started at:.*)$/gm, '<span class="log-dim">$1</span>')
    .replace(/^(  Command:.*)$/gm, '<span class="log-dim">$1</span>');
}

/* ---- Refresh functions ---- */
async function refreshStats() {
  try {
    const resp = await fetch('/api/stats');
    if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
    const d = await resp.json();
    document.getElementById('gpu-container').innerHTML = renderGPUs(d.gpus);
    document.getElementById('cpu-container').innerHTML = renderCPU(d.cpu, d.cpu_temp_c);
    document.getElementById('ram-container').innerHTML = renderRAM(d.ram);
    document.getElementById('model-container').innerHTML = renderModels(d.served_models);
    document.getElementById('last-update').textContent = new Date().toLocaleTimeString();
    document.getElementById('error-banner').style.display = 'none';
  } catch (e) {
    const el = document.getElementById('error-banner');
    el.textContent = 'Could not fetch metrics: ' + e.message;
    el.style.display = 'block';
  }
}

async function refreshLogs() {
  try {
    const resp = await fetch('/api/logs?tail=500');
    if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
    const d = await resp.json();
    const el = document.getElementById('log-output');
    el.innerHTML = highlightLogs(d.logs);
    if (document.getElementById('log-autoscroll').checked) {
      el.scrollTop = el.scrollHeight;
    }
  } catch (e) {
    document.getElementById('log-output').textContent = 'Error fetching logs: ' + e.message;
  }
}

/* ---- Connection info ---- */
document.querySelectorAll('.setting-value.secret').forEach(el => {
  el.addEventListener('click', () => el.classList.toggle('revealed'));
});

async function refreshConnection() {
  try {
    const resp = await fetch('/api/connection');
    if (!resp.ok) return;
    const d = await resp.json();
    const host = d.lan_ip || window.location.hostname;
    const port = d.port || 7171;
    const baseUrl = `http://${host}:${port}/v1`;
    document.getElementById('conn-base-url').textContent = baseUrl;
    document.getElementById('conn-model-id').textContent = d.model || '--';
    const keyEl = document.getElementById('conn-api-key');
    if (d.has_api_key) {
      keyEl.textContent = '(set in config.yaml)';
      keyEl.classList.remove('secret');
    } else {
      keyEl.textContent = '(none — no auth required)';
      keyEl.classList.remove('secret');
    }

    let curl = `curl ${d.has_api_key ? `-H "Authorization: Bearer YOUR_API_KEY" ` : ''}${baseUrl}/models`;
    document.getElementById('conn-curl-cmd').textContent = curl;
  } catch(e) {}
}

/* ---- Polling ---- */
refreshStats();
refreshLogs();
refreshConnection();
setInterval(refreshStats, 2000);
setInterval(refreshLogs, 3000);
</script>
</body>
</html>
"""

if __name__ == "__main__":
    port = int(os.environ.get("DASHBOARD_PORT", "7172"))
    app.run(host="0.0.0.0", port=port, threaded=True)
