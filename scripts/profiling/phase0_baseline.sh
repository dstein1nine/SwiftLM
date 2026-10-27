#!/usr/bin/env bash
# Phase 0 baseline: decode tok/s, TTFT, peak RSS, MLX peak memory and SSD bytes
# read, for one MoE model across the existing streaming configurations.
#
# usage: scripts/profiling/phase0_baseline.sh <model-dir> [out-dir]
#   CONFIGS="resident stream stream_stacked stream_topk4"  (default: all)
#   RUNS=3 MAX_TOKENS=256
#
# Page cache: pread goes through the macOS unified buffer cache, so after one
# pass the expert reads are served from RAM, not NVMe. By default each config
# starts warm: the model files are read once first (WARM=0 skips this). For
# cold numbers set PURGE=1, which runs `sudo purge` before each config.
set -euo pipefail

MODEL_DIR="${1:?model directory}"
OUT="${2:-$(pwd)/.build/bench/phase0-$(date +%Y%m%d-%H%M%S)}"
RUNS="${RUNS:-3}"
MAX_TOKENS="${MAX_TOKENS:-256}"
PORT="${PORT:-5413}"
CONFIGS="${CONFIGS:-resident stream stream_stacked stream_topk4}"
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
# Build first: a stale binary silently benchmarks old code.
(cd "$ROOT" && swift build -c release --product SwiftLM) >&2
BIN="$(cd "$ROOT" && swift build -c release --show-bin-path)/SwiftLM"
mkdir -p "$OUT"

config_args() {
    case "$1" in
        resident)       echo "" ;;
        stream)         echo "--stream-experts" ;;
        stream_stacked) echo "--stream-experts" ;;
        stream_topk4)   echo "--stream-experts" ;;
        stream_stacked_sync|stream_legacy) echo "--stream-experts" ;;
        *) echo "unknown config $1" >&2; exit 2 ;;
    esac
}
config_env() {
    case "$1" in
        stream_stacked) echo "MLX_MOE_STACKED=1" ;;
        stream_topk4)   echo "SWIFTLM_TOP_K=4" ;;
        stream_stacked_sync) echo "MLX_MOE_STACKED=1 MLX_MOE_MISS_SYNC=1" ;;
        stream_legacy)  echo "MLX_MOE_STACKED=0" ;;
        *)              echo "" ;;
    esac
}
disk_read_mb() { iostat -Id disk0 | awk 'NR==3{print $3}'; }  # cumulative MB on disk0

{
    echo "machine: $(sysctl -n machdep.cpu.brand_string), $(( $(sysctl -n hw.memsize) / 1073741824 )) GB"
    echo "macOS:   $(sw_vers -productVersion)"
    echo "commit:  $(git -C "$ROOT" rev-parse --short HEAD) $(git -C "$ROOT" submodule status | awk '{print $2"@"substr($1,1,8)}' | tr '\n' ' ')"
    echo "model:   $MODEL_DIR ($(du -sh "$MODEL_DIR" | cut -f1))"
} | tee "$OUT/env.txt"

for cfg in $CONFIGS; do
    echo "== $cfg"
    if [[ "${PURGE:-0}" == 1 ]]; then
        sudo purge
    elif [[ "${WARM:-1}" == 1 ]]; then
        cat "$MODEL_DIR"/*.safetensors >/dev/null
    fi
    log="$OUT/$cfg.server.log"
    env $(config_env "$cfg") "$BIN" --model "$MODEL_DIR" --port "$PORT" --no-token-echo $(config_args "$cfg") \
        >"$log" 2>&1 &
    pid=$!
    for _ in $(seq 1 900); do
        curl -sf "http://127.0.0.1:$PORT/v1/models" >/dev/null 2>&1 && break
        kill -0 $pid 2>/dev/null || { echo "server died, see $log"; tail -20 "$log"; exit 1; }
        sleep 1
    done
    rss_after_load=$(ps -o rss= -p $pid | tr -d ' ')

    ( peak=0; while kill -0 $pid 2>/dev/null; do
          r=$(ps -o rss= -p $pid 2>/dev/null | tr -d ' ' || echo 0)
          (( r > peak )) && peak=$r && echo "$peak" >"$OUT/$cfg.peak_rss_kb"
          sleep 0.25
      done ) &
    sampler=$!

    d0=$(disk_read_mb)
    python3 -I "$ROOT/scripts/profiling/streaming_bench_client.py" \
        --base "http://127.0.0.1:$PORT" --runs "$RUNS" --max-tokens "$MAX_TOKENS" --label "$cfg" \
        >"$OUT/$cfg.json"
    d1=$(disk_read_mb)

    kill $pid 2>/dev/null; wait $pid 2>/dev/null || true
    kill $sampler 2>/dev/null; wait $sampler 2>/dev/null || true
    python3 -I - "$OUT/$cfg.json" "$rss_after_load" "$(cat "$OUT/$cfg.peak_rss_kb" 2>/dev/null || echo 0)" "$d0" "$d1" <<'PY'
import json, sys
path, rss_load, rss_peak, d0, d1 = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), float(sys.argv[4]), float(sys.argv[5])
s = json.load(open(path))
s["rss_after_load_gb"] = rss_load / 1048576
s["rss_peak_gb"] = rss_peak / 1048576
s["disk0_read_gb_during_bench"] = (d1 - d0) / 1024
json.dump(s, open(path, "w"), indent=2)
PY
    sleep 3
done

python3 -I - "$OUT" $CONFIGS <<'PY'
import json, sys, os
out, cfgs = sys.argv[1], sys.argv[2:]
print(f"\n{'config':16} {'decode tok/s (med [min-max])':30} {'TTFT':>6} {'RSS peak':>9} {'MLX peak':>9} {'disk0 read':>10} det")
for c in cfgs:
    s = json.load(open(os.path.join(out, c + ".json")))
    mp = s["metrics"].get("swiftlm_memory_peak_bytes", 0) / 1e9
    rate = f"{s['decode_tok_s_median']:.2f} [{s['decode_tok_s_min']:.2f}-{s['decode_tok_s_max']:.2f}]"
    m = s["metrics"]
    hits, misses = m.get("swiftlm_stream_expert_hits_total", 0), m.get("swiftlm_stream_expert_misses_total", 0)
    hit = f" hit {hits / (hits + misses):.0%}" if hits + misses else ""
    print(f"{c:16} {rate:30} {s['ttft_s_median']:5.2f}s {s['rss_peak_gb']:7.1f}GB {mp:7.1f}GB {s['disk0_read_gb_during_bench']:8.2f}GB {s['deterministic']}{hit}")
PY
echo "results: $OUT"
