#!/usr/bin/env bash
# KV-pressure test of preloading (docs/preload_theory.md, Lemma K / Section 5a), run ON the pod.
# For each KV cap: restart vLLM with --num-gpu-blocks-override (16-token blocks), confirm the KV size it reports,
# then run the preload preset on arrival seed 2 (full-KV baselines: results/real/preload_seeds/seed2*).
# Finally restore the default server and run the overhead check (small tool results; Section 5b).
# Usage (pod): nohup scripts/kv_pressure.sh > results/logs/kv_pressure.log 2>&1 &
set -uo pipefail
cd /workspace/project
source /opt/venv/bin/activate
ulimit -n 65536
export HF_HOME=/workspace/hf HF_HUB_OFFLINE=1  # model lives on the volume; never re-download to the small root disk
OUT=results/real/preload_kv

while pgrep -f "preload-modes on chai[n]" > /dev/null; do sleep 20; done  # let a running sweep finish

restart() {  # $1 = KV blocks ("" = default)
  pkill -f "vllm serve" || true
  for _ in $(seq 60); do pgrep -f "vllm serve" > /dev/null || break; sleep 2; done
  sleep 5
  local log=results/logs/serve_kv${1:-default}.log
  KV_BLOCKS="$1" PREFIX_CACHE=1 MAXLEN=16384 MAXSEQS=256 SCHED=priority nohup scripts/serve_model.sh > "$log" 2>&1 &
  local spid=$!  # serve_model.sh execs vllm, so this pid is the server for its whole life
  for _ in $(seq 120); do
    curl -sf http://127.0.0.1:8000/health > /dev/null && break
    kill -0 "$spid" 2> /dev/null || break  # server died during startup
    sleep 5
  done
  curl -sf http://127.0.0.1:8000/health > /dev/null || { echo "SERVER_FAILED kv=$1"; tail -20 "$log"; exit 1; }
  echo "SERVER_UP kv=${1:-default} $(grep -o 'GPU KV cache size: [0-9,]* tokens' "$log" | tail -1)"
}

run() {  # $1 = tag, $2 = rate, rest = preload modes (+ extra args after --)
  local tag=$1 rate=$2; shift 2
  python scripts/real_multiagent.py --preset preload --seed 2 --rates "$rate" --duration 180 --out "$OUT/$tag" \
    --preload-modes "$@" || echo "RUN_FAILED $tag r$rate"
}

restart 5000   # 80k tokens: predicted still safe at 0.6/s
run kv80k 0.6 off on
restart 3125   # 50k tokens: predicted saving shrinks (~2.3 s in the engine model)
run kv50k 0.6 off on low
restart 2500   # 40k tokens: predicted harmful at 0.6/s (~-1.3 s), nearly harmless at 0.3/s (~-0.1 s)
run kv40k 0.6 off on low
run kv40k 0.3 off on

restart ""     # default KV (159,904 tokens): overhead check with small tool results at 0.45/s
python scripts/real_multiagent.py --preset preload --seed 2 --rates 0.45 --duration 180 --preload-modes off on \
  --set result_cap=40 --out "$OUT/cap40" || echo "RUN_FAILED cap40"
python scripts/real_multiagent.py --preset preload --seed 2 --rates 0.45 --duration 180 --preload-modes off on \
  --set result_cap=200 --out "$OUT/cap200" || echo "RUN_FAILED cap200"
echo "KV_PRESSURE_EXIT"
