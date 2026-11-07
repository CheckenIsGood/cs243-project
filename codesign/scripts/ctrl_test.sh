#!/usr/bin/env bash
# GPU test of the client-side KV-aware preload controller (docs/controller_theory.md Section 10), run ON the pod.
# For each arrival seed, and each KV cap: restart vLLM with --num-gpu-blocks-override (16-token blocks), then run
# preload modes off / low / ctrl / on on the same arrival trace (paired). Decisive caps first (50k, 40k), then 80k,
# the full cache and 40k at 0.3/s. The controller reads the server's KV capacity from /metrics.
# Usage (pod): nohup scripts/ctrl_test.sh 2 3 4 > results/logs/ctrl_test.log 2>&1 &
set -uo pipefail
cd /workspace/codesign
source /opt/venv/bin/activate
ulimit -n 65536
export HF_HOME=/workspace/hf HF_HUB_OFFLINE=1  # model lives on the volume; never re-download to the root disk
OUT=results/real/preload_ctrl
SEEDS=("${@:-2}")

restart() {  # $1 = KV blocks ("" = default)
  pkill -f "vllm serv[e]" || true
  for _ in $(seq 60); do pgrep -f "vllm serv[e]" > /dev/null || break; sleep 2; done
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

run() {  # $1 = seed, $2 = tag, $3 = rate, rest = preload modes
  local seed=$1 tag=$2 rate=$3; shift 3
  python scripts/real_multiagent.py --preset preload --seed "$seed" --rates "$rate" --duration 180 \
    --out "$OUT/seed$seed/$tag" --preload-modes "$@" || echo "RUN_FAILED seed$seed $tag r$rate"
  echo "RUN_DONE seed$seed $tag r$rate"
}

for s in "${SEEDS[@]}"; do
  restart 3125; run "$s" kv50k 0.6 off ctrl low on
  restart 2500; run "$s" kv40k 0.6 off ctrl low on; run "$s" kv40k 0.3 off ctrl
  restart 5000; run "$s" kv80k 0.6 off ctrl low on
  restart "";   run "$s" full 0.6 off ctrl low on
  echo "SEED_DONE $s"
done
echo "CTRL_TEST_EXIT"
