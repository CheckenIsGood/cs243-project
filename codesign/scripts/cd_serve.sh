#!/usr/bin/env bash
# Start one vLLM server with the codesign scheduler on one GPU (run ON the pod).
# Usage: scripts/cd_serve.sh GPU PORT KV_BLOCKS CONFIG_NAME     (KV_BLOCKS "" = full cache)
# CONFIG_NAME is a key of scripts/cd_configs.json (the CODESIGN env for codesign.vllm_sched).
set -uo pipefail
cd /workspace/codesign
source /opt/venv/bin/activate
export HF_HOME=/workspace/hf HF_HUB_OFFLINE=1
GPU=$1 PORT=$2 KVB=$3 CFG=$4
mkdir -p results/logs results/cd_stats
CONF=$(python -c "import json,sys; c=json.load(open('scripts/cd_configs.json'))['$CFG']['engine']; c['stats_path']='results/cd_stats/gpu$GPU.json'; print(json.dumps(c))")
pkill -f "vllm serv[e].*--port $PORT" || true
for _ in $(seq 30); do pgrep -f "vllm serv[e].*--port $PORT" > /dev/null || break; sleep 2; done
log=results/logs/serve_gpu${GPU}.log
CUDA_VISIBLE_DEVICES=$GPU CODESIGN="$CONF" KV_BLOCKS="$KVB" PORT=$PORT PREFIX_CACHE=1 MAXLEN=${MAXLEN:-16384} MAXSEQS=256 SCHED=priority \
  EXTRA_ARGS="--scheduler-cls codesign.vllm_sched.CodesignScheduler" nohup scripts/serve_model.sh > "$log" 2>&1 &
spid=$!
for _ in $(seq 120); do
  curl -sf http://127.0.0.1:$PORT/health > /dev/null && break
  kill -0 "$spid" 2> /dev/null || break
  sleep 5
done
curl -sf http://127.0.0.1:$PORT/health > /dev/null || { echo "SERVER_FAILED gpu=$GPU"; tail -30 "$log"; exit 1; }
echo "SERVER_UP gpu=$GPU port=$PORT kv=${KVB:-full} cfg=$CFG $(grep -o '\[codesign\] config.*' "$log" | tail -1)"
