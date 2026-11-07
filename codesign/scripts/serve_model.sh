#!/usr/bin/env bash
# Serve the pinned model on localhost (run ON the GPU pod). Records the environment for reproducibility.
# Usage: MODEL=Qwen/Qwen3-8B scripts/serve_model.sh            (EXTRA_ARGS="..." to add flags, e.g. speculation)
set -euo pipefail
MODEL="${MODEL:-Qwen/Qwen3-8B}"
PORT="${PORT:-8000}"
MAXLEN="${MAXLEN:-8192}"
MAXSEQS="${MAXSEQS:-1}"  # 1 = one request at a time (single-task experiments); raise for multi-tenant serving
SCHED="${SCHED:-fcfs}"    # fcfs | priority (per-request priority, lower first)
GPU_UTIL="${GPU_UTIL:-0.88}"
# KV_BLOCKS=N caps the KV cache at N blocks of 16 tokens (vLLM --num-gpu-blocks-override), for KV-pressure runs
KV="${KV_BLOCKS:+--num-gpu-blocks-override $KV_BLOCKS}"
ENVDIR="results/env"; mkdir -p "$ENVDIR"
{
  echo "date: $(date -u +%FT%TZ)"; echo "model: $MODEL"
  python -c 'import vllm, torch; print("vllm:", vllm.__version__); print("torch:", torch.__version__, "cuda:", torch.version.cuda)'
  nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
} | tee "$ENVDIR/serve_env.txt"
# Confirm flag names on the pinned version before relying on them.
vllm serve --help=all 2>/dev/null | grep -E -- "--(no-)?enable-prefix-caching|--max-num-seqs|--speculative-config" | head -5 || true
# PREFIX_CACHE=1: automatic prefix caching ON (KV preloading experiments) and the dev-mode endpoint
# POST /reset_prefix_cache, which the experiment calls before every task run so no run reuses another's KV.
if [ "${PREFIX_CACHE:-0}" = "1" ]; then PC=--enable-prefix-caching; export VLLM_SERVER_DEV_MODE=1
else PC=--no-enable-prefix-caching; fi
echo "prefix_caching: $PC dev_mode=${VLLM_SERVER_DEV_MODE:-0}" | tee -a "$ENVDIR/serve_env.txt"
CMD=(vllm serve "$MODEL" --dtype bfloat16 --max-model-len "$MAXLEN" --max-num-seqs "$MAXSEQS" --scheduling-policy "$SCHED"
     --gpu-memory-utilization "$GPU_UTIL" "$PC" $KV --enable-prompt-tokens-details
     --host 127.0.0.1 --port "$PORT" ${EXTRA_ARGS:-})
echo "${CMD[*]}" | tee "$ENVDIR/serve_cmd.txt"
exec "${CMD[@]}"
