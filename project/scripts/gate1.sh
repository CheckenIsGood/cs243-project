#!/usr/bin/env bash
# Gate 1 driver, run ON the GPU pod from /workspace/project after scripts/pod_setup.sh.
# Starts vLLM in the background, waits for health, runs the real-backend smoke, then profiles and fits costs.
# Usage: scripts/gate1.sh            (SKIP_SMOKE=1 / SKIP_PROFILE=1 to skip a step)
set -euo pipefail
export HF_HOME="${HF_HOME:-/workspace/hf}"
export VIRTUAL_ENV="${VENV:-/opt/venv}" PATH="${VENV:-/opt/venv}/bin:$PATH"
mkdir -p results/env results/calibration logs
if ! curl -sf http://127.0.0.1:8000/health >/dev/null; then
  nohup scripts/serve_model.sh > logs/vllm_serve.log 2>&1 &
  echo "vllm pid $!"
fi
for i in $(seq 1 180); do  # up to 15 min: first run downloads ~16 GB of weights
  curl -sf http://127.0.0.1:8000/health >/dev/null && break
  sleep 5
done
curl -sf http://127.0.0.1:8000/health >/dev/null || { echo "vLLM not healthy; tail of log:"; tail -40 logs/vllm_serve.log; exit 1; }
echo "vLLM healthy"
curl -s http://127.0.0.1:8000/v1/models | tee results/env/models.json; echo
[ -n "${SKIP_SMOKE:-}" ] || python -m deadline_agent.cli run --config configs/smoke.yaml --run-id smoke-gate1 2>&1 | tee logs/smoke.log
[ -n "${SKIP_PROFILE:-}" ] || python -m deadline_agent.cli profile --config configs/smoke.yaml \
  --out results/calibration/costs.json --reps 3 2>&1 | tee logs/profile.log
