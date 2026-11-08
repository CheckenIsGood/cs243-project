#!/usr/bin/env bash
# One-time setup ON the GPU pod (inside /workspace/codesign). Pins vLLM; installs this package.
# The venv lives on the container disk (/opt/venv): fast, but wiped on pod stop, so re-run after a restart.
set -euo pipefail
VLLM_VERSION="${VLLM_VERSION:-0.32.0}"
VENV="${VENV:-/opt/venv}"
[ -x "$VENV/bin/python" ] || python3 -m venv "$VENV"   # system python is PEP 668 "externally managed"
export VIRTUAL_ENV="$VENV" PATH="$VENV/bin:$PATH"
python -m pip install -q uv
# --torch-backend=auto (vLLM's documented uv install) pulls torch + NVIDIA wheels from download.pytorch.org,
# matched to the driver; on the 2026-11-06 RunPod host PyPI's CDN ran at ~2.5 MB/s vs ~84 MB/s there.
# The PyTorch index also mirrors the nvidia-* CUDA wheels; prefer it for those too.
TORCH_INDEX="${TORCH_INDEX:-https://download.pytorch.org/whl/cu130}"
# PyPI's CDN was ~1-2 MB/s on this host; the identical wheel from the GitHub release is verified against PyPI's sha256.
WHEEL="/workspace/wheels/vllm-${VLLM_VERSION}-cp38-abi3-manylinux_2_28_x86_64.whl"
if [ ! -f "$WHEEL" ]; then
  mkdir -p /workspace/wheels
  curl -sL -o "$WHEEL" "https://github.com/vllm-project/vllm/releases/download/v${VLLM_VERSION}/$(basename "$WHEEL")"
fi
SHA=$(curl -s "https://pypi.org/pypi/vllm/${VLLM_VERSION}/json" | python -c "import sys,json; print([f['digests']['sha256'] for f in json.load(sys.stdin)['urls'] if f['filename']=='$(basename "$WHEEL")'][0])")
echo "$SHA  $WHEEL" | sha256sum -c
python -m uv pip install -q "$WHEEL" --extra-index-url "$TORCH_INDEX" --index-strategy unsafe-best-match
python -m uv pip install -q -e '.[dev]'
mkdir -p results/env
python -m uv pip freeze > results/env/pip_freeze.txt
python -c 'import vllm, torch; print("vllm", vllm.__version__, "torch", torch.__version__, "cuda", torch.version.cuda, torch.cuda.is_available())'
python -m pytest -q -x 2>&1 | tail -3   # CPU checks must pass on the pod too
