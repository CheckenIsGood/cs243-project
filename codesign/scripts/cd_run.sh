#!/usr/bin/env bash
# One client run against a codesign server (run ON the pod). Usage: scripts/cd_run.sh PORT CONFIG SEED TAGDIR RATE [DURATION]
set -uo pipefail
cd /workspace/codesign
source /opt/venv/bin/activate
PORT=$1 CFG=$2 SEED=$3 OUTD=$4 RATE=$5 DUR=${6:-180}
read RULE PRE TTLQ KAPPA < <(python -c "import json; c=json.load(open('scripts/cd_configs.json'))['$CFG']; print(c['rule'], c['preload'], c['ttl_q'], c.get('kappa', 0))")
python scripts/real_multiagent.py --preset codesign --seed "$SEED" --rates "$RATE" --duration "$DUR" \
  --base-url http://127.0.0.1:$PORT --out "$OUTD" --rules "$RULE" --preload-modes "$PRE" --ttl-quantile "$TTLQ" --age-kappa "$KAPPA" \
  --tag "$CFG" || echo "RUN_FAILED $CFG seed$SEED $OUTD r$RATE"
echo "RUN_DONE $CFG seed$SEED $OUTD r$RATE"
