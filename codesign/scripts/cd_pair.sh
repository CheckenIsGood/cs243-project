#!/usr/bin/env bash
# Run up to 4 (config, kv) cells in parallel, one per GPU, on ports 9000+g (8001 is the pod's nginx).
# Usage: scripts/cd_pair.sh OUTDIR SEED RATE DUR KVB CFG0 [CFG1 CFG2 CFG3]
set -uo pipefail
cd /workspace/codesign
OUTD=$1 SEED=$2 RATE=$3 DUR=$4 KVB=$5; shift 5
CFGS=("$@")
for g in "${!CFGS[@]}"; do scripts/cd_serve.sh $g $((9000 + g)) "$KVB" "${CFGS[$g]}" & done; wait
for g in "${!CFGS[@]}"; do
  curl -sf http://127.0.0.1:$((9000 + g))/health > /dev/null || { echo "NOT_UP gpu$g"; exit 1; }
done
for g in "${!CFGS[@]}"; do
  scripts/cd_run.sh $((9000 + g)) "${CFGS[$g]}" "$SEED" "$OUTD" "$RATE" "$DUR" > "results/logs/run_$(basename $OUTD)_${CFGS[$g]}_s$SEED.log" 2>&1 &
done
wait
for g in "${!CFGS[@]}"; do
  c=${CFGS[$g]}
  echo "CELL $c kv=${KVB:-full} seed=$SEED rate=$RATE $(grep -h '^{' results/logs/run_$(basename $OUTD)_${c}_s$SEED.log | tail -1 | python3 -c 'import json,sys; d=json.loads(sys.stdin.read() or "{}"); print({k: d.get(k) for k in ("n","lat_mean","lat_p50","lat_p90","error")})')"
  cp results/cd_stats/gpu$g.json "$OUTD/cdstats_${c}_s$SEED.json" 2>/dev/null
  python3 -c "import json; d=json.load(open('$OUTD/cdstats_${c}_s$SEED.json')); print('  stats', {k:v for k,v in d.items() if k not in ('config','t') and v})" 2>/dev/null
done
