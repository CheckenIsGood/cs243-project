#!/usr/bin/env bash
# Full co-design sweep on 4 GPUs (run ON the pod, under nohup). Resumable: a cell whose tasks CSV exists is skipped.
# Rounds of 4 configs (one per GPU) x KV caps x seeds, paired by seed (same arrivals).
set -uo pipefail
cd /workspace/codesign
OUT=results/sweep
RATE=${RATE:-1.0} DUR=${DUR:-90}
ROUNDS=("C0a C1 C3 C6" "C0b C2 C5 C7")
SEEDS=(${SEEDS:-2 3 4 5 6})
CAPS=(${CAPS:-1000 1500 1875 full})
for seed in "${SEEDS[@]}"; do
  for kv in "${CAPS[@]}"; do
    kvb=$kv; [ "$kv" = full ] && kvb=""
    for r in "${ROUNDS[@]}"; do
      todo=()
      for c in $r; do ls $OUT/kv$kv/tasks_*_${c}.csv 2>/dev/null | grep -q "s$seed" || todo+=("$c"); done
      # tasks files are per seed directory
      todo=()
      for c in $r; do [ -n "$(ls $OUT/kv$kv/s$seed/tasks_*_${c}.csv 2>/dev/null)" ] || todo+=("$c"); done
      [ ${#todo[@]} -eq 0 ] && { echo "SKIP s$seed kv$kv $r"; continue; }
      rm -f results/cd_stats/*
      MAXLEN=12288 scripts/cd_pair.sh $OUT/kv$kv/s$seed "$seed" "$RATE" "$DUR" "$kvb" "${todo[@]}" 2>&1 | grep -E "CELL|stats|NOT_UP|FAILED"
      echo "ROUND_DONE s$seed kv$kv ${todo[*]} $(date +%T)"
    done
  done
done
echo SWEEP_EXIT
