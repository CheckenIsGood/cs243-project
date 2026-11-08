#!/usr/bin/env bash
# Paired summaries of the controller GPU test (results/real/preload_ctrl/seed*/<cap>): every mode vs off, and ctrl vs low.
cd "$(dirname "$0")/.."
PY=${PY:-.venv/bin/python}
for d in results/real/preload_ctrl/seed*/*; do
  [ -d "$d" ] || continue
  for rate in 0.6 0.3; do
    [ -f "$d/tasks_r${rate}_fcfs_pre-off.csv" ] || continue
    for m in ctrl low on; do
      [ -f "$d/tasks_r${rate}_fcfs_pre-$m.csv" ] && $PY scripts/analyze_preload.py "$d:off" "$d:$m" --rates $rate | tail -1 | sed "s|^|$d off->$m |"
    done
    [ -f "$d/tasks_r${rate}_fcfs_pre-ctrl.csv" ] && [ -f "$d/tasks_r${rate}_fcfs_pre-low.csv" ] && \
      $PY scripts/analyze_preload.py "$d:low" "$d:ctrl" --rates $rate | tail -1 | sed "s|^|$d low->ctrl |"
  done
done
