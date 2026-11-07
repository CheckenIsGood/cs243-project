#!/usr/bin/env bash
# Hand-off (run ON the pod with setsid): when the main sweep finishes seed 4 (last round s4 kvfull), stop it and
# run seeds 5-8 at the 16k cap only. Appends to the same sweep log.
cd /workspace/codesign
LOG=results/logs/sweep.log
PID=22794 PGID=22792
MARK="ROUND_DONE s4 kvfull C0b C2 C5 C7"
until grep -q "$MARK" $LOG; do
  if ! kill -0 $PID 2>/dev/null; then echo "HANDOFF_ABORT sweep exited before s4 finished $(date +%T)" >> $LOG; exit 1; fi
  sleep 5
done
kill -TERM -- -$PGID 2>/dev/null; sleep 15; kill -KILL -- -$PGID 2>/dev/null
pkill -f "vllm serv[e]" ; sleep 10
rm -rf results/sweep/kv1000/s5   # only a just-started, incomplete round can be there
echo "HANDOFF stopped main sweep, starting seeds 5-8 at kv1000 $(date +%T)" >> $LOG
CAPS=1000 SEEDS="5 6 7 8" scripts/cd_sweep.sh >> $LOG 2>&1
echo "HANDOFF_DONE $(date +%T)" >> $LOG
