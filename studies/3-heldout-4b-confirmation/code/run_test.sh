#!/usr/bin/env bash
# Runs the frozen held-out test: the primary wording (box), then the secondary wording (machine).
# Every GPU step holds the shared GPU lock; the control phase runs in chunks of at most 25 minutes and releases the
# lock between chunks so that other users of the GPU are not starved. Usage:
#   GPU_LOCK=<shared lock file> PY=<python of the restored AI-01 environment> bash code/run_test.sh
set -u
STUDY=$(cd "$(dirname "$0")/.." && pwd)
PY=${PY:?set PY to the python of the restored AI-01 environment}
LOCK=${GPU_LOCK:?set GPU_LOCK to the shared GPU lock file}
export PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
mkdir -p "$STUDY/logs"
for W in box machine; do
  flock "$LOCK" "$PY" -B "$STUDY/code/confirm.py" --wording "$W" --phase baseline >> "$STUDY/logs/test-$W-baseline.log" 2>&1
  code=$?
  if [ $code -ne 0 ]; then echo "baseline $W failed with status $code"; exit $code; fi
  while true; do
    flock "$LOCK" "$PY" -B "$STUDY/code/confirm.py" --wording "$W" --phase controls --max-minutes 25 >> "$STUDY/logs/test-$W-controls.log" 2>&1
    code=$?
    [ $code -eq 0 ] && break
    if [ $code -ne 3 ]; then echo "controls $W failed with status $code"; exit $code; fi
    sleep 60   # let waiting GPU users take the lock
  done
  echo "$W complete"
done
