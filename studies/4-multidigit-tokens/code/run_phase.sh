#!/usr/bin/env bash
# Runs one phase of Study 4 for one candidate under the shared GPU lock, in chunks of at most 25 minutes, releasing the
# lock between chunks. Usage: GPU_LOCK=<lock file> PY=<AI-01 python> bash code/run_phase.sh <phase> <candidate>
set -u
STUDY=$(cd "$(dirname "$0")/.." && pwd)
PY=${PY:?set PY to the python of the restored AI-01 environment}
LOCK=${GPU_LOCK:?set GPU_LOCK to the shared GPU lock file}
PHASE=$1; C=$2
export PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
mkdir -p "$STUDY/logs"
while true; do
  flock "$LOCK" "$PY" -B "$STUDY/code/study4.py" "$PHASE" --candidate "$C" --max-minutes 25 >> "$STUDY/logs/$C-$PHASE.log" 2>&1
  code=$?
  [ $code -eq 0 ] && { echo "$PHASE $C complete"; exit 0; }
  if [ $code -ne 3 ]; then echo "$PHASE $C stopped with status $code"; exit $code; fi
  sleep 60
done
