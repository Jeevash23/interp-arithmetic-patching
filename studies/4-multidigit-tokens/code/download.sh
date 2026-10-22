#!/usr/bin/env bash
# Resumable download of one candidate's weights at its pinned revision into models/<candidate>/, then a check of
# every file against results/hub-files.json. The tokenizer and configuration files are copied from tokenizers/<candidate>/,
# which were checked against the same record. Usage: bash code/download.sh <candidate>
set -u
STUDY=$(cd "$(dirname "$0")/.." && pwd)
PY=${PY:?set PY to the python of the restored AI-01 environment}
C=$1
read -r REPO REV FILES < <("$PY" -B -c "
import sys; sys.path.insert(0, '$STUDY/code'); from candidates import CANDIDATES as K
c = K['$C']; print(c['repository'], c['revision'], ' '.join(c['weight_files']))")
mkdir -p "$STUDY/models/$C"
for f in $(ls "$STUDY/tokenizers/$C"); do cp -p "$STUDY/tokenizers/$C/$f" "$STUDY/models/$C/$f"; done
for f in $FILES; do
  url="https://huggingface.co/$REPO/resolve/$REV/$f"
  want=$("$PY" -B -c "import json; print(json.load(open('$STUDY/results/hub-files.json'))['$C']['files']['$f']['size'])")
  for i in $(seq 1 80); do
    have=$(stat -c %s "$STUDY/models/$C/$f.part" 2>/dev/null || echo 0)
    [ "$have" = "$want" ] && break
    curl -sS -L --fail --speed-limit 50000 --speed-time 45 -C - -o "$STUDY/models/$C/$f.part" "$url"
    echo "$f attempt $i exit $? size $(stat -c %s "$STUDY/models/$C/$f.part")"
    sleep 3
  done
  [ "$(stat -c %s "$STUDY/models/$C/$f.part")" = "$want" ] && mv "$STUDY/models/$C/$f.part" "$STUDY/models/$C/$f"
done
cd "$STUDY" && "$PY" -B code/hub_files.py check-weights "$C" > /dev/null && echo "checked: results/weights-check-$C.json"
