#!/bin/sh
# Start one Study 6 stage on Modal as a standalone process (the AI-03 Study 5 pattern). If the local connection fails
# before the stage starts (no new line in results/stage-log.jsonl), the idle app is stopped and the start retried.
#   systemd-run --user --unit=ai0204-s6-<stage> --working-directory="$PWD" ./run_stage.sh <stage>
stage="$1"
log="logs/$stage.log"
modal="$HOME/.local/bin/modal"
lines() { [ -f results/stage-log.jsonl ] && wc -l < results/stage-log.jsonl || echo 0; }
for attempt in 1 2 3 4 5 6; do
    before=$(lines)
    PYTHONDONTWRITEBYTECODE=1 "$modal" run --detach "code/modal_app.py::$stage" > "$log" 2>&1
    status=$?
    echo "exit $status (attempt $attempt)" >> "$log"
    if [ $status -eq 0 ] || [ "$(lines)" != "$before" ] || \
       ! grep -q -e "Could not connect to the Modal server" -e "Temporary failure in name resolution" "$log"; then
        exit $status
    fi
    app=$(grep -o "ap-[A-Za-z0-9]*" "$log" | head -1)
    [ -n "$app" ] && "$modal" app stop -y "$app" >> "$log" 2>&1
    mv "$log" "logs/$stage-connect-failure-$(date -u +%H%M%S).log"
    sleep 30
done
exit 1
