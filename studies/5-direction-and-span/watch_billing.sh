#!/bin/sh
# Billing watchdog for the AI-02-04 cloud studies (Studies 5 and 6): every 10 minutes, read Modal's billing report;
# if every ai0204-* app together has spent $8, or the workspace's month total has reached $27, stop the running
# ai0204-* apps (Study 5 PROTOCOL.md section 8). Logs to results/billing-watch.jsonl; exits when no ai0204-* app has
# been running for three checks in a row. Run from this folder as a systemd user unit:
#   systemd-run --user --unit=ai0204-billing --working-directory="$PWD" ./watch_billing.sh
modal="$HOME/.local/bin/modal"
idle=0
while true; do
    spend=$("$modal" billing report --for "this month" --json 2>/dev/null | python3 -c '
import json, sys
try:
    rows = json.load(sys.stdin)
    task = sum(float(r["cost"]) for r in rows if r["description"].startswith("ai0204-"))
    month = sum(float(r["cost"]) for r in rows)
    print(round(task, 4), round(month, 4))
except Exception:
    print("unknown unknown")')
    task=${spend% *}; month=${spend#* }
    running=$("$modal" app list --json 2>/dev/null | python3 -c '
import json, sys
try:
    apps = json.load(sys.stdin)
    print(" ".join(a["app_id"] for a in apps if a["description"].startswith("ai0204-") and a["state"] != "stopped"))
except Exception:
    print("unknown")')
    echo "{\"utc\": \"$(date -u +%Y-%m-%dT%H:%M:%SZ)\", \"task_usd\": \"$task\", \"month_usd\": \"$month\", \"running\": \"$running\"}" >> results/billing-watch.jsonl
    if [ "$task" != "unknown" ] && python3 -c "import sys; sys.exit(0 if float('$task') >= 8 or float('$month') >= 27 else 1)"; then
        for app in $running; do "$modal" app stop -y "$app"; done
        echo "{\"utc\": \"$(date -u +%Y-%m-%dT%H:%M:%SZ)\", \"stopped\": \"$running\", \"reason\": \"limit reached\"}" >> results/billing-watch.jsonl
        exit 0
    fi
    if [ -z "$running" ]; then idle=$((idle + 1)); else idle=0; fi
    [ $idle -ge 3 ] && exit 0
    sleep 600
done
