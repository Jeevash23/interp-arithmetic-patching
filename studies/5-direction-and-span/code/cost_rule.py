"""The test-stage cost rule (PROTOCOL.md section 8). After development: project the test stage's cost from the
development timings and decide whether the alpha = 1 sweep at a_end and b_end is dropped from the test.

  python3 -B code/cost_rule.py    # writes results/cost-decision.json (refuses to overwrite)

Spend so far is read from Modal's billing report (every ai0204-study5 app this month). Rates are Modal's list prices
per hour for the worker shape in modal_app.py (one L4, two CPU cores, 12 GiB). Standard library only.
"""
import gzip
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

STUDY = Path(__file__).resolve().parents[1]
LIMIT_USD = 4.50
HOURLY = 0.80 + 2 * 0.0473 + 12 * 0.008          # L4 + 2 cores + 12 GiB, dollars per hour
LOAD_SECONDS = 150                                # container start and model load, per worker (first stage logs)
WORKERS = 4


def records(folder):
    out = []
    for p in sorted((STUDY / 'results' / folder).glob('*.jsonl.gz')):
        with gzip.open(p, 'rt') as fh:
            out += [json.loads(s) for s in fh if s.strip()]
    return out


def main():
    target = STUDY / 'results/cost-decision.json'
    if target.exists():
        raise SystemExit('the cost decision exists; not overwritten')
    dev = records('development') + records('devsites')
    gen = [r['seconds'] for r in dev if r['kind'] in ('direction', 'span', 'baseline')]
    score = [r['seconds'] for r in dev if r['kind'] == 'direction_score']
    g, sc = sum(gen) / len(gen), sum(score) / len(score)
    sel = json.loads((STUDY / 'results/development-selection.json').read_text())
    n_sites = len(sel['direction_sites_test'])
    extra_dir = sum(1 for d in sel['direction_sites_test'] if not d['in_sweep'])
    per_pair_sites = (extra_dir + 2 * n_sites) + 6 * len(sel['span_sites']) + len(sel['span_sites'])  # + self checks
    per_pair_scores = sum(1 for d in sel['direction_sites_test'] if d['role'] in ('first_stage_end', 'prompt_end')) + 1

    def project(roles):
        gens = 2 + 36 * roles + 72 + per_pair_sites
        seconds = 72 * (gens * g + per_pair_scores * sc) + WORKERS * LOAD_SECONDS
        return dict(direction_sweep_roles=roles, generations_per_pair=gens, gpu_hours=seconds / 3600, usd=seconds / 3600 * HOURLY)

    rows = json.loads(subprocess.run([str(Path.home() / '.local/bin/modal'), 'billing', 'report', '--for', 'this month', '--json'],
                                     capture_output=True, text=True, check=True).stdout)
    spent = sum(float(r['cost']) for r in rows if r['description'].startswith('ai0204-study5'))
    full = project(4)
    reduced = project(2)
    drop = spent + full['usd'] > LIMIT_USD
    decision = dict(created_utc=datetime.now(timezone.utc).isoformat(), limit_usd=LIMIT_USD, hourly_usd=HOURLY,
                    study5_spent_usd=round(spent, 4), mean_generation_seconds=g, mean_score_seconds=sc,
                    projection_full=full, projection_reduced=reduced, rule_triggered=drop,
                    test_direction_sweep_roles=['first_stage_end', 'prompt_end'] if drop else ['a_end', 'b_end', 'first_stage_end', 'prompt_end'],
                    note='billing reports lag by up to an hour, so the spend so far is a lower bound')
    target.write_text(json.dumps(decision, indent=1) + '\n')
    print(json.dumps(decision, indent=1))


if __name__ == '__main__':
    main()
