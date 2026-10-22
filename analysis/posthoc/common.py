"""Shared paths and statistics for the post hoc analyses (standard library only).

Every script in this folder reads saved journals from the two original study folders and writes one JSON file to
analysis/posthoc/results/. All of them are post hoc: they were written on 9 October 2026, after every outcome they
analyse had been seen, and they run no model. With --check, a script recomputes its result and compares it with the
saved file instead of writing it (exit status 1 if they differ).
"""
from __future__ import annotations

import hashlib
import json
import math
import random
import sys
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[2]
ORIG = PROJECT / "06-Study-History/original-studies"
S1 = ORIG / "study1-value-transfer-and-readout"          # Study 1
S2 = ORIG / "study2-reasoning-interventions"             # Study 2
HV = S1 / "hidden-value-transfer"                        # Study 1, Qwen3-4B answer-only experiments
READOUT = S1 / "followup-readout"                        # Study 1, Qwen3-0.6B readout experiment
RESULTS = Path(__file__).resolve().parent / "results"

OPS = {"add": lambda x, y: x + y, "sub": lambda x, y: x - y, "mul": lambda x, y: x * y}


def jsonl(path):
    with open(path) as handle:
        return [json.loads(line) for line in handle if line.strip()]


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def rel(path):
    return str(Path(path).resolve().relative_to(PROJECT))


def run_program(task, a=None, b=None):
    """The recipient's two-step program, optionally with one operand replaced."""
    a = task["a"] if a is None else a
    b = task["b"] if b is None else b
    return OPS[task["remaining"]](OPS[task["first"]](a, b), task["k"])


def _binom_cdf(k, n, p):
    """P(X <= k) for X ~ Binomial(n, p)."""
    if k < 0:
        return 0.0
    if k >= n:
        return 1.0
    return sum(math.comb(n, i) * p ** i * (1 - p) ** (n - i) for i in range(k + 1))


def _solve(f, target, lo=0.0, hi=1.0):
    """Bisection for a monotone function f on [lo, hi]."""
    for _ in range(200):
        mid = (lo + hi) / 2
        if (f(mid) - target) * (f(lo) - target) <= 0:
            hi = mid
        else:
            lo = mid
    return (lo + hi) / 2


def exact_interval(k, n, level=0.95):
    """Two-sided Clopper-Pearson interval for k successes in n trials."""
    alpha = 1 - level
    lower = 0.0 if k == 0 else _solve(lambda p: 1 - _binom_cdf(k - 1, n, p), alpha / 2)
    upper = 1.0 if k == n else _solve(lambda p: _binom_cdf(k, n, p), alpha / 2)
    return lower, upper


def upper_bound(k, n, level=0.95):
    """One-sided exact upper confidence bound; for k = 0 it equals 1 - (1 - level)^(1/n)."""
    if k == n:
        return 1.0
    return _solve(lambda p: _binom_cdf(k, n, p), 1 - level)


def cluster_bootstrap(clusters, draws=10000, seed=20261009, level=0.95):
    """Percentile interval for a pooled proportion, resampling whole clusters.

    clusters: list of (successes, trials) per cluster.
    """
    rng = random.Random(seed)
    k = len(clusters)
    stats = []
    for _ in range(draws):
        s = t = 0
        for _ in range(k):
            a, b = clusters[rng.randrange(k)]
            s += a
            t += b
        stats.append(s / t)
    stats.sort()
    lo = stats[int((1 - level) / 2 * draws)]
    hi = stats[int((1 + level) / 2 * draws) - 1]
    return lo, hi


def rate(k, n, clusters=None, cluster_label=None):
    """Count, proportion, exact interval and bounds; optional cluster-level summaries."""
    out = {"k": k, "n": n, "p": k / n, "exact95": list(exact_interval(k, n)), "upper95_one_sided": upper_bound(k, n)}
    if clusters is not None:
        with_success = sum(1 for a, _ in clusters if a)
        out["clusters"] = {"unit": cluster_label, "count": len(clusters), "with_success": with_success,
                           "upper95_one_sided_any_success": upper_bound(with_success, len(clusters))}
        if k:
            out["clusters"]["bootstrap95"] = list(cluster_bootstrap(clusters))
    return out


def write(name, payload, inputs):
    RESULTS.mkdir(parents=True, exist_ok=True)
    payload = {"analysis": name, "status": "post hoc (written 9 October 2026, after all outcomes were seen)",
               "inputs": {rel(p): sha256(p) for p in sorted(set(inputs), key=str)}, **payload}
    path = RESULTS / f"{name}.json"
    text = json.dumps(payload, indent=1, sort_keys=False) + "\n"
    if "--check" in sys.argv[1:]:
        if not path.exists() or path.read_text() != text:
            sys.exit(f"DIFFERS: {rel(path)} is not what the saved journals give")
        print("ok ", rel(path))
        return
    path.write_text(text)
    print("wrote", rel(path))
