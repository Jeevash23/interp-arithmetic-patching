"""Exact binomial (Clopper-Pearson) and donor-cluster bootstrap intervals; standard library and NumPy only."""
from math import comb

import numpy as np

SEED = 20261009
RESAMPLES = 10000


def _cdf(k, n, p):
    return sum(comb(n, i) * p ** i * (1 - p) ** (n - i) for i in range(k + 1))


def _root(f, lo=0.0, hi=1.0):
    """Bisection for a monotone f with a sign change on [lo, hi]."""
    flo = f(lo)
    for _ in range(200):
        mid = (lo + hi) / 2
        fm = f(mid)
        if (fm > 0) == (flo > 0):
            lo, flo = mid, fm
        else:
            hi = mid
    return (lo + hi) / 2


def clopper_pearson(k, n, alpha=0.05):
    if n == 0:
        return [None, None]
    lower = 0.0 if k == 0 else _root(lambda p: (1 - _cdf(k - 1, n, p)) - alpha / 2)
    upper = 1.0 if k == n else _root(lambda p: _cdf(k, n, p) - alpha / 2)
    return [lower, upper]


def cluster_bootstrap(values, groups, resamples=RESAMPLES, seed=SEED):
    """Percentile 95% interval of the pooled mean, resampling donor groups with replacement."""
    values = np.asarray(values, dtype=float)
    keys = sorted(set(groups))
    index = {g: i for i, g in enumerate(keys)}
    sums = np.zeros(len(keys)); counts = np.zeros(len(keys))
    for v, g in zip(values, groups):
        sums[index[g]] += v; counts[index[g]] += 1
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(keys), (resamples, len(keys)))
    stats = sums[draws].sum(axis=1) / counts[draws].sum(axis=1)
    return [float(np.quantile(stats, 0.025)), float(np.quantile(stats, 0.975))]


def proportion(flags, groups):
    """Count, rate, exact interval, donor-cluster interval and the conservative envelope of the two."""
    flags = [bool(f) for f in flags]
    k, n = sum(flags), len(flags)
    cp = clopper_pearson(k, n)
    cl = cluster_bootstrap(flags, groups) if n else [None, None]
    return dict(k=k, n=n, rate=k / n if n else None, exact95=cp, cluster95=cl,
                lower=min(cp[0], cl[0]) if n else None, upper=max(cp[1], cl[1]) if n else None,
                groups=len(set(groups)))


def paired_difference(a, b, groups):
    """Mean of a minus b with a donor-cluster percentile interval."""
    diff = [int(bool(x)) - int(bool(y)) for x, y in zip(a, b)]
    return dict(difference=sum(diff) / len(diff), cluster95=cluster_bootstrap(diff, groups), n=len(diff))
