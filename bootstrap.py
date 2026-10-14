"""Paired comparisons on the same test bodies (generalises paired_stats.py).

paired(rows_a, rows_b): per-body values (field "per_shape" by default) are averaged over the rows
of each set (normally the fine-tuning seeds), then compared body by body.  Returns the mean
relative change (mode="rel", a/b - 1, as in paired_stats.py) or mean difference (mode="diff"),
its 95% bootstrap interval over bodies, and the fraction of bodies on which a is better.
The interval covers test-body sampling only (not pretraining or fine-tuning seed variance).
"""
import numpy as np


def per_body(rows, field="per_shape"):
    xs = [np.asarray(r[field], float) for r in rows]
    if not xs:
        raise ValueError("no rows")
    if any(len(x) != len(xs[0]) for x in xs):
        raise ValueError("rows cover different numbers of bodies")
    return np.mean(xs, 0)


def boot_mean(d, n_boot=4000, seed=0):
    """mean of d and the 95% percentile-bootstrap interval of that mean."""
    d = np.asarray(d, float)
    rng = np.random.default_rng(seed)
    boots = [rng.choice(d, len(d)).mean() for _ in range(n_boot)]
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return float(d.mean()), float(lo), float(hi)


def paired(rows_a, rows_b, field="per_shape", mode="rel", higher_is_better=False, n_boot=4000, seed=0):
    ea, eb = per_body(rows_a, field), per_body(rows_b, field)
    if len(ea) != len(eb):
        raise ValueError("the two sets cover different numbers of bodies")
    d = ea / eb - 1 if mode == "rel" else ea - eb
    m, lo, hi = boot_mean(d, n_boot, seed)
    better = ea > eb if higher_is_better else ea < eb
    seeds = lambda R: sorted({r.get("seed", r.get("ft_seed")) for r in R}, key=str)
    return dict(mean=m, lo=lo, hi=hi, better=float(better.mean()), a=float(ea.mean()), b=float(eb.mean()),
                n_bodies=len(d), seeds_a=seeds(rows_a), seeds_b=seeds(rows_b),
                sig=bool(lo > 0 or hi < 0))


def fmt(r, pct=True):
    s = 100 if pct else 1
    u = "%" if pct else ""
    return (f"{s * r['mean']:+.1f}{u} [{s * r['lo']:+.1f}, {s * r['hi']:+.1f}]{'*' if r['sig'] else ' '} "
            f"better on {100 * r['better']:.0f}% of bodies")
