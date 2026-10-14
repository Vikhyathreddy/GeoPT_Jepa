"""Paired comparisons on the same 100 test bodies (per-shape errors averaged over seeds),
with 95% bootstrap intervals for the mean relative change in error."""
import json
import numpy as np

R = [json.loads(l) for l in open("results/finetune.jsonl")]
rng = np.random.default_rng(0)


def per_shape(v, t, n, frozen=False, shift=0):
    xs = [np.array(r["per_shape"]) for r in R
          if r["variant"] == v and r["task"] == t and r["n"] == n and r["frozen"] == frozen and r["shift"] == shift]
    return np.mean(xs, 0)


def compare(a, b, t, n, **kw):
    ea, eb = per_shape(a, t, n, **kw), per_shape(b, t, n, **kw)
    d = ea / eb - 1                                     # per-shape relative change, a vs b
    boots = [rng.choice(d, len(d)).mean() for _ in range(4000)]
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return d.mean(), lo, hi, (ea < eb).mean()


if __name__ == "__main__":
    out = []
    pairs = [("L", "S"), ("N", "S"), ("J", "S"), ("LT", "S"), ("LJ", "L"), ("LT", "L"), ("LT", "N"), ("L", "N"), ("J", "N")]
    for t in "AB":
        for n in [16, 64, 256]:
            for a, b in pairs:
                m, lo, hi, win = compare(a, b, t, n)
                sig = "*" if (hi < 0 or lo > 0) else " "
                line = f"task {t} n={n:3d}  {a:2s} vs {b:2s}: {100 * m:+6.1f}%  [{100 * lo:+6.1f}, {100 * hi:+6.1f}] {sig}  {a} better on {100 * win:4.0f}% of bodies"
                out.append(dict(task=t, n=n, a=a, b=b, mean=m, lo=lo, hi=hi, win=win))
                print(line)
    json.dump(out, open("results/paired_stats.json", "w"), indent=1)
