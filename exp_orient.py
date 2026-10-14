"""Experiment 2: bodies with a preferred orientation.

In the main experiment the body-to-flow angle is uniform over the full circle, and rotating
the velocity prompt (0 / 90 / 180 degrees) barely changes downstream error.  GeoPT reports that
prompt alignment matters on industrial datasets, where every body faces the flow the same way
(cars, aircraft).  Here we rebuild the labeled set with the flow along each body's long axis,
up to N(0, 10 deg), and repeat the prompt-rotation ablation.  Pretraining is unchanged.

usage: python exp_orient.py data | finetune | report
"""
import os
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
import sys
import numpy as np
from exp_common import load_jsonl, run_jobs, boot_ci, PROCS

DATA = "data/downstream_oriented.npz"
ORIENT_SD = np.radians(10.0)
VARIANTS = ["S", "L", "LT"]
SHIFTS = [0, 90, 180]
NS = [16, 64]
OUT = "results/exp_orient.jsonl"


def _down(seed):
    import world as W
    return W.build_downstream(89, seed, orient_sd=ORIENT_SD)


_D = None


def finetune_job(j):
    global _D
    import finetune as F
    if _D is None:
        _D = F.load(DATA)
    rec, per, _ = F.finetune(_D, j["variant"], j["task"], j["n"], j["seed"], shift=np.radians(j["shift"]))
    return {**rec, **j, "data": "oriented", "per_shape": [round(float(x), 5) for x in per]}


def key(r):
    return (r["variant"], r["task"], r["n"], r["seed"], r["shift"])


def per_shape(R, v, t, n, sh):
    xs = [r["per_shape"] for r in R if (r["variant"], r["task"], r["n"], r["shift"]) == (v, t, n, sh)
          and not r.get("frozen", False)]
    return np.mean(xs, 0) if xs else None


def report():
    import json
    import matplotlib.pyplot as plt
    import analyze as A
    Ro = load_jsonl(OUT)
    Ru = load_jsonl("results/finetune.jsonl")             # uniform orientation (main grid, n=64 only)
    out, lines = [], []
    for name, R in [("uniform", Ru), ("oriented", Ro)]:
        for t in "AB":
            for n in NS:
                for v in VARIANTS:
                    e0 = per_shape(R, v, t, n, 0)
                    if e0 is None:
                        continue
                    for sh in SHIFTS[1:]:
                        es = per_shape(R, v, t, n, sh)
                        if es is None:
                            continue
                        m, lo, hi = boot_ci(es / e0 - 1)
                        out.append(dict(data=name, task=t, n=n, variant=v, shift=sh, err0=float(e0.mean()),
                                        err=float(es.mean()), penalty=m, lo=lo, hi=hi))
                        lines.append(f"{name:8s} task {t} n={n:3d} {v:2s} shift {sh:3d}: {e0.mean():.4f} -> "
                                     f"{es.mean():.4f}  misalignment penalty {100 * m:+6.1f}% "
                                     f"[{100 * lo:+6.1f}, {100 * hi:+6.1f}]")
    print("\n".join(lines))
    json.dump(out, open("results/exp_orient_summary.json", "w"), indent=1)
    fig, axes = plt.subplots(2, 2, figsize=(10.5, 7.6), sharex=True)
    for c, t in enumerate("AB"):
        for r, (name, R) in enumerate([("uniform body orientation", Ru), ("preferred body orientation", Ro)]):
            ax = axes[r, c]
            A.style(ax)
            for v in VARIANTS:
                for n, ls in zip(NS, ["--", "-"]):
                    pts = [(sh, per_shape(R, v, t, n, sh)) for sh in SHIFTS]
                    pts = [(sh, e.mean()) for sh, e in pts if e is not None]
                    if len(pts) > 1:
                        ax.plot(*zip(*pts), ls, color=A.COL[v], marker=A.MRK[v], ms=7, lw=2,
                                markeredgecolor=A.SURF, label=f"{A.NAME[v]}, {n} labels")
            ax.set_xticks(SHIFTS)
            ax.set_xticklabels(["0 (aligned)", "90", "180 (reversed)"])
            ax.set_ylabel("test relative L2 error")
            ax.set_title(f"{A.TASK[t]}\n{name}", loc="left", fontsize=10)
    axes[1, 0].set_xlabel("prompt rotated away from the flow (degrees)")
    axes[1, 1].set_xlabel("prompt rotated away from the flow (degrees)")
    axes[1, 0].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig("figs/exp2_orientation.png", dpi=130)


if __name__ == "__main__":
    stage = sys.argv[1] if len(sys.argv) > 1 else "report"
    if stage == "data":
        from multiprocessing import Pool
        with Pool(PROCS) as pool:
            parts = pool.map(_down, [7000 + i for i in range(4)])     # 356 labeled shapes
        down = {k: np.concatenate([p[k] for p in parts]) if k != "grid" else parts[0][k] for k in parts[0]}
        np.savez(DATA, **down)
        print(f"{DATA}: {down['shapes'].shape[0]} shapes")
    elif stage == "finetune":
        jobs = [dict(variant=v, task=t, n=n, seed=s, shift=sh)
                for t in "BA" for n in NS for v in VARIANTS for sh in SHIFTS for s in (0, 1)]
        run_jobs(jobs, finetune_job, OUT, key)
    else:
        report()
