"""Experiment 1: noisy explicit pretext targets.

Rule of thumb under test: predicting in latent space beats predicting explicit targets once
those targets carry detail the input cannot explain.  We add fixed, zero-mean Gaussian noise
(a noisy "simulator") to the vector-distance targets of the unlabeled training corpus, at
sigma = k * std(vd), and re-pretrain
  L   GeoPT-lifted          every pretext target is explicit (noisy)
  LT  latent-transport JEPA explicit only in its v = 0 anchor (noisy); the latent targets are
                            computed by the EMA network from the clean particle positions
Validation shapes keep clean targets, so the logged val_* losses measure clean accuracy.
k = 0 re-uses ckpt/L.npz, ckpt/LT.npz and the matching rows of results/finetune.jsonl.

usage: python exp_noise.py pretrain | finetune | report
"""
import os
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
import sys
import numpy as np
from exp_common import load_jsonl, run_jobs, boot_ci, PROCS

LEVELS = [0.25, 0.5, 1.0]
VARIANTS = ["L", "LT"]
NS = [16, 64]
OUT = "results/exp_noise.jsonl"
STEPS = int(os.environ.get("PT_STEPS", "3000"))


def tag(v, k):
    return v if k == 0 else f"{v}_noise{k}"


def noisy(Dd, k):
    import pretrain as PT
    tr = Dd["sid"] < PT.N_TRAIN_SHAPES
    sd = float(Dd["vd"][tr].std())
    rng = np.random.default_rng(int(1000 * k))
    vd = Dd["vd"].copy()
    vd[tr] += (k * sd * rng.standard_normal(vd[tr].shape)).astype(np.float32)
    return dict(Dd, vd=vd), sd


def pretrain_job(job):
    import pretrain as PT
    v, k = job
    Dd, sd = noisy(PT.load("data/pretrain.npz"), k)
    print(f"{tag(v, k)}: noise sigma = {k} x {sd:.4f}", flush=True)
    PT.pretrain(v, steps=STEPS, Dd=Dd, out=f"ckpt/{tag(v, k)}.npz")
    return tag(v, k)


_D = None


def finetune_job(j):
    global _D
    import finetune as F
    if _D is None:
        _D = F.load()
    rec, per, _ = F.finetune(_D, tag(j["base"], j["noise"]), j["task"], j["n"], j["seed"])
    return {**rec, **j, "per_shape": [round(float(x), 5) for x in per]}


def key(r):
    return (r["base"], r["noise"], r["task"], r["n"], r["seed"])


def rows():
    """noise > 0 rows from this experiment + k = 0 rows from the main grid."""
    R = load_jsonl(OUT)
    for r in load_jsonl("results/finetune.jsonl"):
        if r["variant"] in VARIANTS and r["n"] in NS and r["shift"] == 0 and not r["frozen"]:
            R.append({**r, "base": r["variant"], "noise": 0.0})
    return R


def per_shape(R, v, k, t, n):
    xs = [r["per_shape"] for r in R if (r["base"], r["noise"], r["task"], r["n"]) == (v, k, t, n)]
    return np.mean(xs, 0) if xs else None


def report():
    import json
    import matplotlib.pyplot as plt
    import analyze as A
    R = rows()
    ks = [0.0] + LEVELS
    out, lines = [], []
    for t in "AB":
        for n in NS:
            for k in ks:
                eL, eT = per_shape(R, "L", k, t, n), per_shape(R, "LT", k, t, n)
                if eL is None or eT is None:
                    continue
                m, lo, hi = boot_ci(eT / eL - 1)
                out.append(dict(task=t, n=n, noise=k, L=float(eL.mean()), LT=float(eT.mean()),
                                LT_vs_L=m, lo=lo, hi=hi, LT_better_frac=float((eT < eL).mean())))
                lines.append(f"task {t} n={n:3d} noise={k:4.2f}  L {eL.mean():.4f}  LT {eT.mean():.4f}  "
                             f"LT vs L {100 * m:+6.1f}% [{100 * lo:+6.1f}, {100 * hi:+6.1f}]")
    print("\n".join(lines))
    json.dump(out, open("results/exp_noise_summary.json", "w"), indent=1)
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2))
    for ax, t in zip(axes, "AB"):
        A.style(ax)
        for v in VARIANTS:
            for n, ls in zip(NS, ["--", "-"]):
                pts = [(k, per_shape(R, v, k, t, n)) for k in ks]
                pts = [(k, e.mean()) for k, e in pts if e is not None]
                if pts:
                    ax.plot(*zip(*pts), ls, color=A.COL[v], marker=A.MRK[v], ms=7, lw=2,
                            markeredgecolor=A.SURF, label=f"{A.NAME[v]}, {n} labels")
        ax.set_xticks(ks)
        ax.set_xlabel("pretext target noise (x std of the clean target)")
        ax.set_ylabel("test relative L2 error")
        ax.set_title(A.TASK[t], loc="left", fontsize=11)
    axes[0].legend(frameon=False, fontsize=8.5)
    fig.tight_layout()
    fig.savefig("figs/exp1_noise.png", dpi=130)


if __name__ == "__main__":
    stage = sys.argv[1] if len(sys.argv) > 1 else "report"
    if stage == "pretrain":
        from multiprocessing import Pool
        jobs = [(v, k) for k in LEVELS for v in VARIANTS if not os.path.exists(f"ckpt/{tag(v, k)}.npz")]
        jobs.sort(key=lambda j: j[0] != "LT")          # LT is ~2x slower: start it first
        with Pool(PROCS) as pool:
            for name in pool.imap_unordered(pretrain_job, jobs):
                print("done", name, flush=True)
    elif stage == "finetune":
        jobs = [dict(base=v, noise=k, task=t, n=n, seed=s)
                for t in "BA" for n in NS for k in LEVELS for v in VARIANTS for s in (0, 1)]
        run_jobs(jobs, finetune_job, OUT, key)
    else:
        report()
