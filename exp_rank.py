"""Experiment 3: train surrogates on an action-ranking loss.

plan_demo.py found that latent-transport JEPA (LT) plans better on task B than GeoPT-lifted (L)
although its field error is higher.  If LT's features really carry more decision-relevant
information, that advantage should survive (or grow) when every surrogate is trained for the
decision itself.  Fine-tuning keeps the usual field loss and adds

    lam * ListNet( softmax(-surrogate cost / tau),  softmax(-true cost / tau) )

over K = 8 candidate flow directions for 2 training bodies per step (costs as in plan_demo.py:
smooth max of surface speed for A, mean plume concentration over 256 random cells for B; both
scaled per body by the spread of its true costs).  The true costs need 36 extra solver runs per
training body - scalar costs only, but more simulation than the field labels; compare variants
at equal lam, not lam > 0 against lam = 0.

Evaluated on the same 40 unseen bodies and 36 actions as plan_demo.py.
usage: python exp_rank.py costs | finetune | report
"""
import os
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
import sys
import numpy as np
from exp_common import load_jsonl, run_jobs, boot_ci, PROCS

VARIANTS = ["S", "L", "LT"]
LAMS = [0.0, 0.1, 1.0]
SEEDS = [0, 1]
N = 64
K, BR, NQ, TAU, BETA = 8, 2, 256, 0.5, 20.0
COSTS = "results/rank_train_costs.npz"
OUT = "results/exp_rank.jsonl"


def pool_ids(seed, n=N):
    """the training pool finetune.fit draws for this seed (same rng sequence)."""
    import model as Mo
    import finetune as F
    rng = np.random.default_rng(seed)
    Mo.init_head({}, rng, "head.task", 1)
    return rng.choice(F.TRAIN, n, replace=False)


def make_rank_loss(Dd, task, table, lam, seed):
    import ag
    import finetune as F
    import plan_demo as PD
    rng = np.random.default_rng(10_000 + seed)
    row = {int(i): r for r, i in enumerate(table["ids"])}
    tc_all = table[task]

    def extra(P, pool, mu, sd):
        logits, targets = [], []
        for i in rng.choice(pool, BR, replace=False):
            ka = rng.choice(len(PD.ANG), K, replace=False)
            R = F.rot(-PD.ANG[ka])
            pts, nrm = F.surf_batch(np.einsum("aij,kj->aki", R, Dd["shapes"][i]), rng)
            if task == "A":
                q = np.einsum("aij,kj->aki", R, Dd["ptsA"][i]).astype(np.float32)
            else:
                j = rng.choice(np.flatnonzero(~Dd["body"][i]), NQ, replace=False)
                q = np.einsum("aij,kj->aki", R, Dd["grid"][j]).astype(np.float32)
            v = np.zeros_like(q)
            v[..., 0] = 0.25
            out = ag.reshape(F.forward(P, pts, nrm, q, v, False), (K, q.shape[1]))
            if task == "A":   # smooth max of q = mu + sd * out, minus the constant mu
                c = ag.scale(ag.logsumexp(ag.scale(out, BETA * sd), 1), 1.0 / BETA)
            else:             # mean concentration over the sampled cells, minus mu
                c = ag.scale(ag.mean_axis(out, 1), sd)
            tc = tc_all[row[int(i)], ka]
            s = tc.std() + 1e-8
            logits.append(ag.reshape(ag.scale(c, -1.0 / (s * TAU)), (1, K)))
            z = -(tc - tc.min()) / (s * TAU)
            targets.append(np.exp(z) / np.exp(z).sum())
        return ag.scale(ag.softmax_xent(ag.concat(logits, 0), np.array(targets, np.float32)), lam)
    return extra


def true_costs_job(args):
    import plan_demo as PD
    return PD.true_costs_one(args)


_D = None


def finetune_job(j):
    global _D
    import ag
    import finetune as F
    import plan_demo as PD
    if _D is None:
        _D = F.load()
    t, lam = j["task"], j["lam"]
    extra = make_rank_loss(_D, t, np.load(COSTS), lam, j["seed"]) if lam > 0 else None
    rec, per, P = F.finetune(_D, j["variant"], t, N, j["seed"], extra=extra, return_params=True)
    P = {k: ag.const(p.data) for k, p in P.items()}
    mu, sd = F.norm_stats(_D, t)
    ids = F.TEST[:PD.N_SHAPES]
    Ct = np.load("results/plan_true_costs.npz")[t]
    Cs = PD.surrogate_costs(P, _D, t, ids, mu, sd)
    chosen = Ct[np.arange(len(ids)), Cs.argmin(1)]
    best = Ct.min(1)
    rank = [np.corrcoef(np.argsort(np.argsort(Cs[k])), np.argsort(np.argsort(Ct[k])))[0, 1] for k in range(len(ids))]
    return {**rec, **j, "regret": float((chosen - best).mean()),
            "within_2pct_of_best": float((chosen <= 1.02 * best).mean()), "spearman": float(np.mean(rank)),
            "per_body_regret": [round(float(x), 5) for x in chosen - best],
            "per_body_spearman": [round(float(x), 4) for x in rank],
            "per_shape": [round(float(x), 5) for x in per]}


def key(r):
    return (r["variant"], r["task"], r["seed"], r["lam"])


def report():
    import json
    import matplotlib.pyplot as plt
    import analyze as A
    R = load_jsonl(OUT)

    def get(v, t, lam, f):
        xs = [r[f] for r in R if (r["variant"], r["task"], r["lam"]) == (v, t, lam)]
        return np.mean(xs, 0) if xs else None

    out, lines = [], []
    for t in "AB":
        for lam in LAMS:
            for v in VARIANTS:
                if get(v, t, lam, "regret") is None:
                    continue
                d = dict(task=t, lam=lam, variant=v, regret=float(get(v, t, lam, "regret")),
                         spearman=float(get(v, t, lam, "spearman")),
                         within_2pct=float(get(v, t, lam, "within_2pct_of_best")),
                         rel_l2=float(get(v, t, lam, "rel_l2")))
                out.append(d)
                lines.append(f"task {t} lam={lam:3.1f} {v:2s}: regret {d['regret']:8.4f}  spearman {d['spearman']:.3f}  "
                             f"within 2% {d['within_2pct']:.2f}  field relL2 {d['rel_l2']:.4f}")
            for a, b in [("LT", "L"), ("LT", "S"), ("L", "S")]:
                ra, rb = get(a, t, lam, "per_body_regret"), get(b, t, lam, "per_body_regret")
                if ra is None or rb is None:
                    continue
                m, lo, hi = boot_ci(np.array(ra) - np.array(rb))
                out.append(dict(task=t, lam=lam, pair=f"{a}-{b}", regret_diff=m, lo=lo, hi=hi))
                lines.append(f"task {t} lam={lam:3.1f} regret {a}-{b} = {m:+.4f}  95% CI [{lo:+.4f}, {hi:+.4f}]")
    print("\n".join(lines))
    json.dump(out, open("results/exp_rank_summary.json", "w"), indent=1)
    fig, axes = plt.subplots(2, 2, figsize=(10.5, 7.2))
    for c, t in enumerate("AB"):
        for r, (f, lab) in enumerate([("regret", "planning regret (lower is better)"),
                                      ("spearman", "action-ranking Spearman (higher is better)")]):
            ax = axes[r, c]
            A.style(ax)
            for v in VARIANTS:
                pts = [(lam, get(v, t, lam, f)) for lam in LAMS]
                pts = [(i, y) for i, (lam, y) in enumerate(pts) if y is not None]
                if pts:
                    ax.plot(*zip(*pts), "-", color=A.COL[v], marker=A.MRK[v], ms=7, lw=2,
                            markeredgecolor=A.SURF, label=A.NAME[v])
            ax.set_xticks(range(len(LAMS)))
            ax.set_xticklabels([f"lam = {lam:g}" for lam in LAMS])
            ax.set_ylabel(lab)
            if r == 0:
                ax.set_title(A.TASK[t], loc="left", fontsize=11)
    axes[0, 0].legend(frameon=False, fontsize=8.5)
    fig.tight_layout()
    fig.savefig("figs/exp3_rank_loss.png", dpi=130)


if __name__ == "__main__":
    stage = sys.argv[1] if len(sys.argv) > 1 else "report"
    if stage == "costs":
        from multiprocessing import Pool
        import finetune as F
        Dd = F.load()
        ids = np.unique(np.concatenate([pool_ids(s) for s in SEEDS]))
        with Pool(PROCS) as pool:
            TA = np.array(pool.map(true_costs_job, [("A", Dd["shapes"][i]) for i in ids]))
            TB = np.array(pool.map(true_costs_job, [("B", Dd["shapes"][i]) for i in ids]))
        np.savez(COSTS, ids=ids, A=TA, B=TB)
        print(f"{COSTS}: {len(ids)} bodies x 36 actions")
    elif stage == "finetune":
        jobs = [dict(variant=v, task=t, seed=s, lam=lam) for t in "BA" for lam in LAMS for v in VARIANTS for s in SEEDS]
        run_jobs(jobs, finetune_job, OUT, key)
    else:
        report()
