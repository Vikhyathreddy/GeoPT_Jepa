"""Paired bootstrap of per-body planning regret (re-uses the fine-tuned surrogates saved by plan_demo)."""
import os
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
import json
import numpy as np
import ag
import finetune as F
import plan_demo as PD

rng = np.random.default_rng(0)
Dd = F.load()
ids = F.TEST[:PD.N_SHAPES]
T = np.load("results/plan_true_costs.npz")
out = {}
for task in "AB":
    Ct = T[task]
    best = Ct.min(1)
    reg, rank_corr = {}, {}
    for v in ["S", "L", "LT"]:
        ck = np.load(f"ckpt/ft_{v}_{task}_n64.npz")
        P = {k: ag.const(ck[k]) for k in ck.files}
        mu, sd = F.norm_stats(Dd, task)
        Cs = PD.surrogate_costs(P, Dd, task, ids, mu, sd)
        reg[v] = Ct[np.arange(len(ids)), Cs.argmin(1)] - best
        # how well does the surrogate rank the 36 candidate actions? (Spearman per body)
        rc = []
        for k in range(len(ids)):
            a = np.argsort(np.argsort(Cs[k]))
            b = np.argsort(np.argsort(Ct[k]))
            rc.append(np.corrcoef(a, b)[0, 1])
        rank_corr[v] = float(np.mean(rc))
    res = {"rank_corr": rank_corr}
    for a, b in [("L", "S"), ("LT", "S"), ("LT", "L")]:
        d = reg[a] - reg[b]
        boots = [rng.choice(d, len(d)).mean() for _ in range(4000)]
        lo, hi = np.percentile(boots, [2.5, 97.5])
        res[f"{a}-{b}"] = dict(mean=float(d.mean()), lo=float(lo), hi=float(hi))
        print(f"task {task}: regret {a} - {b} = {d.mean():+.4f}  95% CI [{lo:+.4f}, {hi:+.4f}]")
    print(f"task {task}: Spearman rank corr surrogate vs solver over 36 actions: {rank_corr}")
    out[task] = res
json.dump(out, open("results/plan_stats.json", "w"), indent=1)
