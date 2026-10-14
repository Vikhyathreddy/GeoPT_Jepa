"""Planning with a learned simulator: the flow direction relative to the body is the *action*.

For each unseen test body a fine-tuned surrogate scores 36 candidate orientations and
picks the cheapest - the JEPA-AC recipe (roll the world model out per candidate
action, take the argmin of a cost) with a GeoPT-style field simulator as the world
model.  The pick is then scored with the real solver.

cost A: peak surface speed max q          (peak suction / structural load)
cost B: plume footprint sum(c) at U = 1  (how much of the domain gets contaminated)
usage: python plan_demo.py S L LT ...
"""
import os
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
import sys
import json
import numpy as np
from multiprocessing import Pool
import ag
import world as W
import finetune as F

ANG = np.radians(np.arange(0, 360, 10))
N_SHAPES = 40


def surrogate_costs(P, Dd, task, ids, mu, sd):
    C = np.zeros((len(ids), len(ANG)))
    R = F.rot(-ANG)                                            # (36,2,2) absolute -> flow frame
    with ag.no_grad():
        for k, i in enumerate(ids):
            shapes = np.einsum("aij,kj->aki", R, Dd["shapes"][i])
            pts, nrm = F.surf_batch(shapes, phase=0.0)
            if task == "A":
                q = np.einsum("aij,kj->aki", R, Dd["ptsA"][i]).astype(np.float32)
                sp = 0.25
            else:
                j = np.flatnonzero(~Dd["body"][i])
                q = np.einsum("aij,kj->aki", R, Dd["grid"][j]).astype(np.float32)
                sp = 0.25                                       # U = 1
            v = np.zeros_like(q)
            v[..., 0] = sp
            out = F.forward(P, pts, nrm, q, v, False).data[..., 0] * sd + mu
            C[k] = out.max(1) if task == "A" else np.clip(out, 0, 1).sum(1)
    return C


def true_costs_one(args):
    task, V = args
    row = []
    for a in ANG:
        if task == "A":
            _, Cp, _ = W.panel_solve(V.astype(np.float64), a)
            row.append(np.sqrt(np.maximum(1 - Cp[::2], 0)).max())
        else:
            _, body, c = W.plume_solve(V.astype(np.float64), a, 1.0)
            row.append(c[~body].sum())
    return row


def fit_one(args):
    variant, task = args
    Dd = F.load()
    P = F.finetune_params(Dd, variant, task, 64, 0)
    return variant, task, {k: p.data for k, p in P.items()}


if __name__ == "__main__":
    variants = sys.argv[1:] or ["S", "L", "LT"]
    Dd = F.load()
    ids = F.TEST[:N_SHAPES]
    tc_path = "results/plan_true_costs.npz"
    with Pool(2) as pool:
        if not os.path.exists(tc_path):
            TA = np.array(pool.map(true_costs_one, [("A", Dd["shapes"][i]) for i in ids]))
            TB = np.array(pool.map(true_costs_one, [("B", Dd["shapes"][i]) for i in ids]))
            np.savez(tc_path, A=TA, B=TB)
        fitted = pool.map(fit_one, [(v, t) for t in "AB" for v in variants])
    for variant, task, Pd in fitted:
        np.savez(f"ckpt/ft_{variant}_{task}_n64.npz", **Pd)
    T = np.load(tc_path)
    out = {}
    for task in "AB":
        Ct = T[task]
        best = Ct.min(1)
        rand_regret = Ct.mean(1) - best
        res = {"random": {"regret": float(rand_regret.mean())}, "mean_best_cost": float(best.mean()),
               "mean_random_cost": float(Ct.mean())}
        for variant, t, Pd in fitted:
            if t != task:
                continue
            P = {k: ag.const(v) for k, v in Pd.items()}
            mu, sd = F.norm_stats(Dd, task)
            Cs = surrogate_costs(P, Dd, task, ids, mu, sd)
            pick = Cs.argmin(1)
            chosen = Ct[np.arange(len(ids)), pick]
            reg = chosen - best
            res[variant] = {"regret": float(reg.mean()),
                            "regret_vs_random": float(reg.mean() / rand_regret.mean()),
                            "frac_candidates_better": float((Ct < chosen[:, None]).mean()),
                            "within_2pct_of_best": float((chosen <= 1.02 * best).mean())}
            print(task, variant, res[variant], flush=True)
        out[task] = res
    json.dump(out, open("results/plan_demo.json", "w"), indent=1)
    print(json.dumps(out, indent=1))
