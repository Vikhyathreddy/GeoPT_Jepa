"""Experiment 3, Stage 1: do latent-transport features plan better?
(spec: experiments/03_action_ranking.md)

usage (from the repo root; N_PROCS sets the worker count):
  python exp3.py data       data/exp3/rank_train.npz, data/exp3/rank_test_costs.npz, sub-grid check
  python exp3.py plandemo   original plan-demo surrogates re-scored with scipy.stats.spearmanr
  python exp3.py probe      linear probes, all 7 backbones -> results/exp3_probe.jsonl
  python exp3.py time       time one R1 and one R3 run (task B)
  python exp3.py runs       R1 and R3 for S, L, LT x A, B x seeds 0-2 -> results/exp3_runs.jsonl
  python exp3.py report     Gate 1 tables -> results/exp3_gate1.json / .md

Choices the spec leaves open (also recorded in EXP3_RESULTS.md):
  * The 8 training actions of a body are 8 distinct candidates (multiples of 10 degrees) drawn
    with np.random.default_rng(0), body by body.
  * Field normalisation (R1/R2) uses the mean/std of the 512 training fields; cost
    standardisation uses the mean/std of the 512 training costs.
  * The 32 x 32 sub-grid shortcut failed its check (within-body Spearman of true sub-grid vs
    full-grid costs as low as 0.958 < 0.99 on the first 10 test bodies; results/
    exp3_data_checks.json), so task B uses all fluid cells: probe features and evaluation costs
    are computed on the full grid.  During R1/R3 training, 256 random fluid cells per sample.
    Surrogate costs for evaluation: A = max predicted q over the 128 surface points; B = sum of
    predicted c clipped to [0, 1] over all fluid cells (exactly plan_demo.py's cost).
  * S and J have no trained decoder: their probe uses a freshly initialised one
    (model.build("S", 100)), the same initialisation fine-tuning seed 0 starts from.
  * Ridge strength: grouped 8-fold cross-validation over the 64 training bodies (8 bodies per
    fold), alpha in 10^[-3..4], minimum mean squared error on the standardised cost.
  * A body whose predicted costs are constant gets rho = 0.
"""
from procs import N_PROCS  # single-threaded BLAS workers; must precede numpy
import os
import sys
import json
import time
import numpy as np
from multiprocessing import Pool

ANG = np.radians(np.arange(0, 360, 10))
N_BODIES, N_ACT = 64, 8
VAL_BODIES = np.arange(56, 64)
TEST = np.arange(256, 356)
TRAIN_DATA = "data/exp3/rank_train.npz"
TEST_COSTS = "data/exp3/rank_test_costs.npz"
BACKBONES = ["S", "N", "J", "L", "LJ", "LT", "LT0"]
RUNS_OUT = "results/exp3_runs.jsonl"
RUNS_KEY = ("regime", "backbone", "task", "seed", "lam", "budget")
PROBE_OUT = "results/exp3_probe.jsonl"
PRED_DIR = "results/exp3_preds"
STEPS, LR = 1000, 1e-3
BB, BA, NQ = 4, 4, 256          # batch: 4 bodies x 4 of their actions; 256 fluid cells per sample (B)


def eval_cells(body_mask):
    """task-B probe/evaluation cells of one body: the fluid cells of the full grid, or of the
    32 x 32 sub-grid if the sub-grid check passed."""
    ok = json.load(open("results/exp3_data_checks.json"))["subgrid_ok"]
    return np.flatnonzero(~body_mask & (sub_mask() if ok else True))


def sub_mask():
    """the 32 x 32 sub-grid (every other cell in both directions) of world.grid_centers' ij order."""
    i, j = np.meshgrid(np.arange(64), np.arange(64), indexing="ij")
    return ((i % 2 == 0) & (j % 2 == 0)).ravel()


# ----------------------------------------------------------------------------- data
def _solve(args):
    import world as W
    V, angs = args
    V = V.astype(np.float64)
    qs, cs, cA, cB, cBs = [], [], [], [], []
    sub = sub_mask()
    for a in angs:
        _, Cp, _ = W.panel_solve(V, a)
        q = np.sqrt(np.maximum(1 - Cp[::2], 0))
        _, body, c = W.plume_solve(V, a, 1.0)
        qs.append(q)
        cs.append(c)
        cA.append(q.max())
        cB.append(c[~body].sum())
        cBs.append(c[~body & sub].sum())
    return np.array(qs), np.array(cs), np.array(cA), np.array(cB), np.array(cBs)


def build_data():
    from scipy.stats import spearmanr
    os.makedirs("data/exp3", exist_ok=True)
    D = np.load("data/downstream.npz")
    rng = np.random.default_rng(0)
    acts = np.array([rng.choice(len(ANG), N_ACT, replace=False) for _ in range(N_BODIES)])
    with Pool(N_PROCS) as pool:
        tr = pool.map(_solve, [(D["shapes"][i], ANG[acts[i]]) for i in range(N_BODIES)])
        te = pool.map(_solve, [(D["shapes"][i], ANG) for i in TEST])
    q, c, cA, cB, _ = map(np.array, zip(*tr))
    np.savez(TRAIN_DATA, bodies=np.arange(N_BODIES), act=acts, alpha=ANG[acts].astype(np.float32),
             q=q.astype(np.float32), c=c.astype(np.float32), costA=cA, costB=cB)
    _, _, tA, tB, tBs = map(np.array, zip(*te))
    np.savez(TEST_COSTS, bodies=TEST, alpha=ANG, costA=tA, costB=tB, costB_sub=tBs)
    old = np.load("results/plan_true_costs.npz")
    rho = [spearmanr(tBs[k], tB[k])[0] for k in range(len(TEST))]
    chk = dict(first40_max_abs_diff_A=float(np.abs(old["A"] - tA[:40]).max()),
               first40_max_abs_diff_B=float(np.abs(old["B"] - tB[:40]).max()),
               subgrid_spearman_first10_min=float(np.min(rho[:10])),
               subgrid_spearman_all100_min=float(np.min(rho)), subgrid_spearman_all100_mean=float(np.mean(rho)),
               subgrid_ok=bool(np.min(rho[:10]) >= 0.99),
               train_solver_runs_per_task=int(acts.size), test_solver_runs_per_task=int(tA.size))
    json.dump(chk, open("results/exp3_data_checks.json", "w"), indent=1)
    print(json.dumps(chk, indent=1))


# ----------------------------------------------------------------------------- model pieces
def load_downstream():
    D = np.load("data/downstream.npz")
    return {k: D[k] for k in D.files}


def backbone(name, seed):
    import model as Mo
    from finetune import ENC, DEC
    P = Mo.build("S", 100 + seed)
    if name != "S":
        ck = np.load(f"ckpt/{name}.npz")
        for k in P:
            if k.startswith(ENC) or (name != "J" and k.startswith(DEC)):
                P[k].data = ck[k].astype(np.float32).copy()
    return P


def inputs(Dd, bodies, angs, task, cells=None, rng=None):
    """geometry, queries and prompt in each action's flow-aligned frame (as plan_demo.py).
    cells: list of grid-cell index arrays (task B), one per sample, all the same length."""
    import finetune as F
    R = F.rot(-np.asarray(angs, np.float64))
    shapes = np.einsum("nij,nkj->nki", R, Dd["shapes"][bodies])
    pts, nrm = F.surf_batch(shapes, rng) if rng is not None else F.surf_batch(shapes, phase=0.0)
    if task == "A":
        q = np.einsum("nij,nkj->nki", R, Dd["ptsA"][bodies])
    else:
        q = np.einsum("nij,nkj->nki", R, np.stack([Dd["grid"][c] for c in cells]))
    q = q.astype(np.float32)
    v = np.zeros_like(q)
    v[..., 0] = 0.25
    return pts, nrm, q, v


def decoder_out(P, pts, nrm, q, v):
    import model as Mo
    geo, _ = Mo.encode(P, pts, nrm)
    return geo, Mo.decode(P, geo, q, v)


def pool_points(h):
    """[mean, log-mean-exp] over the probe points: (n, Q, D) -> (n, 2D)."""
    import ag
    Q = h.shape[1]
    lme = ag.add(ag.logsumexp(h, 1), ag.const(np.full((1, h.shape[2]), -np.log(Q), np.float32)))
    return ag.concat([ag.mean_axis(h, 1), lme], -1)


def kpi_head_init(P, rng):
    import model as Mo
    Mo._lin(P, rng, "kpi.1", 2 * Mo.D, Mo.HID)
    Mo._lin(P, rng, "kpi.2", Mo.HID, 1)


def kpi_head(P, pooled):
    import ag
    import model as Mo
    return Mo.lin(P, ag.gelu(Mo.lin(P, pooled, "kpi.1")), "kpi.2")


def ranknet(chat, cost, groups, T=1.0):
    """mean over same-group pairs (k, l) with c_k != c_l of softplus(-sign(c_k - c_l)(chat_k - chat_l)/T)."""
    import ag
    n = len(cost)
    cols, coef = [], []
    for g in np.unique(groups):
        idx = np.flatnonzero(groups == g)
        for a in range(len(idx)):
            for b in range(a + 1, len(idx)):
                k, l = idx[a], idx[b]
                if cost[k] == cost[l]:
                    continue
                m = np.zeros(n, np.float32)
                m[k], m[l] = 1.0, -1.0
                cols.append(m)
                coef.append(-np.sign(cost[k] - cost[l]) / T)
    if not cols:
        return None
    M = ag.const(np.stack(cols, 1))                                   # (n, pairs)
    d = ag.linear(ag.reshape(chat, (1, n)), M)                         # chat_k - chat_l
    sp = ag.softplus(ag.mul(d, ag.const(np.array(coef, ag.DT)[None])))
    return ag.reshape(ag.mean_axis(sp, 1), ())


# ----------------------------------------------------------------------------- training (R1, R3)
def stats(TR, task):
    y = TR["q"].ravel() if task == "A" else np.concatenate(
        [TR["c"][i, k][~TR["body"][i]] for i in range(N_BODIES) for k in range(N_ACT)])
    c = TR["cost" + task].ravel()
    return float(y.mean()), float(y.std()), float(c.mean()), float(c.std())


def load_train():
    TR = dict(np.load(TRAIN_DATA))
    TR["body"] = np.load("data/downstream.npz")["body"][:N_BODIES]
    return TR


def train(regime, name, task, seed, lam=0.0, steps=STEPS, bodies=np.arange(N_BODIES)):
    import ag
    import model as Mo
    from pretrain import cosine
    Dd, TR = load_downstream(), load_train()
    mu, sd, cmu, csd = stats(TR, task)
    rng = np.random.default_rng(seed)
    P = backbone(name, seed)
    if regime in ("R1", "R2"):
        Mo.init_head(P, rng, "head.task", 1)
    else:
        kpi_head_init(P, rng)
    opt = ag.AdamW(P, lr=LR)
    fluid = [np.flatnonzero(~TR["body"][i]) for i in range(N_BODIES)]
    for it in range(steps):
        bs = rng.choice(bodies, BB, replace=False)
        ks = np.array([rng.choice(N_ACT, BA, replace=False) for _ in bs])
        b_idx, k_idx = np.repeat(bs, BA), ks.ravel()
        cells = [rng.choice(fluid[i], NQ, replace=False) for i in b_idx] if task == "B" else None
        pts, nrm, q, v = inputs(Dd, b_idx, TR["alpha"][b_idx, k_idx], task, cells, rng)
        _, h = decoder_out(P, pts, nrm, q, v)
        cost = TR["cost" + task][b_idx, k_idx]
        if regime in ("R1", "R2"):
            y = TR["q"][b_idx, k_idx] if task == "A" else np.stack([TR["c"][i, k][c] for i, k, c in zip(b_idx, k_idx, cells)])
            out = Mo.lin(P, h, "head.task")
            loss = ag.mse(out, ((y - mu) / sd)[..., None].astype(np.float32))
            if regime == "R2":
                raise NotImplementedError("R2 is Stage 2")
        else:
            chat = kpi_head(P, pool_points(h))
            z = ((cost - cmu) / csd).astype(np.float32)
            loss = ag.scale(ag.mse(chat, z[:, None]), 0.1)
            rn = ranknet(chat, cost, b_idx)
            if rn is not None:
                loss = ag.add(loss, rn)
        loss.backward()
        opt.step(cosine(it, steps, LR, warm=50))
    return P, (mu, sd, cmu, csd)


# ----------------------------------------------------------------------------- evaluation
def predict_costs(P, regime, task, norm, ids=TEST):
    import ag
    import model as Mo
    Dd = load_downstream()
    mu, sd, _, _ = norm
    C = np.zeros((len(ids), len(ANG)))
    with ag.no_grad():
        for k, i in enumerate(ids):
            cells = [eval_cells(Dd["body"][i])] * len(ANG) if task == "B" else None
            pts, nrm, q, v = inputs(Dd, np.full(len(ANG), i), ANG, task, cells)
            _, h = decoder_out(P, pts, nrm, q, v)
            if regime in ("R1", "R2"):
                out = Mo.lin(P, h, "head.task").data[..., 0] * sd + mu
                C[k] = out.max(1) if task == "A" else np.clip(out, 0, 1).sum(1)
            else:
                C[k] = kpi_head(P, pool_points(h)).data[:, 0]
    return C


def metrics(Cp, Ct):
    from scipy.stats import spearmanr
    rho, reg, top = [], [], []
    for p, t in zip(Cp, Ct):
        r = spearmanr(p, t)[0] if np.ptp(p) > 0 else 0.0
        rho.append(0.0 if np.isnan(r) else float(r))
        ch = t[np.argmin(p)]
        reg.append(float((ch - t.min()) / (t.mean() - t.min())))
        top.append(float(ch <= 1.02 * t.min()))
    return dict(rho=float(np.mean(rho)), regret=float(np.mean(reg)), top1=float(np.mean(top)),
                per_body_rho=[round(x, 5) for x in rho], per_body_regret=[round(x, 5) for x in reg],
                per_body_top1=top)


def true_costs(task):
    return np.load(TEST_COSTS)["cost" + task]


def run_job(j):
    import finetune as F
    t0 = time.time()
    P, norm = train(j["regime"], j["backbone"], j["task"], j["seed"], j["lam"])
    Cp = predict_costs(P, j["regime"], j["task"], norm)
    os.makedirs(PRED_DIR, exist_ok=True)
    np.save(f"{PRED_DIR}/{j['regime']}_{j['backbone']}_{j['task']}_s{j['seed']}_lam{j['lam']}_{j['budget']}.npy", Cp)
    rec = metrics(Cp, true_costs(j["task"]))
    if j["regime"] in ("R1", "R2"):
        err, per, _ = F.evaluate(P, F.get_data(), j["task"], 0.0, False, norm[0], norm[1])
        rec.update(field_rel_l2=err, per_shape=[round(float(x), 5) for x in per])
    rec["sec"] = round(time.time() - t0, 1)
    return rec


def run_jobs():
    return [dict(regime=r, backbone=b, task=t, seed=s, lam=0.0, budget="64x8")
            for t in "BA" for r in ("R1", "R3") for b in ("LT", "L", "S") for s in (0, 1, 2)]


# ----------------------------------------------------------------------------- linear probe
def probe_feats(P, task, bodies, angs_per_body):
    """[mean, max] over probe points of the decoder output, and over the encoder tokens."""
    import ag
    Dd = load_downstream()
    dec, enc = [], []
    with ag.no_grad():
        for i, angs in zip(bodies, angs_per_body):
            cells = [eval_cells(Dd["body"][i])] * len(angs) if task == "B" else None
            pts, nrm, q, v = inputs(Dd, np.full(len(angs), i), angs, task, cells)
            geo, h = decoder_out(P, pts, nrm, q, v)
            dec.append(np.concatenate([h.data.mean(1), h.data.max(1)], -1))
            enc.append(np.concatenate([geo.data.mean(1), geo.data.max(1)], -1))
    return np.concatenate(dec), np.concatenate(enc)


def ridge_fit(X, y, groups, alphas=10.0 ** np.arange(-3, 4.5, 0.5), folds=8):
    ug = np.unique(groups)
    fold_of = {g: k % folds for k, g in enumerate(ug)}
    f = np.array([fold_of[g] for g in groups])

    def fit(Xa, ya, a):
        m, s = Xa.mean(0), Xa.std(0) + 1e-8
        Z = (Xa - m) / s
        w = np.linalg.solve(Z.T @ Z + a * np.eye(Z.shape[1]), Z.T @ (ya - ya.mean()))
        return lambda Xb: ((Xb - m) / s) @ w + ya.mean()

    cv = []
    for a in alphas:
        err = []
        for k in range(folds):
            pred = fit(X[f != k], y[f != k], a)(X[f == k])
            err.append(((pred - y[f == k]) ** 2).mean())
        cv.append(np.mean(err))
    best = float(alphas[int(np.argmin(cv))])
    return fit(X, y, best), best, [float(x) for x in cv]


def probe_job(j):
    TR = load_train()
    P = backbone(j["backbone"], 0)
    _, _, cmu, csd = stats(TR, j["task"])
    Xd, Xe = probe_feats(P, j["task"], np.arange(N_BODIES), TR["alpha"])
    y = ((TR["cost" + j["task"]] - cmu) / csd).ravel()
    groups = np.repeat(np.arange(N_BODIES), N_ACT)
    Td, Te = probe_feats(P, j["task"], TEST, [ANG] * len(TEST))
    out = {}
    for name, X, T in (("dec", Xd, Td), ("enc", Xe, Te)):
        f, a, cv = ridge_fit(X, y, groups)
        m = metrics(f(T).reshape(len(TEST), len(ANG)), true_costs(j["task"]))
        out.update({f"{name}_{k}": v for k, v in m.items()})
        out[f"{name}_alpha"], out[f"{name}_cv_mse"] = a, min(cv)
    return out


# ----------------------------------------------------------------------------- original plan demo
def plan_demo_scipy():
    """the original single-action surrogates (ckpt/ft_*_n64.npz), re-scored with scipy's spearmanr."""
    import ag
    import finetune as F
    import plan_demo as PD
    Dd = F.load()
    ids = F.TEST[:PD.N_SHAPES]
    T = np.load("results/plan_true_costs.npz")
    out = {}
    for t in "AB":
        mu, sd = F.norm_stats(Dd, t)
        for v in ("S", "L", "LT"):
            ck = np.load(f"ckpt/ft_{v}_{t}_n64.npz")
            Cs = PD.surrogate_costs({k: ag.const(ck[k]) for k in ck.files}, Dd, t, ids, mu, sd)
            m = metrics(Cs, T[t])
            argsort_rho = np.mean([np.corrcoef(np.argsort(np.argsort(Cs[k])), np.argsort(np.argsort(T[t][k])))[0, 1]
                                   for k in range(len(ids))])
            out[f"{t}_{v}"] = dict(m, argsort_rho=float(argsort_rho))
            print(t, v, "scipy rho %.4f (argsort %.4f)  norm. regret %.4f  top1 %.3f"
                  % (m["rho"], argsort_rho, m["regret"], m["top1"]), flush=True)
    json.dump(out, open("results/exp3_plan_demo_scipy.json", "w"), indent=1)


# ----------------------------------------------------------------------------- Gate 1 report
def report():
    import bootstrap as BS
    from runner import load_rows
    R, Pr = load_rows(RUNS_OUT), load_rows(PROBE_OUT)
    pd = json.load(open("results/exp3_plan_demo_scipy.json"))
    out, L = {}, []
    L.append("Linear probes (100 test bodies x 36 actions): within-body Spearman rho / normalised regret / top-1 within 2%\n")
    L.append("| task | backbone | decoder features | encoder features |\n|---|---|---|---|")
    for t in "AB":
        for b in BACKBONES:
            r = [x for x in Pr if x["task"] == t and x["backbone"] == b]
            if r:
                r = r[0]
                L.append(f"| {t} | {b} | {r['dec_rho']:.3f} / {r['dec_regret']:.3f} / {r['dec_top1']:.2f} | "
                         f"{r['enc_rho']:.3f} / {r['enc_regret']:.3f} / {r['enc_top1']:.2f} |")
    for t in "AB":
        for f in ("dec", "enc"):
            g = lambda b: [dict(per_shape=x[f"{f}_per_body_rho"]) for x in Pr if x["task"] == t and x["backbone"] == b]
            for a, b in (("LT", "L"), ("LT", "S"), ("L", "S")):
                if g(a) and g(b):
                    out[f"probe_{f}_{t}_{a}-{b}"] = BS.paired(g(a), g(b), mode="diff", higher_is_better=True)
    L.append("\nProbe paired differences in rho (95% bootstrap CI over the 100 test bodies)\n")
    L.append("| task | features | LT - L | LT - S | L - S |\n|---|---|---|---|---|")
    for t in "AB":
        for f in ("dec", "enc"):
            c = [out.get(f"probe_{f}_{t}_{p}") for p in ("LT-L", "LT-S", "L-S")]
            L.append(f"| {t} | {f} | " + " | ".join(BS.fmt(x, pct=False) if x else "-" for x in c) + " |")
    for reg in ("R1", "R3"):
        L.append(f"\n{reg}: mean over fine-tune seeds 0-2 (rho / normalised regret / top-1 within 2%"
                 f"{' / field rel-L2' if reg == 'R1' else ''})\n")
        L.append("| task | backbone | seeds | rho | regret | top-1 |" + (" field rel-L2 |" if reg == "R1" else "")
                 + "\n|---|---|---|---|---|---|" + ("---|" if reg == "R1" else ""))
        for t in "AB":
            for b in ("S", "L", "LT"):
                rr = [x for x in R if (x["regime"], x["backbone"], x["task"]) == (reg, b, t)]
                if rr:
                    L.append(f"| {t} | {b} | {len(rr)} | {np.mean([x['rho'] for x in rr]):.3f} | "
                             f"{np.mean([x['regret'] for x in rr]):.3f} | {np.mean([x['top1'] for x in rr]):.2f} |"
                             + (f" {np.mean([x['field_rel_l2'] for x in rr]):.4f} |" if reg == "R1" else ""))
        L.append(f"\n{reg} paired differences (per-body values averaged over seeds; 95% CI over bodies)\n")
        L.append("| task | metric | LT - L | LT - S | L - S |\n|---|---|---|---|---|")
        for t in "AB":
            for met, hib in (("rho", True), ("regret", False)):
                g = lambda b: [dict(per_shape=x[f"per_body_{met}"], seed=x["seed"]) for x in R
                               if (x["regime"], x["backbone"], x["task"]) == (reg, b, t)]
                cells = []
                for a, b in (("LT", "L"), ("LT", "S"), ("L", "S")):
                    if g(a) and g(b):
                        c = BS.paired(g(a), g(b), mode="diff", higher_is_better=hib)
                        out[f"{reg}_{t}_{met}_{a}-{b}"] = c
                        cells.append(BS.fmt(c, pct=False))
                    else:
                        cells.append("-")
                L.append(f"| {t} | {met} | " + " | ".join(cells) + " |")
    L.append("\nOriginal plan-demo surrogates (single-action training, seed 0, first 40 test bodies), "
             "re-scored with scipy.stats.spearmanr\n")
    L.append("| task | backbone | rho (scipy) | rho (argsort, as reported) | normalised regret | top-1 |\n|---|---|---|---|---|---|")
    for k, v in pd.items():
        t, b = k.split("_")
        L.append(f"| {t} | {b} | {v['rho']:.3f} | {v['argsort_rho']:.3f} | {v['regret']:.3f} | {v['top1']:.2f} |")
    json.dump(out, open("results/exp3_gate1.json", "w"), indent=1)
    text = "\n".join(L)
    open("results/exp3_gate1.md", "w").write(text + "\n")
    print(text)
    figures(R, Pr)


def figures(R, Pr):
    import matplotlib.pyplot as plt
    import analyze as A
    import bootstrap as BS
    groups = [("probe, decoder", lambda b, t: [dict(per_shape=x["dec_per_body_rho"]) for x in Pr if (x["backbone"], x["task"]) == (b, t)]),
              ("probe, encoder", lambda b, t: [dict(per_shape=x["enc_per_body_rho"]) for x in Pr if (x["backbone"], x["task"]) == (b, t)]),
              ("R1 field", lambda b, t: [dict(per_shape=x["per_body_rho"], seed=x["seed"]) for x in R if (x["regime"], x["backbone"], x["task"]) == ("R1", b, t)]),
              ("R3 KPI only", lambda b, t: [dict(per_shape=x["per_body_rho"], seed=x["seed"]) for x in R if (x["regime"], x["backbone"], x["task"]) == ("R3", b, t)])]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), sharey=False)
    for ax, t in zip(axes, "AB"):
        A.style(ax)
        bbs = [b for b in BACKBONES]
        w = 0.8 / len(bbs)
        for gi, (gname, get) in enumerate(groups):
            for bi, b in enumerate(bbs):
                rows = get(b, t)
                if not rows:
                    continue
                m, lo, hi = BS.boot_mean(BS.per_body(rows))
                x = gi + (bi - (len(bbs) - 1) / 2) * w
                ax.errorbar([x], [m], yerr=[[m - lo], [hi - m]], marker=A.MRK[b], ms=7, color=A.COL[b], ls="none",
                            capsize=2, markeredgecolor=A.SURF, label=A.NAME[b] if gi == 0 or b in ("S", "L", "LT") else None)
        ax.set_xticks(range(len(groups)))
        ax.set_xticklabels([g for g, _ in groups])
        ax.set_ylabel("within-body Spearman rho, 100 test bodies x 36 actions")
        ax.set_title(A.TASK[t], loc="left", fontsize=11)
    h, l = [], []
    for hh, ll in zip(*axes[0].get_legend_handles_labels()):
        if ll not in l:
            h.append(hh)
            l.append(ll)
    fig.legend(h, l, frameon=False, fontsize=8.5, loc="lower center", ncol=7)
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    fig.savefig("figs/exp3_ranking.png", dpi=130)
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    for ax, t in zip(axes, "AB"):
        A.style(ax)
        for b in ("S", "L", "LT"):
            rr = [x for x in R if (x["regime"], x["backbone"], x["task"]) == ("R1", b, t)]
            if not rr:
                continue
            ax.plot([x["field_rel_l2"] for x in rr], [x["rho"] for x in rr], marker=A.MRK[b], ms=7, ls="none",
                    color=A.COL[b], alpha=0.5, markeredgecolor=A.SURF)
            ax.plot([np.mean([x["field_rel_l2"] for x in rr])], [np.mean([x["rho"] for x in rr])], marker=A.MRK[b],
                    ms=11, ls="none", color=A.COL[b], markeredgecolor=A.SURF, label=f"{A.NAME[b]} (R1; small = single seeds)")
        ax.set_xlabel("field relative L2 error on the original single-action test set")
        ax.set_ylabel("within-body Spearman rho")
        ax.set_title(A.TASK[t], loc="left", fontsize=11)
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, frameon=False, fontsize=8.5, loc="lower center", ncol=3)
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    fig.savefig("figs/exp3_accuracy_vs_ranking.png", dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    import runner
    stage = sys.argv[1] if len(sys.argv) > 1 else "report"
    if stage == "data":
        build_data()
    elif stage == "plandemo":
        plan_demo_scipy()
    elif stage == "probe":
        runner.run([dict(backbone=b, task=t) for t in "BA" for b in BACKBONES], probe_job, PROBE_OUT,
                   ("backbone", "task"))
    elif stage == "time":
        for reg in ("R1", "R3"):
            t0 = time.time()
            r = run_job(dict(regime=reg, backbone="L", task="B", seed=0, lam=0.0, budget="64x8"))
            print(reg, "task B one run: %.0f s" % (time.time() - t0), {k: v for k, v in r.items() if not isinstance(v, list)})
    elif stage == "runs":
        runner.run(run_jobs(), run_job, RUNS_OUT, RUNS_KEY)
    elif stage == "report":
        report()
    else:
        sys.exit(__doc__)
