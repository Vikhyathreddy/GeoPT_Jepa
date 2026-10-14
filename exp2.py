"""Experiment 2, Stage 1: preferred orientation and prompt alignment
(spec: experiments/02_orientation_prompts.md).

usage (from the repo root; N_PROCS sets the worker count):
  python exp2.py data       data/exp2/downstream_canon.npz and data/exp2/pretrain_canon.npz
  python exp2.py checks     default data bit-identical; canonical vs original bodies (area, perimeter)
  python exp2.py pretrain   Lc: L pretrained on pretrain_canon.npz -> ckpt/exp2/Lc_s0.npz
  python exp2.py gates      G1-G3 fine-tuning runs -> results/exp2_finetune.jsonl
  python exp2.py report     Gate tables -> results/exp2_gates.json / .md
  python exp2.py diag       G1 diagnostic: train vs test error at n=64 -> results/exp2_diag.json
  python exp2.py sweep      scratch n=256 control, zero-prompt runs, Stage 2 shift sweep, Stage 3a (q.1 frozen)
  python exp2.py report2    Stage 2/3a tables -> results/exp2_stage2.json / .md, figs/exp2_shift_curves.png

Choices the spec leaves open (also recorded in EXP2_RESULTS.md):
  * New flow angles come from np.random.default_rng(chunk seed + 50_000): alpha_A then alpha_B
    for each body, uniform in (-30, +30) degrees.
  * Mean-predictor baseline (G1): the constant mean of the labels over all 256 TRAIN bodies (the
    same mean used for normalisation); test rel-L2 is computed per body as for the models.
"""
from procs import N_PROCS  # single-threaded BLAS workers; must precede numpy
import os
import sys
import json
import numpy as np
from multiprocessing import Pool

DOWN = "data/exp2/downstream_canon.npz"
PRE = "data/exp2/pretrain_canon.npz"
LC = "ckpt/exp2/Lc_s0.npz"
FT_OUT = "results/exp2_finetune.jsonl"
FT_KEY = ("variant", "data", "task", "n", "seed", "shift", "prompt_mode", "freeze")
FLOW_RANGE_DEG = 30.0


def ckpt_of(v):
    """S scratch; L / Lc pretraining seed 0; Ls<k> = L pretraining seed k (ckpt/exp1/L_clean_s<k>.npz)."""
    if v.startswith("Ls"):
        return f"ckpt/exp1/L_clean_s{int(v[2:])}.npz"
    return {"S": None, "L": "ckpt/L.npz", "Lc": LC}[v]


# ----------------------------------------------------------------------------- data
def _down(seed):
    import world as W
    return W.build_downstream(89, seed, canonical=True, flow_range=np.radians(FLOW_RANGE_DEG))


def _pre(seed):
    import world as W
    return W.build_pretrain(100, n_dyn=8, seed=seed, rotate=False)


def build_data():
    os.makedirs("data/exp2", exist_ok=True)
    with Pool(N_PROCS) as pool:
        parts = pool.map(_down, [5000 + i for i in range(4)])             # as make_data.py
        down = {k: np.concatenate([p[k] for p in parts]) if k != "grid" else parts[0][k] for k in parts[0]}
        np.savez(DOWN, **down)
        print(f"{DOWN}: {down['shapes'].shape[0]} shapes", flush=True)
        parts = pool.map(_pre, [1000 + i for i in range(21)])
    off, merged = 0, {k: [] for k in ("shapes", "X", "v", "vd", "sid")}
    for p in parts:
        for k in ("shapes", "X", "v", "vd"):
            merged[k].append(p[k])
        merged["sid"].append(p["sid"] + off)
        off += len(p["shapes"])
    pre = {k: np.concatenate(v) for k, v in merged.items()}
    np.savez(PRE, **pre)
    print(f"{PRE}: {pre['shapes'].shape[0]} shapes, {pre['X'].shape[0]} samples", flush=True)


# ----------------------------------------------------------------------------- checks
def _area_perimeter(V):
    x, y = V[:, 0].astype(np.float64), V[:, 1].astype(np.float64)
    area = 0.5 * np.abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))
    return area, np.linalg.norm(np.roll(V, -1, 0) - V, axis=1).sum()


def checks():
    import world as W
    out = {}
    orig = np.load("data/downstream.npz")
    d = W.build_downstream(89, 5000)                                        # one chunk, defaults
    out["downstream_default_bit_identical"] = all(np.array_equal(d[k], orig[k][:89] if k != "grid" else orig[k])
                                                  for k in d)
    origp = np.load("data/pretrain.npz")
    p = W.build_pretrain(100, n_dyn=8, seed=1000)
    out["pretrain_default_bit_identical"] = all(np.array_equal(p[k], origp[k][:len(p[k])]) for k in p)
    canon, cpre = np.load(DOWN), np.load(PRE)
    errs = np.array([np.subtract(_area_perimeter(a), _area_perimeter(b))
                     for a, b in zip(canon["shapes"], orig["shapes"])])
    out["downstream_area_max_abs_diff"] = float(np.abs(errs[:, 0]).max())
    out["downstream_perimeter_max_abs_diff"] = float(np.abs(errs[:, 1]).max())
    errs = np.array([np.subtract(_area_perimeter(a), _area_perimeter(b))
                     for a, b in zip(cpre["shapes"], origp["shapes"])])
    out["pretrain_area_max_abs_diff"] = float(np.abs(errs[:, 0]).max())
    out["pretrain_perimeter_max_abs_diff"] = float(np.abs(errs[:, 1]).max())
    out["same_UB"] = bool(np.array_equal(canon["UB"], orig["UB"]))
    out["alpha_range_deg"] = [float(np.degrees(canon[k]).min()) for k in ("alphaA", "alphaB")] + \
                             [float(np.degrees(canon[k]).max()) for k in ("alphaA", "alphaB")]
    # body orientation: principal (second-moment) axis of the canonical bodies vs +x
    ang = []
    for V in canon["shapes"].astype(np.float64):
        c = V - V.mean(0)
        w, U = np.linalg.eigh(c.T @ c)
        a = np.degrees(np.arctan2(U[1, -1], U[0, -1])) % 180
        ang.append(min(a, 180 - a))
    out["canon_principal_axis_deg_median_p90_max"] = [float(np.median(ang)), float(np.percentile(ang, 90)),
                                                      float(np.max(ang))]
    ok = out["downstream_default_bit_identical"] and out["pretrain_default_bit_identical"] and \
        max(out[k] for k in out if k.endswith("_max_abs_diff")) < 1e-6
    out["all_ok"] = bool(ok)
    json.dump(out, open("results/exp2_checks.json", "w"), indent=1)
    print(json.dumps(out, indent=1))
    if not ok:
        sys.exit("exp2 checks FAILED")


# ----------------------------------------------------------------------------- fine-tuning
def finetune_job(j):
    import finetune as F
    rec, per, _ = F.finetune(None, j["variant"], j["task"], j["n"], j["seed"], shift=np.radians(j["shift"]),
                             freeze=tuple(j["freeze"].split(",")) if j["freeze"] else (),
                             ckpt=ckpt_of(j["variant"]), frame="absolute", prompt_mode=j["prompt_mode"],
                             data=DOWN)
    return {**rec, "per_shape": [round(float(x), 5) for x in per]}


def gate_jobs():
    return [dict(variant=v, data="canon30", task=t, n=64, seed=s, shift=0, prompt_mode=m, freeze="")
            for t in "BA" for v in ("Lc", "L", "S") for m in ("normal", "zero") for s in (0, 1)]


def mean_predictor(Dd, task):
    import finetune as F
    mu, _ = F.norm_stats(Dd, task)
    if task == "A":
        y = Dd["Cp"][F.TEST]
        return np.linalg.norm(mu - y, axis=1) / np.linalg.norm(y, axis=1)
    return np.array([np.linalg.norm(mu - Dd["c"][i][~Dd["body"][i]]) / np.linalg.norm(Dd["c"][i][~Dd["body"][i]])
                     for i in F.TEST])


def report():
    import bootstrap as BS
    import finetune as F
    from runner import load_rows
    R = load_rows(FT_OUT)
    Dd = F.load(DOWN)
    sel = lambda v, t, m: [r for r in R if (r["variant"], r["task"], r["n"], r["shift"], r["prompt_mode"], r["freeze"])
                           == (v, t, 64, 0, m, "")]
    out, L = {}, []
    L.append("G1: learnability (n = 64, shift 0) vs the mean predictor\n")
    L.append("| task | mean predictor | S | S vs mean predictor |\n|---|---|---|---|")
    for t in "AB":
        mp = mean_predictor(Dd, t)
        rs = sel("S", t, "normal")
        cmp = BS.paired(rs, [dict(per_shape=list(mp))])
        out[f"G1_{t}"] = dict(mean_predictor=float(mp.mean()), S=cmp["a"], S_vs_mean=cmp)
        L.append(f"| {t} | {mp.mean():.4f} | {cmp['a']:.4f} | {BS.fmt(cmp)} |")
    L.append("\nG2: zero prompt vs aligned prompt (n = 64, 2 fine-tune seeds); positive = zero prompt is worse\n")
    L.append("| task | variant | aligned | zero prompt | zero vs aligned |\n|---|---|---|---|---|")
    for t in "AB":
        for v in ("S", "L", "Lc"):
            a, z = sel(v, t, "normal"), sel(v, t, "zero")
            if a and z:
                c = BS.paired(z, a)
                out[f"G2_{t}_{v}"] = c
                L.append(f"| {t} | {v} | {c['b']:.4f} | {c['a']:.4f} | {BS.fmt(c)} |")
    L.append("\nG3: Lc vs L at shift 0 (negative = Lc better)\n")
    L.append("| task | L | Lc | Lc vs L |\n|---|---|---|---|")
    for t in "AB":
        a, b = sel("Lc", t, "normal"), sel("L", t, "normal")
        if a and b:
            c = BS.paired(a, b)
            out[f"G3_{t}"] = c
            L.append(f"| {t} | {c['b']:.4f} | {c['a']:.4f} | {BS.fmt(c)} |")
    L.append("\nAll variants at shift 0 vs scratch (negative = better than scratch)\n")
    L.append("| task | variant | error | vs S |\n|---|---|---|---|")
    for t in "AB":
        for v in ("L", "Lc"):
            a, b = sel(v, t, "normal"), sel("S", t, "normal")
            if a and b:
                c = BS.paired(a, b)
                out[f"vsS_{t}_{v}"] = c
                L.append(f"| {t} | {v} | {c['a']:.4f} | {BS.fmt(c)} |")
    json.dump(out, open("results/exp2_gates.json", "w"), indent=1)
    text = "\n".join(L)
    open("results/exp2_gates.md", "w").write(text + "\n")
    print(text)


# ----------------------------------------------------------------------------- G1 diagnostic
def diag_job(j):
    import finetune as F
    D = F.get_data(None, DOWN)
    P, loss = F.fit(D, j["variant"], j["task"], 64, j["seed"], ckpt=ckpt_of(j["variant"]), frame="absolute")
    pool = np.random.default_rng(j["seed"])                      # replay fit()'s pool draw
    import model as Mo
    Mo.init_head({}, pool, "head.task", 1)
    ids = pool.choice(F.TRAIN, 64, replace=False)
    mu, sd = F.norm_stats(D, j["task"])
    tr, _, _ = F.evaluate(P, D, j["task"], 0.0, False, mu, sd, ids=ids, frame="absolute")
    te, per, _ = F.evaluate(P, D, j["task"], 0.0, False, mu, sd, frame="absolute")
    return dict(train_rel_l2=tr, test_rel_l2=te, final_train_loss=float(loss.data), per_shape=list(per))


def diag():
    from multiprocessing import Pool as P_
    jobs = [dict(variant=v, task=t, seed=s) for v in ("S", "L", "Lc") for t in "AB" for s in (0, 1)]
    with P_(N_PROCS) as pool:
        res = pool.map(diag_job, jobs)
    out = [{**j, **{k: v for k, v in r.items() if k != "per_shape"}} for j, r in zip(jobs, res)]
    json.dump(out, open("results/exp2_diag.json", "w"), indent=1)
    for r in out:
        print(f"{r['variant']:2s} {r['task']} seed {r['seed']}: train rel-L2 {r['train_rel_l2']:.4f}  "
              f"test {r['test_rel_l2']:.4f}  final train loss (normalised MSE) {r['final_train_loss']:.4f}")


# ----------------------------------------------------------------------------- Stage 2 / 3a
SHIFTS = (0, 20, 45, 90, 180)
SEEDS = (0, 1, 2)


def sweep_jobs():
    J = []
    job = lambda v, t, n, s, sh, m="normal", fr="": dict(variant=v, data="canon30", task=t, n=n, seed=s, shift=sh,
                                                       prompt_mode=m, freeze=fr)
    for t in "BA":                                   # scratch control that uses the prompt (n = 256)
        for s in SEEDS:
            J += [job("S", t, 256, s, sh) for sh in (0, 90, 180)] + [job("S", t, 256, s, 0, "zero")]
    for t in "BA":                                   # zero-prompt reference for every (variant, n, seed)
        for n in (16, 64):
            J += [job(v, t, n, s, 0, "zero") for v in ("S", "L", "Lc") for s in SEEDS]
    for t in "BA":                                   # Stage 2 shift sweep
        for n in (16, 64):
            J += [job(v, t, n, s, sh) for v in ("S", "L", "Lc") for sh in SHIFTS for s in SEEDS]
    for t in "BA":                                   # Stage 3a: prompt-reading layer q.1 frozen
        J += [job(v, t, 64, s, sh, fr="q.1") for v in ("S", "L", "Lc") for sh in (0, 45, 180) for s in SEEDS]
    return J


def step3_jobs():
    """the rest of the Stage 2/3a grid plus L pretraining seed 1, confirmatory n = 64 runs first."""
    job = lambda v, t, n, s, sh, m="normal", fr="": dict(variant=v, data="canon30", task=t, n=n, seed=s, shift=sh,
                                                       prompt_mode=m, freeze=fr)
    ls1 = [job("Ls1", t, 64, s, sh) for t in "BA" for sh in SHIFTS for s in SEEDS] + \
          [job("Ls1", t, 64, s, 0, "zero") for t in "BA" for s in SEEDS]
    J = sweep_jobs()
    prio = lambda j: (0 if j["n"] == 64 and not j["freeze"] else 2 if j["freeze"] else 1, j["task"] != "B")
    J = sorted(J, key=prio)
    n64 = [j for j in J if prio(j)[0] == 0]
    return n64 + ls1 + [j for j in J if prio(j)[0] > 0]


def seed_jobs(seeds=(2, 3, 4)):
    """step 4: L pretraining seeds 2-4 at n = 64, shifts 0/90/180 + zero prompt, ft seeds 0-2."""
    job = lambda v, t, s, sh, m="normal": dict(variant=v, data="canon30", task=t, n=64, seed=s, shift=sh,
                                               prompt_mode=m, freeze="")
    return [job(f"Ls{k}", t, s, sh) for k in seeds for t in "BA" for sh in (0, 90, 180) for s in SEEDS] + \
           [job(f"Ls{k}", t, s, 0, "zero") for k in seeds for t in "BA" for s in SEEDS]


def report2():
    import bootstrap as BS
    from runner import load_rows
    R = load_rows(FT_OUT)
    sel = lambda v, t, n, sh, m="normal", fr="": [r for r in R if (r["variant"], r["task"], r["n"], r["shift"],
                                                                   r["prompt_mode"], r["freeze"]) == (v, t, n, sh, m, fr)]
    per = lambda rows: np.mean([r["per_shape"] for r in rows], 0)
    mean = lambda rows: float(np.mean([r["rel_l2"] for r in rows]))
    out, L = {"penalty": [], "did": [], "benefit_lost": [], "seed_floor": [], "freeze": []}, []
    L.append("Shift penalty Delta_V(shift) = err(shift) - err(0), per body, averaged over fine-tune seeds 0-2; "
             "95% bootstrap CI over the 100 test bodies (covers bodies only, not seeds).\n")
    L.append("| task | n | variant | err(0) | 20 | 45 | 90 | 180 | zero prompt |\n|---|---|---|---|---|---|---|---|---|")
    for t in "AB":
        for n in (16, 64, 256):
            for v in ("S", "L", "Ls1", "Lc"):
                r0 = sel(v, t, n, 0)
                if len(r0) < 3:
                    continue
                cells = []
                for sh in SHIFTS[1:]:
                    rs = sel(v, t, n, sh)
                    if len(rs) < 3:
                        cells.append("-")
                        continue
                    c = BS.paired(rs, r0, mode="diff")
                    out["penalty"].append(dict(task=t, n=n, variant=v, shift=sh, **c))
                    cells.append(f"{c['mean']:+.4f} [{c['lo']:+.4f}, {c['hi']:+.4f}]{'*' if c['sig'] else ''}")
                rz = sel(v, t, n, 0, "zero")
                z = BS.paired(rz, r0, mode="diff") if len(rz) >= 3 else None
                L.append(f"| {t} | {n} | {v} | {mean(r0):.4f} | " + " | ".join(cells) +
                         (f" | {z['mean']:+.4f} [{z['lo']:+.4f}, {z['hi']:+.4f}]{'*' if z['sig'] else ''} |" if z else " | - |"))
    L.append("\nDifference-in-differences Delta_V(shift) - Delta_S(shift) at the same n (95% CI over bodies)\n")
    L.append("| task | n | variant | 20 | 45 | 90 | 180 |\n|---|---|---|---|---|---|---|")
    for t in "AB":
        for n in (16, 64):
            for v in ("L", "Ls1", "Lc"):
                cells = []
                for sh in SHIFTS[1:]:
                    rv0, rvs, rs0, rss = sel(v, t, n, 0), sel(v, t, n, sh), sel("S", t, n, 0), sel("S", t, n, sh)
                    if min(map(len, (rv0, rvs, rs0, rss))) < 3:
                        cells.append("-")
                        continue
                    d = (per(rvs) - per(rv0)) - (per(rss) - per(rs0))
                    m, lo, hi = BS.boot_mean(d)
                    out["did"].append(dict(task=t, n=n, variant=v, shift=sh, mean=m, lo=lo, hi=hi))
                    cells.append(f"{m:+.4f} [{lo:+.4f}, {hi:+.4f}]{'*' if lo > 0 or hi < 0 else ''}")
                L.append(f"| {t} | {n} | {v} | " + " | ".join(cells) + " |")
    L.append("\nFraction of the prompt's benefit lost, (err(shift) - err(0)) / (err(zero) - err(0)), "
             "from seed-averaged mean errors; 95% CI from a bootstrap over bodies of the ratio of means\n")
    L.append("| task | n | variant | 20 | 45 | 90 | 180 |\n|---|---|---|---|---|---|---|")
    rng = np.random.default_rng(0)
    for t in "AB":
        for n in (16, 64):
            for v in ("L", "Ls1", "Lc"):
                r0, rz = sel(v, t, n, 0), sel(v, t, n, 0, "zero")
                if len(r0) < 3 or len(rz) < 3:
                    continue
                e0, ez = per(r0), per(rz)
                cells = []
                for sh in SHIFTS[1:]:
                    rs = sel(v, t, n, sh)
                    if len(rs) < 3:
                        cells.append("-")
                        continue
                    es = per(rs)
                    f = (es.mean() - e0.mean()) / (ez.mean() - e0.mean())
                    bs = []
                    for _ in range(4000):
                        k = rng.integers(0, len(e0), len(e0))
                        bs.append((es[k].mean() - e0[k].mean()) / (ez[k].mean() - e0[k].mean()))
                    lo, hi = np.percentile(bs, [2.5, 97.5])
                    out["benefit_lost"].append(dict(task=t, n=n, variant=v, shift=sh, frac=float(f), lo=float(lo), hi=float(hi)))
                    cells.append(f"{100 * f:+.1f}% [{100 * lo:+.1f}, {100 * hi:+.1f}]")
                L.append(f"| {t} | {n} | {v} | " + " | ".join(cells) + " |")
    L.append("\nSeed-level noise floor at shift 0: largest relative difference in test error between two fine-tune "
             "seeds of the same variant (mean errors)\n")
    L.append("| task | n | variant | errors by seed | max pairwise relative difference |\n|---|---|---|---|---|")
    for t in "AB":
        for n in (16, 64, 256):
            for v in ("S", "L", "Ls1", "Lc"):
                r0 = sorted(sel(v, t, n, 0), key=lambda r: r["seed"])
                if len(r0) < 3:
                    continue
                e = [r["rel_l2"] for r in r0]
                mx = max(abs(a / b - 1) for a in e for b in e)
                out["seed_floor"].append(dict(task=t, n=n, variant=v, errors=e, max_rel_diff=mx))
                L.append(f"| {t} | {n} | {v} | {', '.join(f'{x:.4f}' for x in e)} | {100 * mx:.1f}% |")
    L.append("\nStage 3a: q.1 frozen (n = 64); shift penalty err(shift) - err(0) within the frozen runs, "
             "and frozen vs unfrozen at the same shift\n")
    L.append("| task | variant | err(0), q.1 frozen | penalty 45 | penalty 180 | frozen vs free at 0 | at 180 |\n|---|---|---|---|---|---|---|")
    for t in "AB":
        for v in ("S", "L", "Lc"):
            f0 = sel(v, t, 64, 0, fr="q.1")
            if len(f0) < 3:
                continue
            cells = []
            for sh in (45, 180):
                fs = sel(v, t, 64, sh, fr="q.1")
                if len(fs) < 3:
                    cells.append("-")
                    continue
                c = BS.paired(fs, f0, mode="diff")
                out["freeze"].append(dict(task=t, variant=v, shift=sh, **c))
                cells.append(f"{c['mean']:+.4f} [{c['lo']:+.4f}, {c['hi']:+.4f}]{'*' if c['sig'] else ''}")
            for sh in (0, 180):
                a, b = sel(v, t, 64, sh, fr="q.1"), sel(v, t, 64, sh)
                if len(a) >= 3 and len(b) >= 3:
                    cells.append(BS.fmt(BS.paired(a, b)))
                else:
                    cells.append("-")
            L.append(f"| {t} | {v} | {mean(f0):.4f} | " + " | ".join(cells) + " |")
    out["prereg"], pl = prereg(sel, per)
    L += pl
    json.dump(out, open("results/exp2_stage2.json", "w"), indent=1)
    text = "\n".join(L)
    open("results/exp2_stage2.md", "w").write(text + "\n")
    print(text)
    figure2(R, sel, mean)


def se_seed(sel, v, t, n):
    """SE of a 3-seed difference from the pooled fine-tune-seed SD of the variant's mean errors (all shifts)."""
    res = [np.asarray([r["rel_l2"] for r in sel(v, t, n, sh)]) for sh in SHIFTS]
    res = [x - x.mean() for x in res if len(x) >= 2]
    dof = sum(len(x) - 1 for x in res)
    return float(np.sqrt(sum((x ** 2).sum() for x in res) / dof) * np.sqrt(2 / 3))


def prereg(sel, per):
    """the checks pre-registered in EXP2_RESULTS.md (amendment of 2026-10-14) before the n = 64 runs."""
    import bootstrap as BS
    out, L = {}, ["\nPre-registered checks (amendment of 2026-10-14)\n"]
    L.append("| check | variant | Delta(180) = err(180) - err(0) [95% CI over bodies] | 2 x SE_seed | outcome |\n|---|---|---|---|---|")
    p6 = []
    for v in ("L", "Ls1", "Lc"):
        c = BS.paired(sel(v, "A", 64, 180), sel(v, "A", 64, 0), mode="diff")
        se = se_seed(sel, v, "A", 64)
        ok = abs(c["mean"]) <= 2 * se
        p6.append(ok)
        out[f"P6_{v}"] = dict(c, se_seed=se, holds=ok)
        L.append(f"| P6 task A 180 ~ 0 | {v} | {c['mean']:+.4f} [{c['lo']:+.4f}, {c['hi']:+.4f}] | {2 * se:.4f} | {'holds' if ok else 'FAILS'} |")
    p7 = []
    for v in ("L", "Ls1"):
        c = BS.paired(sel(v, "B", 64, 180), sel(v, "B", 64, 0), mode="diff")
        ok = c["hi"] < 0
        p7.append(ok)
        out[f"P7_{v}"] = dict(c, holds=ok)
        L.append(f"| P7 task B reversed beats aligned | {v} | {c['mean']:+.4f} [{c['lo']:+.4f}, {c['hi']:+.4f}] | {2 * se_seed(sel, v, 'B', 64):.4f} | {'holds' if ok else 'fails'} |")
    out["P6_holds"], out["P7_holds"] = all(p6), all(p7)
    # verdict rule: difference-in-differences vs scratch at the same n
    L.append("\nDifference-in-differences Delta_V(shift) - Delta_S(shift) used by the verdict rule (95% CI over bodies)\n")
    L.append("| task | n | variant | 20 | 45 | 90 | 180 |\n|---|---|---|---|---|---|---|")
    pos, did = {}, {}
    for t in "AB":
        for n in (16, 64):
            for v in ("L", "Ls1", "Lc"):
                cells = []
                for sh in SHIFTS[1:]:
                    rv0, rvs, rs0, rss = sel(v, t, n, 0), sel(v, t, n, sh), sel("S", t, n, 0), sel("S", t, n, sh)
                    if min(map(len, (rv0, rvs, rs0, rss))) < 3:
                        cells.append("-")
                        continue
                    m, lo, hi = BS.boot_mean((per(rvs) - per(rv0)) - (per(rss) - per(rs0)))
                    did[(t, n, v, sh)] = (m, lo, hi)
                    cells.append(f"{m:+.4f} [{lo:+.4f}, {hi:+.4f}]{'*' if lo > 0 or hi < 0 else ''}")
                if any(c != "-" for c in cells):
                    L.append(f"| {t} | {n} | {v} | " + " | ".join(cells) + " |")
    support = any(all(did.get((t, 64, v, sh), (0, 0, 0))[1] > 0 for v in ("L", "Ls1"))
                  for t in "AB" for sh in (20, 45, 90))
    refuted = not any(lo > 0 for (t, n, v, sh), (m, lo, hi) in did.items() if v in ("L", "Lc"))
    out["verdict"] = "support" if support else "refutation" if refuted else "unresolved"
    out["positive_did"] = [dict(task=t, n=n, variant=v, shift=sh, mean=m, lo=lo, hi=hi)
                           for (t, n, v, sh), (m, lo, hi) in did.items() if lo > 0]
    out["negative_did"] = [dict(task=t, n=n, variant=v, shift=sh, mean=m, lo=lo, hi=hi)
                           for (t, n, v, sh), (m, lo, hi) in did.items() if hi < 0]
    L.append(f"\nP6 (task A 180 costs about nothing, L s0, L s1, Lc): {'holds' if out['P6_holds'] else 'FAILS'}. "
             f"P7 (task B reversed beats aligned at n = 64 in both L seeds): {'holds' if out['P7_holds'] else 'fails'}. "
             f"Verdict by the pre-registered rule: {out['verdict']}.")
    return out, L


def figure2(R, sel, mean):
    import matplotlib.pyplot as plt
    import analyze as A
    col = dict(A.COL, Lc="#7a5cd6", Ls1=A.COL["L"])
    mrk = dict(A.MRK, Lc="h", Ls1="^")
    name = dict(A.NAME, Lc="GeoPT-lifted, canonical pretraining", Ls1="GeoPT-lifted, pretraining seed 1")
    fig, axes = plt.subplots(2, 2, figsize=(11, 8.2), sharex=True)
    for c, t in enumerate("AB"):
        for r_, n in enumerate((16, 64)):
            ax = axes[r_, c]
            A.style(ax)
            series = [(v, n, "-") for v in ("S", "L", "Lc")] + ([("Ls1", 64, ":"), ("S", 256, "--")] if n == 64 else [])
            ends = []
            for v, nn, ls in series:
                pts = [(sh, sel(v, t, nn, sh)) for sh in SHIFTS]
                pts = [(sh, np.mean([r["rel_l2"] for r in rs]), min(r["rel_l2"] for r in rs), max(r["rel_l2"] for r in rs))
                       for sh, rs in pts if len(rs) >= 3]
                if not pts:
                    continue
                x, m, lo, hi = map(np.array, zip(*pts))
                lab = name[v] + (f", n = {nn}" if nn != n else "")
                ax.fill_between(x, lo, hi, color=col[v], alpha=0.12, lw=0)
                ax.plot(x, m, ls, color=col[v], marker=mrk[v], ms=7, lw=2, markeredgecolor=A.SURF, label=lab)
                rz = sel(v, t, nn, 0, "zero")
                if len(rz) >= 3:
                    ax.plot([200], [mean(rz)], marker=mrk[v], ms=7, color=col[v], ls="none", alpha=0.6,
                            markeredgecolor=A.SURF)
                ends.append((m[-1], lab))
            ax.set_xticks(list(SHIFTS) + [200])
            ax.set_xticklabels(["0", "20", "45", "90", "180", "no\nprompt"])
            ax.set_ylabel("test relative L2 error")
            ax.set_title(f"{A.TASK[t]}\n{n} labelled bodies (canonical bodies, flow +-30 deg)", loc="left", fontsize=10)
    for ax in axes[1]:
        ax.set_xlabel("prompt rotated away from the true flow direction (degrees)")
    h, l = axes[0, 1].get_legend_handles_labels()
    h2, l2 = axes[1, 1].get_legend_handles_labels()
    for hh, ll in zip(h2, l2):
        if ll not in l:
            h.append(hh)
            l.append(ll)
    fig.legend(h, l, frameon=False, fontsize=8.5, loc="lower center", ncol=4)
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    fig.savefig("figs/exp2_shift_curves.png", dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    import runner
    stage = sys.argv[1] if len(sys.argv) > 1 else "report"
    if stage == "data":
        build_data()
    elif stage == "checks":
        checks()
    elif stage == "pretrain":
        import pretrain as PT
        os.makedirs("ckpt/exp2", exist_ok=True)
        PT.pretrain("L", steps=3000, seed=0, Dd=PT.load(PRE), out=LC)
    elif stage == "gates":
        runner.run(gate_jobs(), finetune_job, FT_OUT, FT_KEY)
    elif stage == "report":
        report()
    elif stage == "diag":
        diag()
    elif stage == "sweep":
        runner.run(sweep_jobs(), finetune_job, FT_OUT, FT_KEY)
    elif stage == "step3":
        runner.run(step3_jobs(), finetune_job, FT_OUT, FT_KEY)
    elif stage == "seeds":
        runner.run(seed_jobs(), finetune_job, FT_OUT, FT_KEY)
    elif stage == "report2":
        report2()
    else:
        sys.exit(__doc__)
