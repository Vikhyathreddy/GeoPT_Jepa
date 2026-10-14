"""Experiment 2, Stage 1: preferred orientation and prompt alignment
(spec: experiments/02_orientation_prompts.md).

usage (from the repo root; N_PROCS sets the worker count):
  python exp2.py data       data/exp2/downstream_canon.npz and data/exp2/pretrain_canon.npz
  python exp2.py checks     default data bit-identical; canonical vs original bodies (area, perimeter)
  python exp2.py pretrain   Lc: L pretrained on pretrain_canon.npz -> ckpt/exp2/Lc_s0.npz
  python exp2.py gates      G1-G3 fine-tuning runs -> results/exp2_finetune.jsonl
  python exp2.py report     Gate tables -> results/exp2_gates.json / .md

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
    else:
        sys.exit(__doc__)
