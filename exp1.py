"""Experiment 1, Stage 1: noisy pretext targets (spec: experiments/01_noisy_targets.md).

usage (from the repo root; N_PROCS sets the worker count):
  python exp1.py corpora     build data/exp1/pretrain_<tag>.npz + results/exp1_corpora.json
  python exp1.py identity    label noise leaves LT/LT0/J training bit-identical (50 steps)
  python exp1.py pretrain    ckpt/exp1/<V>_<tag>_s0.npz, rows in results/exp1_pretrain.jsonl
  python exp1.py finetune    rows in results/exp1_finetune.jsonl
  python exp1.py report      Gate 1 tables (results/exp1_gate1.json) and figs/exp1_noise_curves.png
  Stage 2 (user decision of 2026-10-14: option d, LJ at two label levels, a targeted second seed):
  python exp1.py corpora2    data/exp1/pretrain_lblfrz4.npz, _lblfrz8.npz (lblfrs4 = fresh noise per batch)
  python exp1.py identity2   label noise at sigma 4, 8 and fresh noise leaves LT/LT0/J bit-identical
  python exp1.py pretrain2   L on lblfrz4/8/frs4, LJ on lblfrz2/4, seed 1 of L (clean, lblfrz2), N and LT (clean)
  python exp1.py finetune2   n=64, A/B, ft seeds 0,1 for every Stage 2 checkpoint
  python exp1.py diag        no training: L-vs-N table; L/LJ pretext error split by wall-hit/free, near/far
  python exp1.py report2     Stage 2 tables (results/exp1_stage2.json/.md) and figs/exp1_noise_curves.png

Choices the spec leaves open (also recorded in EXP1_RESULTS.md):
  * Noise is applied to every sample of the corpus, validation bodies included; the clean targets
    are stored (vd_clean, X_clean) and pretrain() logs the explicit heads against them (valclean_*).
  * Label noise uses the fixed scale 0.483 (std of clean vd[:,1:]) and one N(0,1) draw (seed 123)
    shared by all label-noise levels; trajectory noise uses one N(0,1) draw (seed 124) shared by
    all position-noise levels.
  * Measured target-space noise = std(vd - vd_clean) over t = 1,2 / std(vd_clean[:,1:]).
"""
from procs import N_PROCS  # single-threaded BLAS workers; must precede numpy
import os
import sys
import json
import numpy as np
from multiprocessing import Pool

CLEAN_STD = 0.483
STAGE1 = {  # tag -> (family, level)
    "lblfrz05": ("label", 0.5), "lblfrz2": ("label", 2.0),
    "pos015": ("trajectory", 0.15), "pos030": ("trajectory", 0.30),
}
STAGE2 = {"lblfrz4": ("label", 4.0), "lblfrz8": ("label", 8.0), "lblfrs4": ("label", 4.0)}
FRESH = {"lblfrs4"}                     # noise redrawn every time a sample is drawn (batch_fn)
CONDS = {**STAGE1, **STAGE2}
LATENT_ONLY = ("LT", "LT0", "J")       # never read vd[:,1:]
STEPS = 3000
CORPUS = "data/exp1/pretrain_{}.npz"
CKPT = "ckpt/exp1/{}_{}_s{}.npz"
PRE_OUT = "results/exp1_pretrain.jsonl"
FT_OUT = "results/exp1_finetune.jsonl"
FT_KEY = ("variant", "tag", "pretrain_seed", "task", "n", "ft_seed")


def corpus_path(tag):
    return "data/pretrain.npz" if tag == "clean" or tag in FRESH else CORPUS.format(tag)


def fresh_noise(level):
    """batch_fn: vd[:,1:] += level * 0.483 * N(0,1), redrawn for every batch (own generator)."""
    def f(batch, rng):
        vd = batch["vd"].copy()
        vd[:, 1:] += (level * CLEAN_STD * rng.standard_normal(vd[:, 1:].shape)).astype(np.float32)
        return dict(batch, vd=vd)
    return f


def ckpt_path(v, tag, seed=0):
    """the clean seed-0 models are the original ckpt/L.npz, ckpt/LT.npz; under label noise the
    latent-only variants are bit-identical to clean (see `identity`), so they reuse it too."""
    if seed == 0 and (tag == "clean" or (CONDS.get(tag, ("",))[0] == "label" and v in LATENT_ONLY)):
        return f"ckpt/{v}.npz"
    return CKPT.format(v, tag, seed)


# ----------------------------------------------------------------------------- corpora
def _recompute(args):
    import world as W
    V, X = args
    return np.stack([W.vector_distance(V.astype(np.float64), X[t].astype(np.float64)) for t in range(len(X))]
                    ).astype(np.float32)


def build_corpora(conds=STAGE1, meta_path="results/exp1_corpora.json"):
    import pretrain as PT
    os.makedirs("data/exp1", exist_ok=True)
    D = PT.load("data/pretrain.npz")
    E_lbl = np.random.default_rng(123).standard_normal(D["vd"][:, 1:].shape).astype(np.float32)
    E_pos = np.random.default_rng(124).standard_normal(D["X"][:, 1:].shape).astype(np.float32)
    meta = {}
    for tag, (fam, lvl) in conds.items():
        if tag in FRESH:
            continue
        C = dict(D, vd_clean=D["vd"])
        if fam == "label":
            vd = D["vd"].copy()
            vd[:, 1:] += (lvl * CLEAN_STD) * E_lbl
            C["vd"] = vd
        else:
            X = D["X"].copy()
            X[:, 1:] += lvl * E_pos
            with Pool(N_PROCS) as pool:
                vd1 = pool.map(_recompute, [(D["shapes"][D["sid"][i]], X[i, 1:]) for i in range(len(X))],
                               chunksize=256)
            vd = D["vd"].copy()
            vd[:, 1:] = np.array(vd1)
            C.update(X=X, X_clean=D["X"], vd=vd)
        dv = C["vd"][:, 1:] - D["vd"][:, 1:]
        changed = np.abs(dv).max(-1) > 1e-5                       # per particle and step
        meta[tag] = dict(family=fam, level=lvl,
                         target_noise=float(dv.std() / D["vd"][:, 1:].std()),
                         target_noise_t1=float(dv[:, 0].std() / D["vd"][:, 1].std()),
                         target_noise_t2=float(dv[:, 1].std() / D["vd"][:, 2].std()),
                         frac_targets_changed=float(changed.mean()))
        np.savez(CORPUS.format(tag), **C)
        print(tag, meta[tag], flush=True)
    os.makedirs("results", exist_ok=True)
    if os.path.exists(meta_path):
        meta = {**json.load(open(meta_path)), **meta}
    json.dump(meta, open(meta_path, "w"), indent=1)


# ----------------------------------------------------------------------------- identity check
def identity(steps=50, conds=STAGE1, path="results/exp1_identity.json"):
    import pretrain as PT
    D = PT.load("data/pretrain.npz")
    out = {}
    for tag in [t for t, (f, _) in conds.items() if f == "label"]:
        fresh = tag in FRESH
        N = D if fresh else PT.load(CORPUS.format(tag))
        for v in LATENT_ONLY:
            Pc, _ = PT.pretrain(v, steps=steps, Dd=D, log_every=10 ** 9)
            Pn, _ = PT.pretrain(v, steps=steps, Dd=N, log_every=10 ** 9,
                                batch_fn=fresh_noise(conds[tag][1]) if fresh else None)
            same = all(np.array_equal(Pc[k].data, Pn[k].data) for k in Pc)
            out[f"{v}_{tag}"] = same
            print(v, tag, "identical after", steps, "steps:", same, flush=True)
    json.dump(dict(steps=steps, identical=out), open(path, "w"), indent=1)
    if not all(out.values()):
        sys.exit("identity check FAILED: latent-only variants differ under label noise")


# ----------------------------------------------------------------------------- pretraining
def pretrain_job(j):
    import pretrain as PT
    out = ckpt_path(j["variant"], j["tag"], j["pretrain_seed"])
    os.makedirs(os.path.dirname(out), exist_ok=True)
    Dd, fn = PT.load(corpus_path(j["tag"])), None
    if j["tag"] in FRESH:
        Dd, fn = dict(Dd, vd_clean=Dd["vd"]), fresh_noise(CONDS[j["tag"]][1])
    _, hist = PT.pretrain(j["variant"], steps=STEPS, seed=j["pretrain_seed"], Dd=Dd, out=out, batch_fn=fn)
    return {"ckpt": out, "final": hist[-1], "history": hist}


def pretrain_jobs():
    J = []
    for tag, (fam, _) in STAGE1.items():
        J.append(dict(variant="L", tag=tag, pretrain_seed=0))
        if fam != "label":
            J.append(dict(variant="LT", tag=tag, pretrain_seed=0))
    return sorted(J, key=lambda j: j["variant"] != "LT")       # the slow LT runs first


# ----------------------------------------------------------------------------- fine-tuning
def finetune_job(j):
    import finetune as F
    rec, per, _ = F.finetune(None, j["variant"], j["task"], j["n"], j["ft_seed"],
                             ckpt=ckpt_path(j["variant"], j["tag"], j["pretrain_seed"]))
    return {**rec, "per_shape": [round(float(x), 5) for x in per]}


def pretrain2_jobs():
    J = [dict(variant="LT", tag="clean", pretrain_seed=1)]                       # slowest first
    J += [dict(variant=v, tag=g, pretrain_seed=0) for v, g in
          (("LJ", "lblfrz4"), ("LJ", "lblfrz2"), ("L", "lblfrz4"), ("L", "lblfrz8"), ("L", "lblfrs4"))]
    J += [dict(variant=v, tag=g, pretrain_seed=1) for v, g in (("L", "clean"), ("L", "lblfrz2"), ("N", "clean"))]
    return J


def finetune2_jobs():
    return [dict(j, task=t, n=64, ft_seed=s) for t in "BA" for j in pretrain2_jobs() for s in (0, 1)]


def finetune_jobs():
    return [dict(variant=v, tag=tag, pretrain_seed=0, task=t, n=64, ft_seed=s)
            for t in "BA" for tag, (fam, _) in STAGE1.items()
            for v in (["L"] if fam == "label" else ["LT", "L"]) for s in (0, 1)]


def rows():
    """exp1 rows + references from results/finetune.jsonl: the clean L/LT/S/N (the original
    seed-0 checkpoints, reproduced exactly on this machine) and LT under label noise (identical
    to clean LT by the identity check)."""
    from runner import load_rows
    R = load_rows(FT_OUT)
    for r in load_rows("results/finetune.jsonl"):
        if r["n"] != 64 or r["shift"] != 0 or r["frozen"] or r["variant"] not in ("S", "N", "L", "LT", "LJ"):
            continue
        base = dict(r, ft_seed=r["seed"], pretrain_seed=0, source="results/finetune.jsonl")
        R.append(dict(base, tag="clean"))
        if r["variant"] == "LT":
            for tag, (fam, _) in CONDS.items():
                if fam == "label":
                    R.append(dict(base, tag=tag, source="results/finetune.jsonl (identical clean LT)"))
    return R


# ----------------------------------------------------------------------------- Gate 1 report
def report():
    import bootstrap as BS
    from runner import load_rows
    R = rows()
    meta = json.load(open("results/exp1_corpora.json"))
    pre = {(r["variant"], r["tag"]): r for r in load_rows(PRE_OUT)}
    clean_hist = {v: json.load(open(f"ckpt/{v}.json"))[-1] for v in ("L", "LT")}
    sel = lambda v, tag, t: [r for r in R if (r["variant"], r["tag"], r["task"], r["n"]) == (v, tag, t, 64)]
    tags = ["clean"] + list(STAGE1)
    out = dict(conditions={}, tables=[])
    lines = ["| task | condition | target noise | L | LT | LT vs L | L vs clean L | LT vs clean LT |",
             "|---|---|---|---|---|---|---|---|"]
    for t in "AB":
        for tag in tags:
            rl, rt = sel("L", tag, t), sel("LT", tag, t)
            row = dict(task=t, tag=tag, target_noise=0.0 if tag == "clean" else meta[tag]["target_noise"])
            if rl:
                row["L"] = float(np.mean([r["rel_l2"] for r in rl]))
            if rt:
                row["LT"] = float(np.mean([r["rel_l2"] for r in rt]))
            if rl and rt:
                row["LT_vs_L"] = BS.paired(rt, rl)
            if tag != "clean":
                if rl:
                    row["L_vs_clean"] = BS.paired(rl, sel("L", "clean", t))
                if rt:
                    row["LT_vs_clean"] = BS.paired(rt, sel("LT", "clean", t))
            out["tables"].append(row)
            f = lambda k: BS.fmt(row[k]) if k in row else "-"
            lines.append(f"| {t} | {tag} | {row['target_noise']:.2f} | {row.get('L', float('nan')):.4f} | "
                         f"{row.get('LT', float('nan')):.4f} | {f('LT_vs_L')} | {f('L_vs_clean')} | {f('LT_vs_clean')} |")
    refs = {t: {v: float(np.mean([r["rel_l2"] for r in sel(v, "clean", t)])) for v in ("S", "N")} for t in "AB"}
    out["references"] = refs
    plines = ["| condition | family | measured target noise (t=1 / t=2 / pooled) | targets changed | "
              "L val (noisy targets) | L val vs clean targets | LT val_tgt_std |", "|---|---|---|---|---|---|---|"]
    for tag in tags:
        if tag == "clean":
            hL = clean_hist["L"]
            plines.append(f"| clean | - | 0 | 0 | {hL['val_lift']:.5f} | {hL['val_lift']:.5f} | "
                          f"{clean_hist['LT']['val_tgt_std']:.4f} |")
            continue
        m, pl, pt = meta[tag], pre.get(("L", tag)), pre.get(("LT", tag))
        out["conditions"][tag] = dict(meta=m, L_final=pl and pl["final"], LT_final=pt and pt["final"])
        lt_std = pt["final"]["val_tgt_std"] if pt else clean_hist["LT"]["val_tgt_std"]
        plines.append(f"| {tag} | {m['family']} | {m['target_noise_t1']:.3f} / {m['target_noise_t2']:.3f} / "
                      f"{m['target_noise']:.3f} | {100 * m['frac_targets_changed']:.1f}% | "
                      f"{pl['final']['val_lift'] if pl else float('nan'):.5f} | "
                      f"{pl['final'].get('valclean_lift', float('nan')) if pl else float('nan'):.5f} | "
                      f"{lt_std:.4f}{'' if pt else ' (clean LT)'} |")
    json.dump(out, open("results/exp1_gate1.json", "w"), indent=1)
    text = "\n".join(lines) + "\n\nscratch / native-geometry references (n=64): " + json.dumps(refs) + \
        "\n\n" + "\n".join(plines)
    open("results/exp1_gate1.md", "w").write(text + "\n")
    print(text)
    figure(R, meta, refs)


# ----------------------------------------------------------------------------- Stage 2: diagnostics (no training)
NEAR = 0.15                              # |vd(x0)| below this = near-body particle


def _share_retained(eL, eN, eLc, rng, n_boot=4000):
    """(mean N - mean L) / (mean N - mean L_clean) and a bootstrap-over-bodies 95% CI."""
    f = (eN.mean() - eL.mean()) / (eN.mean() - eLc.mean())
    bs = []
    for _ in range(n_boot):
        k = rng.integers(0, len(eL), len(eL))
        bs.append((eN[k].mean() - eL[k].mean()) / (eN[k].mean() - eLc[k].mean()))
    lo, hi = np.percentile(bs, [2.5, 97.5])
    return float(f), float(lo), float(hi)


def l_vs_n(R):
    """every L/LJ condition against native geometry N of the same pretraining seed (n = 64, A and B)."""
    import bootstrap as BS
    rng = np.random.default_rng(0)
    sel = lambda v, tag, t, ps: [r for r in R if (r["variant"], r["tag"], r["task"], r["n"], r["pretrain_seed"])
                                 == (v, tag, t, 64, ps)]
    per = lambda rows: np.mean([r["per_shape"] for r in rows], 0)
    out, lines = [], ["| task | variant | condition | pretrain seed | error | N | vs N | share of clean advantage over N retained |",
                      "|---|---|---|---|---|---|---|---|"]
    order = ["clean", "lblfrz05", "lblfrz2", "lblfrz4", "lblfrz8", "lblfrs4", "pos015", "pos030"]
    for t in "AB":
        for v in ("L", "LJ"):
            for ps in (0, 1):
                rN, rc = sel("N", "clean", t, ps), sel(v, "clean", t, ps)
                if not rN or not rc:
                    continue
                for tag in order:
                    rl = sel(v, tag, t, ps)
                    if not rl:
                        continue
                    c = BS.paired(rl, rN)
                    f, lo, hi = _share_retained(per(rl), per(rN), per(rc), rng)
                    out.append(dict(task=t, variant=v, tag=tag, pretrain_seed=ps, err=c["a"], N=c["b"], vs_N=c,
                                    retained=f, retained_lo=lo, retained_hi=hi))
                    lines.append(f"| {t} | {v} | {tag} | s{ps} | {c['a']:.4f} | {c['b']:.4f} | {BS.fmt(c)} | "
                                 f"{100 * f:.0f}% [{100 * lo:.0f}, {100 * hi:.0f}] |")
    return out, lines


def _dyn_share(args):
    import world as W
    V, X, v, vd = args
    V = V.astype(np.float64)
    free = np.stack([W.vector_distance(V, (X[0] + t * v).astype(np.float64)) for t in (1, 2)])
    return ((vd[1:] - free) ** 2).sum(), np.abs(X[2] - (X[0] + 2 * v)).max(-1) > 1e-4


def pretext_split(ckpts):
    """explicit-head error against the clean targets on the 800 validation samples, split into
    particles that hit the wall within two steps vs free ones, and near-body vs far."""
    import ag
    import model as Mo
    import pretrain as PT
    D = PT.load("data/pretrain.npz")
    va = np.flatnonzero(D["sid"] >= PT.N_TRAIN_SHAPES)
    X, v, vd = D["X"][va], D["v"][va], D["vd"][va]
    hit = np.linalg.norm(X[:, 2] - (X[:, 0] + 2 * v), axis=-1) > 1e-4             # (800, 192)
    near = np.linalg.norm(vd[:, 0], axis=-1) < NEAR
    tgt = np.concatenate([vd[:, 0], vd[:, 1], vd[:, 2]], -1)                        # (800, 192, 6)
    # how much of the t = 1,2 target variance comes from sticking (vs straight-line free flight)?
    with Pool(N_PROCS) as pool:
        res = pool.map(_dyn_share, [(D["shapes"][D["sid"][i]], D["X"][i], D["v"][i], D["vd"][i]) for i in va])
    sig = sum(r[0] for r in res)
    tot = ((vd[:, 1:] - vd[:, 1:].mean((0, 1, 2))) ** 2).sum()
    stats = dict(n_samples=len(va), hit_fraction=float(hit.mean()), near_fraction=float(near.mean()),
                 hit_fraction_near=float(hit[near].mean()), sticking_signal_share_of_t12_variance=float(sig / tot))
    rows = []
    for name, path, head in ckpts:
        ck = np.load(path)
        P = {k: ag.const(ck[k]) for k in ck.files}
        out = []
        with ag.no_grad():
            for s0 in range(0, len(va), 32):
                b = PT.make_batch(D, va[s0:s0 + 32], None, phase=0.0)
                geo, _ = Mo.encode(P, b["pts"], b["nrm"])
                out.append(Mo.lin(P, Mo.decode(P, geo, b["X"][:, 0], b["v"]), head).data)
        se = (np.concatenate(out) - tgt) ** 2                                       # (800, 192, 6)
        r = dict(name=name, ckpt=path)
        for ch_name, ch in (("all", slice(0, 6)), ("t12", slice(2, 6)), ("t0", slice(0, 2))):
            e = se[..., ch].mean(-1)
            r[ch_name] = dict(mse=float(e.mean()), hit=float(e[hit].mean()), free=float(e[~hit].mean()),
                              near=float(e[near].mean()), far=float(e[~near].mean()),
                              hit_share_of_error=float(e[hit].sum() / e.sum()))
        rows.append(r)
        print(name, {k: round(v, 6) for k, v in r["all"].items()}, flush=True)
    return stats, rows


def diag():
    from runner import load_rows
    R = rows()
    out, lines = l_vs_n(R)
    ckpts = [("L clean s0", "ckpt/L.npz", "head.lift"), ("LJ clean s0", "ckpt/LJ.npz", "head.lift")]
    for r in sorted(load_rows(PRE_OUT), key=lambda r: (r["variant"], r["pretrain_seed"], r["tag"])):
        if r["variant"] in ("L", "LJ"):
            ckpts.append((f"{r['variant']} {r['tag']} s{r['pretrain_seed']}", r["ckpt"], "head.lift"))
    stats, split = pretext_split(ckpts)
    json.dump(dict(l_vs_n=out, pretext_stats=stats, pretext_split=split), open("results/exp1_stage2_diag.json", "w"), indent=1)
    base = {r["name"].split(" ")[0]: r for r in split if r["name"].endswith("clean s0")}
    pl = [f"Validation samples: {stats['n_samples']}; wall-hitting particles {100 * stats['hit_fraction']:.1f}%, "
          f"near-body (|vd0| < {NEAR}) {100 * stats['near_fraction']:.1f}%; sticking signal = "
          f"{100 * stats['sticking_signal_share_of_t12_variance']:.2f}% of the t = 1,2 target variance.\n",
          "| checkpoint | MSE (all 6 outputs) | hit | free | hit share of error | near | far | "
          "hit / clean hit | free / clean free |", "|---|---|---|---|---|---|---|---|---|"]
    for r in split:
        a, c = r["all"], base[r["name"].split(" ")[0]]["all"]
        pl.append(f"| {r['name']} | {a['mse']:.5f} | {a['hit']:.5f} | {a['free']:.5f} | {100 * a['hit_share_of_error']:.0f}% | "
                  f"{a['near']:.5f} | {a['far']:.5f} | {a['hit'] / c['hit']:.2f} | {a['free'] / c['free']:.2f} |")
    text = "L and LJ against native geometry N (same pretraining seed; n = 64; mean over fine-tune seeds 0, 1; " \
           "95% CI over bodies)\n\n" + "\n".join(lines) + \
           "\n\nPretext error against clean targets, split by particle type\n\n" + "\n".join(pl)
    open("results/exp1_stage2_diag.md", "w").write(text + "\n")
    print(text)


def report2():
    import bootstrap as BS
    R = rows()
    sel = lambda v, tag, t, ps=0: [r for r in R if (r["variant"], r["tag"], r["task"], r["n"], r["pretrain_seed"])
                                   == (v, tag, t, 64, ps)]
    out, L = {}, []
    pairs = [("label sweep: L vs clean L", [(("L", g, 0), ("L", "clean", 0)) for g in ("lblfrz05", "lblfrz2", "lblfrz4", "lblfrz8", "lblfrs4")]),
             ("fresh vs frozen noise at sigma = 4 (L)", [(("L", "lblfrs4", 0), ("L", "lblfrz4", 0))]),
             ("LT (clean, reused) vs L under label noise; positive = L better",
              [(("LT", g, 0), ("L", g, 0)) for g in ("clean", "lblfrz2", "lblfrz4", "lblfrz8", "lblfrs4")]),
             ("LJ vs its clean model", [(("LJ", g, 0), ("LJ", "clean", 0)) for g in ("lblfrz2", "lblfrz4")]),
             ("LJ vs L at the same condition; negative = LJ better", [(("LJ", g, 0), ("L", g, 0)) for g in ("clean", "lblfrz2", "lblfrz4")]),
             ("second pretraining seed", [(("L", "clean", 1), ("L", "clean", 0)), (("L", "lblfrz2", 1), ("L", "clean", 1)),
                                          (("L", "lblfrz2", 0), ("L", "clean", 0)), (("N", "clean", 1), ("N", "clean", 0)),
                                          (("LT", "clean", 1), ("LT", "clean", 0)), (("LT", "clean", 1), ("N", "clean", 1)),
                                          (("LT", "clean", 0), ("N", "clean", 0)), (("L", "clean", 1), ("N", "clean", 1))])]
    for title, items in pairs:
        L.append(f"\n{title}\n\n| task | a | b | error a | error b | a vs b |\n|---|---|---|---|---|---|")
        for t in "AB":
            for a, b in items:
                ra, rb = sel(a[0], a[1], t, a[2]), sel(b[0], b[1], t, b[2])
                if not ra or not rb:
                    continue
                c = BS.paired(ra, rb)
                key = f"{t}:{a[0]}_{a[1]}_s{a[2]}-vs-{b[0]}_{b[1]}_s{b[2]}"
                out[key] = c
                L.append(f"| {t} | {a[0]} {a[1]} s{a[2]} | {b[0]} {b[1]} s{b[2]} | {c['a']:.4f} | {c['b']:.4f} | {BS.fmt(c)} |")
    json.dump(out, open("results/exp1_stage2.json", "w"), indent=1)
    text = "\n".join(L)
    open("results/exp1_stage2.md", "w").write(text + "\n")
    print(text)
    meta = json.load(open("results/exp1_corpora.json"))
    refs = {t: {v: float(np.mean([r["rel_l2"] for r in sel(v, "clean", t)])) for v in ("S", "N")} for t in "AB"}
    figure(R, meta, refs)


def figure(R, meta, refs):
    import matplotlib.pyplot as plt
    import analyze as A
    fams = ["label", "trajectory"]
    fig, axes = plt.subplots(2, 2, figsize=(10.5, 8.0))
    for c, t in enumerate("AB"):
        for r_, fam in enumerate(fams):
            ax = axes[r_, c]
            A.style(ax)
            tags = ["clean"] + sorted([g for g, (f, _) in CONDS.items() if f == fam and g not in FRESH and g in meta],
                                      key=lambda g: meta[g]["target_noise"])
            xs = [0.0] + [meta[g]["target_noise"] for g in tags[1:]]
            for v, ls in (("S", ":"), ("N", "--")):
                ax.axhline(refs[t][v], color=A.COL[v], ls=ls, lw=1.3, label=f"{A.NAME[v]} (noise-free reference)")
            ends = []
            for v in ("L", "LT", "LJ"):
                pts = []
                for x, g in zip(xs, tags):
                    e = [r["rel_l2"] for r in R if (r["variant"], r["tag"], r["task"], r["n"]) == (v, g, t, 64)]
                    if e:
                        pts.append((x, np.mean(e), np.min(e), np.max(e)))
                if pts:
                    x_, m_, lo_, hi_ = map(np.array, zip(*pts))
                    ax.fill_between(x_, lo_, hi_, color=A.COL[v], alpha=0.12, lw=0)
                    ax.plot(x_, m_, "-", color=A.COL[v], marker=A.MRK[v], ms=8, lw=2, markeredgecolor=A.SURF,
                            label=A.NAME[v])
                    ends.append((m_[-1], x_[-1], v))
                for g in [g for g in FRESH if CONDS[g][0] == fam]:              # fresh noise: open marker
                    e = [r["rel_l2"] for r in R if (r["variant"], r["tag"], r["task"], r["n"]) == (v, g, t, 64)]
                    if e and v != "LT":
                        ax.plot([CONDS[g][1]], [np.mean(e)], marker=A.MRK[v], ms=9, ls="none", markerfacecolor="none",
                                markeredgecolor=A.COL[v], mew=1.6, label=f"{A.NAME[v]}, fresh noise each batch")
            lo_y, hi_y = ax.get_ylim()
            gap, last = (hi_y - lo_y) * 0.06, None
            for y, x, v in sorted(ends):                       # direct labels, pushed apart vertically
                yy = y if last is None or y - last > gap else last + gap
                last = yy
                ax.annotate(A.NAME[v], (x, y), xytext=(x + 0.04 * max(xs), yy), fontsize=8.5, color=A.INK2,
                            va="center", arrowprops=dict(arrowstyle="-", color=A.GRID, lw=0.8),
                            bbox=dict(boxstyle="round,pad=0.15", fc=A.SURF, ec="none"))
            ax.set_xlim(-0.05 * max(xs), max(xs) * 1.55)
            ax.set_xlabel("measured target-space noise / clean target std")
            ax.set_ylabel("test relative L2 error, 64 labels")
            note = " (LT never reads these targets: clean LT reused)" if fam == "label" else ""
            ax.set_title(f"{A.TASK[t]}\n{fam} noise{note}", loc="left", fontsize=10)
    h, l = [], []
    for ax in (axes[0, 0], axes[1, 0]):
        for hh, ll in zip(*ax.get_legend_handles_labels()):
            if ll not in l:
                h.append(hh)
                l.append(ll)
    fig.legend(h, l, frameon=False, fontsize=8.5, loc="lower center", ncol=4)
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    fig.savefig("figs/exp1_noise_curves.png", dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    import runner
    stage = sys.argv[1] if len(sys.argv) > 1 else "report"
    if stage == "corpora":
        build_corpora()
    elif stage == "identity":
        identity()
    elif stage == "pretrain":
        runner.run(pretrain_jobs(), pretrain_job, PRE_OUT, ("variant", "tag", "pretrain_seed"), require_per=False)
    elif stage == "finetune":
        runner.run(finetune_jobs(), finetune_job, FT_OUT, FT_KEY)
    elif stage == "report":
        report()
    elif stage == "corpora2":
        build_corpora(STAGE2)
    elif stage == "identity2":
        identity(conds=STAGE2, path="results/exp1_identity2.json")
    elif stage == "pretrain2":
        runner.run(pretrain2_jobs(), pretrain_job, PRE_OUT, ("variant", "tag", "pretrain_seed"), require_per=False)
    elif stage == "finetune2":
        runner.run(finetune2_jobs(), finetune_job, FT_OUT, FT_KEY)
    elif stage == "diag":
        diag()
    elif stage == "report2":
        report2()
    else:
        sys.exit(__doc__)
