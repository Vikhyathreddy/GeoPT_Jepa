"""Experiment 1, Stage 1: noisy pretext targets (spec: experiments/01_noisy_targets.md).

usage (from the repo root; N_PROCS sets the worker count):
  python exp1.py corpora     build data/exp1/pretrain_<tag>.npz + results/exp1_corpora.json
  python exp1.py identity    label noise leaves LT/LT0/J training bit-identical (50 steps)
  python exp1.py pretrain    ckpt/exp1/<V>_<tag>_s0.npz, rows in results/exp1_pretrain.jsonl
  python exp1.py finetune    rows in results/exp1_finetune.jsonl
  python exp1.py report      Gate 1 tables (results/exp1_gate1.json) and figs/exp1_noise_curves.png

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
LATENT_ONLY = ("LT", "LT0", "J")       # never read vd[:,1:]
STEPS = 3000
CORPUS = "data/exp1/pretrain_{}.npz"
CKPT = "ckpt/exp1/{}_{}_s{}.npz"
PRE_OUT = "results/exp1_pretrain.jsonl"
FT_OUT = "results/exp1_finetune.jsonl"
FT_KEY = ("variant", "tag", "pretrain_seed", "task", "n", "ft_seed")


def corpus_path(tag):
    return "data/pretrain.npz" if tag == "clean" else CORPUS.format(tag)


def ckpt_path(v, tag, seed=0):
    """the clean seed-0 models are the original ckpt/L.npz, ckpt/LT.npz; under label noise the
    latent-only variants are bit-identical to clean (see `identity`), so they reuse it too."""
    if seed == 0 and (tag == "clean" or (STAGE1.get(tag, ("",))[0] == "label" and v in LATENT_ONLY)):
        return f"ckpt/{v}.npz"
    return CKPT.format(v, tag, seed)


# ----------------------------------------------------------------------------- corpora
def _recompute(args):
    import world as W
    V, X = args
    return np.stack([W.vector_distance(V.astype(np.float64), X[t].astype(np.float64)) for t in range(len(X))]
                    ).astype(np.float32)


def build_corpora():
    import pretrain as PT
    os.makedirs("data/exp1", exist_ok=True)
    D = PT.load("data/pretrain.npz")
    E_lbl = np.random.default_rng(123).standard_normal(D["vd"][:, 1:].shape).astype(np.float32)
    E_pos = np.random.default_rng(124).standard_normal(D["X"][:, 1:].shape).astype(np.float32)
    meta = {}
    for tag, (fam, lvl) in STAGE1.items():
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
    json.dump(meta, open("results/exp1_corpora.json", "w"), indent=1)


# ----------------------------------------------------------------------------- identity check
def identity(steps=50):
    import pretrain as PT
    D = PT.load("data/pretrain.npz")
    out = {}
    for tag in [t for t, (f, _) in STAGE1.items() if f == "label"]:
        N = PT.load(CORPUS.format(tag))
        for v in LATENT_ONLY:
            Pc, _ = PT.pretrain(v, steps=steps, Dd=D, log_every=10 ** 9)
            Pn, _ = PT.pretrain(v, steps=steps, Dd=N, log_every=10 ** 9)
            same = all(np.array_equal(Pc[k].data, Pn[k].data) for k in Pc)
            out[f"{v}_{tag}"] = same
            print(v, tag, "identical after", steps, "steps:", same, flush=True)
    json.dump(dict(steps=steps, identical=out), open("results/exp1_identity.json", "w"), indent=1)
    if not all(out.values()):
        sys.exit("identity check FAILED: latent-only variants differ under label noise")


# ----------------------------------------------------------------------------- pretraining
def pretrain_job(j):
    import pretrain as PT
    out = ckpt_path(j["variant"], j["tag"], j["pretrain_seed"])
    os.makedirs(os.path.dirname(out), exist_ok=True)
    _, hist = PT.pretrain(j["variant"], steps=STEPS, seed=j["pretrain_seed"], Dd=PT.load(corpus_path(j["tag"])),
                          out=out)
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
        if r["n"] != 64 or r["shift"] != 0 or r["frozen"] or r["variant"] not in ("S", "N", "L", "LT"):
            continue
        base = dict(r, ft_seed=r["seed"], pretrain_seed=0, source="results/finetune.jsonl")
        R.append(dict(base, tag="clean"))
        if r["variant"] == "LT":
            for tag, (fam, _) in STAGE1.items():
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


def figure(R, meta, refs):
    import matplotlib.pyplot as plt
    import analyze as A
    fams = ["label", "trajectory"]
    fig, axes = plt.subplots(2, 2, figsize=(10.5, 8.0))
    for c, t in enumerate("AB"):
        for r_, fam in enumerate(fams):
            ax = axes[r_, c]
            A.style(ax)
            tags = ["clean"] + [g for g, (f, _) in STAGE1.items() if f == fam]
            xs = [0.0] + [meta[g]["target_noise"] for g in tags[1:]]
            for v, ls in (("S", ":"), ("N", "--")):
                ax.axhline(refs[t][v], color=A.COL[v], ls=ls, lw=1.3, label=f"{A.NAME[v]} (noise-free reference)")
            ends = []
            for v in ("L", "LT"):
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
    h, l = axes[1, 0].get_legend_handles_labels()
    fig.legend(h, l, frameon=False, fontsize=8.5, loc="lower center", ncol=4)
    fig.tight_layout(rect=(0, 0.05, 1, 1))
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
    else:
        sys.exit(__doc__)
