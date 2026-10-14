"""Turn results/finetune.jsonl + pretraining logs into figures and a summary table."""
import json
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SURF, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
COL = {"S": "#8a8984", "N": "#2a78d6", "J": "#eb6834", "L": "#1baf7a", "LJ": "#eda100",
       "LT": "#e87ba4", "LT0": "#008300"}
MRK = {"S": "o", "N": "s", "J": "D", "L": "^", "LJ": "v", "LT": "P", "LT0": "X"}
NAME = {"S": "Scratch", "N": "Native geometry", "J": "Geo-JEPA", "L": "GeoPT-lifted",
        "LJ": "Lifted + JEPA (joint)", "LT": "Latent-transport JEPA", "LT0": "LT, no anchor"}
TASK = {"A": "Task A - surface speed (potential flow)", "B": "Task B - plume (advection-diffusion)"}

plt.rcParams.update({"font.size": 10, "axes.edgecolor": INK2, "axes.labelcolor": INK2,
                     "xtick.color": INK2, "ytick.color": INK2, "axes.titlecolor": INK,
                     "figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF,
                     "axes.spines.top": False, "axes.spines.right": False})


def load(path="results/finetune.jsonl"):
    return [json.loads(l) for l in open(path)]


def agg(R, **f):
    xs = [r["rel_l2"] for r in R if all(r[k] == v for k, v in f.items())]
    return (np.mean(xs), np.min(xs), np.max(xs), len(xs)) if xs else (np.nan,) * 3 + (0,)


def style(ax):
    ax.grid(True, color=GRID, lw=0.8)
    ax.set_axisbelow(True)


def fig_main(R, out):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
    for ax, t in zip(axes, "AB"):
        style(ax)
        ends = []
        for v in ["S", "N", "J", "L", "LJ", "LT"]:
            ns = [16, 64, 256]
            m = [agg(R, variant=v, task=t, n=n, shift=0, frozen=False) for n in ns]
            mu = np.array([x[0] for x in m])
            lo = np.array([x[1] for x in m])
            hi = np.array([x[2] for x in m])
            ax.fill_between(ns, lo, hi, color=COL[v], alpha=0.12, lw=0)
            ax.plot(ns, mu, "-", color=COL[v], lw=2, marker=MRK[v], ms=8,
                    markeredgecolor=SURF, markeredgewidth=1.5, label=NAME[v])
            ends.append((mu[-1], v))
        # direct labels at the right end, de-collided
        ends.sort()
        ys = []
        lo_, hi_ = ax.get_ylim()
        gap = (np.log(hi_) - np.log(lo_)) * 0.05 if ax.get_yscale() == "log" else (hi_ - lo_) * 0.055
        for y, v in ends:
            yy = y if not ys or y - ys[-1] > gap else ys[-1] + gap
            ys.append(yy)
            ax.annotate(NAME[v], (256, y), xytext=(300, yy), fontsize=8.5, color=INK2, va="center",
                        arrowprops=dict(arrowstyle="-", color=GRID, lw=0.8))
        ax.set_xscale("log", base=2)
        ax.set_xticks([16, 64, 256])
        ax.set_xticklabels(["16", "64", "256"])
        ax.set_xlim(13, 256 * 3.2)
        ax.set_xlabel("labeled training simulations")
        ax.set_ylabel("test relative L2 error (lower is better)")
        ax.set_title(TASK[t], loc="left", fontsize=11)
    axes[0].legend(frameon=False, fontsize=8.5, loc="upper right")
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)


def fig_prompt(R, out):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.6))
    for ax, t in zip(axes, "AB"):
        style(ax)
        for v in ["S", "L", "LT"]:
            sh = [0, 90, 180]
            m = [agg(R, variant=v, task=t, n=64, shift=s, frozen=False) for s in sh]
            mu = [x[0] for x in m]
            ax.errorbar(sh, mu, yerr=[[x[0] - x[1] for x in m], [x[2] - x[0] for x in m]], color=COL[v],
                        lw=2, marker=MRK[v], ms=8, capsize=3, markeredgecolor=SURF, label=NAME[v])
            ax.annotate(NAME[v], (180, mu[-1]), xytext=(8, 0), textcoords="offset points",
                        fontsize=8.5, color=INK2, va="center")
        ax.set_xticks([0, 90, 180])
        ax.set_xticklabels(["0 (aligned)", "90", "180 (reversed)"])
        ax.set_xlim(-15, 250)
        ax.set_xlabel("prompt rotated away from the true flow direction (degrees)")
        ax.set_ylabel("test relative L2 error, 64 labels")
        ax.set_title(TASK[t], loc="left", fontsize=11)
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, frameon=False, fontsize=8.5, loc="lower center", ncol=3)
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    fig.savefig(out, dpi=130)
    plt.close(fig)


def fig_frozen(R, out):
    vs = ["S", "N", "J", "L", "LJ", "LT"]
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6), sharey=True)
    for ax, t in zip(axes, "AB"):
        style(ax)
        ax.grid(True, axis="y", lw=0)
        for i, v in enumerate(vs):
            fz = agg(R, variant=v, task=t, n=64, shift=0, frozen=True)
            ft = agg(R, variant=v, task=t, n=64, shift=0, frozen=False)
            ax.barh(i + 0.18, fz[0], height=0.34, color=COL[v], alpha=0.45, edgecolor=SURF, lw=2)
            ax.barh(i - 0.18, ft[0], height=0.34, color=COL[v], edgecolor=SURF, lw=2)
            ax.text(fz[0], i + 0.18, f" {fz[0]:.3f} frozen enc.", va="center", fontsize=8, color=INK2)
            ax.text(ft[0], i - 0.18, f" {ft[0]:.3f} full FT", va="center", fontsize=8, color=INK2)
        ax.set_yticks(range(len(vs)))
        ax.set_yticklabels([NAME[v] for v in vs])
        ax.invert_yaxis()
        ax.set_xlim(0, ax.get_xlim()[1] * 1.45)
        ax.set_xlabel("test relative L2 error, 64 labels")
        ax.set_title(TASK[t], loc="left", fontsize=11)
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)


def fig_collapse(out):
    fig, ax = plt.subplots(figsize=(7.2, 4.3))
    style(ax)
    for v in ["J", "LJ", "LT", "LT0"]:
        p = f"ckpt/{v}.json"
        if not os.path.exists(p):
            continue
        H = json.load(open(p))
        it = [h["it"] for h in H]
        sd = [h["val_tgt_std"] / H[0]["val_tgt_std"] for h in H]   # metrics differ in scale -> relative
        ax.plot(it, sd, "-", color=COL[v], lw=2, marker=MRK[v], ms=7, markeredgecolor=SURF, label=NAME[v])
        ax.annotate(NAME[v], (it[-1], sd[-1]), xytext=(6, 0), textcoords="offset points", fontsize=8.5,
                    color=INK2, va="center")
    ax.set_xlim(right=ax.get_xlim()[1] * 1.35)
    ax.set_xlabel("pretraining step")
    ax.set_ylabel("target-latent spread across inputs\n(relative to initialisation)")
    ax.set_ylim(0, 1.15)
    ax.set_title("Collapse probe: does the JEPA target stay informative?", loc="left", fontsize=11)
    ax.legend(frameon=False, fontsize=8.5, loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=4)
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)


def table(R):
    lines = ["| variant | A n=16 | A n=64 | A n=256 | B n=16 | B n=64 | B n=256 |", "|---|---|---|---|---|---|---|"]
    for v in ["S", "N", "J", "L", "LJ", "LT"]:
        row = [NAME[v]]
        for t in "AB":
            for n in [16, 64, 256]:
                m = agg(R, variant=v, task=t, n=n, shift=0, frozen=False)
                s0 = agg(R, variant="S", task=t, n=n, shift=0, frozen=False)[0]
                row.append(f"{m[0]:.3f} ({100 * (m[0] / s0 - 1):+.0f}%)" if v != "S" else f"{m[0]:.3f}")
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def equivalent_data(R, v, t):
    """labels the variant needs to match scratch@256 (log-linear interpolation of its curve)."""
    target = agg(R, variant="S", task=t, n=256, shift=0, frozen=False)[0]
    ns = np.array([16, 64, 256])
    e = np.array([agg(R, variant=v, task=t, n=n, shift=0, frozen=False)[0] for n in ns])
    if e[-1] > target:
        return None
    for i in range(2):
        if e[i] >= target >= e[i + 1]:
            f = (np.log(e[i]) - np.log(target)) / (np.log(e[i]) - np.log(e[i + 1]))
            return float(np.exp(np.log(ns[i]) + f * (np.log(ns[i + 1]) - np.log(ns[i]))))
    return float(ns[0]) if e[0] <= target else None


if __name__ == "__main__":
    R = load()
    os.makedirs("figs", exist_ok=True)
    fig_main(R, "figs/fig1_data_efficiency.png")
    fig_prompt(R, "figs/fig2_prompt_shift.png")
    fig_frozen(R, "figs/fig3_frozen_vs_finetune.png")
    fig_collapse("figs/fig4_collapse.png")
    print(table(R))
    for t in "AB":
        print(t, {v: equivalent_data(R, v, t) for v in ["N", "J", "L", "LJ", "LT"]})
    for t in "AB":
        for v in ["S", "L", "LT"]:
            print("prompt", t, v, [round(agg(R, variant=v, task=t, n=64, shift=s, frozen=False)[0], 4) for s in (0, 90, 180)])
    for t in "AB":
        print("LT0", t, round(agg(R, variant="LT0", task=t, n=64, shift=0, frozen=False)[0], 4))
        print("frozen", t, {v: round(agg(R, variant=v, task=t, n=64, shift=0, frozen=True)[0], 4)
                            for v in ["S", "N", "J", "L", "LJ", "LT"]})
