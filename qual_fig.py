"""Qualitative predictions on one unseen body (models fine-tuned with 64 labels)."""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import ag
import world as W
import finetune as F

SURF, INK, INK2 = "#fcfcfb", "#0b0b0b", "#52514e"
COL = {"S": "#8a8984", "L": "#1baf7a", "LT": "#e87ba4"}
NAME = {"S": "Scratch", "L": "GeoPT-lifted", "LT": "Latent-transport JEPA"}


def params(v, t):
    ck = np.load(f"ckpt/ft_{v}_{t}_n64.npz")
    return {k: ag.const(ck[k]) for k in ck.files}


if __name__ == "__main__":
    Dd = F.load()
    i = 270
    V, G, body = Dd["shapes"][i], Dd["grid"], Dd["body"][i]
    j = np.flatnonzero(~body)
    fig, axes = plt.subplots(2, 4, figsize=(15, 7.9))
    # ---- task B plume
    preds = {}
    with ag.no_grad():
        for v in COL:
            P = params(v, "B")
            mu, sd = F.norm_stats(Dd, "B")
            pts, nrm = F.surf_batch(Dd["shapesB"][[i]], phase=0.0)
            q = Dd["gridB"][i, j][None]
            pr = F.prompts(Dd, "B", np.array([i]), 0.0, len(j))
            preds[v] = F.forward(P, pts, nrm, q, pr, False).data[0, :, 0] * sd + mu
    truth = Dd["c"][i, j]

    def show(ax, f, title, cmap="viridis", lo=0, hi=1):
        img = np.full(len(G), np.nan)
        img[j] = f
        im = ax.imshow(img.reshape(W.GRID, W.GRID).T, origin="lower", extent=[-1, 1, -1, 1], cmap=cmap, vmin=lo, vmax=hi)
        ax.fill(V[:, 0], V[:, 1], color="0.82", ec=INK2, lw=0.8)
        a = Dd["alphaB"][i]
        ax.arrow(-0.85, 0.85, 0.25 * np.cos(a), 0.25 * np.sin(a), width=0.02, color="white", ec=INK, lw=0.5)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(title, fontsize=10, color=INK, loc="left")
        return im

    show(axes[0, 0], truth, "Task B ground truth (solver)")
    for k, v in enumerate(COL):
        e = np.linalg.norm(preds[v] - truth) / np.linalg.norm(truth)
        show(axes[0, k + 1], preds[v], f"{NAME[v]}  (rel. L2 {e:.3f})")
        show(axes[1, k + 1], np.abs(preds[v] - truth), f"|error|, {NAME[v]}", cmap="magma", lo=0, hi=0.3)
    # ---- task A surface speed along the contour
    ax = axes[1, 0]
    ax.grid(True, color="#e4e3df", lw=0.8)
    yA = Dd["Cp"][i]
    s = np.arange(len(yA)) / len(yA)
    ax.plot(s, yA, "-", color=INK, lw=2.4, label="solver")
    with ag.no_grad():
        for v in COL:
            P = params(v, "A")
            mu, sd = F.norm_stats(Dd, "A")
            pts, nrm = F.surf_batch(Dd["shapesA"][[i]], phase=0.0)
            q = Dd["ptsA_c"][[i]]
            out = F.forward(P, pts, nrm, q, F.prompts(Dd, "A", np.array([i]), 0.0, q.shape[1]), False).data[0, :, 0] * sd + mu
            ax.plot(s, out, "-", color=COL[v], lw=2, label=NAME[v])
    ax.set_xlabel("position along the body contour", color=INK2)
    ax.set_ylabel("surface speed q / U", color=INK2)
    ax.set_title("Task A on the same body", fontsize=10, color=INK, loc="left")
    ax.legend(frameon=False, fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=2)
    fig.suptitle("Unseen test body, models fine-tuned on 64 labeled simulations (arrow = flow direction)",
                 fontsize=11, color=INK, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig("figs/fig6_qualitative.png", dpi=110)
    print("saved")
