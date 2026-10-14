"""What does the velocity prompt retrieve?  (toy analogue of GeoPT Fig. 7c)

Row 1: GeoPT-lifted pretext head, predicted distance-to-wall after 2 transport steps
       under a *uniform* prompt (it was only ever trained on i.i.d. per-point prompts).
Row 2: GeoPT-lifted decoder features, prompt-dependent part (feature minus its mean over prompts), PC1.
Row 3: latent-transport JEPA decoder features, same.
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import ag
import model as Mo
import world as W
import finetune as F

SURF, INK, INK2 = "#fcfcfb", "#0b0b0b", "#52514e"
ANGS = [0, 90, 180]


def load_params(v):
    ck = np.load(f"ckpt/{v}.npz")
    return {k: ag.const(ck[k]) for k in ck.files}


def features(P, V, Xq, ang, head=None):
    pts, nrm = F.surf_batch(V[None], phase=0.0)
    a = np.radians(ang)
    v = np.broadcast_to(0.25 * np.array([np.cos(a), np.sin(a)], np.float32), (1, len(Xq), 2)).copy()
    with ag.no_grad():
        geo, _ = Mo.encode(P, pts, nrm)
        r = Mo.decode(P, geo, Xq[None].astype(np.float32), v)
        out = Mo.lin(P, r, head).data[0] if head else None
    return r.data[0], out


if __name__ == "__main__":
    Dd = F.load()
    i = 262
    V = Dd["shapes"][i]
    G, body = Dd["grid"], Dd["body"][i]
    Xq = G[~body]
    PL, PT = load_params("L"), load_params("LT")
    rows = []
    # row 1: pretext prediction vs. exact transport target
    pred, true = [], []
    for ang in ANGS:
        _, out = features(PL, V, Xq, ang, head="head.lift")
        pred.append(np.linalg.norm(out[:, 4:6], axis=1))
        a = np.radians(ang)
        vv = np.broadcast_to(0.25 * np.array([np.cos(a), np.sin(a)]), Xq.shape).copy()
        X1, st = W.transport_step(V.astype(np.float64), Xq.astype(np.float64), vv, np.zeros(len(Xq), bool))
        X2, st = W.transport_step(V.astype(np.float64), X1, vv, st)
        true.append(np.linalg.norm(W.vector_distance(V.astype(np.float64), X2), axis=1))
    rows.append(("GeoPT pretext output:\npredicted wall distance after transport", pred, "viridis"))
    for name, P in [("GeoPT-lifted features:\nspatially varying\nprompt effect, PC1", PL), ("Latent-transport JEPA features:\nspatially varying\nprompt effect, PC1", PT)]:
        F_ = [features(P, V, Xq, ang)[0] for ang in ANGS]
        # isolate what the prompt changes: subtract each point's mean feature over prompts
        Fm = np.mean(F_, axis=0)
        Dlt = [f - Fm for f in F_]
        Dlt = [d - d.mean(0) for d in Dlt]      # drop the spatially-uniform "which prompt" code
        _, _, Vt = np.linalg.svd(np.concatenate(Dlt), full_matrices=False)
        pcs = [d @ Vt[0] for d in Dlt]
        rows.append((name, pcs, "coolwarm"))
    err = np.mean([np.abs(p - t).mean() for p, t in zip(pred, true)])
    fig, axes = plt.subplots(3, 3, figsize=(10.5, 10.2))
    for r, (name, fields, cmap) in enumerate(rows):
        lo = min(f.min() for f in fields)
        hi = max(f.max() for f in fields)
        if cmap == "coolwarm":
            m = max(abs(lo), abs(hi))
            lo, hi = -m, m
        for c, (ang, f) in enumerate(zip(ANGS, fields)):
            ax = axes[r, c]
            img = np.full(len(G), np.nan)
            img[~body] = f
            ax.imshow(img.reshape(W.GRID, W.GRID).T, origin="lower", extent=[-1, 1, -1, 1], cmap=cmap, vmin=lo, vmax=hi)
            ax.fill(V[:, 0], V[:, 1], color="0.82", ec=INK2, lw=0.8)
            a = np.radians(ang)
            ax.arrow(0.0, 0.0, 0.3 * np.cos(a), 0.3 * np.sin(a), width=0.025, color="white", ec=INK, lw=0.6,
                     length_includes_head=True, zorder=5)
            ax.set_xticks([])
            ax.set_yticks([])
            if r == 0:
                ax.set_title(f"prompt direction {ang} deg", fontsize=10, color=INK)
            if c == 0:
                ax.set_ylabel(name, fontsize=9.5, color=INK2)
    fig.suptitle(f"What the dynamics prompt retrieves (unseen test body; pretext MAE vs exact transport = {err:.3f})",
                 fontsize=11, color=INK, x=0.02, ha="left")
    fig.tight_layout()
    fig.savefig("figs/fig5_prompt_fields.png", dpi=110)
    print("pretext mean abs err vs exact transport:", round(float(err), 4))
