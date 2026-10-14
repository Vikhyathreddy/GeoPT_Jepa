"""Shortcut probe: how much of each pretext target can be predicted WITHOUT the geometry?

A pretext only teaches geometry/physics to the extent its target cannot be guessed from
the query alone.  For each objective we fit a geometry-blind ridge regressor
(1024 random Fourier features of the query inputs the predictor gets for free) on
2000 held-out-shape samples and report held-out R^2.  High R^2 = shortcut.

  N   target: vector distance at x                 blind input: x
  L   target: vector distances along trajectory    blind input: x0, v
  LT  target: (normalised) latent at x1, x2        blind input: x0, v
  J   target: (normalised) latent of masked patch  blind input: patch centre
Latent targets are produced by the trained online networks (proxy for their EMA targets).
"""
import numpy as np
import ag
import model as Mo
import pretrain as PT

rng = np.random.default_rng(0)


def rff(Z, W, b):
    return np.cos(Z @ W + b)


def blind_r2(Zin, Y, ntr):
    W = rng.normal(0, 3.0, (Zin.shape[1], 1024))
    b = rng.uniform(0, 2 * np.pi, 1024)
    Phi = rff(Zin, W, b)
    A, B = Phi[:ntr], Phi[ntr:]
    Ya, Yb = Y[:ntr], Y[ntr:]
    mu = Ya.mean(0)
    coef = np.linalg.solve(A.T @ A + 1e-2 * np.eye(A.shape[1]), A.T @ (Ya - mu))
    pred = B @ coef + mu
    return 1 - ((pred - Yb) ** 2).sum() / ((Yb - Yb.mean(0)) ** 2).sum()


def params(v):
    ck = np.load(f"ckpt/{v}.npz")
    return {k: ag.const(ck[k]) for k in ck.files}


if __name__ == "__main__":
    Dd = PT.load("data/pretrain.npz")
    va = np.flatnonzero(Dd["sid"] >= PT.N_TRAIN_SHAPES)          # unseen shapes
    # one sample per distinct shape, so train/test rows never share a geometry
    first = {}
    for i in va:
        first.setdefault(Dd["sid"][i], i)
    idx = rng.choice(np.array(list(first.values())), 64, replace=False)
    b = PT.make_batch(Dd, idx, rng, phase=0.0)
    X, v, vd = b["X"], b["v"], b["vd"]
    sel = rng.choice(X.shape[2], 40, replace=False)               # 40 points per sample -> 2560 rows
    flat = lambda a: a.reshape(-1, a.shape[-1])
    x0, x1, x2 = flat(X[:, 0][:, sel]), flat(X[:, 1][:, sel]), flat(X[:, 2][:, sel])
    vv = flat(v[:, sel])
    ntr = 2000
    res = {}
    res["N  explicit: vector distance"] = blind_r2(x0, flat(vd[:, 0][:, sel]), ntr)
    res["L  explicit: trajectory vector distances"] = blind_r2(
        np.concatenate([x0, vv], 1), np.concatenate([flat(vd[:, t][:, sel]) for t in range(3)], 1), ntr)
    P = params("LT")
    with ag.no_grad():
        geo, _ = Mo.encode(P, b["pts"], b["nrm"])
        z = np.zeros_like(v)
        t1 = Mo.plain_ln(Mo.decode(P, geo, X[:, 1], z).data)[:, sel]
        t2 = Mo.plain_ln(Mo.decode(P, geo, X[:, 2], z).data)[:, sel]
    res["LT latent: representation at x1, x2"] = blind_r2(np.concatenate([x0, vv], 1),
                                                          np.concatenate([flat(t1), flat(t2)], 1), ntr)
    # sanity: same latents but give the regressor the true landing points instead of (x0, v)
    res["LT latent, blind input = true landing points"] = blind_r2(np.concatenate([x1, x2], 1),
                                                                   np.concatenate([flat(t1), flat(t2)], 1), ntr)
    Pj = params("J")
    with ag.no_grad():
        tg, cen = Mo.encode(Pj, b["pts"], b["nrm"])
    Yj = Mo.plain_ln(tg.data).reshape(-1, Mo.D)
    res["J  latent: masked-patch representation"] = blind_r2(cen.reshape(-1, 2), Yj, int(0.8 * len(Yj)))
    # --- near-body points only (|vector distance| < 0.15): where geometry should matter most
    near = np.linalg.norm(flat(vd[:, 0][:, sel]), axis=1) < 0.15
    nidx = np.flatnonzero(near)
    ntr_n = int(0.78 * len(nidx))
    res["L  explicit, near-body points only"] = blind_r2(np.concatenate([x0, vv], 1)[nidx],
        np.concatenate([flat(vd[:, t][:, sel]) for t in range(3)], 1)[nidx], ntr_n)
    res["LT latent, near-body points only"] = blind_r2(np.concatenate([x0, vv], 1)[nidx],
        np.concatenate([flat(t1), flat(t2)], 1)[nidx], ntr_n)
    # --- the geometry-AWARE trained networks on the same held-out rows, for reference
    def r2(pred, Y):
        return 1 - ((pred - Y) ** 2).sum() / ((Y - Y.mean(0)) ** 2).sum()
    PL = params("L")
    with ag.no_grad():
        gL, _ = Mo.encode(PL, b["pts"], b["nrm"])
        oL = Mo.lin(PL, Mo.decode(PL, gL, X[:, 0], v), "head.lift").data[:, sel]
        hT = ag.gelu(Mo.lin(P, Mo.decode(P, geo, X[:, 0], v), "lt.1"))
        p1, p2 = Mo.lin(P, hT, "lt.o1").data[:, sel], Mo.lin(P, hT, "lt.o2").data[:, sel]
    YL = np.concatenate([flat(vd[:, t][:, sel]) for t in range(3)], 1)
    res["L  network (sees geometry)"] = r2(flat(oL)[ntr:], YL[ntr:])
    YT = np.concatenate([flat(t1), flat(t2)], 1)
    res["LT network (sees geometry)"] = r2(np.concatenate([flat(p1), flat(p2)], 1)[ntr:], YT[ntr:])
    for k, r in res.items():
        print(f"{k:48s} R^2 = {r:.3f}")
    import json
    json.dump({k: float(r) for k, r in res.items()}, open("results/shortcut_probe.json", "w"), indent=1)
