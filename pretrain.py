"""Pretrain one variant on unlabeled geometry only.  usage: python pretrain.py VARIANT [steps]"""
import os
import sys
import time
import json
import numpy as np
import ag
import model as Mo
import world as W

N_TRAIN_SHAPES = 2000


def load(path):
    d = np.load(path)
    return {k: d[k] for k in d.files}


def surf_batch(shapes, rng=None, phase=None):
    pts, nrm = [], []
    for V in shapes:
        ph = rng.uniform() if phase is None else phase
        p, n = W.resample_surface(V.astype(np.float64), ph)
        pts.append(p)
        nrm.append(n)
    return np.array(pts, np.float32), np.array(nrm, np.float32)


def make_batch(Dd, idx, rng, phase=None):
    pts, nrm = surf_batch(Dd["shapes"][Dd["sid"][idx]], rng, phase)
    return dict(pts=pts, nrm=nrm, X=Dd["X"][idx], v=Dd["v"][idx], vd=Dd["vd"][idx])


def cosine(it, steps, lr, warm=100):
    if it < warm:
        return lr * (it + 1) / warm
    return 0.5 * lr * (1 + np.cos(np.pi * (it - warm) / max(1, steps - warm)))


def pretrain(variant, steps=2500, B=16, lr=1e-3, seed=0, Dd=None, log_every=250, out=None):
    rng = np.random.default_rng(seed)
    P = Mo.build(variant, seed)
    ema = variant in ("J", "LJ", "LT", "LT0")
    Pt = Mo.copy_params(P) if ema else None
    opt = ag.AdamW(P, lr=lr)
    tr = np.flatnonzero(Dd["sid"] < N_TRAIN_SHAPES)
    va = np.flatnonzero(Dd["sid"] >= N_TRAIN_SHAPES)
    hist, t0 = [], time.time()
    for it in range(steps):
        idx = rng.choice(tr, B, replace=False)
        loss, log = Mo.pretext_loss(variant, P, Pt, make_batch(Dd, idx, rng), rng)
        loss.backward()
        opt.step(cosine(it, steps, lr))
        if ema:
            Mo.ema_update(Pt, P, 0.99 + 0.009 * it / steps)
        if it % log_every == 0 or it == steps - 1:
            vrng = np.random.default_rng(123)
            with ag.no_grad():
                _, vlog = Mo.pretext_loss(variant, P, Pt, make_batch(Dd, va[:64], vrng, 0.0), vrng)
            rec = {"it": it, "sec": round(time.time() - t0, 1), **{k: round(v, 5) for k, v in log.items()},
                   **{"val_" + k: round(v, 5) for k, v in vlog.items()}}
            hist.append(rec)
            print(variant, rec, flush=True)
    if out:
        np.savez(out, **{k: v.data for k, v in P.items()})
        with open(out.replace(".npz", ".json"), "w") as f:
            json.dump(hist, f)
    return P, hist


if __name__ == "__main__":
    v = sys.argv[1]
    steps = int(sys.argv[2]) if len(sys.argv) > 2 else 2500
    Dd = load("data/pretrain.npz")
    os.makedirs("ckpt", exist_ok=True)
    pretrain(v, steps=steps, Dd=Dd, out=f"ckpt/{v}.npz")
