"""Pretrain one variant on unlabeled geometry only.

usage: python pretrain.py VARIANT [steps]                      (writes ckpt/VARIANT.npz, as before)
       python pretrain.py VARIANT --steps 3000 --seed 0 --data data/pretrain.npz --out ckpt/x.npz \
                                  --train-shapes 2000
"""
import os
import time
import json
import argparse
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


EXPLICIT = ("nat", "lift", "anchor")   # pretext losses on explicit vector-distance targets


def clean_view(Dd):
    """the corpus with its stored clean targets (vd_clean, and X_clean if present) swapped in."""
    C = dict(Dd, vd=Dd["vd_clean"])
    if "X_clean" in Dd:
        C["X"] = Dd["X_clean"]
    return C


def pretrain(variant, steps=2500, B=16, lr=1e-3, seed=0, Dd=None, log_every=250, out=None,
             batch_fn=None, train_shapes=N_TRAIN_SHAPES):
    """batch_fn(batch, rng_aux) -> batch: optional transform of every training batch (e.g. fresh
    label noise); rng_aux is seeded separately, so initialisation and batch order match the plain run.
    If Dd holds vd_clean, the explicit heads' validation losses are also logged against the clean
    targets (keys valclean_*).  train_shapes: train on samples with sid < train_shapes only;
    validation always uses sid >= N_TRAIN_SHAPES."""
    rng = np.random.default_rng(seed)
    rng_aux = np.random.default_rng(seed + 10_000) if batch_fn is not None else None
    P = Mo.build(variant, seed)
    ema = variant in ("J", "LJ", "LT", "LT0")
    Pt = Mo.copy_params(P) if ema else None
    opt = ag.AdamW(P, lr=lr)
    tr = np.flatnonzero(Dd["sid"] < train_shapes)
    va = np.flatnonzero(Dd["sid"] >= N_TRAIN_SHAPES)
    hist, t0 = [], time.time()
    for it in range(steps):
        idx = rng.choice(tr, B, replace=False)
        batch = make_batch(Dd, idx, rng)
        if batch_fn is not None:
            batch = batch_fn(batch, rng_aux)
        loss, log = Mo.pretext_loss(variant, P, Pt, batch, rng)
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
            if "vd_clean" in Dd:
                crng = np.random.default_rng(123)
                with ag.no_grad():
                    _, clog = Mo.pretext_loss(variant, P, Pt, make_batch(clean_view(Dd), va[:64], crng, 0.0), crng)
                rec.update({"valclean_" + k: round(v, 5) for k, v in clog.items() if k in EXPLICIT})
            hist.append(rec)
            print(variant, rec, flush=True)
    if out:
        np.savez(out, **{k: v.data for k, v in P.items()})
        with open(out.replace(".npz", ".json"), "w") as f:
            json.dump(hist, f)
    return P, hist


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("variant")
    ap.add_argument("steps_pos", nargs="?", type=int, help="old positional form: number of steps")
    ap.add_argument("--steps", type=int, default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--data", default="data/pretrain.npz")
    ap.add_argument("--out", default=None)
    ap.add_argument("--train-shapes", type=int, default=N_TRAIN_SHAPES)
    a = ap.parse_args()
    steps = a.steps if a.steps is not None else (a.steps_pos if a.steps_pos is not None else 2500)
    if a.out is None and (a.seed != 0 or a.data != "data/pretrain.npz" or a.train_shapes != N_TRAIN_SHAPES):
        ap.error("non-default --seed/--data/--train-shapes need an explicit --out (keeps the original ckpt/<V>.npz)")
    out = a.out or f"ckpt/{a.variant}.npz"
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    pretrain(a.variant, steps=steps, seed=a.seed, Dd=load(a.data), out=out, train_shapes=a.train_shapes)
