"""Fine-tune a (pre)trained backbone on a labeled physics task and evaluate.

task A: surface speed q = |Vt|/U from potential flow  (elliptic, reversible: q(alpha) == q(alpha+pi))
task B: steady plume concentration on a 64x64 grid     (transport-dominated, directional)

Downstream samples are expressed in the flow-aligned frame (as in industrial CFD
datasets, where the freestream is along the vehicle axis): geometry and query points
are rotated by -alpha, so the body-to-flow angle still varies over the full circle but
the global-rotation nuisance is removed.  The prompt (dynamics condition, GeoPT's V_S)
is then a uniform velocity of magnitude 0.25*U along +x - or deliberately rotated by
`shift` degrees for the prompt-alignment ablation.
"""
import time
import numpy as np
import ag
import model as Mo
from pretrain import surf_batch, cosine

TRAIN = np.arange(256)
TEST = np.arange(256, 356)
ENC = ("tok.", "enc.")
DEC = ("q.", "dec.")


def rot(theta):
    c, s = np.cos(theta), np.sin(theta)
    return np.stack([np.stack([c, -s], -1), np.stack([s, c], -1)], -2)


def load(path="data/downstream.npz"):
    d = np.load(path)
    d = {k: d[k] for k in d.files}
    RA, RB = rot(-d["alphaA"]), rot(-d["alphaB"])
    d["shapesA"] = np.einsum("nij,nkj->nki", RA, d["shapes"]).astype(np.float32)
    d["ptsA_c"] = np.einsum("nij,nkj->nki", RA, d["ptsA"]).astype(np.float32)
    d["shapesB"] = np.einsum("nij,nkj->nki", RB, d["shapes"]).astype(np.float32)
    d["gridB"] = np.einsum("nij,kj->nki", RB, d["grid"]).astype(np.float32)
    return d


def norm_stats(Dd, task):
    if task == "A":
        y = Dd["Cp"][TRAIN]
    else:
        y = Dd["c"][TRAIN][~Dd["body"][TRAIN]]
    return float(y.mean()), float(y.std())


def prompts(Dd, task, ids, shift, Q):
    sp = np.full(len(ids), 0.25) if task == "A" else 0.25 * Dd["UB"][ids]
    v = (sp[:, None] * np.array([np.cos(shift), np.sin(shift)])[None]).astype(np.float32)
    return np.broadcast_to(v[:, None, :], (len(ids), Q, 2)).copy()


def train_batch(Dd, task, ids, rng, shift, nq=256):
    pts, nrm = surf_batch(Dd["shapes" + task][ids], rng)
    if task == "A":
        q, y = Dd["ptsA_c"][ids], Dd["Cp"][ids]
    else:
        q, y = [], []
        for i in ids:
            j = rng.choice(np.flatnonzero(~Dd["body"][i]), nq, replace=False)
            q.append(Dd["gridB"][i, j])
            y.append(Dd["c"][i, j])
        q, y = np.array(q), np.array(y)
    return pts, nrm, q, prompts(Dd, task, ids, shift, q.shape[1]), y


def forward(P, pts, nrm, q, v, frozen):
    if frozen:
        with ag.no_grad():
            geo, _ = Mo.encode(P, pts, nrm)
        geo = ag.const(geo.data)
    else:
        geo, _ = Mo.encode(P, pts, nrm)
    return Mo.lin(P, Mo.decode(P, geo, q, v), "head.task")


def evaluate(P, Dd, task, shift, frozen, mu, sd, ids=TEST, keep_pred=0):
    errs, preds = [], []
    with ag.no_grad():
        if task == "A":
            for s in range(0, len(ids), 20):
                b = ids[s:s + 20]
                pts, nrm = surf_batch(Dd["shapesA"][b], phase=0.0)
                q, y = Dd["ptsA_c"][b], Dd["Cp"][b]
                out = forward(P, pts, nrm, q, prompts(Dd, task, b, shift, q.shape[1]), frozen).data[..., 0] * sd + mu
                errs += list(np.linalg.norm(out - y, axis=1) / np.linalg.norm(y, axis=1))
                preds += list(out)
        else:
            for i in ids:
                j = np.flatnonzero(~Dd["body"][i])
                pts, nrm = surf_batch(Dd["shapesB"][[i]], phase=0.0)
                q, y = Dd["gridB"][i, j][None], Dd["c"][i, j]
                out = forward(P, pts, nrm, q, prompts(Dd, task, np.array([i]), shift, len(j)), frozen).data[0, :, 0] * sd + mu
                errs.append(np.linalg.norm(out - y) / np.linalg.norm(y))
                preds.append(out)
    return float(np.mean(errs)), np.array(errs), preds[:keep_pred]


def fit(Dd, variant, task, n, seed, shift=0.0, frozen=False, steps=1000, B=8, lr=1e-3, freeze=()):
    rng = np.random.default_rng(seed)
    P = Mo.build("S", 100 + seed)                        # fresh encoder + decoder
    if variant != "S":
        ck = np.load(f"ckpt/{variant}.npz")
        load_dec = (not frozen) and variant != "J"       # J only trains the encoder
        for k in P:
            if k.startswith(ENC) or (load_dec and k.startswith(DEC)):
                P[k].data = ck[k].astype(np.float32).copy()
    Mo.init_head(P, rng, "head.task", 1)
    train = {k: p for k, p in P.items()
             if not (frozen and k.startswith(ENC)) and not (freeze and k.startswith(tuple(freeze)))}
    opt = ag.AdamW(train, lr=lr)
    mu, sd = norm_stats(Dd, task)
    ids_pool = rng.choice(TRAIN, n, replace=False)
    for it in range(steps):
        ids = rng.choice(ids_pool, B, replace=n < B)
        pts, nrm, q, v, y = train_batch(Dd, task, ids, rng, shift)
        out = forward(P, pts, nrm, q, v, frozen)
        loss = ag.mse(out, ((y - mu) / sd)[..., None].astype(np.float32))
        loss.backward()
        opt.step(cosine(it, steps, lr, warm=50))
    return P, loss


def finetune_params(Dd, variant, task, n, seed, **kw):
    return fit(Dd, variant, task, n, seed, **kw)[0]


def finetune(Dd, variant, task, n, seed, shift=0.0, frozen=False, steps=1000, B=8, lr=1e-3, keep_pred=0,
             freeze=()):
    t0 = time.time()
    P, loss = fit(Dd, variant, task, n, seed, shift, frozen, steps, B, lr, freeze)
    mu, sd = norm_stats(Dd, task)
    err, per, preds = evaluate(P, Dd, task, shift, frozen, mu, sd, keep_pred=keep_pred)
    return dict(variant=variant, task=task, n=n, seed=seed, shift=round(float(np.degrees(shift))),
                frozen=frozen, rel_l2=err, final_train_loss=float(loss.data),
                sec=round(time.time() - t0, 1)), per, preds
