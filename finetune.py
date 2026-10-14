"""Fine-tune a (pre)trained backbone on a labeled physics task and evaluate.

task A: surface speed q = |Vt|/U from potential flow  (elliptic, reversible: q(alpha) == q(alpha+pi))
task B: steady plume concentration on a 64x64 grid     (transport-dominated, directional)

Downstream samples are expressed in the flow-aligned frame (as in industrial CFD
datasets, where the freestream is along the vehicle axis): geometry and query points
are rotated by -alpha, so the body-to-flow angle still varies over the full circle but
the global-rotation nuisance is removed.  The prompt (dynamics condition, GeoPT's V_S)
is then a uniform velocity of magnitude 0.25*U along +x - or deliberately rotated by
`shift` degrees for the prompt-alignment ablation.

Options (defaults reproduce the behaviour above exactly):
  frame="absolute"  no rotation of geometry or queries; the prompt is per sample,
                    0.25*U*(cos(alpha + shift), sin(alpha + shift)), with U = 1 for task A
  prompt_mode="zero" prompt set to 0
  ckpt=, encoder_only=  checkpoint path (default ckpt/<variant>.npz) and whether to load only the
                    encoder (default: variant == "J")
  Dd=None, data=    load the downstream file from `data` (default data/downstream.npz)
"""
import time
import numpy as np
import ag
import model as Mo
from pretrain import surf_batch, cosine

DATA = "data/downstream.npz"
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


_CACHE = {}


def get_data(Dd=None, data=None):
    """Dd if given, else the (cached) downstream file at `data` (default DATA)."""
    if Dd is not None:
        return Dd
    path = data or DATA
    if path not in _CACHE:
        _CACHE[path] = load(path)
    return _CACHE[path]


def prompts(Dd, task, ids, shift, Q):
    sp = np.full(len(ids), 0.25) if task == "A" else 0.25 * Dd["UB"][ids]
    v = (sp[:, None] * np.array([np.cos(shift), np.sin(shift)])[None]).astype(np.float32)
    return np.broadcast_to(v[:, None, :], (len(ids), Q, 2)).copy()


def prompt_for(Dd, task, ids, shift, Q, frame="flow", mode="normal"):
    """prompts() in the flow frame; the per-sample flow velocity in the absolute frame; or zeros."""
    if frame == "flow":
        v = prompts(Dd, task, ids, shift, Q)
    elif frame == "absolute":
        sp = np.full(len(ids), 0.25) if task == "A" else 0.25 * Dd["UB"][ids]
        a = Dd["alpha" + task][ids].astype(np.float64) + shift
        v = (sp[:, None] * np.stack([np.cos(a), np.sin(a)], -1)).astype(np.float32)
        v = np.broadcast_to(v[:, None, :], (len(ids), Q, 2)).copy()
    else:
        raise ValueError(f"frame {frame!r}")
    if mode == "zero":
        return np.zeros_like(v)
    if mode != "normal":
        raise ValueError(f"prompt_mode {mode!r}")
    return v


def shapes_for(Dd, task, frame="flow"):
    return Dd["shapes" + task] if frame == "flow" else Dd["shapes"]


def queries_A(Dd, frame="flow"):
    return Dd["ptsA_c"] if frame == "flow" else Dd["ptsA"]


def grid_for(Dd, i, frame="flow"):
    return Dd["gridB"][i] if frame == "flow" else Dd["grid"]


def train_batch(Dd, task, ids, rng, shift, nq=256, frame="flow", prompt_mode="normal"):
    pts, nrm = surf_batch(shapes_for(Dd, task, frame)[ids], rng)
    if task == "A":
        q, y = queries_A(Dd, frame)[ids], Dd["Cp"][ids]
    else:
        q, y = [], []
        for i in ids:
            j = rng.choice(np.flatnonzero(~Dd["body"][i]), nq, replace=False)
            q.append(grid_for(Dd, i, frame)[j])
            y.append(Dd["c"][i, j])
        q, y = np.array(q), np.array(y)
    return pts, nrm, q, prompt_for(Dd, task, ids, shift, q.shape[1], frame, prompt_mode), y


def forward(P, pts, nrm, q, v, frozen):
    if frozen:
        with ag.no_grad():
            geo, _ = Mo.encode(P, pts, nrm)
        geo = ag.const(geo.data)
    else:
        geo, _ = Mo.encode(P, pts, nrm)
    return Mo.lin(P, Mo.decode(P, geo, q, v), "head.task")


def evaluate(P, Dd, task, shift, frozen, mu, sd, ids=TEST, keep_pred=0, frame="flow", prompt_mode="normal",
             data=None):
    Dd = get_data(Dd, data)
    errs, preds = [], []
    with ag.no_grad():
        if task == "A":
            for s in range(0, len(ids), 20):
                b = ids[s:s + 20]
                pts, nrm = surf_batch(shapes_for(Dd, task, frame)[b], phase=0.0)
                q, y = queries_A(Dd, frame)[b], Dd["Cp"][b]
                v = prompt_for(Dd, task, b, shift, q.shape[1], frame, prompt_mode)
                out = forward(P, pts, nrm, q, v, frozen).data[..., 0] * sd + mu
                errs += list(np.linalg.norm(out - y, axis=1) / np.linalg.norm(y, axis=1))
                preds += list(out)
        else:
            for i in ids:
                j = np.flatnonzero(~Dd["body"][i])
                pts, nrm = surf_batch(shapes_for(Dd, task, frame)[[i]], phase=0.0)
                q, y = grid_for(Dd, i, frame)[j][None], Dd["c"][i, j]
                v = prompt_for(Dd, task, np.array([i]), shift, len(j), frame, prompt_mode)
                out = forward(P, pts, nrm, q, v, frozen).data[0, :, 0] * sd + mu
                errs.append(np.linalg.norm(out - y) / np.linalg.norm(y))
                preds.append(out)
    return float(np.mean(errs)), np.array(errs), preds[:keep_pred]


def fit(Dd, variant, task, n, seed, shift=0.0, frozen=False, steps=1000, B=8, lr=1e-3, freeze=(),
        ckpt=None, encoder_only=None, frame="flow", prompt_mode="normal", data=None):
    Dd = get_data(Dd, data)
    rng = np.random.default_rng(seed)
    P = Mo.build("S", 100 + seed)                        # fresh encoder + decoder
    if variant != "S" or ckpt is not None:
        ck = np.load(ckpt or f"ckpt/{variant}.npz")
        if encoder_only is None:
            encoder_only = variant == "J"                # J only trains the encoder
        load_dec = (not frozen) and not encoder_only
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
        pts, nrm, q, v, y = train_batch(Dd, task, ids, rng, shift, frame=frame, prompt_mode=prompt_mode)
        out = forward(P, pts, nrm, q, v, frozen)
        loss = ag.mse(out, ((y - mu) / sd)[..., None].astype(np.float32))
        loss.backward()
        opt.step(cosine(it, steps, lr, warm=50))
    return P, loss


def finetune_params(Dd, variant, task, n, seed, **kw):
    return fit(Dd, variant, task, n, seed, **kw)[0]


def finetune(Dd, variant, task, n, seed, shift=0.0, frozen=False, steps=1000, B=8, lr=1e-3, keep_pred=0,
             freeze=(), ckpt=None, encoder_only=None, frame="flow", prompt_mode="normal", data=None):
    t0 = time.time()
    Dd = get_data(Dd, data)
    P, loss = fit(Dd, variant, task, n, seed, shift, frozen, steps, B, lr, freeze,
                  ckpt=ckpt, encoder_only=encoder_only, frame=frame, prompt_mode=prompt_mode)
    mu, sd = norm_stats(Dd, task)
    err, per, preds = evaluate(P, Dd, task, shift, frozen, mu, sd, keep_pred=keep_pred, frame=frame,
                               prompt_mode=prompt_mode)
    rec = dict(variant=variant, task=task, n=n, seed=seed, shift=round(float(np.degrees(shift))),
               frozen=frozen, rel_l2=err, final_train_loss=float(loss.data),
               sec=round(time.time() - t0, 1))
    for k, v, default in [("frame", frame, "flow"), ("prompt_mode", prompt_mode, "normal"), ("ckpt", ckpt, None)]:
        if v != default:                                 # non-default options are recorded
            rec[k] = v
    return rec, per, preds
