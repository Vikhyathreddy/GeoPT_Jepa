"""Backbone + pretext objectives.

Backbone (shared by every variant)
  geometry encoder : 256 surface points -> 32 patch tokens (mini-PointNet) -> 2 self-attn blocks
  query decoder    : query (x, y, prompt velocity v) -> 2 cross-attn blocks over geometry tokens
  heads            : linear readouts (pretext or task)

Pretext variants
  N   native geometry  : predict vector distance at x                       (GeoPT's failing baseline)
  L   GeoPT lifted     : given per-point velocity v, predict vector distance
                         at x0, x1, x2 along the sticking trajectory          (GeoPT, Eq. 6)
  J   Geo-JEPA         : mask 2 arcs of patches, predict EMA-target latents of
                         the masked patches from visible ones                 (I-JEPA recipe on geometry)
  LJ  joint            : L + 0.1 * J                                          (hybrid 1: shared encoder)
  LT  latent transport : JEPA-style "world model" whose *action* is GeoPT's
                         synthetic velocity: from (x0, v) predict the EMA-target
                         latent of the static point x_t it will be carried to,
                         plus an explicit vector-distance anchor at v=0       (hybrid 2)
  LT0 LT without anchor (collapse probe)
"""
import numpy as np
import ag
from ag import linear, layernorm, add, gelu, const

D, H, HID = 64, 4, 128
NP, PS = 32, 8
NF = 6
FF = 2 + 4 * NF
QIN = FF + 3
VSCALE = 2.0  # prompt velocities (|v|<=0.5) are rescaled to O(1)


def ff(p):
    out = [p]
    for k in range(NF):
        out += [np.sin((2 ** k) * np.pi * p), np.cos((2 ** k) * np.pi * p)]
    return np.concatenate(out, -1).astype(np.float32)


# ------------------------------------------------------------------ init
def _lin(P, rng, name, i, o, s=1.0):
    P[name + ".W"] = ag.param(rng.normal(0, s / np.sqrt(i), (i, o)))
    P[name + ".b"] = ag.param(np.zeros(o))


def _ln(P, name):
    P[name + ".g"] = ag.param(np.ones(D))
    P[name + ".b"] = ag.param(np.zeros(D))


def _block(P, rng, name, cross, nblocks):
    s = 1.0 / np.sqrt(2 * nblocks)
    _ln(P, name + ".ln1")
    if cross:
        _ln(P, name + ".lnkv")
    for m in ("q", "k", "v"):
        _lin(P, rng, f"{name}.attn.{m}", D, D)
    _lin(P, rng, name + ".attn.o", D, D, s)
    _ln(P, name + ".ln2")
    _lin(P, rng, name + ".fc1", D, HID)
    _lin(P, rng, name + ".fc2", HID, D, s)


def init_encoder(P, rng):
    _lin(P, rng, "tok.1", 4, D)
    _lin(P, rng, "tok.2", D, D)
    _lin(P, rng, "tok.pos", FF, D)
    for i in range(2):
        _block(P, rng, f"enc.{i}", False, 2)
    _ln(P, "enc.lnf")


def init_decoder(P, rng):
    _lin(P, rng, "q.1", QIN, D)
    _lin(P, rng, "q.2", D, D)
    for i in range(2):
        _block(P, rng, f"dec.{i}", True, 2)
    _ln(P, "dec.lnf")


def init_head(P, rng, name, out):
    _lin(P, rng, name, D, out)


def init_jepa_predictor(P, rng):
    _lin(P, rng, "pred.in", D, D)
    _lin(P, rng, "pred.pos", FF, D)
    P["pred.mask"] = ag.param(rng.normal(0, 0.02, D))
    for i in range(2):
        _block(P, rng, f"pred.{i}", False, 2)
    _ln(P, "pred.lnf")
    _lin(P, rng, "pred.out", D, D)


def init_lt_predictor(P, rng):
    _lin(P, rng, "lt.1", D, HID)
    _lin(P, rng, "lt.o1", HID, D)
    _lin(P, rng, "lt.o2", HID, D)


# ------------------------------------------------------------------ forward
def lin(P, x, n):
    return linear(x, P[n + ".W"], P[n + ".b"])


def ln(P, x, n):
    return layernorm(x, P[n + ".g"], P[n + ".b"])


def mha(P, xq, xkv, n):
    B, Nq, _ = xq.shape
    Nk = xkv.shape[1]
    dh = D // H

    def split(t, m):
        return ag.transpose(ag.reshape(t, (B, m, H, dh)), (0, 2, 1, 3))

    o = ag.attention(split(lin(P, xq, n + ".q"), Nq), split(lin(P, xkv, n + ".k"), Nk),
                     split(lin(P, xkv, n + ".v"), Nk))
    o = ag.reshape(ag.transpose(o, (0, 2, 1, 3)), (B, Nq, D))
    return lin(P, o, n + ".o")


def block(P, x, n, ctx=None):
    h = ln(P, x, n + ".ln1")
    kv = h if ctx is None else ln(P, ctx, n + ".lnkv")
    x = add(x, mha(P, h, kv, n + ".attn"))
    h = ln(P, x, n + ".ln2")
    return add(x, lin(P, gelu(lin(P, h, n + ".fc1")), n + ".fc2"))


def tokens(P, pts, nrm):
    B = pts.shape[0]
    pp = pts.reshape(B, NP, PS, 2)
    cen = pp.mean(2)
    loc = np.concatenate([(pp - cen[:, :, None]) * 4.0, nrm.reshape(B, NP, PS, 2)], -1).astype(np.float32)
    h = lin(P, gelu(lin(P, const(loc), "tok.1")), "tok.2")
    h = ag.mean_axis(h, 2)
    return add(h, lin(P, const(ff(cen)), "tok.pos")), cen


def encode(P, pts, nrm, keep=None):
    x, cen = tokens(P, pts, nrm)
    if keep is not None:
        x = ag.gather_tokens(x, keep)
    for i in range(2):
        x = block(P, x, f"enc.{i}")
    return ln(P, x, "enc.lnf"), cen


def query_features(qpos, qvel):
    sp = np.linalg.norm(qvel, axis=-1, keepdims=True)
    return np.concatenate([ff(qpos), qvel * VSCALE, sp * VSCALE], -1).astype(np.float32)


def decode(P, geo, qpos, qvel):
    q = lin(P, gelu(lin(P, const(query_features(qpos, qvel)), "q.1")), "q.2")
    for i in range(2):
        q = block(P, q, f"dec.{i}", ctx=geo)
    return ln(P, q, "dec.lnf")


def jepa_predict(P, ctx, cen, drop):
    B, Nk, _ = ctx.shape
    Nd = drop.shape[1]
    c = lin(P, ctx, "pred.in")
    mpos = cen[np.arange(B)[:, None], drop]
    m = add(lin(P, const(ff(mpos)), "pred.pos"), P["pred.mask"])
    x = ag.concat([c, m], 1)
    for i in range(2):
        x = block(P, x, f"pred.{i}")
    x = ln(P, x, "pred.lnf")
    x = ag.gather_tokens(x, np.broadcast_to(np.arange(Nk, Nk + Nd), (B, Nd)))
    return lin(P, x, "pred.out")


def sample_masks(rng, B, blk=6):
    keep, drop = [], []
    for _ in range(B):
        s1 = rng.integers(NP)
        s2 = (s1 + blk + rng.integers(0, NP - 2 * blk + 1)) % NP
        d = np.unique(np.concatenate([(s1 + np.arange(blk)) % NP, (s2 + np.arange(blk)) % NP]))
        drop.append(d)
        keep.append(np.setdiff1d(np.arange(NP), d))
    return np.array(keep), np.array(drop)


def plain_ln(x):
    mu = x.mean(-1, keepdims=True)
    return ((x - mu) / np.sqrt(x.var(-1, keepdims=True) + 1e-5)).astype(np.float32)


# ------------------------------------------------------------------ build & losses
def build(variant, seed):
    rng = np.random.default_rng(seed)
    P = {}
    init_encoder(P, rng)
    init_decoder(P, rng)
    if variant in ("N", "LT", "LT0"):
        init_head(P, rng, "head.nat", 2)
    if variant in ("L", "LJ"):
        init_head(P, rng, "head.lift", 6)
    if variant in ("J", "LJ"):
        init_jepa_predictor(P, rng)
    if variant in ("LT", "LT0"):
        init_lt_predictor(P, rng)
    return P


def pretext_loss(variant, P, Pt, batch, rng, lam_j=0.1, lam_lt=0.1):
    """returns (loss T, dict of logged scalars)."""
    pts, nrm, X, v, vd = batch["pts"], batch["nrm"], batch["X"], batch["v"], batch["vd"]
    terms, log = [], {}
    needs_full = variant in ("N", "L", "LJ", "LT", "LT0")
    if needs_full:
        geo, _ = encode(P, pts, nrm)
    if variant == "N":
        out = lin(P, decode(P, geo, X[:, 0], np.zeros_like(v)), "head.nat")
        terms.append(ag.mse(out, vd[:, 0]))
        log["nat"] = float(terms[-1].data)
    if variant in ("L", "LJ"):
        out = lin(P, decode(P, geo, X[:, 0], v), "head.lift")
        tgt = np.concatenate([vd[:, 0], vd[:, 1], vd[:, 2]], -1)
        terms.append(ag.mse(out, tgt))
        log["lift"] = float(terms[-1].data)
    if variant in ("J", "LJ"):
        keep, drop = sample_masks(rng, pts.shape[0])
        ctx, cen = encode(P, pts, nrm, keep)
        pred = jepa_predict(P, ctx, cen, drop)
        with ag.no_grad():
            tg, _ = encode(Pt, pts, nrm)
        tg = plain_ln(tg.data[np.arange(pts.shape[0])[:, None], drop])
        jl = ag.mse(pred, tg)
        terms.append(jl if variant == "J" else ag.scale(jl, lam_j))
        log["jepa"] = float(jl.data)
        log["tgt_std"] = float(tg.std(axis=(0, 1)).mean())
    if variant in ("LT", "LT0"):
        rv = decode(P, geo, X[:, 0], v)
        with ag.no_grad():
            gt, _ = encode(Pt, pts, nrm)
            z = np.zeros_like(v)
            r1 = decode(Pt, gt, X[:, 1], z).data
            r2 = decode(Pt, gt, X[:, 2], z).data
        t1, t2 = plain_ln(r1), plain_ln(r2)
        h = gelu(lin(P, rv, "lt.1"))
        l1 = ag.mse(lin(P, h, "lt.o1"), t1)
        l2 = ag.mse(lin(P, h, "lt.o2"), t2)
        lat = ag.scale(add(l1, l2), 0.5)
        log["latent"] = float(lat.data)
        # collapse probe: spread of the target latent *across points* (per channel std, averaged)
        log["tgt_std"] = float(r2.reshape(-1, D).std(0).mean())
        if variant == "LT":
            r0 = decode(P, geo, X[:, 0], np.zeros_like(v))
            an = ag.mse(lin(P, r0, "head.nat"), vd[:, 0])
            terms.append(an)
            log["anchor"] = float(an.data)
            terms.append(ag.scale(lat, lam_lt))
        else:
            terms.append(lat)
    loss = ag.total(*terms)
    log["loss"] = float(loss.data)
    return loss, log


def ema_update(Pt, P, m):
    for k, t in Pt.items():
        t.data = (m * t.data + (1 - m) * P[k].data).astype(np.float32)


def copy_params(P):
    return {k: const(v.data.copy()) for k, v in P.items()}
