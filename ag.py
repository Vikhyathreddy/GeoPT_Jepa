"""Minimal reverse-mode autograd on NumPy (no torch/jax available in this sandbox).

Only the ops needed for small transformers: fused linear, GELU, LayerNorm,
multi-head attention core, reshape/transpose/concat/gather, reductions, MSE.
Every op is checked against finite differences in test_ag.py.
"""
import numpy as np

DT = np.float32
_GRAD = [True]


def set_dtype(dt):
    global DT
    DT = dt


class no_grad:
    def __enter__(self):
        self.prev = _GRAD[0]
        _GRAD[0] = False

    def __exit__(self, *a):
        _GRAD[0] = self.prev


class T:
    __slots__ = ("data", "grad", "prev", "back", "req")

    def __init__(self, data, prev=()):
        self.data = data
        self.grad = None
        self.back = None
        if _GRAD[0] and any(p.req for p in prev):
            self.req = True
            self.prev = prev
        else:
            self.req = False
            self.prev = ()

    @property
    def shape(self):
        return self.data.shape

    def backward(self):
        topo, seen, stack = [], set(), [(self, False)]
        while stack:
            node, done = stack.pop()
            if done:
                topo.append(node)
                continue
            if id(node) in seen:
                continue
            seen.add(id(node))
            stack.append((node, True))
            for p in node.prev:
                if p.req and id(p) not in seen:
                    stack.append((p, False))
        self.grad = np.ones_like(self.data)
        for node in reversed(topo):
            if node.back is not None and node.grad is not None:
                node.back(node.grad)
                if node.prev:  # free intermediate graph
                    node.back = None
                    node.prev = ()


def param(arr):
    p = T(np.asarray(arr, DT))
    p.req = True
    return p


def const(arr):
    return T(np.asarray(arr, DT))


def _acc(p, g):
    if p.req:
        p.grad = g if p.grad is None else p.grad + g


def _unb(g, shape):
    while g.ndim > len(shape):
        g = g.sum(0)
    for i, s in enumerate(shape):
        if s == 1 and g.shape[i] != 1:
            g = g.sum(i, keepdims=True)
    return g


def _mk(data, prev, back):
    out = T(data, prev)
    if out.req:
        out.back = back
    return out


# ---------------------------------------------------------------- elementwise
def add(a, b):
    def back(g):
        _acc(a, _unb(g, a.data.shape))
        _acc(b, _unb(g, b.data.shape))
    return _mk(a.data + b.data, (a, b), back)


def sub(a, b):
    def back(g):
        _acc(a, _unb(g, a.data.shape))
        _acc(b, _unb(-g, b.data.shape))
    return _mk(a.data - b.data, (a, b), back)


def mul(a, b):
    def back(g):
        _acc(a, _unb(g * b.data, a.data.shape))
        _acc(b, _unb(g * a.data, b.data.shape))
    return _mk(a.data * b.data, (a, b), back)


def scale(a, c):
    def back(g):
        _acc(a, g * DT(c))
    return _mk(a.data * DT(c), (a,), back)


_C = np.sqrt(2.0 / np.pi)


def gelu(x):
    xd = x.data
    c, k = DT(_C), DT(0.044715 * _C)
    x2 = xd * xd                      # NB: xd**3 is ~70x slower in float32
    th = np.tanh(xd * (c + k * x2))
    y = DT(0.5) * xd * (DT(1.0) + th)

    def back(g):
        d = DT(0.5) * (DT(1.0) + th) + DT(0.5) * xd * (DT(1.0) - th * th) * (c + DT(3.0) * k * x2)
        _acc(x, g * d)
    return _mk(y, (x,), back)


# ---------------------------------------------------------------- linear algebra
def linear(x, W, b=None):
    i, o = W.data.shape
    x2 = x.data.reshape(-1, i)
    y = x2 @ W.data
    if b is not None:
        y = y + b.data
    y = y.reshape(x.data.shape[:-1] + (o,))

    def back(g):
        g2 = g.reshape(-1, o)
        if x.req:
            _acc(x, (g2 @ W.data.T).reshape(x.data.shape))
        _acc(W, x2.T @ g2)
        if b is not None:
            _acc(b, g2.sum(0))
    prev = (x, W) if b is None else (x, W, b)
    return _mk(y, prev, back)


def layernorm(x, gamma=None, beta=None, eps=1e-5):
    xd = x.data
    mu = xd.mean(-1, keepdims=True)
    xc = xd - mu
    var = (xc * xc).mean(-1, keepdims=True)
    rstd = 1.0 / np.sqrt(var + eps)
    xh = xc * rstd
    y = xh
    if gamma is not None:
        y = y * gamma.data + beta.data

    def back(g):
        gxh = g * gamma.data if gamma is not None else g
        if x.req:
            gx = rstd * (gxh - gxh.mean(-1, keepdims=True) - xh * (gxh * xh).mean(-1, keepdims=True))
            _acc(x, gx)
        if gamma is not None:
            n = xd.shape[-1]
            _acc(gamma, (g * xh).reshape(-1, n).sum(0))
            _acc(beta, g.reshape(-1, n).sum(0))
    prev = (x,) if gamma is None else (x, gamma, beta)
    return _mk(y.astype(DT), prev, back)


def attention(q, k, v):
    """q (B,H,Q,dh), k/v (B,H,P,dh) -> (B,H,Q,dh); softmax(q k^T / sqrt(dh)) v."""
    sc = DT(1.0 / np.sqrt(q.data.shape[-1]))
    s = (q.data @ k.data.swapaxes(-1, -2)) * sc
    s = s - s.max(-1, keepdims=True)
    a = np.exp(s)
    a /= a.sum(-1, keepdims=True)
    o = a @ v.data

    def back(g):
        ga = g @ v.data.swapaxes(-1, -2)
        _acc(v, a.swapaxes(-1, -2) @ g)
        gs = a * (ga - (ga * a).sum(-1, keepdims=True))
        _acc(q, (gs @ k.data) * sc)
        _acc(k, (gs.swapaxes(-1, -2) @ q.data) * sc)
    return _mk(o, (q, k, v), back)


# ---------------------------------------------------------------- shape ops
def reshape(x, shape):
    def back(g):
        _acc(x, g.reshape(x.data.shape))
    return _mk(x.data.reshape(shape), (x,), back)


def transpose(x, axes):
    inv = np.argsort(axes)

    def back(g):
        _acc(x, g.transpose(inv))
    return _mk(x.data.transpose(axes), (x,), back)


def concat(xs, axis):
    sizes = [x.data.shape[axis] for x in xs]
    cuts = np.cumsum(sizes)[:-1]

    def back(g):
        for x, gi in zip(xs, np.split(g, cuts, axis=axis)):
            _acc(x, gi)
    return _mk(np.concatenate([x.data for x in xs], axis=axis), tuple(xs), back)


def gather_tokens(x, idx):
    """x (B,N,d), idx (B,K) int -> (B,K,d)."""
    bi = np.arange(x.data.shape[0])[:, None]

    def back(g):
        gx = np.zeros_like(x.data)
        np.add.at(gx, (bi, idx), g)
        _acc(x, gx)
    return _mk(x.data[bi, idx], (x,), back)


def mean_axis(x, axis):
    n = x.data.shape[axis]

    def back(g):
        _acc(x, np.broadcast_to(np.expand_dims(g, axis) / DT(n), x.data.shape).copy())
    return _mk(x.data.mean(axis), (x,), back)


# ---------------------------------------------------------------- losses
def mse(pred, target):
    """mean squared error against a constant (stop-gradient) target array."""
    diff = pred.data - target
    n = diff.size

    def back(g):
        _acc(pred, (2.0 / n) * diff * g)
    return _mk(np.asarray((diff * diff).mean(), DT), (pred,), back)


def logsumexp(x, axis=-1):
    """log(sum(exp(x))) along `axis` (the axis is removed)."""
    m = x.data.max(axis, keepdims=True)
    e = np.exp(x.data - m)
    s = e.sum(axis, keepdims=True)
    y = (np.log(s) + m).squeeze(axis)

    def back(g):
        _acc(x, np.expand_dims(g, axis) * (e / s))
    return _mk(y.astype(DT), (x,), back)


def softplus(x):
    """log(1 + exp(x)), computed stably."""
    xd = x.data

    def back(g):
        _acc(x, g / (DT(1.0) + np.exp(-xd)))
    return _mk(np.logaddexp(DT(0.0), xd).astype(DT), (x,), back)


def total(*terms):
    out = terms[0]
    for t in terms[1:]:
        out = add(out, t)
    return out


# ---------------------------------------------------------------- optimiser
class AdamW:
    def __init__(self, params, lr=1e-3, b1=0.9, b2=0.999, eps=1e-8, wd=0.01, clip=1.0):
        self.p = params  # dict name -> T
        self.lr, self.b1, self.b2, self.eps, self.wd, self.clip = lr, b1, b2, eps, wd, clip
        self.m = {k: np.zeros_like(v.data) for k, v in params.items()}
        self.v = {k: np.zeros_like(v.data) for k, v in params.items()}
        self.t = 0

    def step(self, lr=None):
        lr = self.lr if lr is None else lr
        self.t += 1
        gs = {k: p.grad for k, p in self.p.items() if p.grad is not None}
        norm = np.sqrt(sum(float((g * g).sum()) for g in gs.values()))
        c = min(1.0, self.clip / (norm + 1e-12)) if self.clip else 1.0
        bc1, bc2 = 1 - self.b1 ** self.t, 1 - self.b2 ** self.t
        for k, g in gs.items():
            p = self.p[k]
            g = g * c
            self.m[k] = self.b1 * self.m[k] + (1 - self.b1) * g
            self.v[k] = self.b2 * self.v[k] + (1 - self.b2) * g * g
            upd = (self.m[k] / bc1) / (np.sqrt(self.v[k] / bc2) + self.eps)
            if p.data.ndim >= 2 and self.wd:
                upd = upd + self.wd * p.data
            p.data = (p.data - lr * upd).astype(DT)
            p.grad = None
        return norm
