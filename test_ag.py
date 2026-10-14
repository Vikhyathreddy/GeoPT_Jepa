"""Finite-difference gradient checks for every op in ag.py (float64)."""
import numpy as np
import ag

ag.set_dtype(np.float64)
rng = np.random.default_rng(0)


def check(name, build, shapes, idx=None):
    ps = [ag.param(rng.normal(size=s)) for s in shapes]
    R = None

    def f():
        nonlocal R
        out = build(*ps)
        if R is None:
            R = rng.normal(size=out.data.shape)
        return float((out.data * R).sum()), out

    _, out = f()
    loss = _sumall(ag.mul(out, ag.const(R)))
    loss.backward()
    worst = 0.0
    for p in ps:
        g = p.grad
        num = np.zeros_like(p.data)
        it = np.nditer(p.data, flags=["multi_index"])
        for _ in it:
            i = it.multi_index
            old = p.data[i]
            p.data[i] = old + 1e-6
            fp, _ = f()
            p.data[i] = old - 1e-6
            fm, _ = f()
            p.data[i] = old
            num[i] = (fp - fm) / 2e-6
        err = np.abs(g - num).max() / (np.abs(num).max() + 1e-8)
        worst = max(worst, err)
    print(f"{name:14s} max rel err {worst:.2e}  {'OK' if worst < 1e-5 else 'FAIL'}")
    return worst < 1e-5


def _sumall(x):
    # reduce to scalar through supported ops
    while x.data.ndim > 0:
        x = ag.scale(ag.mean_axis(x, 0), x.data.shape[0])
    return x


B, H, Q, P, dh = 2, 2, 3, 4, 5
ok = True
ok &= check("add-bcast", lambda a, b: ag.add(a, b), [(2, 3, 4), (4,)])
ok &= check("sub", lambda a, b: ag.sub(a, b), [(3, 4), (3, 4)])
ok &= check("mul-bcast", lambda a, b: ag.mul(a, b), [(2, 3, 4), (1, 3, 1)])
ok &= check("scale", lambda a: ag.scale(a, 0.7), [(3, 2)])
ok &= check("gelu", lambda a: ag.gelu(a), [(4, 5)])
ok &= check("linear", lambda x, W, b: ag.linear(x, W, b), [(2, 3, 4), (4, 5), (5,)])
ok &= check("layernorm", lambda x, g, b: ag.layernorm(x, g, b), [(2, 3, 6), (6,), (6,)])
ok &= check("layernorm-np", lambda x: ag.layernorm(x), [(3, 6)])
ok &= check("attention", lambda q, k, v: ag.attention(q, k, v), [(B, H, Q, dh), (B, H, P, dh), (B, H, P, dh)])
ok &= check("reshape", lambda x: ag.reshape(x, (6, 4)), [(2, 3, 4)])
ok &= check("transpose", lambda x: ag.transpose(x, (0, 2, 1, 3)), [(2, 3, 4, 2)])
ok &= check("concat", lambda a, b: ag.concat([a, b], 1), [(2, 3, 4), (2, 2, 4)])
idx = np.array([[0, 2], [3, 1]])
ok &= check("gather", lambda x: ag.gather_tokens(x, idx), [(2, 4, 3)])
ok &= check("mean_axis", lambda x: ag.mean_axis(x, 2), [(2, 3, 4, 2)])
tgt = rng.normal(size=(3, 4))
ok &= check("mse", lambda x: ag.mse(x, tgt), [(3, 4)])
ok &= check("logsumexp", lambda x: ag.logsumexp(x, 1), [(2, 5, 3)])
ok &= check("logsumexp-last", lambda x: ag.logsumexp(x, -1), [(3, 4)])
ok &= check("softplus", lambda x: ag.softplus(ag.scale(x, 3.0)), [(4, 5)])
# a composite: two-layer net with attention, to exercise graph bookkeeping (shared nodes)
def comp(x, W1, W2):
    h = ag.gelu(ag.linear(x, W1))
    q = ag.reshape(h, (1, 1, 3, 4))
    o = ag.attention(q, q, q)
    o = ag.reshape(o, (3, 4))
    return ag.add(ag.linear(o, W2), ag.linear(h, W2))
ok &= check("composite", comp, [(3, 2), (2, 4), (4, 2)])
print("ALL OK" if ok else "SOME FAILED")
