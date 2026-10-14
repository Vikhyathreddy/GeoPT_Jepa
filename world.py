"""2D toy physics world.

* random smooth bodies (star-shaped radial curves, stretched + rotated)
* GeoPT pretext: vector-distance features along synthetic straight-line
  trajectories that stick when they hit the body (paper Eq. 4-5)
* downstream task A (elliptic, global): potential-flow surface pressure Cp,
  constant-strength source-panel method
* downstream task B (transport-like, directional): steady advection-diffusion
  plume emitted by the body into the potential-flow velocity field
"""
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from matplotlib.path import Path

NV = 256          # polygon vertices / panels
NS = 256          # surface samples fed to the model (32 patches x 8)
GRID = 64         # plume grid
KAPPA = 0.01      # plume diffusivity


# ----------------------------------------------------------------- geometry
def make_shape(rng):
    th = np.linspace(0, 2 * np.pi, NV, endpoint=False)
    r = np.ones(NV)
    for k in range(2, 6):
        r += rng.uniform(-1, 1) * 0.18 / k ** 0.6 * np.cos(k * th + rng.uniform(0, 2 * np.pi))
    r = np.maximum(r, 0.45)
    p = np.stack([r * np.cos(th), r * np.sin(th)], 1)
    p[:, 0] *= rng.uniform(1.0, 2.2)                     # elongation
    p *= rng.uniform(0.32, 0.5) / np.abs(p).max()        # size
    b = rng.uniform(0, 2 * np.pi)                        # rotation
    R = np.array([[np.cos(b), -np.sin(b)], [np.sin(b), np.cos(b)]])
    p = p @ R.T + rng.uniform(-0.08, 0.08, 2)
    return p  # CCW polygon (NV,2)


def edges(V):
    A = V
    B = np.roll(V, -1, 0)
    e = B - A
    L = np.linalg.norm(e, axis=1)
    t = e / L[:, None]
    n = np.stack([t[:, 1], -t[:, 0]], 1)  # outward for CCW
    return A, B, L, t, n


def inside(V, X):
    return Path(V).contains_points(X)


def resample_surface(V, phase=0.0, ns=NS):
    """arc-length resampling with a phase offset; returns points and outward normals."""
    A, B, L, t, n = edges(V)
    s = np.concatenate([[0], np.cumsum(L)])
    per = s[-1]
    q = (np.arange(ns) + phase) / ns * per
    j = np.clip(np.searchsorted(s, q, side="right") - 1, 0, len(L) - 1)
    u = (q - s[j]) / L[j]
    pts = A[j] + u[:, None] * (B[j] - A[j])
    # smooth normals: average of neighbouring edge normals
    nn = n[j] + n[(j + 1) % len(L)] * u[:, None] + n[(j - 1) % len(L)] * (1 - u[:, None])
    nn /= np.linalg.norm(nn, axis=1, keepdims=True)
    return pts, nn


def vector_distance(V, X):
    """vector from X to its closest boundary point (Faugeras-Gomes vector distance)."""
    A, B, L, _, _ = edges(V)
    e = B - A                                    # (E,2)
    w = X[:, None, :] - A[None]                  # (Q,E,2)
    tt = np.clip((w * e[None]).sum(-1) / (L ** 2)[None], 0, 1)
    C = A[None] + tt[..., None] * e[None]        # (Q,E,2)
    d2 = ((X[:, None, :] - C) ** 2).sum(-1)
    k = d2.argmin(1)
    return C[np.arange(len(X)), k] - X


def transport_step(V, X, v, stuck):
    """x <- x + v, but stop at the first boundary crossing and stick there."""
    A, B, _, _, _ = edges(V)
    e = B - A
    w = A[None] - X[:, None, :]                  # (Q,E,2)
    rxe = v[:, 0:1] * e[None, :, 1] - v[:, 1:2] * e[None, :, 0]
    ok = np.abs(rxe) > 1e-12
    rxe_s = np.where(ok, rxe, 1.0)
    s = (w[..., 0] * e[None, :, 1] - w[..., 1] * e[None, :, 0]) / rxe_s
    u = (w[..., 0] * v[:, 1:2] - w[..., 1] * v[:, 0:1]) / rxe_s
    hit = ok & (s > 1e-9) & (s <= 1) & (u >= 0) & (u <= 1)
    s = np.where(hit, s, np.inf).min(1)
    newhit = np.isfinite(s) & ~stuck
    Xn = X.copy()
    free = ~stuck & ~newhit
    Xn[free] = X[free] + v[free]
    Xn[newhit] = X[newhit] + s[newhit, None] * v[newhit]
    return Xn, stuck | newhit


def pretext_samples(V, rng, n_dyn=8, qv=160, qs=32, vmax=0.5, tau=2):
    """GeoPT lifted-pretext samples for one geometry: n_dyn random velocity fields."""
    out = []
    A, B, L, t, n = edges(V)
    for _ in range(n_dyn):
        cand = rng.uniform(-1, 1, (qv * 3, 2))
        cand = cand[~inside(V, cand)][:qv]
        j = rng.choice(len(L), qs, p=L / L.sum())
        u = rng.uniform(0, 1, qs)
        surf = A[j] + u[:, None] * (B[j] - A[j]) + 1e-3 * n[j]
        X0 = np.concatenate([cand, surf])
        rr = vmax * np.sqrt(rng.uniform(0, 1, len(X0)))
        ang = rng.uniform(0, 2 * np.pi, len(X0))
        v = np.stack([rr * np.cos(ang), rr * np.sin(ang)], 1)   # i.i.d. per point
        xs, vds = [X0], [vector_distance(V, X0)]
        stuck = np.zeros(len(X0), bool)
        X = X0
        for _ in range(tau):
            X, stuck = transport_step(V, X, v, stuck)
            xs.append(X)
            vds.append(vector_distance(V, X))
        out.append((np.stack(xs, 0), v, np.stack(vds, 0)))
    return out


# ----------------------------------------------------------------- task A: panel method
def _source_influence(X, A, t, n, L):
    d = X[:, None, :] - A[None]
    xi = (d * t[None]).sum(-1)
    eta = (d * n[None]).sum(-1)
    u_xi = np.log((xi ** 2 + eta ** 2) / ((xi - L[None]) ** 2 + eta ** 2)) / (4 * np.pi)
    u_eta = (np.arctan2(eta, xi - L[None]) - np.arctan2(eta, xi)) / (2 * np.pi)
    ux = u_xi * t[None, :, 0] + u_eta * n[None, :, 0]
    uy = u_xi * t[None, :, 1] + u_eta * n[None, :, 1]
    return ux, uy


def panel_solve(V, alpha, U=1.0):
    A, B, L, t, n = edges(V)
    M = 0.5 * (A + B)
    ux, uy = _source_influence(M, A, t, n, L)
    idx = np.arange(len(L))
    ux[idx, idx] = 0.5 * n[:, 0]   # self-influence: sigma/2 along the outward normal
    uy[idx, idx] = 0.5 * n[:, 1]
    Uinf = U * np.array([np.cos(alpha), np.sin(alpha)])
    An = ux * n[:, 0:1] + uy * n[:, 1:2]
    sigma = np.linalg.solve(An, -(n @ Uinf))
    At = ux * t[:, 0:1] + uy * t[:, 1:2]
    np.fill_diagonal(At, 0.0)
    Vt = t @ Uinf + At @ sigma
    Cp = 1 - (Vt / U) ** 2
    return M, Cp, sigma


def panel_velocity(V, sigma, alpha, X, U=1.0):
    A, B, L, t, n = edges(V)
    ux, uy = _source_influence(X, A, t, n, L)
    return np.stack([U * np.cos(alpha) + ux @ sigma, U * np.sin(alpha) + uy @ sigma], 1)


# ----------------------------------------------------------------- task B: plume
def grid_centers(N=GRID):
    h = 2.0 / N
    c = -1 + h / 2 + h * np.arange(N)
    XX, YY = np.meshgrid(c, c, indexing="ij")
    return np.stack([XX.ravel(), YY.ravel()], 1), h


def plume_solve(V, alpha, U, N=GRID, kappa=KAPPA):
    """steady u.grad(c) = kappa lap(c); c=1 in body, c=0 at inflow, zero-gradient outflow."""
    X, h = grid_centers(N)
    body = inside(V, X)
    _, _, sigma = panel_solve(V, alpha, U)
    vel = np.zeros_like(X)
    vel[~body] = panel_velocity(V, sigma, alpha, X[~body], U)
    unk = np.flatnonzero(~body)
    pos = -np.ones(N * N, int)
    pos[unk] = np.arange(len(unk))
    rows, cols, vals = [], [], []
    rhs = np.zeros(len(unk))
    I, J = np.unravel_index(unk, (N, N))
    ux, uy = vel[unk, 0], vel[unk, 1]
    aP = 4 * kappa + (np.abs(ux) + np.abs(uy)) * h
    nb = [  # (di, dj, coefficient, outflow-if)
        (1, 0, -kappa + np.minimum(ux, 0) * h, ux > 0),
        (-1, 0, -kappa - np.maximum(ux, 0) * h, ux < 0),
        (0, 1, -kappa + np.minimum(uy, 0) * h, uy > 0),
        (0, -1, -kappa - np.maximum(uy, 0) * h, uy < 0),
    ]
    for di, dj, a, outflow in nb:
        In, Jn = I + di, J + dj
        out = (In < 0) | (In >= N) | (Jn < 0) | (Jn >= N)
        aP = aP + np.where(out & outflow, a, 0.0)             # zero-gradient ghost
        lin = np.ravel_multi_index((np.clip(In, 0, N - 1), np.clip(Jn, 0, N - 1)), (N, N))
        isbody = ~out & body[lin]
        rhs -= np.where(isbody, a, 0.0)                        # c=1 inside body
        m = ~out & ~body[lin]
        rows.append(np.arange(len(unk))[m])
        cols.append(pos[lin[m]])
        vals.append(a[m])
    rows.append(np.arange(len(unk)))
    cols.append(np.arange(len(unk)))
    vals.append(aP)
    Amat = sp.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
                         shape=(len(unk), len(unk)))
    c = spla.spsolve(Amat, rhs)
    full = np.ones(N * N)
    full[unk] = c
    return X, body, full


# ----------------------------------------------------------------- datasets
def build_pretrain(n_shapes, n_dyn, seed):
    rng = np.random.default_rng(seed)
    shapes, X, Vv, VD, sid = [], [], [], [], []
    for i in range(n_shapes):
        V = make_shape(rng)
        shapes.append(V)
        for xs, v, vds in pretext_samples(V, rng, n_dyn=n_dyn):
            X.append(xs)
            Vv.append(v)
            VD.append(vds)
            sid.append(i)
    return dict(shapes=np.array(shapes, np.float32), X=np.array(X, np.float32),
                v=np.array(Vv, np.float32), vd=np.array(VD, np.float32), sid=np.array(sid))


def build_downstream(n_shapes, seed):
    rng = np.random.default_rng(seed)
    shapes, aA, CpM, Cp, aB, UB, C, Body = [], [], [], [], [], [], [], []
    for i in range(n_shapes):
        V = make_shape(rng)
        a = rng.uniform(0, 2 * np.pi)
        M, cp, _ = panel_solve(V, a)
        b = rng.uniform(0, 2 * np.pi)
        U = rng.uniform(0.5, 1.5)
        X, body, c = plume_solve(V, b, U)
        shapes.append(V)
        aA.append(a)
        CpM.append(M[::2])
        Cp.append(np.sqrt(np.maximum(1 - cp[::2], 0)))   # surface speed q = |Vt|/U
        aB.append(b)
        UB.append(U)
        C.append(c)
        Body.append(body)
    f = np.float32
    return dict(shapes=np.array(shapes, f), alphaA=np.array(aA, f), ptsA=np.array(CpM, f),
                Cp=np.array(Cp, f), alphaB=np.array(aB, f), UB=np.array(UB, f),
                c=np.array(C, f), body=np.array(Body), grid=grid_centers()[0].astype(f))
