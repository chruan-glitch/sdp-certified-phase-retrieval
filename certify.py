"""Phase 2: SDP certification of  max_{||delta||_2<=eps} ||f(y+delta) - sigma0 x||_2^2.

Formulation
-----------
* First-layer elimination: W0 = U S V^T (thin SVD), W0 delta ranges over {G w : ||w||_2 <= eps},
  G = U S (h x r).  So  zhat_0 = c + G w  with c = W0 y + b0.  (No inverse of W0 W0^T needed.)
* Every quantity is an AFFINE function of  qt = (1, w, z'_1, ..., z'_K)  where the z' are the
  post-activations of the UNSTABLE neurons only.  Stable neurons are substituted out exactly:
      u <= 0 : z' = 0          l >= 0 : z' = zhat (affine)          l < 0 < u : new variable z'.
  Hence with no unstable neuron the relaxation is an exact trust-region problem (S-lemma).
* M ~ qt qt^T,  M >= 0,  M[0,0] = 1.  Each constraint is  qt^T K qt (<=,=) 0  with
  K = sym(a b^T), i.e.  <K, M> = a^T M b.   Per unstable neuron (zhat = s^T qt, z' = e^T qt):
      z' >= 0,  z' >= zhat                                   (linear)
      z'(z' - zhat) = 0                                      (ReLU complementarity)
      (zhat - l)(zhat - u) <= 0                              (bounds)
      z' <= u/(u-l) (zhat - l)                               (triangle)
      [rlt=True]  z'(u - zhat) >= 0,  (z' - zhat)(zhat - l) >= 0   (products of valid inequalities)
  plus  ||w||^2 <= eps^2.  Objective  <Q, M>,  Q = R^T R,  R = F - sigma0 x e0^T  (F: output map).
  The optimal value is an upper bound on the true maximum.

Rigorous dual evaluation
------------------------
Solvers return multipliers only up to tolerance, so the raw optimum is NOT a certificate.  From the
multipliers (lam >= 0 on inequalities, mu free on equalities) we rebuild the dual slack
      Y = sum_j lam_j K_j + sum_k mu_k K_k + lam_ball K_ball - Q + t (I - B e0 e0^T)
(t >= 0 multiplies the valid constraint ||qt||^2 <= B, B = 1 + eps^2 + sum_unstable u_i^2) and set
the free multiplier nu of M[0,0]=1 EXACTLY by a Schur complement:  Z = Y + nu e0 e0^T >= 0.
Weak duality then gives  max <= nu  for any lam >= 0, any mu, any t >= 0 -- the solver only
supplies a good guess; there is no separate "dual equality" left to violate.  lam is clipped at 0
and lam_min(Z) is checked numerically (a residual shortfall kappa is paid for with kappa*B).
The result is a certified bound up to floating-point arithmetic.
"""
import time
import numpy as np
import cvxpy as cp
from .bounds import best_sigma0, forward
from .phaselift import _solver_list

# Clarabel keeps a DENSE (svec_dim x svec_dim) scaling block for the PSD cone: 8 * svec_dim^2 bytes
# (N=256 -> svec_dim=32896 -> 8.7 GB; a failed allocation aborts the whole Python process, it cannot be caught).
# Above this size Clarabel is skipped and the next solver (SCS) is used.  16000 -> about 2 GB, i.e. N <= 178.
CLARABEL_MAX_SVEC = 16000


def _build(weights, ls, us, y, x, eps, sigma0, rlt, reduce=True):
    W0, b0 = weights[0]
    U, sv, _ = np.linalg.svd(W0, full_matrices=False)
    keep = sv > 1e-12 * sv[0]                      # drop exactly-null directions only
    G = U[:, keep] * sv[keep]
    r = G.shape[1]
    L = len(weights)
    n_unst = sum(int(np.sum((l < 0) & (u > 0))) for l, u in zip(ls, us))
    N = 1 + r + n_unst
    e0 = np.zeros(N); e0[0] = 1.0
    S = np.zeros((W0.shape[0], N))
    S[:, 0] = W0 @ y + b0
    S[:, 1:1 + r] = G
    A_in, B_in, A_eq, B_eq = [], [], [], []
    ge = lambda a, b: (A_in.append(-a), B_in.append(b))      # (a.q)(b.q) >= 0
    le = lambda a, b: (A_in.append(a), B_in.append(b))       # (a.q)(b.q) <= 0
    eq = lambda a, b: (A_eq.append(a), B_eq.append(b))       # (a.q)(b.q) = 0
    nxt = 1 + r
    Bnorm = 1.0 + eps ** 2
    for l in range(L - 1):
        lo, hi = ls[l], us[l]
        Z = np.zeros((S.shape[0], N))
        for i in range(S.shape[0]):
            if hi[i] <= 0:
                continue
            if lo[i] >= 0:
                Z[i] = S[i]
                continue
            e = np.zeros(N); e[nxt] = 1.0; nxt += 1
            s, li, ui = S[i], lo[i], hi[i]
            ge(e, e0)                                  # z' >= 0
            ge(e - s, e0)                              # z' >= zhat
            eq(e, e - s)                               # z'(z' - zhat) = 0
            le(s - li * e0, s - ui * e0)               # (zhat-l)(zhat-u) <= 0
            ge(ui / (ui - li) * (s - li * e0) - e, e0)  # triangle
            if rlt:
                ge(e, ui * e0 - s)                     # z'(u - zhat) >= 0
                ge(e - s, s - li * e0)                 # (z'-zhat)(zhat-l) >= 0
            Bnorm += ui ** 2
            Z[i] = e
        Wn, bn = weights[l + 1]
        S = Wn @ Z
        S[:, 0] += bn
    R = S.copy()
    R[:, 0] -= sigma0 * x
    arr = lambda v: np.array(v) if len(v) else np.zeros((0, N))
    A_in, B_in, A_eq, B_eq = arr(A_in), arr(B_in), arr(A_eq), arr(B_eq)
    r_full = r
    if reduce:
        # EXACT dimension reduction of the w-block.  w enters only through (i) the ball ||w|| <= eps (rotation invariant)
        # and (ii) the w-parts of the constraint vectors and of the output map R.  Let P be an orthonormal basis of the
        # span of all those w-parts; w = P w' + w_perp, nothing depends on w_perp, and ||w'|| <= ||w|| <= eps.
        # So substituting w = P w' (r -> r_eff <= #unstable + n_out) gives the same optimum with a much smaller PSD cone.
        Wm = np.vstack([m_[:, 1:1 + r] for m_ in (A_in, B_in, A_eq, B_eq, R) if len(m_)])
        _, sv, Vt = np.linalg.svd(Wm, full_matrices=False)
        keep = sv > 1e-11 * max(float(sv[0]), 1e-300) if sv.size and sv[0] > 0 else np.array([True])
        P = Vt[keep].T if sv.size and sv[0] > 0 else np.eye(r)[:, :1]
        if P.shape[1] < r:
            proj = lambda m_: m_ if not len(m_) else np.hstack([m_[:, :1], m_[:, 1:1 + r] @ P, m_[:, 1 + r:]])
            A_in, B_in, A_eq, B_eq, R = proj(A_in), proj(B_in), proj(A_eq), proj(B_eq), proj(R)
            r = P.shape[1]
            N = 1 + r + n_unst
    return dict(N=N, r=r, r_full=r_full, Q=R.T @ R, A_in=A_in, B_in=B_in, A_eq=A_eq, B_eq=B_eq,
                Bnorm=Bnorm, n_unstable=n_unst)


def _symsum(A, B, w):
    if len(A) == 0:
        return 0.0
    S = A.T @ (w[:, None] * B)
    return 0.5 * (S + S.T)


def certify_full(weights, bounds, y, x, eps, sigma0=None, rlt=False, solver=None, verbose=False, reduce=True):
    """Return a dict with the certified `bound` and diagnostics.  bounds = (ls, us)."""
    ls, us = bounds
    y = np.asarray(y, dtype=np.float64).ravel()
    x = np.asarray(x, dtype=np.float64).ravel()
    if sigma0 is None:
        sigma0 = best_sigma0(weights, y, x)
    if eps <= 0:
        clean = float(np.sum((forward(weights, y) - sigma0 * x) ** 2))
        return dict(bound=clean, primal_value=clean, gap=0.0, status="eps=0", sigma0=sigma0)
    d = _build(weights, ls, us, y, x, eps, sigma0, rlt, reduce)
    N, r, Q = d["N"], d["r"], d["Q"]
    M = cp.Variable((N, N), PSD=True)
    ball = cp.trace(M[1:1 + r, 1:1 + r]) - eps ** 2 * M[0, 0] <= 0
    cons = [M[0, 0] == 1, ball]
    c_in = c_eq = None
    if len(d["A_in"]):
        c_in = cp.sum(cp.multiply(d["A_in"] @ M, d["B_in"]), axis=1) <= 0
        cons.append(c_in)
    if len(d["A_eq"]):
        c_eq = cp.sum(cp.multiply(d["A_eq"] @ M, d["B_eq"]), axis=1) == 0
        cons.append(c_eq)
    prob = cp.Problem(cp.Minimize(-cp.sum(cp.multiply(Q, M))), cons)
    t0 = time.perf_counter()
    err = None
    for s in _solver_list(solver):
        if s == "CLARABEL" and N * (N + 1) // 2 > CLARABEL_MAX_SVEC:
            continue                                   # would need several GB -> process crash; see top of file
        try:
            prob.solve(solver=s, verbose=verbose)
            if prob.status in ("optimal", "optimal_inaccurate"):
                break
        except cp.error.SolverError as ex:
            err = ex
    solve_time = time.perf_counter() - t0
    info = dict(status=prob.status, solver=s if 'optimal' in str(prob.status) else None, time=solve_time,
                N=N, r=r, n_unstable=d["n_unstable"], rlt=rlt, sigma0=sigma0)
    if prob.status not in ("optimal", "optimal_inaccurate") or ball.dual_value is None:
        return dict(bound=np.inf, primal_value=np.nan, gap=np.inf, **info)

    # ---- rigorous dual evaluation from the multipliers (see module docstring)
    lam_ball = max(float(ball.dual_value), 0.0)
    lam = np.maximum(np.asarray(c_in.dual_value).ravel(), 0.0) if c_in is not None else np.zeros(0)
    mu = np.asarray(c_eq.dual_value).ravel() if c_eq is not None else np.zeros(0)
    K_ball = np.zeros((N, N)); idx = np.arange(1, 1 + r)
    K_ball[idx, idx] = 1.0; K_ball[0, 0] = -eps ** 2
    Y0 = -Q + lam_ball * K_ball + _symsum(d["A_in"], d["B_in"], lam) + _symsum(d["A_eq"], d["B_eq"], mu)
    KB = np.eye(N); KB[0, 0] = 1.0 - d["Bnorm"]
    lamD = np.linalg.eigvalsh(Y0[1:, 1:])[0]
    best = None
    for delta in (1e-10, 1e-9, 1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3):
        t = max(0.0, delta - lamD)
        Y = Y0 + t * KB
        D, b, a = Y[1:, 1:], Y[1:, 0], Y[0, 0]
        try:
            np.linalg.cholesky(D)
        except np.linalg.LinAlgError:
            continue
        nu = float(b @ np.linalg.solve(D, b) - a)       # Schur complement: Z = Y + nu e0 e0^T >= 0
        Z = Y.copy(); Z[0, 0] += nu
        lmZ = float(np.linalg.eigvalsh(Z)[0])
        kappa = max(0.0, -lmZ)                           # pay any numerical shortfall: Z + kappa I >= 0
        bound = nu + kappa * d["Bnorm"]
        if best is None or bound < best["bound"]:
            best = dict(bound=bound, t=t, delta=delta, lam_min_Z=lmZ)
    if best is None:
        return dict(bound=np.inf, primal_value=-float(prob.value), gap=np.inf, **info)
    pv = -float(prob.value)
    return dict(bound=best["bound"], primal_value=pv, gap=best["bound"] - pv, t=best["t"],
                delta=best["delta"], lam_min_Z=best["lam_min_Z"], **info)


def certify(weights, bounds, y, x, eps, **kw):
    """Interface used by the experiments: certified SDP upper bound (float)."""
    return certify_full(weights, bounds, y, x, eps, **kw)["bound"]


def certify_both_signs(weights, bounds, y, x, eps, **kw):
    """Solve for sigma = +1 and -1 and keep the smaller bound (still valid: max_d min_s <= min_s max_d)."""
    rs = [certify_full(weights, bounds, y, x, eps, sigma0=s, **kw) for s in (1.0, -1.0)]
    return min(rs, key=lambda r: r["bound"]), rs
