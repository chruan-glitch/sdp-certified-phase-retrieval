"""PhaseLift baseline (trace minimisation) in CVXPY, real-valued signals.

    min Tr(X)  s.t.  X >= 0,  A(X) = y            (eta = 0)
    min Tr(X)  s.t.  X >= 0,  ||A(X) - y||_2 <= eta   (noisy, l2 form; use eta = eps)

The l2 form is used because the perturbation set is ||delta||_2 <= eps: the true X = x x^T then
satisfies ||A(X) - y||_2 <= eps, so it is feasible. (An l1 constraint with eta = eps would NOT be
guaranteed feasible, since ||delta||_1 can be as large as sqrt(m) * eps.)
"""
import time
import numpy as np
import cvxpy as cp


def _solver_list(prefer=None):
    avail = cp.installed_solvers()
    order = [prefer] if prefer else []
    order += ["MOSEK", "CLARABEL", "SCS"]
    return [s for s in order if s in avail]


def phaselift(A, y, eta=0.0, solver=None, verbose=False):
    """Solve PhaseLift. Returns (x_hat, info) with x_hat = sqrt(lam1) * v1 (sign is arbitrary)."""
    A = np.asarray(A, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64).ravel()
    m, n = A.shape
    X = cp.Variable((n, n), PSD=True)
    AX = cp.sum(cp.multiply(A @ X, A), axis=1)      # (A(X))_k = a_k^T X a_k
    cons = [AX == y] if eta <= 0 else [cp.norm(AX - y, 2) <= eta]
    prob = cp.Problem(cp.Minimize(cp.trace(X)), cons)
    t0 = time.perf_counter()
    last_err = None
    for s in _solver_list(solver):
        try:
            prob.solve(solver=s, verbose=verbose)
            break
        except cp.error.SolverError as e:
            last_err = e
    else:
        raise RuntimeError(f"no solver succeeded: {last_err}")
    elapsed = time.perf_counter() - t0
    if X.value is None:
        return None, dict(status=prob.status, solver=s, time=elapsed)
    Xv = 0.5 * (X.value + X.value.T)
    lam, V = np.linalg.eigh(Xv)
    x_hat = np.sqrt(max(lam[-1], 0.0)) * V[:, -1]
    info = dict(status=prob.status, solver=s, time=elapsed, trace=float(np.trace(Xv)),
                eig_top2=(float(lam[-1]), float(lam[-2]) if n > 1 else 0.0),
                resid=float(np.linalg.norm(np.sum((A @ Xv) * A, axis=1) - y)))
    return x_hat, info


def sign_aligned_error(x_hat, x):
    """Sign-invariant squared l2 error min_sigma ||x_hat - sigma x||^2."""
    return float(min(np.sum((x_hat - x) ** 2), np.sum((x_hat + x) ** 2)))
