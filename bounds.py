"""Pre-activation bounds and IBP output bound over the l2 ball ||delta||_2 <= eps.

`weights` is a list [(W_0, b_0), ..., (W_{L-1}, b_{L-1})] of float64 numpy arrays, with
W_l of shape (out, in). Network: z_0 = y + delta, zhat_l = W_l z_l + b_l, z_{l+1} = ReLU(zhat_l),
f = zhat_{L-1} (last layer affine, no ReLU).

Layer 0 is EXACT for the l2 ball:  zhat_{0,i} in [c_i - eps*||W_{0,i}||_2, c_i + eps*||W_{0,i}||_2],
c = W_0 y + b_0.  Deeper layers use interval bound propagation (IBP).
"""
import numpy as np


def get_weights(model):
    """Extract float64 [(W, b), ...] from a PhaseNet (see model.py)."""
    return [(l.weight.detach().cpu().double().numpy(), l.bias.detach().cpu().double().numpy())
            for l in model.linears]


def _affine_interval(W, b, lo, hi):
    mid, rad = 0.5 * (lo + hi), 0.5 * (hi - lo)
    c = W @ mid + b
    r = np.abs(W) @ rad
    return c - r, c + r


def propagate(weights, y, eps):
    """Return (ls, us, out_lo, out_hi): pre-activation bounds of every HIDDEN layer and the
    interval of the (affine) output layer."""
    y = np.asarray(y, dtype=np.float64).ravel()
    W0, b0 = weights[0]
    c = W0 @ y + b0
    r = eps * np.linalg.norm(W0, axis=1)
    lo, hi = c - r, c + r
    if len(weights) == 1:                      # purely linear network
        return [], [], lo, hi
    ls, us = [lo], [hi]
    for W, b in weights[1:]:
        zlo, zhi = np.maximum(lo, 0.0), np.maximum(hi, 0.0)
        lo, hi = _affine_interval(W, b, zlo, zhi)
        ls.append(lo); us.append(hi)
    out_lo, out_hi = ls.pop(), us.pop()        # last affine layer is the output, not a ReLU layer
    return ls, us, out_lo, out_hi


def preactivation_bounds(weights, y, eps):
    """Interface for certify(): bounds(weights, y, eps) -> (ls, us) for the hidden layers."""
    ls, us, _, _ = propagate(weights, y, eps)
    return ls, us


def ibp_output_bound(weights, y, eps, x, sigma0):
    """IBP upper bound on max_{||delta||<=eps} ||f(y+delta) - sigma0 x||_2^2 (output box)."""
    _, _, lo, hi = propagate(weights, y, eps)
    t = sigma0 * np.asarray(x, dtype=np.float64).ravel()
    return float(np.sum(np.maximum((lo - t) ** 2, (hi - t) ** 2)))


def forward(weights, y):
    z = np.asarray(y, dtype=np.float64).ravel()
    for i, (W, b) in enumerate(weights):
        z = W @ z + b
        if i < len(weights) - 1:
            z = np.maximum(z, 0.0)
    return z


def best_sigma0(weights, y, x):
    """sigma0 = better sign at delta = 0:  argmin_s ||f(y) - s x||^2 = sign(<f(y), x>)."""
    return 1.0 if float(forward(weights, y) @ np.asarray(x).ravel()) >= 0 else -1.0


def preactivations(weights, y):
    """Pre-activations zhat_l of every HIDDEN layer at input y (for bound-containment checks)."""
    z = np.asarray(y, dtype=np.float64).ravel()
    zh = []
    for W, b in weights[:-1]:
        z = W @ z + b
        zh.append(z)
        z = np.maximum(z, 0.0)
    return zh


def check_bounds_contain(weights, y, deltas, ls, us, tol=1e-9):
    """True iff the pre-activations at y + delta lie in [l, u] for every delta in `deltas` (K, m)."""
    for d in np.atleast_2d(deltas):
        for zh, l, u in zip(preactivations(weights, np.asarray(y).ravel() + d), ls, us):
            if np.any(zh < l - tol) or np.any(zh > u + tol):
                return False
    return True
