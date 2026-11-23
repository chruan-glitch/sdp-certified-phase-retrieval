"""l2-ball PGD attack on the network, with the SAME perturbation set as the SDP: ||delta||_2 <= eps.

Two objectives:
  "fixed"     max ||f(y+delta) - sigma0 x||^2     <- directly comparable with the SDP bound (4)
  "invariant" max min_sigma ||f(y+delta) - sigma x||^2   <- empirical worst case of the real metric (2)

Use a float64 copy of the model (copy.deepcopy(model).double()) and float64 tensors when comparing
against SDP bounds, so the sanity check "PGD <= SDP bound" is not polluted by float32 noise.
"""
import torch
from .model import sign_invariant_sq_error


def clean_sigma0(model, y, x):
    """Better sign at delta = 0 (argmin_s ||f(y) - s x||^2 = sign <f(y), x>). Shape (B,)."""
    with torch.no_grad():
        s = (model(y) * x).sum(1)
    return torch.where(s >= 0, torch.ones_like(s), -torch.ones_like(s))


def _errors(model, y, d, x, objective, sigma0):
    pred = model(y + d)
    if objective == "fixed":
        return ((pred - sigma0[:, None] * x) ** 2).sum(1)
    if objective == "invariant":
        return sign_invariant_sq_error(pred, x)
    raise ValueError(objective)


def pgd_attack(model, y, x, eps, objective="fixed", sigma0=None, steps=200, restarts=10, seed=0):
    """y: (B, m), x: (B, n) torch tensors. Returns (best_err (B,), best_delta (B, m)).

    Normalised-gradient ascent with decaying step, projection onto the l2 ball, random restarts on
    the sphere. The best objective value over ALL iterates and restarts is kept (so the returned
    value is a genuine lower bound on the max, never an over-estimate)."""
    model.eval()
    B, m = y.shape
    if objective == "fixed" and sigma0 is None:
        sigma0 = clean_sigma0(model, y, x)
    with torch.no_grad():
        best_err = _errors(model, y, torch.zeros_like(y), x, objective, sigma0).clone()   # delta = 0
    best_delta = torch.zeros_like(y)
    if eps <= 0:
        return best_err, best_delta
    g = torch.Generator().manual_seed(seed)
    for _ in range(restarts):
        d = torch.randn(B, m, generator=g, dtype=y.dtype)
        d = eps * d / d.norm(dim=1, keepdim=True)
        for t in range(steps + 1):
            d = d.detach().requires_grad_(True)
            err = _errors(model, y, d, x, objective, sigma0)
            with torch.no_grad():
                imp = err > best_err
                best_err = torch.where(imp, err.detach(), best_err)
                best_delta = torch.where(imp[:, None], d.detach(), best_delta)
            if t == steps:
                break
            (gr,) = torch.autograd.grad(err.sum(), d)
            alpha = eps * (0.3 * (1.0 - t / steps) + 0.02)
            with torch.no_grad():
                d = d.detach() + alpha * gr / (gr.norm(dim=1, keepdim=True) + 1e-12)
                d = d * torch.clamp(eps / (d.norm(dim=1, keepdim=True) + 1e-12), max=1.0)
    return best_err, best_delta
