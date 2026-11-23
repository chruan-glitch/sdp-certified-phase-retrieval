"""Torch-free sanity checks for PhaseLift and the bounds module.
Run from the parent directory:   python -m pr_sdp.sanity_numpy
"""
import time
import numpy as np
from .common import make_A, measure, load_signals, y_norm_stats
from .phaselift import phaselift, sign_aligned_error
from .bounds import (propagate, preactivation_bounds, ibp_output_bound, forward, best_sigma0,
                     check_bounds_contain)

ok_all = True


def check(name, cond, detail=""):
    global ok_all
    ok_all &= bool(cond)
    print(f"[{'PASS' if cond else 'FAIL'}] {name} {detail}")


rng = np.random.default_rng(0)

# ---------------- PhaseLift ----------------
for n in (16, 64):
    A = make_A(n, 4, seed=1)
    Xs, _ = load_signals("synthetic", side=int(np.sqrt(n)), n_train=5, n_test=2, seed=2)
    x = Xs[0]
    y = measure(A, x)[0]
    xh, info = phaselift(A, y)                                   # noise-free
    check(f"PhaseLift noise-free n={n}", sign_aligned_error(xh, x) < 1e-3,
          f"err={sign_aligned_error(xh, x):.2e} solver={info['solver']} t={info['time']:.2f}s "
          f"status={info['status']} lam2/lam1={info['eig_top2'][1] / info['eig_top2'][0]:.1e}")
    eps = 0.05 * np.linalg.norm(y)
    d = rng.standard_normal(y.shape); d *= eps / np.linalg.norm(d)
    xh2, info2 = phaselift(A, y + d, eta=eps)                    # noisy, l2 constraint, eta = eps
    check(f"PhaseLift noisy (l2, eta=eps) n={n}", xh2 is not None,
          f"err={sign_aligned_error(xh2, x):.2e} status={info2['status']} t={info2['time']:.2f}s")

# ---------------- bounds on a random network ----------------
n, m, h1, h2 = 64, 256, 32, 32
A = make_A(n, 4, seed=3)
Xs, _ = load_signals("synthetic", side=8, n_train=3, n_test=2, seed=4)
x = Xs[0]; y = measure(A, x)[0]
print("||y|| stats on 200 synthetic samples:",
      {k: round(v, 3) for k, v in y_norm_stats(measure(A, load_signals('synthetic', 8, 200, 2, 5)[0])).items()})
dims = [m, h1, h2, n]
weights = [(rng.standard_normal((o, i)) / np.sqrt(i), 0.1 * rng.standard_normal(o))
           for i, o in zip(dims[:-1], dims[1:])]
sigma0 = best_sigma0(weights, y, x)

eps = 0.3
ls, us, out_lo, out_hi = propagate(weights, y, eps)
W0, b0 = weights[0]
c = W0 @ y + b0
hit_hi = hit_lo = True
for i in range(h1):
    di = eps * W0[i] / np.linalg.norm(W0[i])
    hit_hi &= abs((W0[i] @ (y + di) + b0[i]) - us[0][i]) < 1e-9
    hit_lo &= abs((W0[i] @ (y - di) + b0[i]) - ls[0][i]) < 1e-9
check("layer-0 l2 bounds are attained (exact)", hit_hi and hit_lo)

D = rng.standard_normal((3000, m)); D /= np.linalg.norm(D, axis=1, keepdims=True)
D *= eps * rng.uniform(0, 1, (3000, 1)) ** (1 / m)
check("random deltas stay inside [l, u] (all hidden layers)", check_bounds_contain(weights, y, D, ls, us))
outs = np.array([forward(weights, y + d) for d in D])
check("outputs inside IBP output box", np.all(outs >= out_lo - 1e-9) and np.all(outs <= out_hi + 1e-9))
errs = ((outs - sigma0 * x) ** 2).sum(1)
ibp = ibp_output_bound(weights, y, eps, x, sigma0)
check("IBP output bound >= sampled errors", errs.max() <= ibp + 1e-9, f"max sampled={errs.max():.4f} ibp={ibp:.4f}")

ls0, us0, lo0, hi0 = propagate(weights, y, 0.0)
clean = float(((forward(weights, y) - sigma0 * x) ** 2).sum())
check("eps=0: bounds collapse, IBP bound == clean error",
      all(np.allclose(l, u) for l, u in zip(ls0, us0)) and abs(ibp_output_bound(weights, y, 0.0, x, sigma0) - clean) < 1e-9,
      f"clean={clean:.6f}")
n_unst = sum(int(np.sum((l < 0) & (u > 0))) for l, u in zip(ls, us))
print(f"unstable neurons at eps={eps}: {n_unst} / {h1 + h2}")
print("ALL PASSED" if ok_all else "SOME CHECKS FAILED")
