"""Torch-free sanity checks for the SDP certification (random ReLU networks).
Run from the parent directory:   python -m pr_sdp.sanity_certify
"""
import numpy as np
from .bounds import preactivation_bounds, ibp_output_bound, forward, best_sigma0
from .certify import certify_full, certify_both_signs

rng = np.random.default_rng(0)
ok_all = True


def check(name, cond, detail=""):
    global ok_all
    ok_all &= bool(cond)
    print(f"[{'PASS' if cond else 'FAIL'}] {name} {detail}")


def err_and_grad(weights, y, t):
    z, pre, L = y, [], len(weights)
    for i, (W, b) in enumerate(weights):
        zh = W @ z + b
        pre.append(zh)
        z = np.maximum(zh, 0.0) if i < L - 1 else zh
    g = 2.0 * (z - t)
    for i in reversed(range(L)):
        if i < L - 1:
            g = g * (pre[i] > 0)
        g = weights[i][0].T @ g
    return float(np.sum((z - t) ** 2)), g


def np_pgd(weights, y, x, sigma0, eps, steps=300, restarts=20, seed=0):
    """numpy PGD on the fixed-sign objective (lower bound on the true max)."""
    r, t, best = np.random.default_rng(seed), sigma0 * x, -np.inf
    for _ in range(restarts):
        d = r.standard_normal(y.shape); d *= eps / np.linalg.norm(d)
        for k in range(steps + 1):
            E, g = err_and_grad(weights, y + d, t)
            best = max(best, E)
            if k == steps:
                break
            d = d + eps * (0.3 * (1 - k / steps) + 0.02) * g / (np.linalg.norm(g) + 1e-12)
            nd = np.linalg.norm(d)
            if nd > eps:
                d *= eps / nd
    return best


def make_net(m, hidden, n):
    dims = [m, *hidden, n]
    return [(rng.standard_normal((o, i)) / np.sqrt(i), 0.1 * rng.standard_normal(o))
            for i, o in zip(dims[:-1], dims[1:])]


for (m, hidden, n) in [(64, (16, 16), 16), (64, (24,), 16)]:
    print(f"\n=== random net m={m} hidden={hidden} n={n} ===")
    weights = make_net(m, hidden, n)
    x = rng.standard_normal(n); x /= np.linalg.norm(x)
    y = (rng.standard_normal((m, n)) @ x) ** 2
    sigma0 = best_sigma0(weights, y, x)
    clean = float(np.sum((forward(weights, y) - sigma0 * x) ** 2))
    print(f"clean fixed-sign error {clean:.4f}")
    print(f"{'eps':>7} {'unst':>4} {'PGD LB':>10} {'SDP':>10} {'SDP+RLT':>10} {'IBP':>10} {'gap(prim)':>10} {'t':>9} {'time':>6}")
    for eps in (0.0, 1e-3, 0.1, 0.5, 1.0):
        bnds = preactivation_bounds(weights, y, eps)
        res = certify_full(weights, bnds, y, x, eps, sigma0=sigma0, rlt=False)
        resr = certify_full(weights, bnds, y, x, eps, sigma0=sigma0, rlt=True)
        lb = np_pgd(weights, y, x, sigma0, eps) if eps > 0 else clean
        ibp = ibp_output_bound(weights, y, eps, x, sigma0)
        print(f"{eps:7.3f} {res.get('n_unstable', 0):4d} {lb:10.4f} {res['bound']:10.4f} {resr['bound']:10.4f} "
              f"{ibp:10.4f} {res.get('gap', 0):10.2e} {res.get('t', 0):9.2e} {res.get('time', 0):6.2f}")
        tol = 1e-6 * max(1.0, abs(lb))
        check(f"eps={eps}: PGD <= SDP bound", lb <= res["bound"] + tol)
        check(f"eps={eps}: PGD <= SDP+RLT bound", lb <= resr["bound"] + tol)
        check(f"eps={eps}: SDP >= clean error", res["bound"] >= clean - 1e-9)
        check(f"eps={eps}: RLT not looser", resr["bound"] <= res["bound"] + 1e-5 * max(1, res["bound"]))
        if eps == 1e-3 and res["n_unstable"] == 0:
            check("tiny eps, no unstable neurons: SDP exact (== PGD)", abs(res["bound"] - lb) < 1e-4 * max(1, lb))
        if eps == 0.5:
            r2 = certify_full(weights, bnds, y, x, eps, sigma0=sigma0, solver="SCS")
            print(f"   cross-check SCS: {r2['bound']:.4f}  (Clarabel {res['bound']:.4f})")
            check("SCS bound still >= PGD", lb <= r2["bound"] + tol)
            best, both = certify_both_signs(weights, bnds, y, x, eps)
            print(f"   both signs: {[round(r['bound'], 4) for r in both]} -> {best['bound']:.4f}")

print("\nALL PASSED" if ok_all else "\nSOME CHECKS FAILED")
