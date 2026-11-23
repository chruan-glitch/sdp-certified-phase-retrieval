"""Phase 1 driver: data/params, PhaseLift baseline, MLP training, PGD, pre-activation bounds.

    python -m pr_sdp.run_phase1 --data mnist --hidden 64 --epochs 60
    python -m pr_sdp.run_phase1 --data synthetic          # no download needed (for testing)

Writes <out>/ckpt.pt and <out>/phase1_results.json.
"""
import argparse, copy, json, os, time
import numpy as np
import torch

from .common import make_A, measure, load_signals, y_norm_stats, eps_grid_abs
from .phaselift import phaselift, sign_aligned_error
from .model import PhaseNet, train, save_checkpoint, sign_invariant_sq_error
from .pgd import pgd_attack, clean_sigma0
from .bounds import get_weights, propagate, ibp_output_bound, check_bounds_contain


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="mnist", choices=["mnist", "synthetic"])
    ap.add_argument("--side", type=int, default=8)
    ap.add_argument("--hidden", type=int, nargs="+", default=[64])
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--n_train", type=int, default=20000)
    ap.add_argument("--n_test", type=int, default=500)
    ap.add_argument("--n_eval", type=int, default=5, help="test samples for PGD/PhaseLift/bounds checks")
    ap.add_argument("--rel_levels", type=float, nargs="+", default=[0.005, 0.01, 0.02, 0.05, 0.1])
    ap.add_argument("--pgd_steps", type=int, default=200)
    ap.add_argument("--pgd_restarts", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="out_phase1")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    # ---- 1. data & conventions: A is drawn ONCE and shared by training, test and PhaseLift
    n = a.side ** 2
    A = make_A(n, 4, seed=a.seed)
    m = A.shape[0]
    Xtr, Xte = load_signals(a.data, a.side, a.n_train, a.n_test, seed=a.seed)
    Ytr, Yte = measure(A, Xtr), measure(A, Xte)
    stats = y_norm_stats(Yte)
    eps_abs, eps_rel = eps_grid_abs(Yte, a.rel_levels)
    print(f"n={n} m={m} train={len(Xtr)} test={len(Xte)}")
    print("||y||_2 over test set:", {k: round(float(v), 3) for k, v in stats.items()})
    print("common absolute eps grid:", np.round(eps_abs, 4), " (eps_rel for plots:", eps_rel, ")")

    # ---- 2. train (sign-invariant loss)
    model = PhaseNet(m, n, tuple(a.hidden))
    hist = train(model, Ytr, Xtr, Yte, Xte, epochs=a.epochs, device=dev, seed=a.seed)
    save_checkpoint(os.path.join(a.out, "ckpt.pt"), model, A, meta=vars(a))

    # ---- 3. float64 copy for all evaluation / attacks / bounds
    m64 = copy.deepcopy(model).cpu().double().eval()
    weights = get_weights(m64)
    Yt = torch.as_tensor(Yte, dtype=torch.float64)
    Xt = torch.as_tensor(Xte, dtype=torch.float64)
    with torch.no_grad():
        clean = sign_invariant_sq_error(m64(Yt), Xt)
    print(f"clean sign-invariant test error: mean {clean.mean():.5f}  median {clean.median():.5f}")

    # ---- 4. per-sample checks on the first n_eval test samples
    rng = np.random.default_rng(a.seed + 1)
    rows = []
    for i in range(min(a.n_eval, len(Xte))):
        y, x = Yte[i], Xte[i]
        yt, xt = Yt[i:i + 1], Xt[i:i + 1]
        xh, info0 = phaselift(A, y)                              # noise-free PhaseLift on clean y
        row0 = dict(sample=i, eps=0.0, phaselift_clean_err=sign_aligned_error(xh, x),
                    phaselift_clean_time=info0["time"], net_clean_err=float(clean[i]))
        print(f"\nsample {i}: net clean {clean[i]:.5f} | PhaseLift clean {row0['phaselift_clean_err']:.2e} "
              f"({info0['time']:.2f}s, {info0['solver']})")
        rows.append(row0)
        sigma0 = clean_sigma0(m64, yt, xt)
        for eps, er in zip(eps_abs, eps_rel):
            e_fix, d_fix = pgd_attack(m64, yt, xt, eps, "fixed", sigma0, a.pgd_steps, a.pgd_restarts, seed=i)
            e_inv, d_inv = pgd_attack(m64, yt, xt, eps, "invariant", None, a.pgd_steps, a.pgd_restarts, seed=i)
            ls, us, _, _ = propagate(weights, y, eps)
            ibp = ibp_output_bound(weights, y, eps, x, float(sigma0))
            rand = rng.standard_normal(m); rand *= eps / np.linalg.norm(rand)
            probes = np.vstack([d_fix.numpy(), d_inv.numpy(), rand[None]])
            contained = check_bounds_contain(weights, y, probes, ls, us)
            # PhaseLift (l2 noise constraint, eta = eps) on the PGD-perturbed and on a random input
            xh_p, ip = phaselift(A, y + d_inv.numpy()[0], eta=eps)
            xh_r, ir = phaselift(A, y + rand, eta=eps)
            row = dict(sample=i, eps=float(eps), eps_rel=float(er), pgd_fixed=float(e_fix),
                       pgd_invariant=float(e_inv), ibp_output_bound=ibp, bounds_contain_pgd=bool(contained),
                       n_unstable=int(sum(np.sum((l < 0) & (u > 0)) for l, u in zip(ls, us))),
                       phaselift_on_pgd=sign_aligned_error(xh_p, x), phaselift_on_random=sign_aligned_error(xh_r, x),
                       phaselift_time=ip["time"])
            rows.append(row)
            flag = "" if (contained and float(e_fix) <= ibp + 1e-9) else "   <-- CHECK FAILED"
            print(f"  eps_rel={er:.3f}  PGD fixed {float(e_fix):.4f}  PGD inv {float(e_inv):.4f}  IBP {ibp:.4f}  "
                  f"PhaseLift@PGD {row['phaselift_on_pgd']:.4f}  unstable {row['n_unstable']}{flag}")

    with open(os.path.join(a.out, "phase1_results.json"), "w") as f:
        json.dump(dict(args=vars(a), y_norm_stats={k: float(v) for k, v in stats.items()},
                       eps_abs=eps_abs.tolist(), eps_rel=eps_rel.tolist(), rows=rows,
                       train_hist=hist), f, indent=1)
    print("\nsaved to", a.out)


if __name__ == "__main__":
    main()
