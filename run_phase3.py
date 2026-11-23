"""Phase 3 driver: trained network + cheap baselines + first SDP certification round.

    python -m pr_sdp.run_phase3 --ckpt out_phase1/ckpt.pt --n_eval 5
    python -m pr_sdp.run_phase3 --ckpt out_phase1/ckpt.pt --n_eval 10 --lipsdp --rlt on

Steps (all on float64 copies of the trained weights, same absolute eps grid as Phase 1):
  1. Clean performance: network sign-invariant error on the whole test set vs. noise-free PhaseLift
     on the first n_eval samples.
  2. Layer-wise pre-activation bounds per eps (width, #unstable / always-active / always-inactive).
  3. Cheap baselines:  Lipschitz  (||f(y)-s0 x|| + L eps)^2  with L = product of spectral norms
     (and, with --lipsdp, LipSDP-Neuron; min of the two is used), and the IBP output bound.
  4. First certification: SDP (and SDP+RLT) on n_eval samples x eps grid, plus PGD lower bounds
     (fixed-sign: comparable to the bounds; sign-invariant: empirical worst case of the real metric),
     with the Phase-2 sanity checks repeated on the trained network.
  5. Timing + a summary table (per-eps means) for the report.
Writes <out>/phase3_results.json. Exit code 1 if a sanity check fails.
"""
import argparse, copy, json, os, time
import numpy as np
import cvxpy as cp
import torch

from .common import measure, load_signals, y_norm_stats, eps_grid_abs
from .model import load_checkpoint, sign_invariant_sq_error
from .pgd import pgd_attack
from .phaselift import phaselift, sign_aligned_error, _solver_list
from .bounds import get_weights, propagate, preactivation_bounds, ibp_output_bound, forward, best_sigma0
from .certify import certify_full
from .run_phase2 import np_pgd

OK = True
CHECKS = []


def check(name, cond, detail=""):
    global OK
    cond = bool(cond)
    OK &= cond
    CHECKS.append(dict(name=name, passed=cond, detail=detail))
    if not cond:
        print(f"    [FAIL] {name} {detail}")
    return cond


# ----------------------------------------------------------------------------- Lipschitz constants
def lipschitz_spectral(weights):
    """Product of spectral norms (ReLU is 1-Lipschitz)."""
    return float(np.prod([np.linalg.norm(W, 2) for W, _ in weights]))


def lipschitz_lipsdp(weights, solver=None, verbose=False):
    """LipSDP-Neuron (Fazlyab et al. 2019) for ReLU (slopes in [0,1]):
        min gamma  s.t.  gamma E0^T E0 - El^T Wo^T Wo El - (B^T T A + A^T T B) + 2 B^T T B >= 0,  T = diag(t) >= 0
    L = sqrt(gamma).  Returns min(spectral, sdp * (1 + 1e-5)) -- never worse than the spectral product."""
    Ws = [W for W, _ in weights[:-1]]
    Wo = weights[-1][0]
    spec = lipschitz_spectral(weights)
    if not Ws:
        return spec
    d = [Ws[0].shape[1]] + [W.shape[0] for W in Ws]            # d_0 .. d_l
    D, P = sum(d), sum(d[1:])
    off = np.concatenate([[0], np.cumsum(d)])                  # column offsets of x_0..x_l
    A, B = np.zeros((P, D)), np.zeros((P, D))
    r = 0
    for k, W in enumerate(Ws):
        A[r:r + d[k + 1], off[k]:off[k] + d[k]] = W
        B[r:r + d[k + 1], off[k + 1]:off[k + 1] + d[k + 1]] = np.eye(d[k + 1])
        r += d[k + 1]
    E0 = np.zeros((d[0], D)); E0[:, :d[0]] = np.eye(d[0])
    El = np.zeros((d[-1], D)); El[:, off[-2]:] = np.eye(d[-1])
    C1 = El.T @ (Wo.T @ Wo) @ El
    gam, t = cp.Variable(), cp.Variable(P, nonneg=True)
    T = cp.diag(t)
    M = gam * (E0.T @ E0) - C1 - (B.T @ T @ A + A.T @ T @ B) + 2 * (B.T @ T @ B)
    prob = cp.Problem(cp.Minimize(gam), [(M + M.T) / 2 >> 0])
    for s in _solver_list(solver):
        if s == "CLARABEL" and D * (D + 1) // 2 > 16000:
            continue          # dense (svec x svec) PSD block would need many GB and kills the process (see certify.py)
        try:
            prob.solve(solver=s, verbose=verbose)
            if prob.status in ("optimal", "optimal_inaccurate"):
                return min(spec, float(np.sqrt(max(gam.value, 0.0))) * (1 + 1e-5))
        except cp.error.SolverError:
            pass
    return spec


def lip_bound(weights, y, x, sigma0, eps, L):
    return float((np.linalg.norm(forward(weights, y) - sigma0 * x) + L * eps) ** 2)


# ----------------------------------------------------------------------------- helpers
def layer_stats(weights, y, eps):
    ls, us, _, _ = propagate(weights, y, eps)
    out = []
    for l, u in zip(ls, us):
        out.append(dict(width_mean=float(np.mean(u - l)), unstable=int(np.sum((l < 0) & (u > 0))),
                        active=int(np.sum(l >= 0)), inactive=int(np.sum(u <= 0)), size=int(l.size)))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="out_phase1/ckpt.pt")
    ap.add_argument("--n_eval", type=int, default=5, help="test samples for certification (5-10)")
    ap.add_argument("--rlt", choices=["off", "on", "both"], default="both")
    ap.add_argument("--lipsdp", action="store_true", help="also compute the LipSDP-Neuron constant (slow for wide nets)")
    ap.add_argument("--pgd_steps", type=int, default=300)
    ap.add_argument("--pgd_restarts", type=int, default=20)
    ap.add_argument("--solver", default=None)
    ap.add_argument("--solvers", nargs="*", default=["CLARABEL", "SCS", "MOSEK"],
                    help="cross-check solvers on sample 0 at the middle eps (empty = skip)")
    ap.add_argument("--rel_levels", type=float, nargs="+", default=None,
                    help="override the eps_rel grid stored in the checkpoint")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="out_phase3")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    # ---- load model, data, eps grid (identical to Phase 1)
    model, A, meta = load_checkpoint(a.ckpt)
    side, data, seed = meta.get("side", 8), meta.get("data", "mnist"), meta.get("seed", 0)
    _, Xte = load_signals(data, side, meta.get("n_train", 20000), meta.get("n_test", 500), seed=seed)
    Yte = measure(A, Xte)
    rel = a.rel_levels or meta.get("rel_levels", [0.005, 0.01, 0.02, 0.05, 0.1])
    eps_abs, eps_rel = eps_grid_abs(Yte, rel)
    stats = y_norm_stats(Yte)
    m64 = copy.deepcopy(model).cpu().double().eval()
    weights = get_weights(m64)
    n_eval = min(a.n_eval, len(Xte))
    print(f"model dims {model.dims} | data={data} | test={len(Xte)} | n_eval={n_eval}")
    print("||y||_2 stats:", {k: round(float(v), 3) for k, v in stats.items()})
    print("eps_abs:", np.round(eps_abs, 4), " eps_rel:", eps_rel)
    print("installed solvers:", cp.installed_solvers(), "| priority:", _solver_list(a.solver))

    # ---- 1. clean performance
    Yt, Xt = torch.as_tensor(Yte, dtype=torch.float64), torch.as_tensor(Xte, dtype=torch.float64)
    with torch.no_grad():
        clean_net = sign_invariant_sq_error(m64(Yt), Xt).numpy()
    print(f"\n[1] network clean sign-invariant error (all {len(Xte)} test): mean {clean_net.mean():.5f} "
          f"median {np.median(clean_net):.5f}")
    pl = []
    for i in range(n_eval):
        xh, info = phaselift(A, Yte[i])
        pl.append(dict(sample=i, phaselift_err=sign_aligned_error(xh, Xte[i]), phaselift_time=info["time"],
                       net_err=float(clean_net[i])))
    print(f"    first {n_eval}: network mean {np.mean([p['net_err'] for p in pl]):.5f} | "
          f"PhaseLift (noise-free) mean {np.mean([p['phaselift_err'] for p in pl]):.2e}, "
          f"time {np.mean([p['phaselift_time'] for p in pl]):.3f}s/sample")

    # ---- 3a. Lipschitz constants
    L_spec = lipschitz_spectral(weights)
    L = L_spec
    L_sdp = None
    if a.lipsdp:
        t0 = time.perf_counter()
        L_sdp = lipschitz_lipsdp(weights, a.solver)
        L = min(L_spec, L_sdp)
        print(f"\n[3] Lipschitz: spectral product {L_spec:.4f} | LipSDP-Neuron {L_sdp:.4f} ({time.perf_counter() - t0:.1f}s)")
    else:
        print(f"\n[3] Lipschitz: spectral product {L_spec:.4f}  (use --lipsdp for LipSDP)")
    # empirical sanity: observed local slope must not exceed L
    rng = np.random.default_rng(a.seed)
    ratios = []
    for _ in range(2000):
        i = rng.integers(len(Yte)); d = rng.standard_normal(Yte.shape[1]); d *= 0.05 * np.linalg.norm(Yte[i]) / np.linalg.norm(d)
        ratios.append(np.linalg.norm(forward(weights, Yte[i] + d) - forward(weights, Yte[i])) / np.linalg.norm(d))
    check("Lipschitz constant >= sampled slope", max(ratios) <= L * (1 + 1e-6), f"max slope {max(ratios):.4f} L {L:.4f}")
    print(f"    max sampled slope {max(ratios):.4f} <= L = {L:.4f}")

    # ---- 2/4. per-sample layer bounds + baselines + certification
    rlt_modes = {"off": [False], "on": [True], "both": [False, True]}[a.rlt]
    rows = []
    cross_eps_idx = len(eps_abs) // 2
    for i in range(n_eval):
        y, x = Yte[i], Xte[i]
        yt, xt = Yt[i:i + 1], Xt[i:i + 1]
        sigma0 = best_sigma0(weights, y, x)
        clean = float(np.sum((forward(weights, y) - sigma0 * x) ** 2))
        print(f"\n=== sample {i}: ||y||={np.linalg.norm(y):.3f} sigma0={sigma0:+.0f} clean fixed-sign err {clean:.5f}")
        print(f"{'eps_rel':>7} {'unst':>5} {'PGD fix':>9} {'PGD inv':>9} {'SDP':>9} {'SDP+RLT':>9} {'IBP':>9} {'Lip':>9} "
              f"{'SDP/PGD':>7} {'gap':>8} {'t':>7} {'t_sdp':>6} {'flip':>4}")
        # eps = 0 reference row (SDP must equal the clean error)
        r0 = certify_full(weights, preactivation_bounds(weights, y, 0.0), y, x, 0.0, sigma0=sigma0)
        check(f"[s{i}] eps=0: SDP bound == clean error", abs(r0["bound"] - clean) < 1e-9)
        prev_lb = -np.inf
        for k, (eps, er) in enumerate(zip(eps_abs, eps_rel)):
            bnds = preactivation_bounds(weights, y, eps)
            res = {rl: certify_full(weights, bnds, y, x, eps, sigma0=sigma0, rlt=rl, solver=a.solver) for rl in rlt_modes}
            main_r = res[rlt_modes[0]]
            best_b = min(r["bound"] for r in res.values())
            lb, d_best = np_pgd(weights, y, x, sigma0, eps, a.pgd_steps, a.pgd_restarts, seed=a.seed + i)
            inv, _ = pgd_attack(m64, yt, xt, eps, "invariant", None, a.pgd_steps, a.pgd_restarts, seed=a.seed + i)
            flip = bool(float(forward(weights, y + d_best) @ x) * sigma0 < 0)
            ibp = ibp_output_bound(weights, y, eps, x, sigma0)
            row = dict(sample=i, eps=float(eps), eps_rel=float(er), sigma0=sigma0, clean_fixed=clean,
                       pgd_fixed=lb, pgd_invariant=float(inv), sign_flip=flip, ibp=ibp,
                       lipschitz=lip_bound(weights, y, x, sigma0, eps, L),
                       lipschitz_spectral=lip_bound(weights, y, x, sigma0, eps, L_spec),
                       sdp=res[False]["bound"] if False in res else None,
                       sdp_rlt=res[True]["bound"] if True in res else None,
                       primal=main_r.get("primal_value"), gap=main_r.get("gap"), t_corr=main_r.get("t"),
                       lam_min_Z=main_r.get("lam_min_Z"), status=main_r["status"], solver=main_r.get("solver"),
                       n_unstable=main_r.get("n_unstable"), N=main_r.get("N"),
                       time_sdp=res[False]["time"] if False in res else None,
                       time_rlt=res[True]["time"] if True in res else None,
                       layers=layer_stats(weights, y, eps))
            rows.append(row)
            with open(os.path.join(a.out, "phase3_rows.jsonl"), "a") as fp:      # survives ^C / crashes
                fp.write(json.dumps(row, default=float) + "\n")
            f = lambda v: f"{v:9.4f}" if v is not None else f"{'-':>9}"
            print(f"{er:7.3f} {row['n_unstable']:5d} {lb:9.4f} {float(inv):9.4f} {f(row['sdp'])} {f(row['sdp_rlt'])} "
                  f"{ibp:9.4f} {row['lipschitz']:9.4f} {best_b / max(lb, 1e-12):7.2f} {main_r.get('gap', 0):8.1e} "
                  f"{main_r.get('t', 0):7.0e} {main_r.get('time', 0):6.2f} {'YES' if flip else '':>4}")

            pre = f"[s{i} eps_rel={er:.3g}] "
            tol = 1e-6 * max(1.0, lb)
            for rl, r in res.items():
                nm = "SDP+RLT" if rl else "SDP"
                check(pre + f"{nm} finite", np.isfinite(r["bound"]), f"status={r['status']}")
                check(pre + f"PGD <= {nm}", lb <= r["bound"] + tol, f"PGD={lb:.6g} bound={r['bound']:.6g}")
                check(pre + f"{nm} >= clean error", r["bound"] >= clean - 1e-9)
            if len(res) == 2:
                check(pre + "RLT not looser", res[True]["bound"] <= res[False]["bound"] + 1e-5 * max(1.0, res[False]["bound"]))
            check(pre + "PGD fixed <= IBP", lb <= ibp + tol)
            check(pre + "PGD fixed <= Lipschitz", lb <= row["lipschitz"] + tol)
            check(pre + "PGD invariant <= PGD fixed (+tol)", float(inv) <= lb + 1e-6 * max(1.0, lb) + 1e-9,
                  f"inv={float(inv):.6g} fixed={lb:.6g}")
            if row["n_unstable"] == 0:
                check(pre + "no unstable neurons: SDP exact (== PGD)", abs(main_r["bound"] - lb) <= 1e-3 * max(1.0, lb),
                      f"SDP={main_r['bound']:.6g} PGD={lb:.6g}")

            if i == 0 and k == cross_eps_idx and a.solvers:
                print("    solver cross-check:")
                for s in a.solvers:
                    if s not in cp.installed_solvers():
                        print(f"      {s:9s} not installed -> skipped"); continue
                    r2 = certify_full(weights, bnds, y, x, eps, sigma0=sigma0, solver=s)
                    print(f"      {s:9s} bound {r2['bound']:.6f} primal {r2.get('primal_value', np.nan):.6f} "
                          f"{r2['status']} {r2.get('time', 0):.2f}s")
                    check(pre + f"{s} bound >= PGD", lb <= r2["bound"] + tol)
                    row.setdefault("cross", {})[s] = r2["bound"]

    # ---- 5. summary per eps
    print("\n" + "=" * 100 + "\n[5] per-eps means over samples (squared l2 errors)")
    print(f"{'eps_rel':>7} {'PGD inv':>9} {'PGD fix':>9} {'SDP':>9} {'SDP+RLT':>9} {'IBP':>9} {'Lip':>9} {'#unst':>6} "
          f"{'flip%':>6} {'t_sdp':>6}")
    summ = []
    for er in eps_rel:
        R = [r for r in rows if abs(r["eps_rel"] - er) < 1e-12]
        mean = lambda key: float(np.mean([r[key] for r in R if r[key] is not None])) if any(r[key] is not None for r in R) else None
        s = dict(eps_rel=float(er), eps=R[0]["eps"], pgd_invariant=mean("pgd_invariant"), pgd_fixed=mean("pgd_fixed"),
                 sdp=mean("sdp"), sdp_rlt=mean("sdp_rlt"), ibp=mean("ibp"), lipschitz=mean("lipschitz"),
                 n_unstable=mean("n_unstable"), flip_frac=float(np.mean([r["sign_flip"] for r in R])),
                 time_sdp=mean("time_sdp"), time_rlt=mean("time_rlt"))
        summ.append(s)
        g = lambda v: f"{v:9.4f}" if v is not None else f"{'-':>9}"
        print(f"{er:7.3f} {g(s['pgd_invariant'])} {g(s['pgd_fixed'])} {g(s['sdp'])} {g(s['sdp_rlt'])} {g(s['ibp'])} "
              f"{g(s['lipschitz'])} {s['n_unstable']:6.1f} {100 * s['flip_frac']:6.0f} {s['time_sdp'] or 0:6.2f}")
    ts = [r["time_sdp"] for r in rows if r["time_sdp"] is not None]
    tr = [r["time_rlt"] for r in rows if r["time_rlt"] is not None]
    print(f"\nSDP solve time: median {np.median(ts):.2f}s max {np.max(ts):.2f}s" if ts else "")
    if tr:
        print(f"SDP+RLT solve time: median {np.median(tr):.2f}s max {np.max(tr):.2f}s")
    if ts:
        per = float(np.median(ts))
        print(f"-> 1 hour ~ {int(3600 / per)} solves = e.g. {int(3600 / per / len(eps_abs))} samples x {len(eps_abs)} eps")
    vac = [r for r in rows if r["sdp"] is not None and r["sdp"] > 3.5]
    if vac:
        print(f"note: {len(vac)}/{len(rows)} SDP bounds > 3.5 (near the sign-flip level ~4): likely vacuous -> consider RLT/fine-tuning")

    n_pass = sum(c["passed"] for c in CHECKS)
    print(f"\nchecks: {n_pass}/{len(CHECKS)} passed")
    for c in CHECKS:
        if not c["passed"]:
            print("  FAILED:", c["name"], c["detail"])
    print("ALL PASSED" if OK else "SOME CHECKS FAILED")
    with open(os.path.join(a.out, "phase3_results.json"), "w") as fo:
        json.dump(dict(args=vars(a), dims=model.dims, y_norm_stats={k: float(v) for k, v in stats.items()},
                       eps_abs=eps_abs.tolist(), eps_rel=list(map(float, eps_rel)),
                       clean_net_mean=float(clean_net.mean()), clean_net_median=float(np.median(clean_net)),
                       phaselift_clean=pl, lipschitz=dict(spectral=L_spec, lipsdp=L_sdp, used=L),
                       rows=rows, summary=summ, checks=CHECKS, solvers=cp.installed_solvers(), cvxpy=cp.__version__),
                  fo, indent=1, default=float)
    print("saved to", a.out)
    raise SystemExit(0 if OK else 1)


if __name__ == "__main__":
    main()
