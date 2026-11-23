"""Phase 2 driver (staged, resumable): SDP certification prototype -- sanity checks, RLT trial, solver cross-check, timing.

Everything that is computed is appended to <out>/cache.jsonl, one record per (instance, eps, stage).  A stage that is
already in the cache is skipped, so you can stop / crash / reconnect and simply re-run the same command.

STAGES (run any subset, in any order, in separate Colab cells / sessions):
    pgd    cheap, numpy only: clean error, PGD lower bound (fixed sign), sign flip, IBP, Lipschitz, #unstable neurons,
           estimated SDP size N and Clarabel memory.   -> seconds
    sdp    plain SDP certificate (with the rigorous dual evaluation)            -> the expensive stage
    rlt    SDP + RLT certificate                                                -> the expensive stage
    cross  other solvers (--solvers) on the middle eps of each instance (+ both signs with --both_signs)
    report reads the cache only: sanity checks, RLT trial, timing projection, writes phase2_results.json.  No solving.

    python -m project.run_phase2 --stages pgd                                  # random nets, seconds
    python -m project.run_phase2 --stages sdp rlt cross report                 # random nets (default: all stages)
    python -m project.run_phase2 --ckpt out_mnist_h128/ckpt.pt --n_eval 1 --solver SCS --solvers SCS --stages pgd
    python -m project.run_phase2 --ckpt out_mnist_h128/ckpt.pt --n_eval 1 --solver SCS --solvers SCS --stages sdp
    python -m project.run_phase2 --ckpt out_mnist_h128/ckpt.pt --n_eval 1 --solver SCS --solvers SCS --stages rlt cross report

Sanity checks done by `report` (exit code 1 if one fails):
    1. PGD <= SDP (and <= SDP+RLT)            4. RLT not looser than plain SDP
    2. SDP >= clean error (incl. eps = 0)      5. tiny eps & no unstable neurons: SDP ~= PGD (exact relaxation)
    3. bound finite, >= solver primal          6. every solver in the cross-check gives a bound >= PGD
Memory guard: --max_N skips SDP solves with estimated N above the limit (recorded as skipped, not as failures).
Clarabel needs ~8*(N(N+1)/2)^2 bytes: with a 12 GB Colab use --solver SCS once N is above ~180 (certify.py already
skips Clarabel above that size).
"""
import argparse, gc, json, os, time
import numpy as np
import cvxpy as cp

from .bounds import preactivation_bounds, ibp_output_bound, forward, best_sigma0
from .certify import certify_full, certify_both_signs
from .phaselift import _solver_list

ALL_STAGES = ["pgd", "sdp", "rlt", "cross", "report"]
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


# ----------------------------------------------------------------------------- numpy PGD (fixed sign)
def _err_and_grad(weights, y, t):
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
    """Lower bound on max_{||d||<=eps} ||f(y+d) - sigma0 x||^2 (same perturbation set as the SDP).
    Returns (best_err, best_delta). Best value over all iterates and restarts."""
    t = sigma0 * x
    best, best_d = _err_and_grad(weights, y, t)[0], np.zeros_like(y)
    if eps <= 0:
        return best, best_d
    rng = np.random.default_rng(seed)
    for _ in range(restarts):
        d = rng.standard_normal(y.shape); d *= eps / np.linalg.norm(d)
        for k in range(steps + 1):
            E, g = _err_and_grad(weights, y + d, t)
            if E > best:
                best, best_d = E, d.copy()
            if k == steps:
                break
            d = d + eps * (0.3 * (1 - k / steps) + 0.02) * g / (np.linalg.norm(g) + 1e-12)
            nd = np.linalg.norm(d)
            if nd > eps:
                d *= eps / nd
    return best, best_d


def lipschitz_bound(weights, y, x, sigma0, eps):
    """Cheap baseline (||f(y)-s0 x|| + L eps)^2, L = product of spectral norms (ReLU is 1-Lipschitz)."""
    L = float(np.prod([np.linalg.norm(W, 2) for W, _ in weights]))
    return float((np.linalg.norm(forward(weights, y) - sigma0 * x) + L * eps) ** 2)


# ----------------------------------------------------------------------------- cache
class Cache:
    def __init__(self, path):
        self.path, self.d = path, {}
        if os.path.exists(path):
            for line in open(path):
                line = line.strip()
                if line:
                    r = json.loads(line)
                    self.d[r["key"]] = r
        self.f = open(path, "a")

    @staticmethod
    def key(tag, eps, stage, extra=""):
        return f"{tag}|{eps:.9g}|{stage}{('|' + extra) if extra else ''}"

    def get(self, tag, eps, stage, extra=""):
        return self.d.get(self.key(tag, eps, stage, extra))

    def put(self, tag, eps, stage, rec, extra=""):
        rec = dict(rec, key=self.key(tag, eps, stage, extra), tag=tag, eps=float(eps), stage=stage, extra=extra)
        self.d[rec["key"]] = rec
        self.f.write(json.dumps(rec, default=float) + "\n"); self.f.flush()


def _slim(res):
    """The fields of a certify_full result that we keep (all JSON-safe)."""
    keys = ("bound", "primal_value", "gap", "t", "lam_min_Z", "status", "solver", "time", "N", "n_unstable")
    out = {k: res.get(k) for k in keys}
    for k in ("bound", "primal_value", "gap", "t", "lam_min_Z", "time"):
        if out[k] is not None:
            out[k] = float(out[k])
    return out


# ----------------------------------------------------------------------------- instances
def random_instances(a):
    rng = np.random.default_rng(a.seed)
    out = []
    for (m, hidden, n) in [(64, (16, 16), 16), (64, (24,), 16)]:
        dims = [m, *hidden, n]
        weights = [(rng.standard_normal((o, i)) / np.sqrt(i), 0.1 * rng.standard_normal(o))
                   for i, o in zip(dims[:-1], dims[1:])]
        for j in range(a.n_random):
            x = rng.standard_normal(n); x /= np.linalg.norm(x)
            y = (rng.standard_normal((m, n)) @ x) ** 2
            out.append((f"rand m={m} h={hidden} #{j}", weights, y, x, list(a.eps)))
    return out


def ckpt_instances(a):
    from .common import measure, load_signals, eps_grid_abs
    from .model import load_checkpoint
    from .bounds import get_weights
    model, A, meta = load_checkpoint(a.ckpt)
    side, data, seed = meta.get("side", 8), meta.get("data", "mnist"), meta.get("seed", 0)
    _, Xte = load_signals(data, side, meta.get("n_train", 20000), meta.get("n_test", 500), seed=seed)
    Yte = measure(A, Xte)
    rel = meta.get("rel_levels", [0.005, 0.01, 0.02, 0.05, 0.1])
    eps_abs, _ = eps_grid_abs(Yte, rel)
    if a.eps_idx:
        eps_abs = eps_abs[a.eps_idx]
    weights = get_weights(model)
    print(f"checkpoint {a.ckpt}: dims={model.dims}, data={data}, eps grid {np.round(eps_abs, 4)}")
    idx = range(a.start, min(a.start + a.n_eval, len(Xte)))
    return [(f"{data} test#{i}", weights, Yte[i], Xte[i], list(map(float, eps_abs))) for i in idx]


# ----------------------------------------------------------------------------- stages
def est_size(weights, bnds):
    ls, us = bnds
    n_unst = sum(int(np.sum((l < 0) & (u > 0))) for l, u in zip(ls, us))
    # certify.py reduces the w-block exactly to r_eff <= min(rank W0, #unstable + n_out)  (upper estimate)
    r = min(min(weights[0][0].shape), n_unst + weights[-1][0].shape[0])
    N = 1 + r + n_unst
    return N, n_unst


def stage_pgd(c, inst, a):
    tag, W, y, x, eps_list = inst
    sigma0 = best_sigma0(W, y, x)
    clean = float(np.sum((forward(W, y) - sigma0 * x) ** 2))
    for eps in eps_list:
        if c.get(tag, eps, "pgd"):
            continue
        bnds = preactivation_bounds(W, y, eps)
        N, n_unst = est_size(W, bnds)
        lb, d = np_pgd(W, y, x, sigma0, eps, a.pgd_steps, a.pgd_restarts, seed=a.seed)
        flip = bool(eps > 0 and float(forward(W, y + d) @ x) * sigma0 < 0)
        c.put(tag, eps, "pgd", dict(sigma0=sigma0, clean=clean, pgd_lb=lb, sign_flip=flip,
                                    ibp=ibp_output_bound(W, y, eps, x, sigma0), lipschitz=lipschitz_bound(W, y, x, sigma0, eps),
                                    n_unstable=n_unst, N_est=N, norm_y=float(np.linalg.norm(y))))
        print(f"  [pgd] {tag} eps={eps:.4g}: unstable {n_unst}, est. N={N} (Clarabel would need "
              f"{8 * (N * (N + 1) / 2) ** 2 / 1e9:.1f} GB), PGD {lb:.4f}, IBP {ibp_output_bound(W, y, eps, x, sigma0):.4f}")


def stage_sdp(c, inst, a, rlt):
    stage = "rlt" if rlt else "sdp"
    tag, W, y, x, eps_list = inst
    sigma0 = best_sigma0(W, y, x)
    for eps in eps_list:
        if c.get(tag, eps, stage):
            continue
        bnds = preactivation_bounds(W, y, eps)
        N, n_unst = est_size(W, bnds)
        if a.max_N and N > a.max_N:
            c.put(tag, eps, stage, dict(skipped=f"N={N} > max_N={a.max_N}", N=N, n_unstable=n_unst))
            print(f"  [{stage}] {tag} eps={eps:.4g}: SKIPPED (N={N} > --max_N {a.max_N})")
            continue
        res = certify_full(W, bnds, y, x, eps, sigma0=sigma0, rlt=rlt, solver=a.solver)
        c.put(tag, eps, stage, _slim(res))
        print(f"  [{stage}] {tag} eps={eps:.4g}: N={res.get('N', 0)} bound {res['bound']:.4f} "
              f"gap {res.get('gap', 0):.1e} {res['status']} {res.get('solver')} {res.get('time', 0):.1f}s", flush=True)
        del res
        gc.collect()


def stage_cross(c, inst, a):
    tag, W, y, x, eps_list = inst
    pos = [e for e in eps_list if e > 0]
    if not pos or not a.solvers:
        return
    eps = pos[len(pos) // 2]                                 # middle eps
    sigma0 = best_sigma0(W, y, x)
    bnds = preactivation_bounds(W, y, eps)
    N, _ = est_size(W, bnds)
    for s in a.solvers:
        if c.get(tag, eps, "cross", s):
            continue
        if s not in cp.installed_solvers():
            c.put(tag, eps, "cross", dict(skipped="not installed"), extra=s)
            print(f"  [cross] {s} not installed -> skipped"); continue
        if a.max_N and N > a.max_N:
            c.put(tag, eps, "cross", dict(skipped=f"N={N} > max_N"), extra=s)
            print(f"  [cross] {tag} {s}: SKIPPED (N={N} > --max_N)"); continue
        res = certify_full(W, bnds, y, x, eps, sigma0=sigma0, solver=s)
        c.put(tag, eps, "cross", _slim(res), extra=s)
        note = "" if res.get("solver") in (None, s) else f"  (fell back to {res.get('solver')})"
        print(f"  [cross] {tag} eps={eps:.4g} {s}: bound {res['bound']:.6f} {res['status']} {res.get('time', 0):.1f}s{note}", flush=True)
        del res
        gc.collect()
    if a.both_signs and not c.get(tag, eps, "cross", "both_signs") and not (a.max_N and N > a.max_N):
        best, both = certify_both_signs(W, bnds, y, x, eps, solver=a.solver)
        c.put(tag, eps, "cross", dict(bounds=[float(r["bound"]) for r in both], bound=float(best["bound"])), extra="both_signs")
        print(f"  [cross] both signs {[round(r['bound'], 4) for r in both]} -> {best['bound']:.4f}")
        gc.collect()


# ----------------------------------------------------------------------------- report (cache only)
def _ok(rec):
    return rec is not None and "skipped" not in rec and rec.get("bound") is not None


def report(c, insts, a):
    rows = []
    for inst in insts:
        tag, W, y, x, eps_list = inst
        print(f"\n=== {tag}")
        print(f"{'eps':>8} {'unst':>4} {'N':>4} {'PGD LB':>10} {'SDP':>10} {'SDP+RLT':>10} {'IBP':>10} {'Lip':>10} "
              f"{'gap(prim)':>9} {'t':>8} {'lamZ':>9} {'t_sdp':>7} {'t_rlt':>7} {'flip':>4}")
        for eps in eps_list:
            p, s, r = c.get(tag, eps, "pgd"), c.get(tag, eps, "sdp"), c.get(tag, eps, "rlt")
            if eps <= 0:                                     # eps = 0: no solve needed, bound == clean error
                sigma0 = best_sigma0(W, y, x)
                clean = float(np.sum((forward(W, y) - sigma0 * x) ** 2))
                s = dict(bound=clean, primal_value=clean, gap=0.0, status="eps=0", time=0.0, N=0, n_unstable=0)
                r = dict(s)
                p = p or dict(clean=clean, pgd_lb=clean, sign_flip=False, ibp=clean, lipschitz=clean, n_unstable=0, N_est=0)
            nan = float("nan")
            g = lambda d, k: (d.get(k) if d and d.get(k) is not None else nan)
            row = dict(tag=tag, eps=float(eps), sdp=g(s, "bound") if _ok(s) else nan, sdp_rlt=g(r, "bound") if _ok(r) else nan,
                       pgd_lb=g(p, "pgd_lb"), ibp=g(p, "ibp"), lipschitz=g(p, "lipschitz"), clean=g(p, "clean"),
                       n_unstable=int(g(p, "n_unstable")) if p else 0, N=int(g(s, "N") or g(p, "N_est") or 0),
                       primal=g(s, "primal_value"), gap=g(s, "gap"), t_corr=g(s, "t"), lam_min_Z=g(s, "lam_min_Z"),
                       time_sdp=g(s, "time"), time_rlt=g(r, "time"), solver=(s or {}).get("solver"),
                       sign_flip=bool(p and p.get("sign_flip")), status=(s or {}).get("status", "not run"),
                       status_rlt=(r or {}).get("status", "not run"))
            rows.append(row)
            f = lambda v, w=10: f"{v:{w}.4f}" if np.isfinite(v) else f"{'-':>{w}}"
            print(f"{eps:8.4f} {row['n_unstable']:4d} {row['N']:4d} {f(row['pgd_lb'])} {f(row['sdp'])} {f(row['sdp_rlt'])} "
                  f"{f(row['ibp'])} {f(row['lipschitz'])} {row['gap']:9.2e} {row['t_corr']:8.1e} {row['lam_min_Z']:9.1e} "
                  f"{f(row['time_sdp'], 7)} {f(row['time_rlt'], 7)} {'YES' if row['sign_flip'] else '':>4}")
            lb = row["pgd_lb"]
            if not np.isfinite(lb):
                continue
            tol, pre = 1e-6 * max(1.0, abs(lb)), f"[{tag} eps={eps:.4g}] "
            for nm, b in (("SDP", row["sdp"]), ("SDP+RLT", row["sdp_rlt"])):
                if np.isfinite(b):
                    check(pre + f"{nm} finite", np.isfinite(b))
                    check(pre + f"PGD <= {nm}", lb <= b + tol, f"PGD={lb:.6g} {nm}={b:.6g}")
                    check(pre + f"{nm} >= clean error", b >= row["clean"] - 1e-9, f"clean={row['clean']:.6g}")
                elif (nm == "SDP" and s and "skipped" not in s) or (nm == "SDP+RLT" and r and "skipped" not in r):
                    check(pre + f"{nm} finite", False, "solver returned inf / failed")
            if np.isfinite(row["sdp"]) and np.isfinite(row["sdp_rlt"]):
                check(pre + "RLT not looser", row["sdp_rlt"] <= row["sdp"] + 1e-5 * max(1.0, row["sdp"]))
            pos = [e for e in eps_list if e > 0]
            if np.isfinite(row["sdp"]) and row["n_unstable"] == 0 and eps > 0 and eps == min(pos):
                check(pre + "no unstable neurons: SDP exact (== PGD)", abs(row["sdp"] - lb) <= 1e-3 * max(1.0, lb),
                      f"SDP={row['sdp']:.6g} PGD={lb:.6g}")
            if np.isfinite(row["primal"]) and np.isfinite(row["sdp"]):
                check(pre + "certified bound >= solver primal (weak duality)",
                      row["sdp"] >= row["primal"] - 1e-6 * max(1.0, abs(row["primal"])),
                      f"bound={row['sdp']:.6g} primal={row['primal']:.6g}")
            # cross-check results stored for this eps
            for sname in a.solvers or []:
                cr = c.get(tag, eps, "cross", sname)
                if _ok(cr):
                    used = cr.get("solver")
                    print(f"    cross-check {sname:9s}: bound {cr['bound']:.6f}  primal {cr.get('primal_value') or float('nan'):.6f} "
                          f"{cr['status']} {cr.get('time', 0):.1f}s" + (f"  (actually {used})" if used not in (None, sname) else ""))
                    check(pre + f"{sname} bound >= PGD", lb <= cr["bound"] + tol, f"{sname}={cr['bound']:.6g}")
                    row.setdefault("cross", {})[sname] = cr["bound"]
            bs = c.get(tag, eps, "cross", "both_signs")
            if bs:
                print(f"    both signs: {[round(v, 4) for v in bs['bounds']]} -> {bs['bound']:.4f}")
    return rows, summarise(rows, a)


def summarise(rows, a):
    pos = [r for r in rows if r["eps"] > 0 and np.isfinite(r["sdp"]) and np.isfinite(r["sdp_rlt"])]
    out = {}
    if pos:
        print("\n" + "=" * 78 + "\nRLT early trial (plain SDP vs SDP+RLT, eps > 0)")
        rel = np.array([(r["sdp"] - r["sdp_rlt"]) / max(r["sdp"], 1e-12) for r in pos])
        ts, tr = np.array([r["time_sdp"] for r in pos]), np.array([r["time_rlt"] for r in pos])
        tight = np.array([r["n_unstable"] for r in pos]) > 0
        print(f"  instances with unstable neurons: {int(tight.sum())}/{len(pos)}")
        if tight.any():
            print(f"  relative bound reduction from RLT (unstable only): mean {rel[tight].mean() * 100:.2f}%  "
                  f"median {np.median(rel[tight]) * 100:.2f}%  max {rel[tight].max() * 100:.2f}%")
            print(f"  solve time: SDP median {np.median(ts[tight]):.2f}s | SDP+RLT median {np.median(tr[tight]):.2f}s "
                  f"(x{np.median(tr[tight]) / max(np.median(ts[tight]), 1e-9):.1f})")
        ratio = np.array([r["sdp"] / max(r["pgd_lb"], 1e-12) for r in pos])
        ratio_r = np.array([r["sdp_rlt"] / max(r["pgd_lb"], 1e-12) for r in pos])
        ibp_ratio = np.array([r["ibp"] / max(r["pgd_lb"], 1e-12) for r in pos])
        print(f"  median bound/PGD ratio: SDP {np.median(ratio):.3f} | SDP+RLT {np.median(ratio_r):.3f} | IBP {np.median(ibp_ratio):.3f}")
        rec = bool(tight.any() and np.median(rel[tight]) > 0.01 and np.median(tr[tight]) < 3 * max(np.median(ts[tight]), 1e-9))
        print(f"  heuristic recommendation: {'ENABLE RLT by default' if rec else 'keep RLT off by default (or use only when SDP is vacuous)'}"
              "  [rule: median reduction >1% and < 3x time -- adjust to taste]")
        out.update(rlt_rel_reduction_median=float(np.median(rel[tight])) if tight.any() else None, rlt_recommended=rec)
    timed = [r for r in rows if r["eps"] > 0 and np.isfinite(r["sdp"]) and np.isfinite(r["time_sdp"])]
    if timed:
        print("\n" + "=" * 78 + "\nTiming (sizes the Week 5-6 experiment)")
        t_all, N = np.array([r["time_sdp"] for r in timed]), np.array([r["N"] for r in timed])
        print(f"  SDP size N: min {N.min()} median {int(np.median(N))} max {N.max()}")
        print(f"  single SDP solve: median {np.median(t_all):.2f}s  max {t_all.max():.2f}s  (solver {timed[0]['solver']})")
        per = float(np.median(t_all))
        n_solves = int(a.budget_hours * 3600.0 // max(per, 1e-9))
        print(f"  budget {a.budget_hours:g}h -> ~{n_solves} solves (one sigma0 each; halve for both signs):")
        for ne in (3, 5, 8):
            print(f"     eps grid of {ne} points -> ~{n_solves // ne} samples")
        out.update(solve_time_median=per, solves_in_budget=n_solves)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stages", nargs="+", choices=ALL_STAGES, default=ALL_STAGES,
                    help="which stages to run now (default: all). Finished work is read from <out>/cache.jsonl")
    ap.add_argument("--mode", choices=["random", "ckpt", "both"], default=None,
                    help="default: 'ckpt' if --ckpt is given, else 'random'")
    ap.add_argument("--ckpt", default=None, help="checkpoint from run_phase1 (out_phase1/ckpt.pt)")
    ap.add_argument("--n_eval", type=int, default=5, help="test samples when using --ckpt")
    ap.add_argument("--start", type=int, default=0, help="first test index when using --ckpt")
    ap.add_argument("--eps_idx", type=int, nargs="+", default=None,
                    help="ckpt mode: only these indices of the eps grid (e.g. 0 2 4)")
    ap.add_argument("--n_random", type=int, default=1, help="random (x, y) instances per random network")
    ap.add_argument("--eps", type=float, nargs="+", default=[0.0, 1e-3, 0.1, 0.5, 1.0],
                    help="absolute eps grid for random nets")
    ap.add_argument("--pgd_steps", type=int, default=300)
    ap.add_argument("--pgd_restarts", type=int, default=20)
    ap.add_argument("--solver", default=None, help="preferred solver for the main solves (default MOSEK>CLARABEL>SCS)")
    ap.add_argument("--solvers", nargs="*", default=["CLARABEL", "SCS", "MOSEK"],
                    help="solvers for the cross-check (empty = skip)")
    ap.add_argument("--both_signs", action="store_true", help="also solve sigma=+-1 at the cross-check eps")
    ap.add_argument("--max_N", type=int, default=0, help="skip SDP solves whose estimated size N exceeds this (0 = no limit)")
    ap.add_argument("--budget_hours", type=float, default=2.0, help="for the timing projection")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="out_phase2")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    mode = a.mode or ("ckpt" if a.ckpt else "random")
    if mode in ("ckpt", "both") and not a.ckpt:
        ap.error("--ckpt is required for mode ckpt/both")

    print("installed cvxpy solvers:", cp.installed_solvers(), "| cvxpy", cp.__version__)
    print("solver priority used by certify:", _solver_list(a.solver), "| stages:", a.stages)
    insts = []
    if mode in ("random", "both"):
        insts += random_instances(a)
    if mode in ("ckpt", "both"):
        insts += ckpt_instances(a)
    cache = Cache(os.path.join(a.out, "cache.jsonl"))
    print(f"cache: {len(cache.d)} records already in {a.out}/cache.jsonl")

    t0 = time.perf_counter()
    for stage in [s for s in ALL_STAGES if s in a.stages and s != "report"]:
        print(f"\n##### stage {stage}")
        for inst in insts:
            if stage == "pgd":
                stage_pgd(cache, inst, a)
            elif stage == "sdp":
                stage_sdp(cache, inst, a, rlt=False)
            elif stage == "rlt":
                stage_sdp(cache, inst, a, rlt=True)
            elif stage == "cross":
                stage_cross(cache, inst, a)
    if "report" not in a.stages:
        print(f"\nstages done ({time.perf_counter() - t0:.1f}s). Cache: {os.path.join(a.out, 'cache.jsonl')}")
        return
    rows, summ = report(cache, insts, a)
    n_pass = sum(c["passed"] for c in CHECKS)
    print("\n" + "=" * 78 + f"\nchecks: {n_pass}/{len(CHECKS)} passed")
    for c in CHECKS:
        if not c["passed"]:
            print("  FAILED:", c["name"], c["detail"])
    missing = [s for s in ("pgd", "sdp", "rlt") if not any(r.get("stage") == s for r in cache.d.values())]
    if missing:
        print("note: no cached results for stage(s):", missing, "-> the corresponding columns/checks are empty")
    print("ALL PASSED" if OK else "SOME CHECKS FAILED")
    with open(os.path.join(a.out, "phase2_results.json"), "w") as f:
        json.dump(dict(args=vars(a), solvers=cp.installed_solvers(), cvxpy=cp.__version__, rows=rows,
                       summary=summ, checks=CHECKS), f, indent=1, default=float)
    print("saved to", a.out)
    raise SystemExit(0 if OK else 1)


if __name__ == "__main__":
    main()
