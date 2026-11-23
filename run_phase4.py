"""Phase 4 driver: full certification run, speed benchmark and the tightness plot.   (revised)

    python -m project.run_phase4 --ckpt out_phase1/ckpt.pt --n_samples 20 --rlt both --out out_phase4
    python -m project.run_phase4 --ckpt out_phase1/ckpt.pt --n_samples 20 --both_signs --lipsdp --out out_phase4_full
    python -m project.run_phase4 --out out_phase4 --plot_only          # redo summary/plots from saved results

Changes w.r.t. the first version
  * --rlt {off,on,both} (default both): plain SDP and SDP+RLT are stored and reported SEPARATELY
    (before, the column called "SDP" was the RLT result when --rlt on).
  * Vacuity: primary threshold --vac (default 1.0 = error of the trivial estimator f == 0, since ||x|| = 1),
    plus the sign-flip saturation level 3.5 for the SDP.  Vacuous fractions are reported for SDP, SDP+RLT, IBP
    and Lipschitz, and the largest eps_rel at which each method's median bound is still non-vacuous.
  * Resume safety: the settings that change results (rlt, both_signs, lipsdp, L used, solver, PGD settings, seed,
    eps grid, ckpt) are stored in meta.json["config"] BEFORE anything is overwritten.  A re-run with different
    settings into the same --out stops with a message (use another --out, or --fresh to archive the old results).
    --n_samples / --start may differ (that is how you extend a run).
  * Timing: one warm-up solve before timing; the SAME solver (--solver) is used for SDP and PhaseLift; PhaseLift is
    timed in both forms (noise-free equality, and eta = eps l2 form); SDP time is stored for plain and RLT separately;
    solver names, CPU count and load average are recorded so odd timings can be diagnosed.
  * Summary tables report medians AND means, SDP/PGD ratio (median, mean, max), and both Lipschitz constants.

For every test sample i and every eps on the common absolute grid it records
  * SDP certified upper bound (fixed sign sigma0; optionally also min over sigma=+-1, --both_signs)
  * PGD lower bounds: fixed-sign (comparable to the SDP bound) and sign-invariant (empirical worst case of metric (2))
  * Lipschitz baseline (spectral product and, with --lipsdp, LipSDP), IBP output bound
  * PhaseLift error (l2 noise constraint, eta = eps) on y + delta_PGD and on a random Gaussian perturbation of the same
    norm; PhaseLift clean error
  * timings, and whether PGD triggers an output sign flip (fixed-sign bound then jumps to ~4)
Results are appended to <out>/results.jsonl after every (sample, eps) so an interrupted run can be resumed.
Writes <out>/summary.json, summary.csv, tightness.png/.pdf, timing.png.

FIGURE CAPTION (also printed): see CAPTION below.
"""
import argparse, copy, csv, datetime, json, os, textwrap, time
import numpy as np
import cvxpy as cp
import torch

from .common import measure, load_signals, y_norm_stats, eps_grid_abs
from .model import load_checkpoint, sign_invariant_sq_error
from .pgd import pgd_attack
from .phaselift import phaselift, sign_aligned_error, _solver_list
from .bounds import get_weights, preactivation_bounds, ibp_output_bound, forward, best_sigma0
from .certify import certify_full, certify_both_signs
from .run_phase2 import np_pgd
from .run_phase3 import lipschitz_spectral, lipschitz_lipsdp, lip_bound

CAPTION = ("SDP / SDP+RLT: certified upper bounds on the worst-case fixed-sign squared error of the trained network over "
           "||delta||_2<=eps (SDP+RLT adds pairwise product constraints). PGD: empirical lower bound (fixed-sign, and "
           "sign-invariant worst case). Lipschitz/IBP: cheap upper bounds. The dotted line at 1 is the error of the trivial "
           "estimator f=0; a bound >= 1 is called vacuous. PhaseLift is a different estimator: its curves show its error on "
           "y+delta_PGD (attack crafted against the network) and on a random Gaussian delta of equal norm -- neither a "
           "PhaseLift worst case nor a certified bound.")
VAC_FLIP = 3.5      # fixed-sign error saturates near 4 when the output sign flips (||x||=1, ||f||~1)


# ----------------------------------------------------------------------------- small helpers
def _f(v):
    return float("nan") if v is None else float(v)


def _nanmean(v):
    v = [u for u in v if u is not None and np.isfinite(u)]
    return float(np.mean(v)) if v else float("nan")


def _nanmed(v):
    v = [u for u in v if u is not None and np.isfinite(u)]
    return float(np.median(v)) if v else float("nan")


def _nanmax(v):
    v = [u for u in v if u is not None and np.isfinite(u)]
    return float(np.max(v)) if v else float("nan")


def _vfrac(R, key, thr):
    """Fraction of samples whose bound is >= thr (inf = solver failure counts as vacuous; missing values ignored)."""
    v = [_f(r.get(key)) for r in R]
    v = [u for u in v if not np.isnan(u)]
    return float(np.mean([u >= thr for u in v])) if v else float("nan")


def _max_nonvacuous(summ, key, thr):
    """Largest eps_rel such that the MEDIAN bound is < thr at that eps_rel and at every smaller one."""
    best = None
    for s in sorted(summ, key=lambda s: s["eps_rel"]):
        m = s.get(key + "_median", float("nan"))
        if np.isfinite(m) and m < thr:
            best = s["eps_rel"]
        else:
            break
    return best


def _safe_pl(A, y, x, eta, solver=None):
    """PhaseLift error, solve time, solver name.  A failure must not kill a long run."""
    try:
        xh, info = phaselift(A, y, eta=eta, solver=solver)
        if xh is None:
            return float("nan"), info["time"], info.get("solver")
        return sign_aligned_error(xh, x), info["time"], info.get("solver")
    except Exception as e:
        print("   PhaseLift failed:", e)
        return float("nan"), float("nan"), None


def _config(a, eps_rel, L):
    """Everything that changes the numbers in results.jsonl (n_samples/start/out deliberately excluded)."""
    return dict(schema=2, ckpt=a.ckpt, rlt=a.rlt, both_signs=bool(a.both_signs), lipsdp=bool(a.lipsdp),
                L_used=round(float(L), 6), solver=a.solver, pgd_steps=a.pgd_steps, pgd_restarts=a.pgd_restarts,
                seed=a.seed, eps_rel=[round(float(e), 9) for e in eps_rel])


def _normalise(rows, args):
    """Bring rows from the first run_phase4 version up to the current schema (for --plot_only on old results)."""
    out = []
    for r in rows:
        r = dict(r)
        if "sdp_rlt" not in r:                       # legacy: `sdp` was the RLT result when --rlt on
            if args.get("rlt", "on") == "on":
                r["sdp_rlt"], r["sdp"] = r.get("sdp"), None
            else:
                r["sdp_rlt"] = None
        if "sdp_main" not in r:
            r["sdp_main"] = r["sdp_rlt"] if r.get("sdp_rlt") is not None else r.get("sdp")
        if "time_pl_eps" not in r:
            r["time_pl_eps"] = r.get("time_phaselift")
        r.setdefault("lipschitz_spec", r.get("lipschitz"))
        out.append(r)
    return out


# ----------------------------------------------------------------------------- summary / plots
def summarise(rows, eps_rel, out, clean_net_mean=None, vac=1.0):
    keys = ["pgd_invariant", "pgd_fixed", "sdp", "sdp_rlt", "sdp_main", "sdp_both", "ibp", "lipschitz", "lipschitz_spec",
            "phaselift_pgd", "phaselift_rand", "gap", "time_sdp_plain", "time_sdp_rlt", "time_sdp", "time_pl_noisefree",
            "time_pl_eps", "time_net"]
    summ = []
    for er in eps_rel:
        R = [r for r in rows if abs(r["eps_rel"] - er) < 1e-12]
        if not R:
            continue
        s = dict(eps_rel=float(er), eps=R[0]["eps"], n=len(R))
        for k in keys:
            vals = [r.get(k) for r in R]
            s[k + "_mean"], s[k + "_median"] = _nanmean(vals), _nanmed(vals)
        ratios = [_f(r.get("sdp_main")) / max(_f(r.get("pgd_fixed")), 1e-12) for r in R]
        s["ratio_median"], s["ratio_mean"], s["ratio_max"] = _nanmed(ratios), _nanmean(ratios), _nanmax(ratios)
        s["flip_frac"] = float(np.mean([bool(r.get("sign_flip")) for r in R]))
        s["flip_inv_frac"] = float(np.mean([bool(r.get("sign_flip_inv")) for r in R]))
        for k in ("sdp_main", "sdp", "sdp_rlt", "ibp", "lipschitz", "lipschitz_spec"):
            s["vac_" + k] = _vfrac(R, k, vac)
        s["vac_flip_sdp_main"] = _vfrac(R, "sdp_main", VAC_FLIP)
        s["unstable_mean"] = _nanmean([r.get("n_unstable") for r in R])
        summ.append(s)
    if not summ:
        print("no rows to summarise"); return []

    has_plain = any(np.isfinite(s["sdp_median"]) for s in summ)
    has_rlt = any(np.isfinite(s["sdp_rlt_median"]) for s in summ)
    sdp_label = "SDP+RLT" if has_rlt else "SDP"
    W = 124
    print("\n" + "=" * W + f"\nSUMMARY A: medians over samples (squared l2 errors; vacuous = bound >= {vac:g})")
    print(f"{'eps_rel':>7} {'n':>3} {'PGD inv':>9} {'PGD fix':>9} {'SDP':>9} {'SDP+RLT':>9} {'IBP':>9} {'Lip(used)':>9} "
          f"{'Lip(spec)':>9} {'PL@PGD':>9} {'PL@rand':>9}")
    for s in summ:
        print(f"{s['eps_rel']:7.3f} {s['n']:3d} {s['pgd_invariant_median']:9.4f} {s['pgd_fixed_median']:9.4f} "
              f"{s['sdp_median']:9.4f} {s['sdp_rlt_median']:9.4f} {s['ibp_median']:9.4f} {s['lipschitz_median']:9.4f} "
              f"{s['lipschitz_spec_median']:9.4f} {s['phaselift_pgd_median']:9.4f} {s['phaselift_rand_median']:9.4f}")
    print("\nSUMMARY B: means over samples")
    print(f"{'eps_rel':>7} {'n':>3} {'PGD inv':>9} {'PGD fix':>9} {'SDP':>9} {'SDP+RLT':>9} {'IBP':>9} {'Lip(used)':>9} "
          f"{'PL@PGD':>9} {'PL@rand':>9}")
    for s in summ:
        print(f"{s['eps_rel']:7.3f} {s['n']:3d} {s['pgd_invariant_mean']:9.4f} {s['pgd_fixed_mean']:9.4f} "
              f"{s['sdp_mean']:9.4f} {s['sdp_rlt_mean']:9.4f} {s['ibp_mean']:9.4f} {s['lipschitz_mean']:9.4f} "
              f"{s['phaselift_pgd_mean']:9.4f} {s['phaselift_rand_mean']:9.4f}")
    print(f"\nSUMMARY C: tightness ({sdp_label} / PGD-fixed) and vacuity (% of samples with bound >= {vac:g}; last col >= {VAC_FLIP})")
    print(f"{'eps_rel':>7} {'ratio med':>9} {'ratio mean':>10} {'ratio max':>9} {'#unst':>6} {'flip%':>6} {'vac SDP':>8} "
          f"{'vac +RLT':>8} {'vac IBP':>8} {'vac Lip':>8} {'sdp>=3.5':>8}")
    for s in summ:
        p = lambda k: 100 * s[k]
        print(f"{s['eps_rel']:7.3f} {s['ratio_median']:9.3f} {s['ratio_mean']:10.3f} {s['ratio_max']:9.3f} "
              f"{s['unstable_mean']:6.1f} {100 * s['flip_frac']:6.0f} {p('vac_sdp'):8.0f} {p('vac_sdp_rlt'):8.0f} "
              f"{p('vac_ibp'):8.0f} {p('vac_lipschitz'):8.0f} {p('vac_flip_sdp_main'):8.0f}")
    maxnv = {m: _max_nonvacuous(summ, m, vac) for m in ("sdp", "sdp_rlt", "ibp", "lipschitz", "lipschitz_spec")}
    print(f"\nlargest eps_rel with median bound < {vac:g} (and all smaller ones too): "
          + " | ".join(f"{k}: {v}" for k, v in maxnv.items()))

    allr = lambda k: [r.get(k) for r in rows]
    t_net, t_plz, t_ple = allr("time_net"), allr("time_pl_noisefree"), allr("time_pl_eps")
    t_sp, t_sr, t_sm = allr("time_sdp_plain"), allr("time_sdp_rlt"), allr("time_sdp")
    print(f"\ntiming (median s): SDP plain {_nanmed(t_sp):.3f} | SDP+RLT {_nanmed(t_sr):.3f} | SDP incl. extra signs {_nanmed(t_sm):.3f} | "
          f"PhaseLift noise-free {_nanmed(t_plz):.3f} | PhaseLift eta=eps {_nanmed(t_ple):.3f} | "
          f"network inference {_nanmed(t_net) * 1e3:.3f} ms")
    print("  solvers used: SDP", sorted({str(r.get('sdp_solver')) for r in rows}), "| PhaseLift",
          sorted({str(r.get('pl_solver')) for r in rows}),
          "\n  (compare timings only within one run: same machine, same solver, no other jobs running)")

    with open(os.path.join(out, "summary.json"), "w") as f:
        json.dump(dict(summary=summ, caption=CAPTION, clean_net_mean=clean_net_mean, vac_threshold=vac,
                       vac_flip_threshold=VAC_FLIP, max_nonvacuous_eps_rel=maxnv), f, indent=1, default=float)
    with open(os.path.join(out, "summary.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(summ[0].keys()))
        w.writeheader(); w.writerows(summ)

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed -> skipping plots"); return summ
    x = [s["eps_rel"] for s in summ]
    med = lambda k: [s[k + "_median"] for s in summ]
    ok = lambda k: not all(np.isnan(med(k)))
    fig, ax = plt.subplots(figsize=(7.8, 5.4))
    if ok("sdp_rlt"):
        ax.plot(x, med("sdp_rlt"), "o-", c="C3", lw=2, label="SDP+RLT certified upper bound")
    if ok("sdp"):
        ax.plot(x, med("sdp"), "o--", c="C3", alpha=.6, lw=1.5, label="SDP (no RLT) certified upper bound")
    if ok("sdp_both"):
        ax.plot(x, med("sdp_both"), "x:", c="C3", label=r"SDP, min over $\sigma=\pm1$")
    if ok("lipschitz_spec") and ok("lipschitz") and not np.allclose(med("lipschitz_spec"), med("lipschitz"), equal_nan=True):
        ax.plot(x, med("lipschitz_spec"), "s:", c="C1", alpha=.6, label="Lipschitz (spectral product)")
        ax.plot(x, med("lipschitz"), "s-", c="C1", label="Lipschitz (LipSDP)")
    else:
        ax.plot(x, med("lipschitz"), "s-", c="C1", label="Lipschitz bound")
    ax.plot(x, med("ibp"), "^-", c="C2", label="IBP output bound")
    ax.plot(x, med("pgd_fixed"), "v-", c="C0", label="PGD lower bound (fixed sign)")
    ax.plot(x, med("pgd_invariant"), "v--", c="C0", label="PGD worst case (sign-invariant)")
    ax.plot(x, med("phaselift_pgd"), "d-", c="C4", label=r"PhaseLift on y+$\delta_{PGD}$")
    ax.plot(x, med("phaselift_rand"), "d--", c="C5", label=r"PhaseLift on y+random $\delta$")
    ax.axhline(vac, c="k", ls=":", lw=1, label=f"vacuity level ({vac:g})")
    if clean_net_mean is not None:
        ax.axhline(clean_net_mean, c="gray", ls=":", lw=1, label="network clean error (mean)")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel(r"$\epsilon_{rel}=\epsilon/\overline{\|y\|_2}$"); ax.set_ylabel("squared $\\ell_2$ error")
    ax.set_title("Tightness of the SDP certificate (median over test samples)")
    ax.grid(alpha=.3, which="both"); ax.legend(fontsize=7, loc="best")
    fig.text(0.01, 0.005, "\n".join(textwrap.wrap(CAPTION, 125)), fontsize=6, va="bottom")
    fig.tight_layout(rect=(0, 0.17, 1, 1))
    fig.savefig(os.path.join(out, "tightness.png"), dpi=200); fig.savefig(os.path.join(out, "tightness.pdf"))
    plt.close(fig)

    names = ["network\ninference", "PhaseLift\nnoise-free", "PhaseLift\n" + r"$\eta=\epsilon$", "SDP", "SDP+RLT"]
    vals = [_nanmed(t_net), _nanmed(t_plz), _nanmed(t_ple), _nanmed(t_sp), _nanmed(t_sr)]
    keep = [(n_, v_) for n_, v_ in zip(names, vals) if np.isfinite(v_)]
    fig, ax = plt.subplots(figsize=(6.2, 3.6))
    ax.bar([k[0] for k in keep], [k[1] for k in keep], color=["C2", "C4", "C4", "C3", "C3"][:len(keep)] if len(keep) == 5
           else None)
    ax.set_yscale("log"); ax.set_ylabel("seconds per sample (median)"); ax.grid(alpha=.3, axis="y", which="both")
    fig.tight_layout(); fig.savefig(os.path.join(out, "timing.png"), dpi=200); plt.close(fig)
    print("plots:", os.path.join(out, "tightness.png"), os.path.join(out, "timing.png"))
    print("\nCAPTION:", CAPTION)
    return summ


# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="out_phase1/ckpt.pt")
    ap.add_argument("--n_samples", type=int, default=20)
    ap.add_argument("--start", type=int, default=0, help="first test index")
    ap.add_argument("--rlt", choices=["off", "on", "both"], default="both",
                    help="off: plain SDP only | on: SDP+RLT only | both: store both (default)")
    ap.add_argument("--both_signs", action="store_true", help="also solve sigma=+-1 (doubles SDP time)")
    ap.add_argument("--lipsdp", action="store_true")
    ap.add_argument("--rel_levels", type=float, nargs="+", default=None,
                    help="eps_rel grid (default: the one stored in the checkpoint), e.g. 0.01 0.05 0.1 0.2 0.5")
    ap.add_argument("--vac", type=float, default=1.0, help="vacuity threshold for the summaries (1 = trivial estimator f=0)")
    ap.add_argument("--pgd_steps", type=int, default=300)
    ap.add_argument("--pgd_restarts", type=int, default=20)
    ap.add_argument("--solver", default=None, help="used for SDP, PhaseLift and LipSDP alike (default MOSEK>CLARABEL>SCS)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="out_phase4")
    ap.add_argument("--plot_only", action="store_true")
    ap.add_argument("--fresh", action="store_true", help="archive existing results in --out and start over")
    ap.add_argument("--no_warmup", action="store_true", help="skip the warm-up solves before timing")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    path = os.path.join(a.out, "results.jsonl")
    meta_path = os.path.join(a.out, "meta.json")

    if a.plot_only:
        meta = json.load(open(meta_path))
        rows = _normalise([json.loads(l) for l in open(path)], meta.get("args", {}))
        eps_rel = sorted({r["eps_rel"] for r in rows})
        summarise(rows, eps_rel, a.out, meta.get("clean_net_mean"), a.vac)
        return

    model, A, meta = load_checkpoint(a.ckpt)
    side, data, seed = meta.get("side", 8), meta.get("data", "mnist"), meta.get("seed", 0)
    _, Xte = load_signals(data, side, meta.get("n_train", 20000), meta.get("n_test", 500), seed=seed)
    Yte = measure(A, Xte)
    rel = a.rel_levels or meta.get("rel_levels", [0.005, 0.01, 0.02, 0.05, 0.1])
    eps_abs, eps_rel = eps_grid_abs(Yte, rel)
    m64 = copy.deepcopy(model).cpu().double().eval()
    weights = get_weights(m64)
    idxs = list(range(a.start, min(a.start + a.n_samples, len(Xte))))
    stats = y_norm_stats(Yte)
    rlt_modes = {"off": [False], "on": [True], "both": [False, True]}[a.rlt]
    main_rlt = rlt_modes[-1]                       # the variant used for ratios / validity / both-signs
    print(f"dims {model.dims} | data={data} | {len(idxs)} samples x {len(eps_abs)} eps | rlt={a.rlt} both_signs={a.both_signs}")
    print("||y|| stats:", {k: round(float(v), 3) for k, v in stats.items()}, "| eps_abs:", np.round(eps_abs, 4))
    print("solvers:", cp.installed_solvers(), "priority:", _solver_list(a.solver), "cvxpy", cp.__version__)
    load0 = os.getloadavg() if hasattr(os, "getloadavg") else None
    print(f"machine: {os.cpu_count()} CPUs, load average at start {load0}")

    with torch.no_grad():
        clean_net = sign_invariant_sq_error(m64(torch.as_tensor(Yte)), torch.as_tensor(Xte)).numpy()

    L_spec = lipschitz_spectral(weights)
    L = L_spec
    if a.lipsdp:
        L = min(L_spec, lipschitz_lipsdp(weights, a.solver))
    print(f"Lipschitz: spectral {L_spec:.4f}, used {L:.4f}")

    # ---- resume safety: compare settings BEFORE overwriting meta.json
    cfg = _config(a, eps_rel, L)
    if os.path.exists(path) and os.path.getsize(path) > 0:
        old = json.load(open(meta_path)).get("config") if os.path.exists(meta_path) else None
        if a.fresh or old != cfg:
            if not a.fresh:
                print("\nERROR: results.jsonl in", a.out, "was produced with different settings; reusing it would mix runs.")
                if old is None:
                    print("  (no config recorded -> results from an older run_phase4)")
                else:
                    for k in cfg:
                        if old.get(k) != cfg[k]:
                            print(f"  {k}: stored {old.get(k)!r} vs now {cfg[k]!r}")
                print("  -> use a different --out, or add --fresh to archive the old results and start over.")
                raise SystemExit(2)
            stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            os.replace(path, f"{path}.{stamp}.bak")
            if os.path.exists(meta_path):
                os.replace(meta_path, f"{meta_path}.{stamp}.bak")
            print(f"archived old results as *.{stamp}.bak")
    json.dump(dict(args=vars(a), config=cfg, dims=model.dims, data=data, eps_abs=eps_abs.tolist(),
                   eps_rel=list(map(float, eps_rel)), y_norm_stats={k: float(v) for k, v in stats.items()},
                   clean_net_mean=float(clean_net.mean()), lipschitz=dict(spectral=L_spec, used=L),
                   solvers=cp.installed_solvers(), cvxpy=cp.__version__, n_cpu=os.cpu_count(), loadavg_start=load0),
              open(meta_path, "w"), indent=1, default=float)

    done = set()
    rows = []
    if os.path.exists(path):
        for l in open(path):
            r = json.loads(l); rows.append(r); done.add((r["sample"], r["k"]))
        print(f"resuming: {len(done)} entries already done (same settings)")
    fout = open(path, "a")

    # ---- warm-up so that the first timed solves are not dominated by imports / licence checks / JIT
    if not a.no_warmup and idxs:
        i0, e0 = idxs[0], eps_abs[len(eps_abs) // 2]
        b0 = preactivation_bounds(weights, Yte[i0], e0)
        certify_full(weights, b0, Yte[i0], Xte[i0], e0, rlt=main_rlt, solver=a.solver)
        _safe_pl(A, Yte[i0], Xte[i0], 0.0, a.solver)
        _safe_pl(A, Yte[i0], Xte[i0], float(e0), a.solver)
        print("warm-up done")

    pl_clean_cache = {}
    t_start = time.perf_counter()
    rng = np.random.default_rng(a.seed + 123)
    total = len(idxs) * len(eps_abs)
    tol_ = lambda lb: 1e-6 * max(1.0, lb)

    for i in idxs:
        y, x = Yte[i], Xte[i]
        yt, xt = torch.as_tensor(y[None]), torch.as_tensor(x[None])
        sigma0 = best_sigma0(weights, y, x)
        clean = float(np.sum((forward(weights, y) - sigma0 * x) ** 2))
        with torch.no_grad():                                           # network inference time (float64, 1 sample)
            t0 = time.perf_counter()
            for _ in range(200):
                m64(yt)
            t_net = (time.perf_counter() - t0) / 200
        if i not in pl_clean_cache:                                     # noise-free PhaseLift (equality form), timed
            pl_clean_cache[i] = _safe_pl(A, y, x, 0.0, a.solver)
        for k, (eps, er) in enumerate(zip(eps_abs, eps_rel)):
            if (i, k) in done:
                continue
            bnds = preactivation_bounds(weights, y, eps)
            res_by = {rl: certify_full(weights, bnds, y, x, eps, sigma0=sigma0, rlt=rl, solver=a.solver) for rl in rlt_modes}
            res = res_by[main_rlt]
            sdp_both, sdp_signs, t_both = None, None, 0.0
            if a.both_signs:
                best, both = certify_both_signs(weights, bnds, y, x, eps, rlt=main_rlt, solver=a.solver)
                sdp_both, sdp_signs = best["bound"], [r["bound"] for r in both]
                t_both = sum(r.get("time", 0.0) for r in both)
            lb, d_fix = np_pgd(weights, y, x, sigma0, eps, a.pgd_steps, a.pgd_restarts, seed=a.seed + i)
            inv, d_inv = pgd_attack(m64, yt, xt, eps, "invariant", None, a.pgd_steps, a.pgd_restarts, seed=a.seed + i)
            d_inv = d_inv.numpy()[0]
            flip = bool(float(forward(weights, y + d_fix) @ x) * sigma0 < 0)
            flip_inv = bool(float(forward(weights, y + d_inv) @ x) * sigma0 < 0)
            rd = rng.standard_normal(y.shape); rd *= eps / np.linalg.norm(rd)
            pl_pgd, t_pl, pl_solver = _safe_pl(A, y + d_inv, x, eps, a.solver)
            pl_rand, _, _ = _safe_pl(A, y + rd, x, eps, a.solver)
            plain, rltr = res_by.get(False), res_by.get(True)
            valid = all(lb <= r["bound"] + tol_(lb) for r in res_by.values())
            row = dict(sample=i, k=k, eps=float(eps), eps_rel=float(er), sigma0=sigma0, clean_fixed=clean,
                       clean_net_inv=float(clean_net[i]), rlt_mode=a.rlt,
                       sdp=None if plain is None else plain["bound"], sdp_rlt=None if rltr is None else rltr["bound"],
                       sdp_main=res["bound"], sdp_primal=res.get("primal_value"), gap_dual=res.get("gap"),
                       sdp_status=res["status"], sdp_solver=res.get("solver"), t_corr=res.get("t"),
                       lam_min_Z=res.get("lam_min_Z"), sdp_both=sdp_both, sdp_signs=sdp_signs,
                       n_unstable=res.get("n_unstable", 0), N=res.get("N", 0),
                       pgd_fixed=lb, pgd_invariant=float(inv), sign_flip=flip, sign_flip_inv=flip_inv,
                       ibp=ibp_output_bound(weights, y, eps, x, sigma0),
                       lipschitz=lip_bound(weights, y, x, sigma0, eps, L),
                       lipschitz_spec=lip_bound(weights, y, x, sigma0, eps, L_spec),
                       gap=res["bound"] - lb, valid=bool(valid),
                       phaselift_pgd=pl_pgd, phaselift_rand=pl_rand, phaselift_clean=pl_clean_cache[i][0],
                       pl_solver=pl_solver, time_pl_noisefree=pl_clean_cache[i][1], time_pl_eps=t_pl,
                       time_phaselift=t_pl,                                     # kept for older plotting code
                       time_sdp_plain=None if plain is None else plain.get("time"),
                       time_sdp_rlt=None if rltr is None else rltr.get("time"),
                       time_sdp=res.get("time", 0.0) + t_both, time_net=t_net)
            rows.append(row)
            fout.write(json.dumps(row, default=float) + "\n"); fout.flush()
            elapsed = time.perf_counter() - t_start
            f_ = lambda v: "   n/a " if v is None else f"{v:7.4f}"
            print(f"[{len(rows)}/{total}] s{i} eps_rel={er:.3f} unst={row['n_unstable']:3d}  PGDinv {float(inv):.4f} "
                  f"PGDfix {lb:.4f} SDP {f_(row['sdp'])} SDP+RLT {f_(row['sdp_rlt'])} IBP {row['ibp']:.4f} "
                  f"Lip {row['lipschitz']:.4f} PL@PGD {pl_pgd:.4f} PL@rand {pl_rand:.4f} t_sdp {row['time_sdp']:.2f}s"
                  f"{'  FLIP' if flip else ''}{'  <-- PGD > SDP (BUG?)' if not valid else ''}  [{elapsed / 60:.1f} min]",
                  flush=True)
    fout.close()

    bad = [r for r in rows if not r["valid"]]
    print(f"\nvalidity: {len(rows) - len(bad)}/{len(rows)} entries have PGD <= every computed SDP bound"
          + (" (!!! check above)" if bad else ""))
    if hasattr(os, "getloadavg"):
        print("load average at end:", os.getloadavg())
    summarise(rows, list(map(float, eps_rel)), a.out, float(clean_net.mean()), a.vac)
    print("saved to", a.out)


if __name__ == "__main__":
    main()
