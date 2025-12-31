"""
Experiment 2: Finite-Size Scaling Collapse

Fits collapse parameters (λ_c, ν, κ) from FSS data and computes
bootstrap uncertainties.

Usage:
    python exp_02_collapse.py
    python exp_02_collapse.py --ns 3,4,5,6,7,8 --bootstrap 100
"""

import argparse
import csv
import os

import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.optimize import minimize

THIS_DIR = os.path.dirname(os.path.abspath(__file__))


def load_curve(path):
    """Load (lambda, chi) from CSV."""
    lam, chi = [], []
    with open(path) as f:
        for row in csv.DictReader(f):
            lam.append(float(row["lambda"]))
            chi.append(float(row["chi"]))
    return np.array(lam), np.array(chi)


def collapse_residual(params, data, ref_n):
    """Compute mismatch for collapse fit."""
    lam_c, nu, kappa = params
    if not (0.2 <= nu <= 5) or not (0 <= kappa <= 4):
        return 1e9
    
    ref_lam, ref_chi = data[ref_n]
    ref_x = (ref_lam - lam_c) * (ref_n ** (1/nu))
    ref_y = ref_chi / (ref_n ** kappa)
    
    if not np.all(np.isfinite(ref_x)) or not np.all(np.isfinite(ref_y)):
        return 1e9
    
    f_ref = interp1d(ref_x, ref_y, kind="linear", bounds_error=False, fill_value=np.nan)
    
    total, count = 0, 0
    for n, (lam, chi) in data.items():
        if n == ref_n:
            continue
        x = (lam - lam_c) * (n ** (1/nu))
        y = chi / (n ** kappa)
        y_ref = f_ref(x)
        ok = np.isfinite(y_ref) & (y > 0)
        if np.sum(ok) > 0:
            total += np.mean((y[ok] - y_ref[ok])**2)
            count += 1
    
    return total / max(count, 1)


def fit_collapse(data):
    """Fit collapse with grid search + refinement."""
    ns = sorted(data.keys())
    ref_n = ns[len(ns)//2]
    
    lam_guess = np.mean([lam[np.argmax(chi)] for lam, chi in data.values()])
    chi_maxes = [np.max(chi) for _, chi in data.values()]
    log_fit = np.polyfit(np.log(ns), np.log(chi_maxes), 1)
    kappa_init = np.clip(log_fit[0], 0.3, 2.5)
    
    best = (np.array([lam_guess, 1.0, kappa_init]), 1e9)
    for lam_c in np.linspace(lam_guess-1, lam_guess+1, 21):
        for nu in np.linspace(0.3, 2.5, 15):
            for kappa in np.linspace(0.3, 2.0, 15):
                p = np.array([lam_c, nu, kappa])
                r = collapse_residual(p, data, ref_n)
                if r < best[1]:
                    best = (p, r)
    
    res = minimize(lambda p: collapse_residual(p, data, ref_n), best[0],
                   method="Nelder-Mead", options={"maxiter": 500})
    return res.x if res.fun < best[1] else best[0]


def bootstrap(data, B, seed):
    """Bootstrap uncertainties."""
    rng = np.random.default_rng(seed)
    samples = []
    for _ in range(B):
        boot = {}
        for n, (lam, chi) in data.items():
            idx = np.sort(rng.choice(len(lam), len(lam), replace=True))
            boot[n] = (lam[idx], chi[idx])
        try:
            samples.append(fit_collapse(boot))
        except:
            pass
    return np.array(samples)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--outdir", default="out")
    parser.add_argument("--ns", default="3,4,5,6,7,8")
    parser.add_argument("--bootstrap", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    
    outdir = os.path.join(THIS_DIR, args.outdir)
    ns = [int(x) for x in args.ns.split(",") if x.strip()]
    
    data = {}
    for n in ns:
        path = os.path.join(outdir, f"fss_n{n}.csv")
        if os.path.exists(path):
            data[n] = load_curve(path)
    
    if len(data) < 3:
        print("Need at least 3 sizes. Run exp_01 first.")
        return
    
    print(f"Loaded n = {sorted(data.keys())}")
    
    params = fit_collapse(data)
    lam_c, nu, kappa = params
    print(f"Fit: λ_c={lam_c:.4f}, ν={nu:.3f}, κ={kappa:.3f}")
    
    print(f"Bootstrap ({args.bootstrap} samples)...")
    bs = bootstrap(data, args.bootstrap, args.seed)
    errs = np.std(bs, axis=0) if len(bs) > 10 else [0, 0, 0]
    
    # Save parameters
    with open(os.path.join(outdir, "collapse_params.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["parameter", "value", "std"])
        w.writerow(["lambda_c", f"{lam_c:.6f}", f"{errs[0]:.6f}"])
        w.writerow(["nu", f"{nu:.6f}", f"{errs[1]:.6f}"])
        w.writerow(["kappa", f"{kappa:.6f}", f"{errs[2]:.6f}"])
    
    # Plot
    plt.style.use("seaborn-v0_8-paper")
    fig, ax = plt.subplots(figsize=(8, 5))
    for n in sorted(data.keys()):
        lam, chi = data[n]
        x = (lam - lam_c) * (n ** (1/nu))
        y = chi / (n ** kappa)
        ax.plot(x, y, 'o-', markersize=3, lw=1.5, label=f"n={n}")
    
    ax.set_xlabel(r"$(\lambda - \lambda_c) n^{1/\nu}$", fontsize=12)
    ax.set_ylabel(r"$\chi / n^\kappa$", fontsize=12)
    ax.set_title(f"Data Collapse: $\\lambda_c={lam_c:.2f}$, $\\nu={nu:.2f}$, $\\kappa={kappa:.2f}$")
    ax.grid(True, alpha=0.3)
    ax.legend(frameon=True)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "collapse.png"), dpi=300)
    plt.close()
    
    print(f"Output: {outdir}")


if __name__ == "__main__":
    main()
