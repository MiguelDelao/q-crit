"""
Experiment 3: Gap Proxy (Relaxation Rate)

Computes a relaxation-time proxy for the Liouvillian gap by fitting
exponential decay of distance to steady state.

Usage:
    python exp_03_gap.py
    python exp_03_gap.py --ns 3,4,5,6,7,8 --points 31
"""

import argparse
import csv
import os

import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import expm
from scipy.optimize import curve_fit

from pqs_throughput import (
    commutator_super,
    dissipator_super,
    make_chain_hamiltonian,
    make_dephasing_L_ops,
    projector_bitstring,
)

THIS_DIR = os.path.dirname(os.path.abspath(__file__))


def evolve(L, rho0, dt, steps):
    """Evolve density matrix."""
    d = int(np.sqrt(L.shape[0]))
    prop = expm(L.toarray() * dt)
    rho_vec = rho0.flatten()
    traj = [rho_vec.reshape(d, d)]
    for _ in range(steps):
        rho_vec = prop @ rho_vec
        traj.append(rho_vec.reshape(d, d))
    return traj


def fit_gap(traj, rho_ss, times):
    """Fit exponential decay to get gap proxy."""
    dist = np.array([np.linalg.norm(rho - rho_ss, 'fro') for rho in traj])
    
    mid = len(times) // 2
    t_fit, d_fit = times[mid:], dist[mid:]
    valid = d_fit > 1e-12
    if np.sum(valid) < 5:
        return np.nan
    
    try:
        def exp_decay(t, A, g):
            return A * np.exp(-g * t)
        popt, _ = curve_fit(exp_decay, t_fit[valid] - t_fit[valid][0], d_fit[valid],
                           p0=[d_fit[valid][0], 0.1], bounds=([0, 1e-6], [10, 10]))
        return popt[1]
    except:
        return np.nan


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--outdir", default="out")
    parser.add_argument("--ns", default="3,4,5,6,7,8")
    parser.add_argument("--gamma", type=float, default=0.2)
    parser.add_argument("--theta2", type=float, default=0.5)
    parser.add_argument("--T", type=float, default=20.0)
    parser.add_argument("--steps", type=int, default=400)
    parser.add_argument("--lam-min", type=float, default=0.5)
    parser.add_argument("--lam-max", type=float, default=5.0)
    parser.add_argument("--points", type=int, default=31)
    args = parser.parse_args()
    
    outdir = os.path.join(THIS_DIR, args.outdir)
    os.makedirs(outdir, exist_ok=True)
    
    ns = [int(x) for x in args.ns.split(",") if x.strip()]
    lambdas = np.linspace(args.lam_min, args.lam_max, args.points)
    dt = args.T / args.steps
    times = np.arange(args.steps + 1) * dt
    
    results = {}
    rows = []
    
    # Load λ_c from FSS if available
    lam_c = {}
    summary_path = os.path.join(outdir, "fss_summary.csv")
    if os.path.exists(summary_path):
        with open(summary_path) as f:
            for row in csv.DictReader(f):
                lam_c[int(row["n"])] = float(row["lambda_c"])
    
    for n in ns:
        print(f"\n=== n={n} ===")
        d = 2**n
        rho0 = projector_bitstring("0" * n)
        rho_ss = np.eye(d) / d
        
        H_xx = make_chain_hamiltonian(n, theta_xx=1.0, theta_zz=0.0)
        H_zz = make_chain_hamiltonian(n, theta_xx=0.0, theta_zz=1.0)
        C_xx = commutator_super(H_xx)
        C_zz = commutator_super(H_zz)
        D = dissipator_super(make_dephasing_L_ops(n, gamma=args.gamma), dim=d)
        
        gaps = []
        for i, lam in enumerate(lambdas):
            theta1 = lam * args.gamma
            L = (theta1 * C_xx + args.theta2 * C_zz + D).tocsr()
            traj = evolve(L, rho0, dt, args.steps)
            g = fit_gap(traj, rho_ss, times)
            gaps.append(g)
            rows.append([n, lam, g])
            if i % 5 == 0:
                print(f"  {i+1}/{len(lambdas)} λ={lam:.2f} gap={g:.4f}")
        
        results[n] = (lambdas, np.array(gaps))
    
    # Save data
    with open(os.path.join(outdir, "gap_scan.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["n", "lambda", "gap_proxy"])
        for row in rows:
            w.writerow([row[0], f"{row[1]:.6f}", f"{row[2]:.8f}"])
    
    # Plot
    plt.style.use("seaborn-v0_8-paper")
    fig, ax = plt.subplots(figsize=(8, 5))
    for n in sorted(results.keys()):
        lam, gap = results[n]
        line, = ax.plot(lam, gap, 'o-', markersize=4, lw=1.5, label=f"n={n}")
        if n in lam_c:
            ax.axvline(lam_c[n], color=line.get_color(), ls='--', alpha=0.5, lw=1)
    
    ax.set_xlabel(r"$\lambda$", fontsize=12)
    ax.set_ylabel("Gap Proxy (relaxation rate)", fontsize=12)
    ax.set_title("Relaxation Rate Suppression Near Critical Point", fontsize=14)
    ax.grid(True, alpha=0.3)
    ax.legend(frameon=True)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "gap.png"), dpi=300)
    plt.close()
    
    print(f"\nOutput: {outdir}")


if __name__ == "__main__":
    main()
