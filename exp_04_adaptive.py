"""
Experiment 4: Adaptive Self-Tuning

Demonstrates that a gradient-ascent controller on χ(λ) = |dJ/dλ|
finds and holds the critical window.

Usage:
    python exp_04_adaptive.py
    python exp_04_adaptive.py --ns 3,4,5 --iters 400
"""

import argparse
import csv
import os

import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import UnivariateSpline

from pqs_throughput import (
    compute_J_total_variation,
    commutator_super,
    dissipator_super,
    make_chain_hamiltonian,
    make_dephasing_L_ops,
    projector_bitstring,
)

THIS_DIR = os.path.dirname(os.path.abspath(__file__))


def scan_J(n, lambdas, gamma, theta2, T, steps):
    """Compute J(λ) and χ(λ)."""
    d = 2**n
    rho0 = projector_bitstring("0" * n)
    F = projector_bitstring("0" * n)
    
    H_xx = make_chain_hamiltonian(n, theta_xx=1.0, theta_zz=0.0)
    H_zz = make_chain_hamiltonian(n, theta_xx=0.0, theta_zz=1.0)
    C_xx = commutator_super(H_xx)
    C_zz = commutator_super(H_zz)
    D = dissipator_super(make_dephasing_L_ops(n, gamma=gamma), dim=d)
    
    J = np.zeros(len(lambdas))
    for i, lam in enumerate(lambdas):
        theta1 = lam * gamma
        L = (theta1 * C_xx + theta2 * C_zz + D).tocsr()
        res = compute_J_total_variation(
            L=L, rho0=rho0, F_final=F,
            rho_ss=np.eye(d)/d, T=T, steps=steps,
            use_fast_unital_ss=True
        )
        J[i] = res.J
        if i % 15 == 0:
            print(f"    {i+1}/{len(lambdas)}")
    
    s = 1e-3 * len(lambdas) * (np.max(J)**2 + 1e-12)
    spl = UnivariateSpline(lambdas, J, s=s, k=3)
    chi = np.abs(spl.derivative(1)(lambdas))
    chi[0], chi[-1] = 0, 0
    return J, chi, spl


def adaptive_run(J_spl, lam0, iters, eta, delta, noise, lam_min, lam_max, rng):
    """Run adaptive dynamics."""
    dJ = J_spl.derivative(1)
    lam = lam0
    traj = np.zeros(iters)
    
    for k in range(iters):
        chi_plus = abs(float(dJ(np.clip(lam + delta, lam_min, lam_max))))
        chi_minus = abs(float(dJ(np.clip(lam - delta, lam_min, lam_max))))
        grad = (chi_plus - chi_minus) / (2 * delta)
        lam = np.clip(lam + eta * grad + noise * rng.normal(), lam_min, lam_max)
        traj[k] = lam
    
    return traj


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--outdir", default="out")
    parser.add_argument("--ns", default="3,4,5")
    parser.add_argument("--gamma", type=float, default=0.2)
    parser.add_argument("--theta2", type=float, default=0.5)
    parser.add_argument("--T", type=float, default=6.0)
    parser.add_argument("--steps", type=int, default=180)
    parser.add_argument("--lam-max", type=float, default=5.0)
    parser.add_argument("--lam-points", type=int, default=51)
    parser.add_argument("--iters", type=int, default=400)
    parser.add_argument("--eta", type=float, default=0.25)
    parser.add_argument("--delta", type=float, default=0.08)
    parser.add_argument("--noise", type=float, default=0.12)
    parser.add_argument("--lam0", type=float, default=0.5)
    parser.add_argument("--burnin", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    
    outdir = os.path.join(THIS_DIR, args.outdir)
    os.makedirs(outdir, exist_ok=True)
    
    ns = [int(x) for x in args.ns.split(",") if x.strip()]
    lambdas = np.linspace(0, args.lam_max, args.lam_points)
    rng = np.random.default_rng(args.seed)
    
    results = {}
    summary = []
    
    for n in ns:
        print(f"\n=== n={n} ===")
        print("  Scanning J(λ)...")
        J, chi, J_spl = scan_J(n, lambdas, args.gamma, args.theta2, args.T, args.steps)
        
        # Find λ_c
        m = max(1, len(lambdas)//10)
        mask = np.zeros_like(lambdas, dtype=bool)
        mask[m:-m] = True
        idx = np.argmax(chi * mask)
        lam_c = lambdas[idx]
        
        print("  Running adaptive...")
        traj = adaptive_run(J_spl, args.lam0, args.iters, args.eta, args.delta,
                           args.noise, 0.3, lambdas[-1], rng)
        tail = traj[args.burnin:]
        
        results[n] = {"lambdas": lambdas, "chi": chi, "lam_c": lam_c,
                      "chi_max": chi[idx], "tail": tail}
        summary.append([n, lam_c, chi[idx], np.mean(tail), np.std(tail)])
        print(f"  λ_c={lam_c:.2f}, mean(tail)={np.mean(tail):.2f}")
    
    # Save summary
    with open(os.path.join(outdir, "adaptive_summary.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["n", "lambda_c", "chi_max", "lambda_mean", "lambda_std"])
        for row in summary:
            w.writerow([row[0]] + [f"{x:.4f}" for x in row[1:]])
    
    # Plot
    plt.style.use("seaborn-v0_8-paper")
    fig, axes = plt.subplots(2, len(ns), figsize=(4*len(ns), 6))
    if len(ns) == 1:
        axes = axes.reshape(2, 1)
    
    for i, n in enumerate(ns):
        r = results[n]
        
        # Top: χ landscape
        ax = axes[0, i]
        ax.plot(r["lambdas"], r["chi"], 'C0-', lw=2)
        ax.axvline(r["lam_c"], color='C3', ls='--', lw=2, label=f'$\\lambda_c={r["lam_c"]:.2f}$')
        ax.fill_between(r["lambdas"], 0, r["chi"],
                        where=(np.abs(r["lambdas"] - r["lam_c"]) < 0.5),
                        alpha=0.2, color='C3')
        ax.set_xlabel(r"$\lambda$")
        ax.set_ylabel(r"$\chi(\lambda)$")
        ax.set_title(f"n = {n}", fontweight='bold')
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0, r["lambdas"][-1])
        
        # Bottom: histogram
        ax = axes[1, i]
        ax.hist(r["tail"], bins=30, density=True, alpha=0.7, color='C3', edgecolor='white')
        ax.axvline(r["lam_c"], color='k', ls='--', lw=2)
        ax.set_xlabel(r"$\lambda$")
        ax.set_ylabel("Density")
        ax.set_xlim(0, r["lambdas"][-1])
        ax.grid(True, alpha=0.3)
        ax.text(0.95, 0.95, f'mean = {np.mean(r["tail"]):.2f}',
                transform=ax.transAxes, ha='right', va='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    fig.suptitle("Adaptive Self-Tuning: Controller Finds Critical Window", fontweight='bold')
    plt.tight_layout()
    fig.savefig(os.path.join(outdir, "adaptive.png"), dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"\nOutput: {outdir}")


if __name__ == "__main__":
    main()
