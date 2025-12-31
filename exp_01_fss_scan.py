"""
Experiment 1: Finite-Size Scaling Scan

Computes entropy throughput J(λ) and susceptibility χ(λ) = |dJ/dλ| 
for n-qubit XX+ZZ chains with local dephasing.

Usage:
    python exp_01_fss_scan.py
    python exp_01_fss_scan.py --ns 3,4,5,6,7,8 --points 61
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


def fwhm(x, y):
    """Full width at half maximum."""
    if np.max(y) <= 0:
        return np.nan
    half = 0.5 * np.max(y)
    idx = np.where(y >= half)[0]
    return x[idx[-1]] - x[idx[0]] if len(idx) >= 2 else np.nan


def compute_chi(lambdas, J):
    """Susceptibility via spline derivative."""
    s = 1e-3 * len(lambdas) * (np.max(J)**2 + 1e-12)
    spl = UnivariateSpline(lambdas, J, s=s, k=3)
    chi = np.abs(spl.derivative(1)(lambdas))
    chi[0], chi[-1] = 0, 0
    return chi


def run_scan(n, lambdas, gamma, theta2, T, steps):
    """Run FSS scan for a single system size."""
    d = 2**n
    rho0 = projector_bitstring("0" * n)
    F = projector_bitstring("0" * n)
    
    H_xx = make_chain_hamiltonian(n, theta_xx=1.0, theta_zz=0.0)
    H_zz = make_chain_hamiltonian(n, theta_xx=0.0, theta_zz=1.0)
    C_xx = commutator_super(H_xx)
    C_zz = commutator_super(H_zz)
    D = dissipator_super(make_dephasing_L_ops(n, gamma=gamma), dim=d)
    
    J_arr = np.zeros(len(lambdas))
    for i, lam in enumerate(lambdas):
        theta1 = lam * gamma
        L = (theta1 * C_xx + theta2 * C_zz + D).tocsr()
        res = compute_J_total_variation(
            L=L, rho0=rho0, F_final=F,
            rho_ss=np.eye(d)/d, T=T, steps=steps,
            use_fast_unital_ss=True
        )
        J_arr[i] = res.J
        if i % 10 == 0 or i == len(lambdas)-1:
            print(f"  n={n}: {i+1}/{len(lambdas)} λ={lam:.2f} J={J_arr[i]:.4f}")
    
    return J_arr, compute_chi(lambdas, J_arr)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--outdir", default="out")
    parser.add_argument("--ns", default="3,4,5,6,7,8")
    parser.add_argument("--gamma", type=float, default=0.2)
    parser.add_argument("--theta2", type=float, default=0.5)
    parser.add_argument("--T", type=float, default=6.0)
    parser.add_argument("--steps", type=int, default=180)
    parser.add_argument("--lam-max", type=float, default=6.0)
    parser.add_argument("--points", type=int, default=61)
    args = parser.parse_args()
    
    outdir = os.path.join(THIS_DIR, args.outdir)
    os.makedirs(outdir, exist_ok=True)
    
    ns = [int(x) for x in args.ns.split(",") if x.strip()]
    lambdas = np.linspace(0, args.lam_max, args.points)
    
    results = {}
    summary = []
    
    for n in ns:
        print(f"\n=== n={n} ===")
        J, chi = run_scan(n, lambdas, args.gamma, args.theta2, args.T, args.steps)
        results[n] = (lambdas, J, chi)
        
        # Find peak
        m = max(1, len(lambdas)//10)
        mask = np.zeros_like(lambdas, dtype=bool)
        mask[m:-m] = True
        idx = np.argmax(chi * mask)
        summary.append([n, lambdas[idx], chi[idx], fwhm(lambdas, chi)])
        
        # Save curve
        with open(os.path.join(outdir, f"fss_n{n}.csv"), "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["lambda", "J", "chi"])
            for l, j, c in zip(lambdas, J, chi):
                w.writerow([f"{l:.6f}", f"{j:.8f}", f"{c:.8f}"])
    
    # Save summary
    with open(os.path.join(outdir, "fss_summary.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["n", "lambda_c", "chi_max", "fwhm"])
        for row in summary:
            w.writerow([row[0], f"{row[1]:.4f}", f"{row[2]:.6f}", f"{row[3]:.4f}"])
    
    # Plot J(λ)
    plt.style.use("seaborn-v0_8-paper")
    fig, ax = plt.subplots(figsize=(8, 5))
    for n in ns:
        ax.plot(results[n][0], results[n][1], lw=2, label=f"n={n}")
    ax.set_xlabel(r"$\lambda = |\theta_1|/\gamma$", fontsize=12)
    ax.set_ylabel(r"$J(\lambda)$", fontsize=12)
    ax.set_title("Entropy Throughput", fontsize=14)
    ax.grid(True, alpha=0.3)
    ax.legend(frameon=True)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "fss_J.png"), dpi=300)
    plt.close()
    
    # Plot χ(λ)
    fig, ax = plt.subplots(figsize=(8, 5))
    for n in ns:
        ax.plot(results[n][0], results[n][2], lw=2, label=f"n={n}")
    ax.set_xlabel(r"$\lambda$", fontsize=12)
    ax.set_ylabel(r"$\chi(\lambda) = |dJ/d\lambda|$", fontsize=12)
    ax.set_title("Susceptibility", fontsize=14)
    ax.grid(True, alpha=0.3)
    ax.legend(frameon=True)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "fss_chi.png"), dpi=300)
    plt.close()
    
    print(f"\nOutput: {outdir}")


if __name__ == "__main__":
    main()
