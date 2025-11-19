import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import expm, norm, eigh, sqrtm
from scipy.ndimage import gaussian_filter1d
from tqdm import tqdm
import warnings

# Suppress warnings
warnings.filterwarnings("ignore")

# ==========================================
# 1. PHYSICS ENGINE
# ==========================================

class TwoQubitSystem:
    def __init__(self):
        self.I = np.eye(2)
        self.X = np.array([[0, 1], [1, 0]])
        self.Z = np.array([[1, 0], [0, -1]])
        self.XX = np.kron(self.X, self.X)
        self.ZZ = np.kron(self.Z, self.Z)
        self.Z1 = np.kron(self.Z, self.I)
        self.Z2 = np.kron(self.I, self.Z)
        self.ket0 = np.array([[1], [0]])
        self.ket1 = np.array([[0], [1]])
        self.ket00 = np.kron(self.ket0, self.ket0)
        self.ket11 = np.kron(self.ket1, self.ket1)
        self.rho0 = self.ket00 @ self.ket00.T.conj()
        self.F_00 = self.rho0
        self.F_11 = self.ket11 @ self.ket11.T.conj()
        self.rho_ss = np.eye(4) / 4.0

    def build_liouvillian(self, theta1, theta2, gamma):
        H = theta1 * self.XX + theta2 * self.ZZ
        L_ops = [self.Z1, self.Z2]
        dim_sq = 16
        L_mat = np.zeros((dim_sq, dim_sq), dtype=complex)
        for i in range(dim_sq):
            basis_vec = np.zeros(dim_sq)
            basis_vec[i] = 1.0
            rho_basis = basis_vec.reshape(4, 4)
            d_rho = -1j * (H @ rho_basis - rho_basis @ H)
            for Op in L_ops:
                term = Op @ rho_basis @ Op.T.conj()
                anticomm = 0.5 * (Op.T.conj() @ Op @ rho_basis + rho_basis @ Op.T.conj() @ Op)
                d_rho += gamma * (term - anticomm)
            L_mat[:, i] = d_rho.reshape(-1)
        return L_mat, H

    def get_relative_entropy(self, rho):
        evals = eigh(rho, eigvals_only=True)
        evals = np.clip(evals, 1e-15, 1.0)
        return np.sum(evals * np.log(evals)) + np.log(4.0)

    def run_experiment(self, theta1, theta2, gamma, T, steps, F_final):
        dt = T / steps
        times = np.linspace(0, T, steps)
        L_mat, H = self.build_liouvillian(theta1, theta2, gamma)
        Prop_Fwd = expm(L_mat * dt)
        Prop_Bwd = expm(L_mat.T.conj() * dt)

        rhos = [self.rho0.reshape(-1)]
        curr = rhos[0]
        for _ in range(steps - 1):
            curr = Prop_Fwd @ curr
            rhos.append(curr)

        Es = [None] * steps
        Es[-1] = F_final.reshape(-1)
        curr = Es[-1]
        for i in range(steps - 1, 0, -1):
            curr = Prop_Bwd @ curr
            Es[i-1] = curr

        rel_entropies = []
        coh_list = []

        for i in range(steps):
            r = rhos[i].reshape(4, 4)
            e = Es[i].reshape(4, 4)
            r = (r + r.conj().T) / 2.0
            e = (e + e.conj().T) / 2.0

            try:
                e_reg = e + 1e-12 * np.eye(4)
                sq_e = sqrtm(e_reg)
                num = sq_e @ r @ sq_e
                prob = np.trace(num).real
                rho_s = num / prob if prob > 1e-12 else r
            except Exception:
                rho_s = r

            rho_s = (rho_s + rho_s.conj().T) / 2.0
            evals, evecs = eigh(rho_s)
            evals = np.clip(evals, 0, 1)
            evals = evals / np.sum(evals)
            rho_s = evecs @ np.diag(evals) @ evecs.T.conj()
            rel_entropies.append(self.get_relative_entropy(rho_s))

            c_norm = norm(r, 'fro')
            coh_list.append(norm(H @ r - r @ H, 'fro') / c_norm if c_norm > 1e-9 else 0)

        sigma_pqs = -np.gradient(rel_entropies, dt)
        J_val = np.trapz(np.abs(sigma_pqs), dx=dt)
        return {
            'J': J_val,
            'times': times,
            'sigma': sigma_pqs,
            'avg_coh': np.mean(coh_list),
            'avg_dec': 2 * gamma
        }

# ==========================================
# 2. MAIN ROUTINE
# ==========================================

def main():
    sys = TwoQubitSystem()
    theta2 = 0.5
    gamma_fixed = 0.2
    steps = 200
    T_horizon = 4.0

    print("--- Running Final Simulation ---")

    # 1. Scan Mechanism (High Resolution)
    theta1s = np.linspace(0.0, 0.6, 1000)
    lambdas = theta1s / gamma_fixed
    Js, cohs, decs = [], [], []

    for t1 in tqdm(theta1s):
        res = sys.run_experiment(t1, theta2, gamma_fixed, T_horizon, steps, sys.F_11)
        Js.append(res['J'])
        cohs.append(res['avg_coh'])
        decs.append(res['avg_dec'])

    # 2. Susceptibility
    Js_smooth = gaussian_filter1d(Js, sigma=12)
    dJ = np.gradient(Js_smooth, lambdas)
    dJ = np.abs(dJ)
    dJ[0] = 0

    # 3. Select Balance Point (Critical Lambda)
    rate_diff = np.abs(np.array(cohs) - np.array(decs))
    valid_mask = lambdas > 0.5
    rate_diff[~valid_mask] = 9999
    crit_idx = np.argmin(rate_diff)
    crit_lambda = lambdas[crit_idx]
    best_theta1 = theta1s[crit_idx]

    print(f"Critical Lambda from Panel A: {crit_lambda:.2f}")

    res_00 = sys.run_experiment(best_theta1, theta2, gamma_fixed, T_horizon, steps, sys.F_00)
    res_11 = sys.run_experiment(best_theta1, theta2, gamma_fixed, T_horizon, steps, sys.F_11)

    # 4. Heatmap data
    res_h = 80
    t1_grid = np.linspace(0.1, 0.6, res_h)
    g_grid = np.linspace(0.05, 0.4, res_h)
    heatmap = np.zeros((res_h, res_h))

    print("Generating Phase Diagram...")
    for i, g in enumerate(tqdm(g_grid)):
        for j, t1 in enumerate(t1_grid):
            r = sys.run_experiment(t1, theta2, g, T_horizon, steps, sys.F_11)
            heatmap[i, j] = r['J']

    # --- PLOTTING ---
    plt.style.use('seaborn-v0_8-paper')

    fig = plt.figure(figsize=(12, 10))
    plt.subplots_adjust(hspace=0.35, wspace=0.25, bottom=0.08, top=0.95, left=0.08, right=0.92)

    # Panel A
    ax1 = fig.add_subplot(2, 2, 1)
    ax1.plot(lambdas, Js_smooth, 'b-', alpha=0.6, linewidth=2, label='PQS Entropy Yield')
    scale = max(Js_smooth) / max(dJ)
    ax1.plot(lambdas, dJ * scale, 'b-', linewidth=3, label=r'Susceptibility $\chi$')

    ax2 = ax1.twinx()
    ax2.plot(lambdas, cohs, 'r--', linewidth=1.5, label=r'$\Gamma_{coh}$')
    ax2.plot(lambdas, decs, 'k:', linewidth=1.5, label=r'$\Gamma_{dec}$')
    ax2.set_ylabel('Rates (Hz)', fontsize=12)

    ax1.axvline(crit_lambda, color='gray', linestyle='--', alpha=0.8)

    lns = ax1.get_lines() + ax2.get_lines()
    labs = [l.get_label() for l in lns]
    ax1.legend(lns, labs, loc='lower right', fontsize=10, frameon=True)
    ax1.set_title('(a) Critical Crossover', fontsize=14, fontweight='bold')
    ax1.set_xlabel(r'Control Ratio $\lambda = |\theta_1|/\gamma$', fontsize=12)
    ax1.set_ylabel('Response (a.u.)', fontsize=12)

    # Panel B
    ax3 = fig.add_subplot(2, 2, 2)
    ax3.plot(res_00['times'], res_00['sigma'], 'b-', linewidth=2.5, label=r'Boundary $|00\rangle$')
    ax3.plot(res_11['times'], res_11['sigma'], 'r--', linewidth=2.5, label=r'Boundary $|11\rangle$')
    ax3.fill_between(res_00['times'], res_00['sigma'], color='b', alpha=0.1)
    ax3.fill_between(res_11['times'], res_11['sigma'], color='r', alpha=0.1)
    ax3.set_xlabel('Time $t$', fontsize=12)
    ax3.set_ylabel(r'PQS Entropy Rate $\sigma_{PQS}(t)$', fontsize=12)
    ax3.legend(loc='upper right', fontsize=10)
    ax3.set_title(f'(b) Boundary History at $\lambda={crit_lambda:.2f}$', fontsize=14, fontweight='bold')
    ax3.grid(True, alpha=0.3)

    # Panel C
    ax4 = fig.add_subplot(2, 1, 2)
    im = ax4.imshow(
        heatmap,
        origin='lower',
        aspect='auto',
        cmap='magma',
        extent=[t1_grid[0], t1_grid[-1], g_grid[0], g_grid[-1]]
    )
    cbar = plt.colorbar(im, ax=ax4, aspect=30)
    cbar.set_label('PQS Entropy Yield', rotation=270, labelpad=20, fontsize=12)

    ax4.set_xlabel(r'Coherent Drive Strength $\theta_1$', fontsize=12)
    ax4.set_ylabel(r'Dissipation Rate $\gamma$', fontsize=12)
    ax4.set_title('(c) Phase Diagram', fontsize=14, fontweight='bold')

    # Ensure limits match heatmap
    ax4.set_ylim(g_grid[0], g_grid[-1])
    ax4.set_xlim(t1_grid[0], t1_grid[-1])

    plt.savefig('Final_Publication_Figure.png', dpi=300)
    plt.show()

if __name__ == "__main__":
    main()
