"""PQS entropy-throughput experiments for Part I.

This module implements (i) Lindblad Liouvillian construction in Liouville space,
(ii) Past Quantum State smoothing (unconditional Lindblad setting), and
(iii) the entropy-throughput functional

    J = \int |\sigma_PQS(t)| dt

computed robustly as total variation of the relative entropy along the
smoothed trajectory:

    J = \sum_k | S_k - S_{k-1} |.

Designed for small n-qubit systems (n<=6) using sparse superoperators.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from typing import Iterable, Literal

import numpy as np
import scipy.linalg
import scipy.sparse as sp
import scipy.sparse.linalg as spla


DTYPE = np.complex128


def _sp_eye(n: int) -> sp.csr_matrix:
    return sp.identity(n, format="csr", dtype=DTYPE)


def kron(a: sp.spmatrix, b: sp.spmatrix) -> sp.csr_matrix:
    return sp.kron(a, b, format="csr")


def vec(mat: np.ndarray) -> np.ndarray:
    # Row-stacking (C order) vectorization to match the original Part I codebase
    # convention (see q-crit/part_1/main.py).
    return mat.reshape(-1, order="C")


def unvec(v: np.ndarray, dim: int) -> np.ndarray:
    return v.reshape((dim, dim), order="C")


def paulis() -> tuple[sp.csr_matrix, sp.csr_matrix, sp.csr_matrix, sp.csr_matrix, sp.csr_matrix]:
    I = sp.csr_matrix(np.eye(2, dtype=DTYPE))
    X = sp.csr_matrix(np.array([[0, 1], [1, 0]], dtype=DTYPE))
    Y = sp.csr_matrix(np.array([[0, -1j], [1j, 0]], dtype=DTYPE))
    Z = sp.csr_matrix(np.array([[1, 0], [0, -1]], dtype=DTYPE))
    sm = sp.csr_matrix(np.array([[0, 0], [1, 0]], dtype=DTYPE))  # sigma_- (|0><1|)
    return I, X, Y, Z, sm


def op_on_site(op: sp.csr_matrix, site: int, n: int) -> sp.csr_matrix:
    """Place a single-qubit operator on `site` (0-based) in an n-qubit register."""
    I, *_ = paulis()
    out = None
    for j in range(n):
        factor = op if j == site else I
        out = factor if out is None else kron(out, factor)
    assert out is not None
    return out


def two_site(op_a: sp.csr_matrix, i: int, op_b: sp.csr_matrix, j: int, n: int) -> sp.csr_matrix:
    """Place op_a on i and op_b on j (i!=j), identities elsewhere."""
    I, *_ = paulis()
    out = None
    for k in range(n):
        if k == i:
            factor = op_a
        elif k == j:
            factor = op_b
        else:
            factor = I
        out = factor if out is None else kron(out, factor)
    assert out is not None
    return out


def liouvillian_from_H_and_Ls(H: sp.csr_matrix, L_ops: Iterable[sp.csr_matrix]) -> sp.csr_matrix:
    """Return Liouvillian superoperator in vec convention: d/dt vec(rho)=L vec(rho).

    This implementation uses row-stacking vectorization (order='C'), for which:
      vec(A X B) = (A ⊗ B^T) vec(X).
    """
    d = H.shape[0]
    I_d = _sp_eye(d)
    # -i[H, rho] -> -i( H⊗I - I⊗H^T ) vec(rho)
    L = (-1j) * (kron(H, I_d) - kron(I_d, H.transpose()))

    for Lk in L_ops:
        Lk = sp.csr_matrix(Lk)
        Lk_dag_Lk = (Lk.getH() @ Lk).tocsr()
        # Lk rho Lk† -> (Lk ⊗ (Lk†)^T) vec(rho) = (Lk ⊗ Lk*) vec(rho)
        jump = kron(Lk, Lk.conjugate())
        # -1/2 {L†L, rho} -> -1/2 ( (L†L)⊗I + I⊗(L†L)^T )
        antic = 0.5 * (kron(Lk_dag_Lk, I_d) + kron(I_d, Lk_dag_Lk.transpose()))
        L = (L + jump - antic).tocsr()

    return L.tocsr()


def commutator_super(H: sp.csr_matrix) -> sp.csr_matrix:
    """Return -i[H,·] as a Liouville-space superoperator for row-stacking vec."""
    d = H.shape[0]
    I_d = _sp_eye(d)
    return ((-1j) * (kron(H, I_d) - kron(I_d, H.transpose()))).tocsr()


def dissipator_super(L_ops: Iterable[sp.csr_matrix], dim: int) -> sp.csr_matrix:
    """Return sum_k D[L_k] as a Liouville-space superoperator for row-stacking vec."""
    I_d = _sp_eye(dim)
    D = sp.csr_matrix((dim * dim, dim * dim), dtype=DTYPE)
    for Lk in L_ops:
        Lk = sp.csr_matrix(Lk)
        Lk_dag_Lk = (Lk.getH() @ Lk).tocsr()
        jump = kron(Lk, Lk.conjugate())
        antic = 0.5 * (kron(Lk_dag_Lk, I_d) + kron(I_d, Lk_dag_Lk.transpose()))
        D = (D + jump - antic).tocsr()
    return D.tocsr()


def trace_row(dim: int) -> sp.csr_matrix:
    """Row vector r s.t. r @ vec(rho) = Tr(rho) for row-stacking vec (order='C')."""
    idx = np.array([i * dim + i for i in range(dim)], dtype=int)
    data = np.ones(dim, dtype=float)
    return sp.csr_matrix((data, (np.zeros(dim, dtype=int), idx)), shape=(1, dim * dim))


def stationary_state_trace1(L: sp.csr_matrix, dim: int) -> np.ndarray:
    """Compute a trace-1 stationary state by solving an augmented least-squares system.

    Solve [L; tr] x ≈ [0; 1] in the least-squares sense, then reshape to rho.
    This avoids dense SVD and is typically more stable for small sparse Liouvillians.
    """
    tr = trace_row(dim)
    A = sp.vstack([L, tr], format="csr")
    b = np.zeros(A.shape[0], dtype=DTYPE)
    b[-1] = 1.0
    sol = spla.lsqr(A, b, atol=1e-12, btol=1e-12, iter_lim=2000)
    x = sol[0].astype(DTYPE)
    rho = unvec(x, dim)
    return ensure_density_matrix(rho)


def make_chain_hamiltonian(n: int, theta_xx: float, theta_zz: float) -> sp.csr_matrix:
    I, X, _, Z, _ = paulis()
    d = 2**n
    H = sp.csr_matrix((d, d), dtype=DTYPE)
    for i in range(n - 1):
        H = H + theta_xx * two_site(X, i, X, i + 1, n)
        H = H + theta_zz * two_site(Z, i, Z, i + 1, n)
    return H.tocsr()


def make_dephasing_L_ops(n: int, gamma: float) -> list[sp.csr_matrix]:
    """Local dephasing with rate gamma: L_i = sqrt(gamma) Z_i."""
    _, _, _, Z, _ = paulis()
    return [np.sqrt(gamma) * op_on_site(Z, i, n) for i in range(n)]


def make_thermal_amp_damp_L_ops(n: int, gamma: float, n_th: float) -> list[sp.csr_matrix]:
    """Local thermal amplitude damping.

    Rates:
      gamma_down = gamma*(n_th+1)
      gamma_up   = gamma*n_th

    Lindblad ops per site:
      L_down = sqrt(gamma_down) * sigma_-
      L_up   = sqrt(gamma_up)   * sigma_+

    This is non-unital for n_th != 1/2 and has non-maximally-mixed steady state.
    """
    I, _, _, _, sm = paulis()
    sp_op = sm.getH().tocsr()  # sigma_+ = |1><0|

    gamma_down = gamma * (n_th + 1.0)
    gamma_up = gamma * n_th

    Ls: list[sp.csr_matrix] = []
    for i in range(n):
        if gamma_down > 0:
            Ls.append(np.sqrt(gamma_down) * op_on_site(sm, i, n))
        if gamma_up > 0:
            Ls.append(np.sqrt(gamma_up) * op_on_site(sp_op, i, n))
    return Ls


def projector_bitstring(bitstring: str) -> np.ndarray:
    """|bitstring><bitstring| as dense matrix."""
    n = len(bitstring)
    d = 2**n
    idx = int(bitstring, 2)
    ket = np.zeros((d, 1), dtype=DTYPE)
    ket[idx, 0] = 1.0
    return ket @ ket.conj().T


def ensure_density_matrix(rho: np.ndarray, eps: float = 1e-15) -> np.ndarray:
    rho = (rho + rho.conj().T) / 2.0
    evals, evecs = np.linalg.eigh(rho)
    evals = np.clip(evals.real, 0.0, None)
    tr = float(np.sum(evals))
    if tr <= eps:
        # fallback to maximally mixed
        d = rho.shape[0]
        return np.eye(d, dtype=DTYPE) / d
    evals = evals / tr
    return evecs @ np.diag(evals) @ evecs.conj().T


def sqrt_psd(mat: np.ndarray, eps: float = 1e-15) -> np.ndarray:
    mat = (mat + mat.conj().T) / 2.0
    evals, evecs = np.linalg.eigh(mat)
    evals = np.clip(evals.real, 0.0, None)
    return evecs @ np.diag(np.sqrt(np.maximum(evals, eps))) @ evecs.conj().T


def von_neumann_entropy(rho: np.ndarray, eps: float = 1e-15) -> float:
    evals = np.linalg.eigvalsh((rho + rho.conj().T) / 2.0).real
    evals = np.clip(evals, eps, 1.0)
    return float(-np.sum(evals * np.log(evals)))


def relative_entropy(rho: np.ndarray, rho_ss: np.ndarray, log_rho_ss: np.ndarray | None = None) -> float:
    """S(rho || rho_ss) in nats. Assumes rho_ss full rank (regularize upstream if needed)."""
    rho = ensure_density_matrix(rho)
    rho_ss = ensure_density_matrix(rho_ss)

    evals, evecs = np.linalg.eigh(rho)
    evals = np.clip(evals.real, 1e-15, 1.0)
    log_rho = evecs @ np.diag(np.log(evals)) @ evecs.conj().T

    if log_rho_ss is None:
        evals_ss, evecs_ss = np.linalg.eigh(rho_ss)
        evals_ss = np.clip(evals_ss.real, 1e-15, 1.0)
        log_rho_ss = evecs_ss @ np.diag(np.log(evals_ss)) @ evecs_ss.conj().T

    return float(np.trace(rho @ (log_rho - log_rho_ss)).real)


def estimate_stationary_state(L: sp.csr_matrix, dim: int) -> np.ndarray:
    """Estimate rho_ss by taking the right singular vector of L with smallest singular value."""
    L_dense = L.toarray()
    # Right singular vector corresponding to smallest singular value approximates nullspace.
    _, _, vh = scipy.linalg.svd(L_dense)
    v = vh.conj().T[:, -1]
    rho = unvec(v, dim)
    rho = ensure_density_matrix(rho)
    return rho


@dataclass
class PQSResult:
    times: np.ndarray
    S_series: np.ndarray
    J: float


@dataclass
class PQSObservableResult:
    times: np.ndarray
    expectations: dict[str, np.ndarray]


def compute_pqs_expectations(
    *,
    L: sp.csr_matrix,
    rho0: np.ndarray,
    F_final: np.ndarray,
    observables: dict[str, np.ndarray],
    T: float,
    steps: int,
) -> PQSObservableResult:
    """Compute expectation time series of observables in the PQS-smoothed state.

    This uses the same smoothing construction as `compute_J_total_variation`, but
    instead of returning the entropy path functional, it returns ⟨O⟩_PQS(t) for
    each observable O.
    """
    d = rho0.shape[0]
    dt = T / steps
    times = np.arange(steps, dtype=float) * dt

    Ldag = L.conjugate().transpose().tocsr()
    rhos = spla.expm_multiply(L, vec(rho0), start=0.0, stop=times[-1], num=steps, endpoint=True)
    Es_tau = spla.expm_multiply(Ldag, vec(F_final), start=0.0, stop=times[-1], num=steps, endpoint=True)
    Es = Es_tau[::-1].copy()

    out: dict[str, np.ndarray] = {k: np.zeros(steps, dtype=float) for k in observables.keys()}

    # Pre-cast observables once
    obs_dense: dict[str, np.ndarray] = {k: np.asarray(v, dtype=DTYPE) for k, v in observables.items()}

    for k in range(steps):
        rho_t = ensure_density_matrix(unvec(np.asarray(rhos[k]), d))
        E_raw = unvec(np.asarray(Es[k]), d)
        E_t = (E_raw + E_raw.conj().T) / 2.0

        sqE = sqrt_psd(E_t + 1e-12 * np.eye(d, dtype=DTYPE))
        num = sqE @ rho_t @ sqE
        prob = float(np.trace(num).real)
        rho_s = (num / prob) if prob > 1e-14 else rho_t
        rho_s = ensure_density_matrix(rho_s)

        for name, O in obs_dense.items():
            out[name][k] = float(np.trace(rho_s @ O).real)

    return PQSObservableResult(times=times, expectations=out)


def compute_J_total_variation(
    L: sp.csr_matrix,
    rho0: np.ndarray,
    F_final: np.ndarray,
    rho_ss: np.ndarray,
    T: float,
    steps: int,
    use_fast_unital_ss: bool = True,
) -> PQSResult:
    """Compute J as total variation of S(rho_smooth||rho_ss) along the PQS-smoothed trajectory."""
    d = rho0.shape[0]
    # Match the original Part I discretization convention (dt = T/steps with
    # `steps` grid points, so the final time is (steps-1)*dt < T.
    dt = T / steps
    times = np.arange(steps, dtype=float) * dt

    # Precompute log(rho_ss) if needed.
    log_rho_ss = None
    is_maximally_mixed = False
    if use_fast_unital_ss:
        is_maximally_mixed = np.allclose(rho_ss, np.eye(d) / d, atol=1e-10)

    if not is_maximally_mixed:
        rho_ss_reg = ensure_density_matrix(rho_ss)
        evals_ss, evecs_ss = np.linalg.eigh(rho_ss_reg)
        evals_ss = np.clip(evals_ss.real, 1e-15, 1.0)
        log_rho_ss = evecs_ss @ np.diag(np.log(evals_ss)) @ evecs_ss.conj().T

    Ldag = L.conjugate().transpose().tocsr()

    # Forward rho(t) and backward effect E(t), computed in one call each.
    # expm_multiply with start/stop returns an array of shape (steps, d^2).
    rhos = spla.expm_multiply(L, vec(rho0), start=0.0, stop=times[-1], num=steps, endpoint=True)
    # Backward: evolve from t_f to 0 by using Ldag and reversing times.
    # E(t) = exp(Ldag * (t_f - t)) F, so compute forward in tau=(t_f-t).
    Es_tau = spla.expm_multiply(Ldag, vec(F_final), start=0.0, stop=times[-1], num=steps, endpoint=True)
    Es = Es_tau[::-1].copy()

    S_series = np.zeros(steps, dtype=float)

    for k in range(steps):
        rho_t = ensure_density_matrix(unvec(np.asarray(rhos[k]), d))
        E_raw = unvec(np.asarray(Es[k]), d)
        E_t = (E_raw + E_raw.conj().T) / 2.0
        # PQS smoothing (symmetric)
        sqE = sqrt_psd(E_t + 1e-12 * np.eye(d, dtype=DTYPE))
        num = sqE @ rho_t @ sqE
        prob = float(np.trace(num).real)
        if prob > 1e-14:
            rho_s = num / prob
        else:
            rho_s = rho_t
        rho_s = ensure_density_matrix(rho_s)

        if is_maximally_mixed:
            # S(rho||I/d) = log d - S(rho)
            S_series[k] = np.log(d) - von_neumann_entropy(rho_s)
        else:
            S_series[k] = relative_entropy(rho_s, rho_ss, log_rho_ss=log_rho_ss)

    # Total variation
    J = float(np.sum(np.abs(np.diff(S_series))))
    return PQSResult(times=times, S_series=S_series, J=J)
