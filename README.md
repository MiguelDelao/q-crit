# Reproducible Experiments: Critical Throughput in Open Quantum Systems

Self-contained code for reproducing the numerical experiments in:

> "Entropy Throughput, Dynamic Critical Regimes, and Adaptive Self-Tuning in Open Quantum Systems"

## Quick Start

```bash
pip install -r requirements.txt

# Core experiments (run in order)
python exp_01_fss_scan.py          # ~30 min for n=3-8
python exp_02_collapse.py          # ~2 min
python exp_03_gap.py               # ~20 min for n=3-8  
python exp_04_adaptive.py          # ~10 min

# Robustness checks (appendix figures)
python exp_05_robustness.py        # ~5 min
```

For faster testing, use smaller system sizes:
```bash
python exp_01_fss_scan.py --ns 3,4,5 --points 31
```

## Files

| File | Description | Output |
|------|-------------|--------|
| `pqs_throughput.py` | Core: Lindblad dynamics, PQS smoothing, entropy throughput | - |
| `exp_01_fss_scan.py` | Finite-size scaling: J(λ) and χ(λ) | `fss_*.csv`, `fss_*.png` |
| `exp_02_collapse.py` | Data collapse with bootstrap | `collapse_*.csv`, `collapse.png` |
| `exp_03_gap.py` | Gap proxy (relaxation rate) | `gap_*.csv`, `gap.png` |
| `exp_04_adaptive.py` | Adaptive self-tuning | `adaptive_*.csv`, `adaptive.png` |
| `exp_05_robustness.py` | All robustness checks | `experiment_E*.png` |

## Model

n-qubit XX+ZZ chain with local dephasing:

```
H = θ₁ Σᵢ σˣᵢσˣᵢ₊₁ + θ₂ Σᵢ σᶻᵢσᶻᵢ₊₁
L[ρ] = -i[H,ρ] + γ Σⱼ (ZⱼρZⱼ - ρ)
```

Control parameter: λ = |θ₁|/γ

## Requirements

- Python 3.8+
- numpy, scipy, matplotlib

