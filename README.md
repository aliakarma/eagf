# EAGF: Ethical AI Governance Framework

> **Joint Optimization of Fairness, Privacy, Explainability & Accountability in AI-Based Cybersecurity**

![Python](https://img.shields.io/badge/Python-3.9%2B-blue?style=flat-square)
![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)
![Status](https://img.shields.io/badge/Status-Publication_Ready-brightgreen?style=flat-square)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-red?style=flat-square)
![Reproducibility](https://img.shields.io/badge/Reproducibility-10%2F5_Seeds-success?style=flat-square)


---

## Overview

EAGF is a four-pillar ethical AI governance framework that jointly optimises **fairness**, **privacy (DP-SGD)**, **transparency (SHAP)**, and **accountability (cryptographic audit)** for trustworthy cybersecurity in 5G renewable-energy IoT systems. Evaluated on two case studies: biometric access control (EFR, ResNet-50) and real-world IIoT intrusion detection (Edge-IIoTset, BiLSTM).

**Key Focus**: Governance-aware AI deployment for IoT/IIoT cybersecurity with minimal system overhead.

---

## Key Contributions

- **Four-pillar governance**: Joint optimisation of clarity, fairness, privacy, and accountability with a composite Trust Index
- **Two real-world case studies**: EFR biometric (10,021 images, 158 classes) and Edge-IIoTset (157,800 flow records, binary classification)
- **Pareto trade-off analysis**: 5x5 logarithmic grid quantifying accuracy-fairness-privacy trade-offs
- **Reproducible pipeline**: Deterministic execution with fixed seeds (CS1: 10 seeds, CS2: 5 seeds)

---

## Key Results

### Case Study 1: Biometric Access Control (EFR)

**Architecture**: ResNet-50 (ImageNet-pretrained, GroupNorm for Opacus), 10 seeds (42-51)
**Fairness metric**: Recall Parity (RP)

| Metric | M0 (Baseline) | **M2 (EAGF)** | Delta |
|--------|---------------|---------------|-------|
| Accuracy (%) | 84.63 +/- 1.03 | 78.63 +/- 2.52 | -6.00 pp |
| Recall Parity | 0.786 +/- 0.021 | **0.905 +/- 0.012** | +0.119 (+15.1%) |
| Clarity | 0.929 +/- 0.028 | **0.961 +/- 0.021** | +0.032 (+3.5%) |
| Privacy | 0.243 +/- 0.009 | **0.289 +/- 0.012** | +0.046 |
| Accountability | 0.300 +/- 0.000 | **0.983 +/- 0.000** | +0.683 |
| **Trust Index** | 0.565 +/- 0.007 | **0.785 +/- 0.008** | **+0.220 (+38.9%)** |

Statistical tests: Exact Wilcoxon signed-rank, p = 0.002 (floor at n=10) for all pairwise comparisons.

### Case Study 2: IIoT Intrusion Detection (Edge-IIoTset)

**Architecture**: 2-layer Bidirectional LSTM (hidden=128), 5 seeds (42-46)
**Dataset**: 157,800 flow records, binary (Normal/Attack, 1:5.8 imbalance)
**Fairness metric**: FPR Parity (FPRP), protected groups: protocol type (web, IoT-MQTT, misc)

| Metric | M0 (Baseline) | **M2 (EAGF)** | Delta | Improvement |
|--------|---------------|---------------|-------|-------------|
| Accuracy (macro) | 0.648 +/- 0.025 | **0.665 +/- 0.008** | +0.017 | +2.6% |
| FPR Parity | 0.493 +/- 0.085 | **0.771 +/- 0.057** | +0.278 | **+56.4%** |
| Clarity | 0.692 +/- 0.043 | **0.739 +/- 0.055** | +0.047 | +6.8% |
| Privacy | 0.248 +/- 0.003 | 0.248 +/- 0.003 | +0.001 | Preserved |
| Accountability | 0.000 +/- 0.000 | **0.667 +/- 0.000** | +0.667 | Full coverage |
| **Trust Index** | 0.358 +/- 0.013 | **0.606 +/- 0.011** | +0.248 | **+69.3%** |

Statistical tests: Paired t-test, M0 vs M2 t(4)=32.6, p<0.001.
System overhead: +0.2 ms latency (+11%), +5.8 MB memory (edge-deployable).

### Key Findings

- **CS1**: EAGF raises Trust Index by +38.9% (0.565 to 0.785); accountability contributes 77.6% of the gain
- **CS2**: Trust Index improves +69.3% (0.358 to 0.606); FPR parity improves +56.4% across protocol groups
- **Privacy**: DP-SGD (epsilon=3) reduces MIA AUC from 0.612 to 0.541 in CS1; near-random (0.50) in CS2
- **No pillar is redundant**: Leave-one-out ablation confirms every mechanism degrades TI when removed
- **Edge-deployable**: +0.2 ms latency, +5.8 MB memory; SHAP runs asynchronously (~12 ms batch)  

---

## Repository Structure

```
eagf/
├── README.md
├── requirements.txt
├── setup.py
│
├── run_eagf.py                      # CS1 biometric pipeline entry point
├── run_full_pipeline.py             # CS2 Edge-IIoTset pipeline entry point
│
├── src/
│   ├── models/
│   │   ├── __init__.py              # Model factory (build_model)
│   │   ├── resnet50.py              # ResNet-50 + GroupNorm (CS1)
│   │   ├── lstm.py                  # BiLSTM classifier (CS2)
│   │   └── tabular_mlp.py           # Tabular MLP fallback
│   │
│   ├── training/
│   │   ├── eagf_trainer.py          # Main EAGF training loop with governance
│   │   ├── fairness_loss.py         # Fairness penalty (RP / FPRP)
│   │   └── pareto_trainer.py        # Pareto front exploration (5x5 grid)
│   │
│   ├── evaluation/
│   │   ├── ablation.py              # 6-variant + leave-one-out ablation
│   │   ├── statistics.py            # Wilcoxon (CS1), paired t-test (CS2)
│   │   ├── mia_attack.py            # Yeom loss-threshold MIA
│   │   ├── audit_logger.py          # RSA-SHA256 cryptographic audit
│   │   └── report_generator.py      # Summary report generation
│   │
│   ├── metrics/
│   │   ├── fairness.py              # Recall Parity, FPR Parity (ratio-based)
│   │   ├── privacy.py               # Privacy score (beta*exp(-eps) + (1-beta)*(1-MIA))
│   │   ├── clarity.py               # SHAP-based clarity (Eq. 6-7)
│   │   ├── accountability.py        # Audit + trace + compliance
│   │   └── trust_index.py           # TI = 0.25*(C + F + P + A)
│   │
│   ├── utils/
│   │   ├── data_loader.py           # Dataset loading (dispatches by architecture)
│   │   ├── biometric_pipeline.py    # Image pipeline for ResNet-50 (CS1)
│   │   ├── edge_iiot_loader.py      # Edge-IIoTset loader (CS2)
│   │   ├── preprocessing.py         # Normalisation, DP noise, sample weights
│   │   └── visualisation.py         # Figures
│   │
│   └── baselines/
│       ├── aif360_dp_pipeline.py    # AIF360 reweighing baseline
│       ├── fairlearn_baseline.py    # Fairlearn EG (FPR parity) baseline
│       └── joint_dp_fair_baseline.py
│
├── configs/
│   ├── biometric_default.yaml       # CS1: ResNet-50, 50 epochs, cosine+warmup
│   ├── reiot_real.yaml              # CS2: BiLSTM, 30 epochs, Edge-IIoTset
│   └── compliance_checklist*.yaml   # Compliance templates
│
├── docs/
│   └── EAGF.tex                     # Paper source
│
├── data/
│   └── README.md                    # Data setup instructions
│
└── results/                         # Generated experiment outputs
```

---

## Installation

```bash
git clone https://github.com/aliakarma/eagf.git
cd eagf
python -m venv .venv

# Windows
.venv\Scripts\activate
# Linux / macOS
source .venv/bin/activate

pip install -r requirements.txt
```

---

## Dataset Setup

### CS1: EFR Biometric Dataset

Place preprocessed numpy arrays under `data/biometric/efr_processed/` (see `data/README.md`).
In demo mode (`--demo`), synthetic 3x224x224 images are generated automatically.

### CS2: Edge-IIoTset

Download `ML-EdgeIIoT-dataset.csv` from [IEEE DataPort](https://dx.doi.org/10.21203/rs.3.rs-1433551/v1) and place at `data/real_iot/edge_iiot.csv` (~78 MB, 157.8K rows).

---

## Quick Start

### CS1 — Biometric (single seed, demo data)

```bash
python run_eagf.py --config configs/biometric_default.yaml --seeds 42 --demo
```

### CS2 — IIoT (single seed)

```bash
python run_full_pipeline.py --real_dataset edge_iiot --config configs/reiot_real.yaml --seeds 42
```

---

## Full Experiment (Publication Results)

### CS1: 10 seeds

```bash
python run_eagf.py --config configs/biometric_default.yaml --seeds 42 43 44 45 46 47 48 49 50 51
```

### CS2: 5 seeds

```bash
python run_full_pipeline.py --real_dataset edge_iiot --config configs/reiot_real.yaml --seeds 42 43 44 45 46
```

Both pipelines produce seed-specific results under `results/`, aggregate statistics (mean +/- std, 95% CI), ablation tables, and Pareto front visualisation (5x5 grid, 25 runs).

---

## Method Overview

### Trust Index

`TI = 0.25 * (Clarity + Fairness + Privacy + Accountability)`

| Pillar | CS1 Metric | CS2 Metric | Implementation |
|--------|-----------|-----------|----------------|
| Clarity | SHAP fidelity (Eq. 6-7) | SHAP fidelity | `src/metrics/clarity.py` |
| Fairness | Recall Parity (ratio) | FPR Parity (ratio) | `src/metrics/fairness.py` |
| Privacy | beta*exp(-eps) + (1-beta)*(1-MIA) | Same | `src/metrics/privacy.py` |
| Accountability | Audit + trace + compliance | Same | `src/metrics/accountability.py` |

### Architectures

| Case Study | Model | Key Config |
|-----------|-------|-----------|
| CS1 Biometric | ResNet-50 (ImageNet, GroupNorm, Dropout 0.3) | 50 epochs, batch 32, cosine+warmup(5) |
| CS2 IIoT | 2-layer BiLSTM (hidden=128, LayerNorm head) | 30 epochs, batch 256, cosine+warmup(3) |

---

## Reproducibility

- **CS1**: 10 seeds (42-51), exact Wilcoxon signed-rank (p=0.002 floor at n=10)
- **CS2**: 5 seeds (42-46), paired t-test (t(4)=32.6, p<0.001 for TI)
- **Deterministic**: NumPy + PyTorch seeds fixed per run
- **Convergence**: Early stop when delta_TI < 0.002 over 5 epochs or epsilon budget exhausted

---

## License

MIT License — See [LICENSE](LICENSE) for full terms.

---

**Last Updated**: June 2026 | Python 3.9+ | PyTorch 2.0+
