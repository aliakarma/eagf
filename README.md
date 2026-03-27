# EAGF: Ethical AI Governance Framework for Cybersecurity in Renewable Energy IoT Systems

<div align="center">

[![Python 3.9+](https://img.shields.io/badge/Python-3.9%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![arXiv](https://img.shields.io/badge/arXiv-Preprint-red.svg)](#citation)
[![IEEE](https://img.shields.io/badge/Journal-IEEE%20Submission-blue.svg)](#citation)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](#interactive-notebooks)
[![GitHub](https://img.shields.io/badge/GitHub-aliakarma%2Feagf-blue?logo=github)](https://github.com/aliakarma/eagf)

**A Four-Pillar Framework Integrating Transparency, Fairness, Privacy, and Accountability with a Composite Trust Index**

[Paper](#citation) · [Installation](#installation) · [Quick Start](#quick-start) · [Reproducibility](#reproducibility) · [Results](#results) · [Citation](#citation)

</div>

---

## Table of Contents

1. [Overview](#overview)
2. [Key Contributions](#key-contributions)
3. [Architecture](#architecture)
4. [Repository Structure](#repository-structure)
5. [Installation](#installation)
6. [Quick Start](#quick-start)
7. [Detailed Usage](#detailed-usage)
8. [Reproducibility](#reproducibility)
9. [Results](#results)
10. [Paper–Code Mapping](#papercode-mapping)
11. [Interactive Notebooks](#interactive-notebooks)
12. [Citation](#citation)
13. [License](#license)
14. [Acknowledgements](#acknowledgements)

---

## Overview

**EAGF** (*Ethical AI Governance Framework*) addresses a critical gap in the deployment of AI-driven cybersecurity systems: no existing framework simultaneously operationalises all four governance pillars mandated by the EU AI Act (2024) and NIST AI RMF (2023) — **transparency**, **fairness**, **privacy**, and **accountability** — within a single adversarial-aware lifecycle.

This repository provides the complete implementation of EAGF as presented in the paper:

> **"Ethical AI Governance for Cybersecurity in Renewable Energy IoT Systems: A Four-Pillar Framework Integrating Transparency, Fairness, Privacy, and Accountability with a Composite Trust Index"**
> Salman Jan, Munir Azam Muhammad, Toqeer Ali Syed, Ali Akarma, It Ee Lee, Qamar Wali, Shahid Kamal, Jawad Ali.

EAGF currently includes reproducible experiments on:
- **Tabular fairness/privacy benchmarking:** synthetic biometric-style tabular data and optional UCI Adult support
- **Renewable Energy IoT:** synthetic 5G solar-microgrid anomaly detection across heterogeneous node classes (FDIA, command-injection, DoS attacks)

---

## Key Contributions

| # | Contribution | Code Location |
|---|---|---|
| **C1** | Four-pillar governance framework with formal operational definitions | `src/metrics/` |
| **C2** | Composite Trust Index (TI) with AHP weighting | `src/metrics/trust_index.py` |
| **C3** | Multi-objective Pareto-guided training with formal MOO formulation | `src/training/pareto_trainer.py` |
| **C4** | Dual-domain validation with full six-variant ablation study | `src/evaluation/`, `scripts/run_ablation.sh` |

### Why Joint Governance Matters

The ablation scripts and notebooks compute metrics directly from model outputs and artifacts (no hardcoded boosts). Results vary with data, seeds, and configuration; use the provided pipeline to regenerate all tables and figures.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    EAGF Lifecycle (9 Stages)                    │
│                                                                 │
│  Stage 1: Ethical Design Requirements                           │
│  ┌──────────┬──────────┬──────────┬──────────────────────────┐  │
│  │ Target C │ Target RP│ Target P │ Target A                 │  │
│  │ ≥ 0.80   │ ≥ 0.95   │ ≥ 0.80   │ ≥ 0.85                   │  │
│  └──────────┴──────────┴──────────┴──────────────────────────┘  │
│                                                                 │
│  Stage 2-3: Privacy-Preserving Data Preparation + Bias Audit    │
│                                                                 │
│  Stage 4: Multi-Objective Constrained Training                  │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │  min L_CE(θ)  s.t.  C̃(θ)≥Ĉ, RP̃(θ)≥R̂P, P̃(θ)≥P̂, Ã(θ)≥Â     │   │
│  │                                                          │   │
│  │  Pareto Grid: λ_RP × λ_C  ∈  [10⁻³, 10⁰]  (5×5)          │   │
│  └──────────────────────────────────────────────────────────┘   │
│                                                                 │
│  Stage 5: SHAP Explainability (top-k, latency-bounded)          │
│  Stage 6: Trust Metric Evaluation                               │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │  TI = 0.25·C̃ + 0.25·R̃P + 0.25·P̃ + 0.25·Ã                   │ │
│  └────────────────────────────────────────────────────────────┘ │
│  Stage 7: Pareto-Front Selection                                │
│  Stage 8: Deployment + Governance Hooks                         │
│  Stage 9: Continuous Stakeholder Reporting                      │
└─────────────────────────────────────────────────────────────────┘
```

### Four-Pillar Metric Architecture

```
Transparency (C)          Fairness (RP / FPRP)
──────────────────        ─────────────────────────
ClarityScore(i) =         RP_A/B = Recall_A / Recall_B
  Fidelity(E(i))          FPRP_A/B = FPR_A / FPR_B
  ──────────────          (criterion selected per context)
  1 + Size(E(i))

Privacy (P)               Accountability (A)
──────────────────        ─────────────────────────────
P = α·exp(-ε_eff)         A = (α_audit + α_trace + α_comply) / 3
  + (1-α)·(1-MIA)         α_audit  : cryptographic log completeness
  α = 0.6                 α_trace  : data lineage recoverability
                          α_comply : regulatory checklist score
```

---

## Repository Structure

```
eagf/
│
├── README.md                    ← This file
├── LICENSE                      ← MIT License
├── requirements.txt             ← Python dependencies (pip)
├── environment.yml              ← Conda environment specification
│
├── src/                         ← Core framework implementation
│   ├── __init__.py
│   ├── metrics/
│   │   ├── __init__.py
│   │   ├── clarity.py           ← Transparency metric (C): SHAP-based clarity score
│   │   ├── fairness.py          ← Fairness metrics: RP and FPRP
│   │   ├── privacy.py           ← Privacy metric (P): DP + MIA composite
│   │   ├── accountability.py    ← Accountability metric (A): audit/trace/comply
│   │   └── trust_index.py       ← Composite Trust Index (TI) with AHP weighting
│   │
│   ├── training/
│   │   ├── __init__.py
│   │   ├── fairness_loss.py     ← Recall-parity and clarity penalties
│   │   ├── pareto_trainer.py    ← Pareto-front exploration (5×5 λ grid)
│   │   └── eagf_trainer.py      ← Main PyTorch + Opacus multi-objective trainer
│   │
│   ├── evaluation/
│   │   ├── __init__.py
│   │   ├── ablation.py          ← Six-variant ablation study runner
│   │   ├── baseline.py          ← Baseline model evaluation
│   │   ├── mia_attack.py        ← Membership inference attack (shadow model)
│   │   └── audit_logger.py      ← Cryptographic audit log writer (SHA-256)
│   │
│   └── utils/
│       ├── __init__.py
│       ├── data_loader.py       ← Dataset loading, stratified splitting
│       ├── preprocessing.py     ← Anonymisation, group-balance re-weighting
│       ├── visualisation.py     ← Bar charts, Pareto-front plots
│       └── ahp.py               ← AHP weight derivation from pairwise matrix
│
├── configs/
│   ├── biometric_default.yaml   ← Hyperparameters for biometric case study
│   ├── reiot_default.yaml       ← Hyperparameters for RE-IoT case study
│   └── eagf_thresholds.yaml     ← Stage-1 governance targets
│
├── scripts/
│   ├── run_all.sh               ← Master script: runs all experiments end-to-end
│   ├── run_ablation.sh          ← Six-variant ablation study (M0–M5)
│   ├── run_biometric.sh         ← Biometric case study (Case Study 1)
│   ├── run_reiot.sh             ← RE-IoT case study (Case Study 2)
│   └── run_pareto_search.sh     ← 5×5 Pareto-grid hyperparameter sweep
│
├── notebooks/
│   ├── 01_eagf_demo.ipynb             ← Full framework demo + Figure 3
│   ├── 02_statistical_analysis.ipynb  ← z-test, paired t-test, bootstrap CIs
│   ├── 03_reiot_fairness.ipynb        ← RE-IoT node-class FPRP analysis
│   ├── 04_pareto_front.ipynb          ← Pareto-front MOO visualisation
│   └── 05_trust_index_sensitivity.ipynb ← AHP weights, TI sensitivity
│
├── data/
│   ├── README.md                ← Data access instructions and preprocessing steps
│   ├── biometric/               ← Placeholder; EFR dataset downloaded separately
│   └── reiot/                   ← Synthetic RE-IoT telemetry generation scripts
│
├── results/
│   ├── biometric/               ← Output CSVs and model checkpoints (biometric)
│   └── reiot/                   ← Output CSVs and model checkpoints (RE-IoT)
│
├── figures/                     ← Paper figures (reproduced by visualisation scripts)
│   ├── figure1.png              ← EAGF causal architecture diagram
│   ├── figure2.png              ← RE-IoT integration architecture
│   └── figure3.png              ← Ablation bar chart (Baseline vs. EAGF)
│
└── docs/
    ├── reproducibility.md       ← Detailed reproducibility statement and checklist
    ├── metric_definitions.md    ← Formal metric definitions with equations
    └── regulatory_mapping.md    ← EAGF → EU AI Act / GDPR / NIST / NIS2 mapping
```

---

## Installation

### Prerequisites

| Requirement | Version |
|---|---|
| Python | ≥ 3.9 |
| PyTorch | ≥ 2.0.0 |
| CUDA (optional) | ≥ 11.7 |
| RAM | ≥ 16 GB |
| Disk | ≥ 10 GB (dataset + checkpoints) |

### Option A: pip (recommended for most users)

```bash
# 1. Clone the repository
git clone https://github.com/aliakarma/eagf.git
cd eagf

# 2. Create a virtual environment
python -m venv .venv
source .venv/bin/activate        # Linux / macOS
# .venv\Scripts\activate         # Windows

# 3. Install dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

### Option B: Conda

```bash
# 1. Clone the repository
git clone https://github.com/aliakarma/eagf.git
cd eagf

# 2. Create and activate the Conda environment
conda env create -f environment.yml
conda activate eagf

# 3. Verify installation
python -c "import src; print('EAGF installation verified.')"
```

### Option C: Docker (fully reproducible)

```bash
# Build image
docker build -t eagf:latest .

# Run biometric experiment
docker run --gpus all -v $(pwd)/results:/app/results eagf:latest \
    bash scripts/run_biometric.sh
```

---

## Quick Start

### 1. Download Data

```bash
# Biometric dataset (EFR from Kaggle — requires Kaggle API token)
python src/utils/data_loader.py --dataset efr --output data/biometric/

# RE-IoT synthetic telemetry (generated locally, no download required)
python src/utils/data_loader.py --dataset reiot --output data/reiot/ \
    --nodes 120 --attack-ratio 0.05 --seed 42
```

> **Note:** The EFR dataset is a Kaggle community upload without formal demographic certification. For production use, we recommend CelebA, VGGFace2, or DemogPairs. See `data/README.md` for details.

### 2. Run EAGF (Full Framework)

```bash
# Biometric case study — full EAGF (M5)
python -m src.training.eagf_trainer \
    --config configs/biometric_default.yaml \
    --model eagf \
    --seed 42 \
    --output results/biometric/

# RE-IoT case study — full EAGF (M5)
python -m src.training.eagf_trainer \
    --config configs/reiot_default.yaml \
    --model eagf \
    --seed 42 \
    --output results/reiot/
```

### 3. Run the Full Ablation Study

```bash
# Runs all six variants M0–M5 with three random seeds each
bash scripts/run_ablation.sh --config configs/biometric_default.yaml
```

### 4. Compute Trust Index

```bash
python -m src.metrics.trust_index \
    --clarity   results/biometric/clarity.json \
    --fairness  results/biometric/recall_parity.json \
    --privacy   results/biometric/privacy.json \
    --accountability results/biometric/accountability.json \
    --weights   equal \
    --output    results/biometric/trust_index.json
```

### 5. Reproduce Paper Figures

```bash
# Figure 3: Ablation bar chart
python src/utils/visualisation.py \
    --results results/biometric/ablation_summary.csv \
    --output  figures/figure3.png
```

---

## Detailed Usage

### Configuration Files

All experiment parameters are controlled via YAML config files in `configs/`. Key parameters:

```yaml
# configs/biometric_default.yaml (excerpt)
model:
  architecture: tabular_mlp
  pretrained: false

training:
  epochs: 50
  batch_size: 64
  optimizer: adamw
  lr: 1.0e-4

governance:
  dp_epsilon: 3.0              # DP-SGD privacy budget
  dp_delta: 1.0e-5
  lambda_rp: 0.1               # Recall-parity Lagrangian coefficient
  lambda_c: 0.05               # Clarity Lagrangian coefficient
  pareto_grid_size: 5          # 5×5 grid → 25 training runs

thresholds:                    # Stage-1 governance targets
  min_clarity: 0.80
  min_recall_parity: 0.95
  min_privacy: 0.80
  min_accountability: 0.85

fairness:
  criterion: recall_parity     # "recall_parity" or "fprp"
  protected_groups: [gender, skin_tone]
```

### Running Individual Pillar Models (Ablation)

```bash
# M0: Baseline (no governance)
python -m src.training.eagf_trainer --model baseline --config configs/biometric_default.yaml

# M1: Transparency only (clarity penalty enabled)
python -m src.training.eagf_trainer --model transparency --config configs/biometric_default.yaml

# M2: Fairness only (recall-parity regularisation)
python -m src.training.eagf_trainer --model fairness --config configs/biometric_default.yaml

# M3: Privacy only (DP-SGD)
python -m src.training.eagf_trainer --model privacy --config configs/biometric_default.yaml

# M4: Accountability only (audit logging + compliance)
python -m src.training.eagf_trainer --model accountability --config configs/biometric_default.yaml

# M5: EAGF Full (all pillars, joint MOO)
python -m src.training.eagf_trainer --model eagf --config configs/biometric_default.yaml
```

### Pareto-Front Search

```bash
python -m src.training.pareto_trainer \
    --config configs/biometric_default.yaml \
    --lambda-rp-range 1e-3 1e0 --lambda-rp-steps 5 \
    --lambda-c-range  1e-3 1e0 --lambda-c-steps  5 \
    --output results/biometric/pareto/
```

### Membership Inference Attack (Privacy Stress-Test)

```bash
python -m src.evaluation.mia_attack \
    --model-checkpoint results/biometric/eagf_best.pt \
    --shadow-models 4 \
    --output results/biometric/mia_results.json
```

### Accountability Audit Logging

```bash
python -m src.evaluation.audit_logger \
    --model-checkpoint results/biometric/eagf_best.pt \
    --dataset data/biometric/test/ \
    --log-output results/biometric/audit_log.jsonl \
    --sign-key configs/audit_key.pem
```

---

## Reproducibility

Full reproducibility details are provided in [`docs/reproducibility.md`](docs/reproducibility.md). The key guarantees are:

| Element | Specification |
|---|---|
| Random seeds | 42, 123, 456 (all results are means ± 95% CI over 3 seeds) |
| Bootstrap resamples | n = 1000 (stratified by demographic group) |
| Statistical tests | Accuracy: two-proportion z-test; TI: Wilcoxon signed-rank; C/P/A: bootstrap t-test |
| DP accounting | Rényi moment accountant (Opacus library) |
| SHAP variant | DeepExplainer (biometric); KernelExplainer (RE-IoT) |
| Hardware | Single NVIDIA A100 40 GB (biometric); CPU-only (RE-IoT simulation) |
| Training time | ~4 h per full EAGF run (biometric); ~20 min (RE-IoT) |

### One-Command Full Reproduction

```bash
# Reproduces ALL tables and figures from the paper
bash scripts/run_all.sh --seeds 42 123 456 --output results/
```

Expected outputs:

| Script output | Paper location |
|---|---|
| `results/biometric/ablation_summary.csv` | Ablation summary (computed) |
| `results/biometric/main_results.csv` | Main baseline vs EAGF results |
| `results/reiot/node_class_results.csv` | RE-IoT node-class results |
| `figures/figure3.png` | Ablation bar chart |

---

## Results

All reported metrics are generated at runtime from actual model outputs. To reproduce results, run the pipeline and inspect generated CSV/JSON artifacts in `results/`.

---

## Paper–Code Mapping

| Paper Section | Description | Code Location |
|---|---|---|
| §3.1 (Framework Overview) | 9-stage lifecycle | `src/training/eagf_trainer.py` |
| §3.2 (Clarity Metric, Eq. 1–2) | SHAP-based C score | `src/metrics/clarity.py` |
| §3.3 (Fairness Metric, Eq. 3–6) | RP and FPRP | `src/metrics/fairness.py` |
| §3.3.2 (Criterion Selection) | RP vs. FPRP justification | `src/metrics/fairness.py::select_criterion()` |
| §3.4 (Privacy Metric, Eq. 7) | DP + MIA composite | `src/metrics/privacy.py` |
| §3.5 (Accountability, Eq. 8) | Audit sub-scores | `src/metrics/accountability.py` |
| §3.6 (Trust Index, Eq. 9) | AHP-weighted TI | `src/metrics/trust_index.py` |
| §3.7 (MOO, Eq. 10–11) | Lagrangian training | `src/training/fairness_loss.py` |
| §3.8 (Algorithm 1) | EAGF procedure | `src/training/pareto_trainer.py` |
| §5.1 (Biometric Setup) | Data loading + splits | `src/utils/data_loader.py` |
| §5.1.2 (Ablation) | M0–M5 variants | `src/evaluation/ablation.py` |
| §5.1.3 (Statistics) | z-test, Wilcoxon, bootstrap | `src/evaluation/statistics.py` |
| §5.2 (RE-IoT) | Telemetry simulation | `src/utils/reiot_simulator.py` |
| §6.1 (Trade-off Analysis) | Coupling evidence | `notebooks/04_pareto_front.ipynb` |
| §6.3 (TI Proxy) | Engineering proxy caveat | `docs/reproducibility.md` |
| §6.4 (Regulatory Alignment) | Compliance mapping | `docs/regulatory_mapping.md` |

---

## Interactive Notebooks

<div align="center">

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/aliakarma/eagf/blob/main/notebooks/01_eagf_demo.ipynb)
&nbsp;
[![GitHub](https://img.shields.io/badge/GitHub-aliakarma%2Feagf-blue?logo=github)](https://github.com/aliakarma/eagf)

</div>

Run all five notebooks directly in your browser — no local installation required.
Each notebook is self-contained and installs its own lightweight dependencies.

| # | Notebook | Content | Open |
|:---:|---|---|:---:|
| 1 | `01_eagf_demo.ipynb` | Full framework demo: train M0–M5, ablation table, Figure 3, TI components | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/aliakarma/eagf/blob/main/notebooks/01_eagf_demo.ipynb) |
| 2 | `02_statistical_analysis.ipynb` | Two-proportion z-test, paired t-test, bootstrap CIs, significance matrix | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/aliakarma/eagf/blob/main/notebooks/02_statistical_analysis.ipynb) |
| 3 | `03_reiot_fairness.ipynb` | RE-IoT node-class FPRP analysis, FPR breakdown, operational cost savings | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/aliakarma/eagf/blob/main/notebooks/03_reiot_fairness.ipynb) |
| 4 | `04_pareto_front.ipynb` | 5×5 Pareto-grid MOO, privacy–fairness trade-off scatter, TI heatmap | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/aliakarma/eagf/blob/main/notebooks/04_pareto_front.ipynb) |
| 5 | `05_trust_index_sensitivity.ipynb` | AHP weight derivation, sector-specific weights, TI sensitivity curves | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/aliakarma/eagf/blob/main/notebooks/05_trust_index_sensitivity.ipynb) |

> **Tip:** Click any badge to open the notebook in Google Colab instantly.
> The first cell installs all required packages automatically.


---

## Citation

If you use EAGF in your research, please cite:

```bibtex
@article{jan2025eagf,
  author    = {Jan, Salman and Muhammad, Munir Azam and Syed, Toqeer Ali and
               Akarma, Ali and Lee, It Ee and Wali, Qamar and
               Kamal, Shahid and Ali, Jawad},
  title     = {Ethical {AI} Governance for Cybersecurity in Renewable Energy
               {IoT} Systems: A Four-Pillar Framework Integrating Transparency,
               Fairness, Privacy, and Accountability with a Composite Trust Index},
  journal   = {},
  year      = {2026},
  note      = {Under review},
  doi       = {},
  url       = {}
}
```

---

## License

This project is released under the **MIT License**. See [LICENSE](LICENSE) for full terms.

The following third-party components are used under their respective licences:
- [Opacus](https://opacus.ai/) (Apache 2.0) — DP-SGD implementation
- [SHAP](https://github.com/slundberg/shap) (MIT) — Explainability
- [PyTorch](https://pytorch.org/) (BSD-3-Clause) — Deep learning framework
- [scikit-learn](https://scikit-learn.org/) (BSD-3-Clause) — Baseline models and metrics

---

## Acknowledgements

This research is supported by the **Ministry of Higher Education (MOHE)** under the 2023 Translational Research Program for the Energy Sustainability Focus Area (Project ID: MMUE/240001), the **2024 ASEAN IVO** (Project ID: 2024-02), and **Multimedia University, Malaysia**.

---

<div align="center">
<sub>Built with rigour. Deployed with trust.</sub>
</div>
