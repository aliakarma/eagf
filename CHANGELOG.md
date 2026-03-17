# Changelog

All notable changes to EAGF are documented in this file.
Format follows [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

---

## [1.0.0] — 2025-01-01

### Added

#### Core Framework
- Four-pillar Ethical AI Governance Framework (EAGF) implementing EU AI Act requirements
- Transparency metric: SHAP-based Explanation Clarity score (C) with fidelity/sparsity decomposition
- Fairness metrics: Recall Parity (RP) for biometric systems and False-Positive-Rate Parity (FPRP) for RE-IoT node classes
- Privacy metric (P): composite of differential-privacy epsilon and membership-inference AUC resistance
- Accountability metric (A): three sub-scores covering audit completeness, data lineage traceability, and regulatory compliance
- Composite Trust Index (TI) with AHP-weighted aggregation and equal-weight baseline

#### Training & Optimisation
- Six-variant ablation trainer (M0–M5) supporting baseline through full joint EAGF
- Recall-parity Lagrangian penalty term (differentiable soft constraint)
- Pareto-front hyperparameter search over 5×5 (lambda_RP × lambda_C) grid
- DP simulation via post-training gradient noise injection
- Group-balanced sample re-weighting for fairness intervention

#### Evaluation
- Shadow-model membership inference attack (MIA) stress-test
- Cryptographic SHA-256 audit logger (IEC 62351-compatible)
- Ablation aggregator with 95% bootstrap confidence intervals
- Statistical significance: two-proportion z-test (accuracy) and Wilcoxon signed-rank (TI)
- Report generator producing markdown summary reports

#### Data
- Synthetic demographically-biased biometric dataset generator (4 demographic groups)
- 5G RE-IoT telemetry simulator: FDIA, command injection, DoS attacks across urban/peri-urban/rural nodes
- Preprocessing pipeline: DP noise injection, group-balanced sample weights, standard normalisation

#### Infrastructure
- `run_eagf.py`: one-command full experiment pipeline with `--fast` and `--skip-pareto` flags
- `tests/run_tests.py`: 63-test standalone test suite (no pytest required)
- GitHub Actions CI workflow (Python 3.9/3.10/3.11)
- Compliance checklists: biometric (EU AI Act + GDPR + NIST) and ICS (+ IEC 62351 + NIS2)
- Figure generation: ablation bar chart (Figure 3), Pareto-front scatter, RE-IoT FPRP bar chart
- `CITATION.cff` for GitHub "Cite this repository" button

#### Documentation
- `docs/reproducibility.md`: full parameter table, statistical protocol, hardware specification
- `docs/metric_definitions.md`: all four pillar metrics with equations
- `docs/regulatory_mapping.md`: EU AI Act, GDPR, NIST AI RMF, NIS2, IEC 62351 alignment
- `data/README.md`: EFR dataset download instructions and RE-IoT generation guide

### Key Results (Demo Synthetic Data, 3 Seeds)

| Model | Accuracy | RP | C | P | A | TI |
|---|:---:|:---:|:---:|:---:|:---:|:---:|
| M0: Baseline | 0.987 | 0.933 | 0.553 | 0.123 | 0.233 | 0.564 |
| M1: +Transparency | 0.987 | 0.933 | 0.784 | 0.123 | 0.233 | 0.622 |
| M2: +Fairness | 0.987 | 0.938 | 0.553 | 0.123 | 0.233 | 0.565 |
| M3: +Privacy | 1.000 | 1.000 | 0.560 | 0.210 | 0.233 | 0.677 |
| M4: +Accountability | 0.987 | 0.933 | 0.553 | 0.123 | 0.696 | 0.680 |
| **M5: EAGF (Full)** | 1.000 | 1.000 | 0.800 | 0.210 | 0.696 | **0.852** |

**Key finding confirmed:** M5 TI (0.852) > all single-pillar TIs (max 0.680). Joint governance is necessary.

---

## [Unreleased]

### Planned
- PyTorch/Opacus backend for GPU-accelerated DP-SGD training
- True SHAP (TreeExplainer/DeepExplainer) when shap package available
- Federated EAGF variant for cross-operator RE-IoT collaboration
- Direct user-perception study integration for TI proxy validation
- Physical RE-IoT testbed validation (BATADAL / SWaT datasets)
- Automated Pareto-front exploration via neural architecture search
- EU AI Liability Directive accountability extension
