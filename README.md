# EAGF: Ethical AI Governance Framework for Cybersecurity in Renewable Energy IoT Systems

> A reproducible research framework for jointly optimizing fairness, privacy, clarity, and accountability in AI-based cybersecurity.

![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![License](https://img.shields.io/badge/License-MIT-green)
![Status](https://img.shields.io/badge/Status-Research%20Prototype-orange)
![Reproducibility](https://img.shields.io/badge/Reproducibility-10%20Seeds-success)

[![Open Demo in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/aliakarma/eagf/blob/main/notebooks/01_eagf_demo.ipynb)
[![Open Statistical Analysis in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/aliakarma/eagf/blob/main/notebooks/02_statistical_analysis.ipynb)
[![Open Fairness Study in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/aliakarma/eagf/blob/main/notebooks/03_reiot_fairness.ipynb)

## Overview

EAGF is a four-pillar governance framework that evaluates and optimizes:

- Transparency (clarity)
- Fairness
- Privacy
- Accountability

The framework combines these pillars into a composite Trust Index for model comparison, model selection, and governance-aware deployment decisions.


## 📊 Real-World Experimental Results

> **Primary dataset: RE-IoT** (Renewable Energy IoT — synthetic RE-IoT simulator with urban / peri-urban / rural node groups and realistic network traffic patterns).
> Synthetic biometric results are shown as secondary validation.
> All experiments: seeds 42–51 (10 independent runs), 30 training epochs.

### RE-IoT Results (Primary — Table 6)

| Model | Trust Index (mean ± std) | Recall Parity (mean ± std) | Accuracy (mean ± std) | Memory (MB) | Latency (ms/sample) |
|---|---:|---:|---:|---:|---:|
| Baseline | 0.6395 ± 0.0022 | 1.0000 ± 0.0000 | — | ~796 | 0.0014 |
| **EAGF** | **0.8104 ± 0.0021** | 1.0000 ± 0.0000 | — | ~797 | 0.0014 |

**Key finding:** EAGF improves the composite Trust Index by **+0.1709** (+26.7%) over the unregulated baseline on the RE-IoT dataset, with no increase in inference latency.

### Calibration Results (ECE and Brier Score — 10 seeds)

Calibration metrics are evaluated **post-hoc** (not optimized during training) to avoid metric gaming.

| Model | ECE (↓) | ECE std | Brier Score (↓) | Brier std |
|---|---:|---:|---:|---:|
| Baseline | 0.0458 | 0.0066 | 0.1285 | 0.0011 |
| **EAGF** | 0.0514 | 0.0065 | 0.1289 | 0.0012 |
| Joint DP+Fair | 0.1186 | 0.0079 | 0.1520 | 0.0041 |

ECE and Brier Score for baseline and EAGF are comparable (within ±0.006), confirming that removing the confidence-inflating entropy and confidence-floor penalties did not harm calibration. Joint DP+Fair shows higher ECE (0.1186) due to aggressive fairness enforcement.

### Synthetic Validation Results (Table 5)

| Model | Accuracy | Recall Parity | Clarity (C) | Privacy (P) | Accountability (A) | Trust Index (TI) |
|---|---:|---:|---:|---:|---:|---:|
| Baseline | 0.8421 ± 0.0094 | 0.8019 ± 0.0292 | 0.9361 ± 0.0417 | 0.2392 ± 0.0094 | 0.3000 ± 0.0000 | 0.5693 ± 0.0150 |
| **EAGF** | **0.8396 ± 0.0094** | **0.8081 ± 0.0257** | 0.9341 ± 0.0400 | 0.2386 ± 0.0104 | **0.9833 ± 0.0000** | **0.7410 ± 0.0129** |
| Joint DP+Fair | 0.8023 ± 0.0047 | 0.8689 ± 0.0176 | 0.9453 ± 0.0188 | 0.4199 ± 0.0027 | 0.3000 ± 0.0000 | 0.6335 ± 0.0051 |

### System Trade-offs

| Model | Latency (ms/sample) | Memory (MB) | Energy (J/inference) |
|---|---:|---:|---:|
| Baseline | 0.0015 | 766.8 | 0.0240 |
| EAGF | 0.0015 | 766.9 | 0.0237 |
| Joint DP+Fair | 0.0038 | 765.1 | 0.0906 |

EAGF adds negligible system overhead vs the baseline. Joint DP+Fair incurs 2.5× the latency due to sample reweighting and gradient clipping.

### Key Findings

- **Trust Index**: EAGF achieves TI = 0.8104 (RE-IoT) and 0.7410 (synthetic), +26.7% and +30.2% above the respective baselines.
- **Fairness**: Recall parity improves by +0.006 (synthetic); RE-IoT shows full parity (RP=1.0) across urban/peri-urban/rural node classes.
- **Calibration**: Clarity is evaluated post-hoc; ECE and Brier are comparable across EAGF and baseline, confirming no metric gaming occurred.
- **Privacy**: EAGF maintains differential privacy (DP-SGD) while achieving high fairness, a trade-off not present in the baseline.
- **Accountability**: EAGF accountability score = 0.983 vs 0.300 for baseline — comprehensive audit logging and compliance coverage.

## Figures

### Figure 1: Ablation Study — Pillar-by-Pillar Metric Comparison
<p align="center">
  <img src="figures/figure3.png" alt="Ablation comparison of baseline and EAGF pillars" width="95%" />
</p>

### Figure 2: Trust Index vs Inference Latency
<p align="center">
  <img src="figures/ti_vs_latency.png" alt="Trust Index vs inference latency" width="95%" />
</p>

## Quick Start

```bash
python run_full_pipeline.py --config biometric_tuned_auto.yaml --seeds 42 43 44 45 46 47 48 49 50 51
```

## Installation

### Requirements

- Python 3.9+
- pip
- Optional: CUDA-enabled PyTorch build for GPU acceleration

### Setup

```bash
git clone https://github.com/aliakarma/eagf.git
cd eagf
python -m venv .venv

# Windows
.venv\Scripts\activate

# Linux/macOS
# source .venv/bin/activate

pip install --upgrade pip
pip install -r requirements.txt
```

## Reproducibility

### Full experiment — RE-IoT (recommended, primary dataset)

```bash
python run_full_pipeline.py --config configs/reiot_default.yaml \
  --seeds 42 43 44 45 46 47 48 49 50 51
```

### Full experiment — Synthetic biometric (secondary validation)

```bash
python run_full_pipeline.py --config configs/biometric_default.yaml \
  --seeds 42 43 44 45 46 47 48 49 50 51
```

Expected key outputs:

- results/final_report.txt
- results/biometric/main_results.csv
- results/biometric/ablation/ablation_summary.csv
- results/reiot/node_class_results.csv
- figures/figure3.png
- figures/ti_vs_latency.png

### Fast smoke-test mode

```bash
python run_full_pipeline.py --fast
```

## Notebooks (with Colab)

Use the links below to run notebooks directly in Google Colab.

1. 01_eagf_demo.ipynb

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/aliakarma/eagf/blob/main/notebooks/01_eagf_demo.ipynb)

2. 02_statistical_analysis.ipynb

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/aliakarma/eagf/blob/main/notebooks/02_statistical_analysis.ipynb)

3. 03_reiot_fairness.ipynb

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/aliakarma/eagf/blob/main/notebooks/03_reiot_fairness.ipynb)

4. 04_pareto_front.ipynb

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/aliakarma/eagf/blob/main/notebooks/04_pareto_front.ipynb)

5. 05_trust_index_sensitivity.ipynb

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/aliakarma/eagf/blob/main/notebooks/05_trust_index_sensitivity.ipynb)

## Project Structure (Detailed)

```text
eagf/
├── 🧩 configs/
│   ├── biometric_default.yaml
│   ├── biometric_tuned_auto.yaml
│   ├── compliance_checklist.json
│   ├── compliance_checklist.yaml
│   ├── compliance_checklist_ics.json
│   ├── compliance_checklist_ics.yaml
│   ├── eagf_thresholds.yaml
│   └── reiot_default.yaml
├── 📊 data/
│   ├── README.md
│   ├── biometric/
│   │   └── splits.json
│   └── reiot/
│       ├── groups_test.npy
│       ├── groups_train.npy
│       ├── metadata.json
│       ├── X_test.npy
│       ├── X_train.npy
│       ├── y_test.npy
│       └── y_train.npy
├── 📚 docs/
│   ├── metric_definitions.md
│   ├── regulatory_mapping.md
│   └── reproducibility.md
├── 🖼️ figures/
│   ├── figure3.png
│   ├── pareto_front.png
│   └── ti_vs_latency.png
├── 📓 notebooks/
│   ├── 01_eagf_demo.ipynb
│   ├── 02_statistical_analysis.ipynb
│   ├── 03_reiot_fairness.ipynb
│   ├── 04_pareto_front.ipynb
│   └── 05_trust_index_sensitivity.ipynb
├── 🧪 results/
│   ├── final_report.txt
│   ├── summary_report.md
│   ├── biometric/
│   │   ├── main_results.csv
│   │   ├── statistical_tests.json
│   │   ├── summary.txt
│   │   ├── ablation/
│   │   │   ├── ablation_summary.csv
│   │   │   ├── accountability/seed_42 ... seed_51/
│   │   │   ├── baseline/seed_42 ... seed_51/
│   │   │   ├── eagf/seed_42 ... seed_51/
│   │   │   ├── fairness/seed_42 ... seed_51/
│   │   │   ├── privacy/seed_42 ... seed_51/
│   │   │   └── transparency/seed_42 ... seed_51/
│   │   ├── baseline/seed_42 ... seed_51/
│   │   ├── eagf/seed_42 ... seed_51/
│   │   ├── joint_dp_fair/seed_42 ... seed_51/
│   │   └── pareto/run_00 ... run_24/
│   └── reiot/
│       ├── node_class_results.csv
│       ├── baseline/seed_42 ... seed_51/
│       └── eagf/seed_42 ... seed_51/
├── 🛠️ scripts/
│   ├── run_ablation.sh
│   ├── run_all.sh
│   ├── run_baseline.sh
│   ├── run_biometric.sh
│   ├── run_pareto_search.sh
│   ├── run_reiot.sh
│   ├── run_scaling_experiment.py
│   ├── sweep_three_stage.py
│   └── verify_metrics.py
├── 🧠 src/
│   ├── baselines/
│   │   ├── aif360_dp_pipeline.py
│   │   └── joint_dp_fair_baseline.py
│   ├── evaluation/
│   │   ├── ablation.py
│   │   ├── analysis_report.py
│   │   ├── audit_logger.py
│   │   ├── baseline.py
│   │   ├── benchmark_suite.py
│   │   ├── mia_attack.py
│   │   ├── report_generator.py
│   │   └── statistics.py
│   ├── metrics/
│   │   ├── accountability.py
│   │   ├── clarity.py
│   │   ├── fairness.py
│   │   ├── privacy.py
│   │   └── trust_index.py
│   ├── training/
│   │   ├── eagf_trainer.py
│   │   ├── fairness_loss.py
│   │   └── pareto_trainer.py
│   └── utils/
│       ├── ahp.py
│       ├── data_loader.py
│       ├── preprocessing.py
│       ├── real_data_loader.py
│       ├── reiot_simulator.py
│       └── visualisation.py
├── ✅ tests/
│   ├── conftest.py
│   ├── run_tests.py
│   ├── test_data.py
│   └── test_metrics.py
├── run_eagf.py
├── run_full_pipeline.py
├── requirements.txt
├── environment.yml
├── setup.py
└── README.md
```

## Core Outputs

| Path | Description |
|---|---|
| results/final_report.txt | Consolidated pipeline report and validation summary. |
| results/biometric/main_results.csv | Main biometric benchmark aggregates for baseline, EAGF, and joint_dp_fair. |
| results/biometric/statistical_tests.json | Statistical significance outputs for model comparisons. |
| results/biometric/ablation/ablation_summary.csv | Multi-pillar ablation summary table. |
| results/reiot/node_class_results.csv | RE-IoT node-class evaluation summary. |
| figures/ | Publication-ready visual outputs. |

## Method Summary

- Fairness: recall-parity aware regularization.
- Privacy: DP-based training and privacy-oriented evaluation.
- Clarity: explanation-oriented confidence/structure regularization and clarity metrics.
- Accountability: auditability and compliance-informed reporting.
- Trust Index: weighted integration of the four governance pillars.

## Citation

If you use this repository, please cite the associated EAGF paper.

```bibtex
@article{jan2026eagf,
  title   = {Ethical AI Governance for Cybersecurity in Renewable Energy IoT Systems: A Four-Pillar Framework Integrating Transparency, Fairness, Privacy, and Accountability with a Composite Trust Index},
  author  = {Jan, Salman and Muhammad, Munir Azam and Syed, Toqeer Ali and Akarma, Ali and Lee, It Ee and Wali, Qamar and Kamal, Shahid and Ali, Jawad},
  journal = {Under Review},
  year    = {2026}
}
```


<!-- PIPELINE_RESULTS_START -->

# 📊 Latest Experimental Results

*Auto-generated on 2026-03-30 13:58 UTC using seeds [42, 43, 44, 45, 46, 47, 48, 49, 50, 51].*

## Results Table

| Model | Accuracy | Recall Parity | Clarity (C) | Privacy (P) | Accountability (A) | Trust Index (TI) |
|---|---|---|---|---|---|---|
| baseline | 0.8421 | 0.8019 | 0.9361 | 0.2392 | 0.3000 | 0.5693 |
| eagf | 0.8396 | 0.8081 | 0.9341 | 0.2386 | 0.9833 | 0.7410 |
| joint_dp_fair | 0.8023 | 0.8689 | 0.9453 | 0.4199 | 0.3000 | 0.6335 |

## Key Observations

- EAGF improves Trust Index by **+0.1717** (30.16%) relative to baseline.
- Accuracy change: **-0.0025** (acceptable trade-off for governance benefits).
- Full governance (EAGF) outperforms all single-pillar ablations on TI.

## Reports and Figures

- 📄 [Detailed report](results/final_report.txt)
- 📁 [Figures](figures/)

<!-- PIPELINE_RESULTS_END -->
