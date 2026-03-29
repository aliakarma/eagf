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

## Latest Results (Updated)

The table below reflects the most recent regenerated outputs from results/biometric/main_results.csv using seeds 42-51.

| Model | Accuracy (mean +- std) | Recall Parity (mean +- std) | Clarity (mean +- std) | Privacy (mean +- std) | Accountability (mean +- std) | Trust Index (mean +- std) |
|---|---:|---:|---:|---:|---:|---:|
| baseline | 0.8463 +- 0.0103 | 0.7864 +- 0.0205 | 0.9285 +- 0.0283 | 0.2430 +- 0.0088 | 0.3000 +- 0.0000 | 0.5645 +- 0.0072 |
| eagf | 0.7863 +- 0.0252 | 0.9051 +- 0.0123 | 0.9609 +- 0.0214 | 0.2886 +- 0.0123 | 0.9833 +- 0.0000 | 0.7845 +- 0.0075 |
| joint_dp_fair | 0.7646 +- 0.0313 | 0.9084 +- 0.0176 | 0.9515 +- 0.0163 | 0.2880 +- 0.0127 | 0.3000 +- 0.0000 | 0.6120 +- 0.0087 |

Key observations:

- EAGF reaches the highest Trust Index (0.7845), with a +0.2200 absolute gain vs baseline.
- EAGF strongly improves governance pillars (fairness, privacy, clarity, accountability) at an expected accuracy trade-off.
- joint_dp_fair improves fairness/privacy but does not achieve EAGF-level accountability or Trust Index.

## Figures

### Figure 1: Main Metric Comparison
<p align="center">
  <img src="figures/figure3.png" alt="Main comparison of baseline and EAGF" width="95%" />
</p>

### Figure 2: Pareto Front
<p align="center">
  <img src="figures/pareto_front.png" alt="Pareto front" width="95%" />
</p>

### Figure 3: Trust Index vs Inference Latency
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

### Full experiment (recommended)

```bash
python run_full_pipeline.py --config biometric_tuned_auto.yaml --seeds 42 43 44 45 46 47 48 49 50 51
```

Expected key outputs:

- results/final_report.txt
- results/biometric/main_results.csv
- results/biometric/ablation/ablation_summary.csv
- figures/figure3.png
- figures/pareto_front.png
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
