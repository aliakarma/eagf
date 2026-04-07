# EAGF: Ethical AI Governance Framework

> **Joint Optimization of Fairness, Privacy, Explainability & Accountability in AI-Based Cybersecurity**

![Python](https://img.shields.io/badge/Python-3.9%2B-blue?style=flat-square)
![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)
![Status](https://img.shields.io/badge/Status-Publication_Ready-brightgreen?style=flat-square)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-red?style=flat-square)
![Reproducibility](https://img.shields.io/badge/Reproducibility-5_Seeds-success?style=flat-square)

[![Open Demo in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/aliakarma/eagf/blob/main/notebooks/01_eagf_demo.ipynb)
[![Open Statistical Analysis in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/aliakarma/eagf/blob/main/notebooks/02_statistical_analysis.ipynb)
[![Open Fairness Study in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/aliakarma/eagf/blob/main/notebooks/03_reiot_fairness.ipynb)
[![Open Pareto Analysis in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/aliakarma/eagf/blob/main/notebooks/04_pareto_front.ipynb)
[![Open Sensitivity Analysis in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/aliakarma/eagf/blob/main/notebooks/05_trust_index_sensitivity.ipynb)

---

## 📋 Overview

EAGF is a reproducible research framework that combines **differential privacy (DP-SGD)**, **fairness regularization (false positive rate parity)**, **explainability (SHAP)**, and **audit logging** into a unified governance pipeline. Evaluated on the real-world **Edge-IIoTset** (IEEE 2022) for IoT anomaly detection with multi-objective Pareto trade-off analysis.

**Key Focus**: Cybersecurity for IoT networks with resource-constrained devices. EAGF enables governance-aware AI deployment with minimal system overhead.

---

## 🎯 Key Contributions

- 🏭 **Real-world dataset integration**: Edge-IIoTset (IEEE 2022, 150K samples, 40 network flow features)
- ⚖️ **Multi-objective governance**: Joint optimization of four pillars—fairness, privacy, clarity, accountability
- 📊 **Trust Index metric**: Composite governance score for model selection and comparison
- 📈 **Pareto trade-off analysis**: Quantify accuracy-fairness-privacy trade-offs with front visualization
- 🔄 **Reproducible pipeline**: Deterministic execution, fixed seeds, publication-ready results

---

## 📊 Key Results Summary

**Dataset**: Edge-IIoTset (150K samples, 3 protocol-type protected groups)  
**Baselines**: Unregulated + Joint DP+Fair  
**Statistical Rigor**: 5 independent seeds with 95% CI  

| Metric | Baseline | **EAGF** | Δ | Improvement |
|--------|----------|----------|-------|-------------|
| **Accuracy** | 0.6481 ± 0.0251 | **0.6650 ± 0.0079** | +0.0168 | +2.6% |
| **FPR Parity** | 0.4931 ± 0.0849 | **0.7709 ± 0.0573** | +0.2779 | **+56.4%** ✓ |
| **Clarity** | 0.6918 ± 0.0432 | **0.7390 ± 0.0548** | +0.0472 | +6.8% |
| **Privacy** | 0.2475 ± 0.0030 | **0.2482 ± 0.0025** | +0.0007 | Preserved ✓ |
| **Accountability** | 0.0000 ± 0.0000 | **0.6667 ± 0.0000** | +0.6667 | Full coverage ✓ |
| **Trust Index** | 0.3581 ± 0.0129 | **0.6062 ± 0.0108** | +0.2481 | **+69.3%** ✓ |

### 🔬 Key Findings

✅ **Fairness breakthrough**: FPR parity improved +56.4% across protocol-type groups (web, IoT MQTT, misc)  
✅ **Trust Index surge**: Composite governance metric +69.3%, indicating strong multi-objective alignment  
✅ **Privacy guarantee**: Differential privacy (ε=2.4) maintained with negligible DP-ε change  
✅ **Edge deployment ready**: +0.2ms latency (~11%), +5.8MB memory—suitable for constrained IoT  
✅ **Calibration stable**: ECE and Brier comparable (±0.05), no metric gaming  

---

## 📁 Repository Structure

```
eagf/
├── 📖 README.md                    # This file
├── 📄 requirements.txt             # Dependencies (numpy, pandas, scikit-learn, fairlearn, pyyaml)
├── 🔧 setup.py                     # Package setup
│
├── 🚀 run_eagf.py                  # Single-seed entry point
├── 🚀 run_full_pipeline.py         # Multi-seed experiment runner (MAIN)
│
├── 🧠 src/
│   ├── training/
│   │   ├── eagf_trainer.py         # Main EAGF training loop with governance
│   │   ├── fairness_loss.py         # Fairness penalty (FPR parity)
│   │   └── pareto_trainer.py        # Pareto front exploration
│   │
│   ├── evaluation/
│   │   ├── baseline.py             # Unregulated baseline + Joint DP+Fair
│   │   ├── ablation.py             # Single-pillar ablation study
│   │   ├── report_generator.py      # Multi-seed report + statistics
│   │   ├── audit_logger.py         # Compliance audit trail
│   │   ├── benchmark_suite.py      # System metrics (latency, memory, energy)
│   │   └── statistics.py           # 95% CI, statistical tests
│   │
│   ├── metrics/
│   │   ├── fairness.py             # FPR parity, recall parity, group metrics
│   │   ├── privacy.py              # DP-SGD evaluation, privacy accounting
│   │   ├── clarity.py              # SHAP-based explainability
│   │   ├── accountability.py       # Audit coverage, compliance scoring
│   │   └── trust_index.py          # Composite Trust Index aggregation
│   │
│   ├── utils/
│   │   ├── data_loader.py          # Generic dataset loading
│   │   ├── edge_iiot_loader.py     # Edge-IIoTset specific (protocol_type grouping)
│   │   ├── real_data_loader.py     # Real dataset pipeline
│   │   ├── preprocessing.py        # Feature engineering, normalization
│   │   ├── reiot_simulator.py      # RE-IoT synthetic simulator (optional)
│   │   ├── ahp.py                  # Analytic Hierarchy Process (Trust Index weights)
│   │   └── visualisation.py        # Pareto, Trade-off plots
│   │
│   └── baselines/
│       ├── aif360_dp_pipeline.py   # AIF360 fairness baseline
│       └── joint_dp_fair_baseline.py # Combined DP + fairness baseline
│
├── ⚙️ configs/
│   ├── reiot_real.yaml             # Main: Edge-IIoTset + EAGF governance (RECOMMENDED)
│   ├── reiot_default.yaml          # Alternative RE-IoT config
│   ├── biometric_default.yaml      # Biometric (secondary validation)
│   ├── biometric_tuned_auto.yaml   # Tuned biometric
│   ├── eagf_thresholds.yaml        # Governance thresholds
│   └── compliance_checklist*.yaml  # Compliance templates
│
├── 📊 data/
│   ├── README.md                   # Data documentation
│   └── real_iot/
│       └── edge_iiot.csv           # Edge-IIoTset (150K rows, 40 features) [USER PROVIDED]
│
├── 📓 notebooks/
│   ├── 01_eagf_demo.ipynb          # Quick start demo
│   ├── 02_statistical_analysis.ipynb # Multi-seed statistics
│   ├── 03_reiot_fairness.ipynb     # Fairness deep-dive (protocol_type groups)
│   ├── 04_pareto_front.ipynb       # Pareto front visualization
│   └── 05_trust_index_sensitivity.ipynb # Sensitivity analysis
│
├── 🎨 figures/
│   ├── pareto_front.png            # Accuracy vs. Fairness vs. Privacy
│   ├── ti_vs_latency.png           # Trust Index vs. Inference Latency
│   └── ablation_comparison.png     # Single-pillar vs. multi-pillar
│
├── 📋 docs/
│   ├── metric_definitions.md       # Detailed metric documentation
│   ├── regulatory_mapping.md       # Compliance + GDPR/CCPA alignment
│   └── reproducibility.md          # Detailed reproducibility steps
│
├── 🧪 results/
│   ├── final_report.txt            # ✨ MAIN DELIVERABLE—aggregated results
│   ├── main_results.csv            # Baseline, EAGF, Joint metrics (5 seeds)
│   ├── pareto_results.csv          # Pareto exploration (25 runs)
│   └── [seed-specific subdirs]/     # Individual seed outputs
│
├── 🛠️ scripts/
│   ├── run_all.sh                  # Full pipeline (Edge-IIoT + ablation)
│   ├── run_reiot.sh                # Edge-IIoTset only
│   ├── run_baseline.sh             # Baseline only
│   ├── run_pareto_search.sh        # Pareto front exploration
│   ├── sweep_three_stage.py        # Hyperparameter sweep
│   └── verify_metrics.py           # Metric validation
│
├── ✅ tests/
│   ├── test_data.py                # Data loading & preprocessing
│   ├── test_metrics.py             # Metric computation
│   ├── conftest.py                 # Pytest fixtures
│   └── run_tests.py                # Test runner
│
├── 🐳 Dockerfile                   # Container setup
├── 🌍 environment.yml              # Conda environment (optional)
├── 📚 CONTRIBUTING.md              # Contribution guidelines
├── 📜 LICENSE                      # MIT License
├── 📝 CHANGELOG.md                 # Version history
└── 📋 CITATION.cff                 # BibTeX citation metadata
```

---

## 🖼️ Visualizations & Results

### Figure 1: Pareto Front — Trade-off Analysis
<p align="center">
  <img src="figures/pareto_front.png" alt="Pareto front: Accuracy vs Fairness vs Privacy" width="90%" />
  <br><em>Multi-objective optimization frontier showing EAGF solutions (orange) vs. baseline (blue) across accuracy-fairness-privacy space.</em>
</p>

### Figure 2: Trust Index vs. Inference Latency
<p align="center">
  <img src="figures/ti_vs_latency.png" alt="Trust Index vs inference latency comparison" width="90%" />
  <br><em>System efficiency comparison: EAGF maintains high governance (TI=0.61) with minimal latency overhead (+0.2ms/sample).</em>
</p>

### Figure 3: Ablation Study — Pillar-by-Pillar Comparison
<p align="center">
  <img src="figures/figure3.png" alt="Ablation comparison of EAGF pillars" width="90%" />
  <br><em>Impact of each governance pillar on Trust Index. Multi-pillar integration significantly outperforms single-pillar approaches.</em>
</p>

---

## 📓 Interactive Notebooks

Run notebooks directly in Google Colab without local setup:

| Notebook | Purpose | Badge |
|----------|---------|-------|
| **01_eagf_demo.ipynb** | 5-minute quick start | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/aliakarma/eagf/blob/main/notebooks/01_eagf_demo.ipynb) |
| **02_statistical_analysis.ipynb** | Multi-seed statistics, hypothesis tests | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/aliakarma/eagf/blob/main/notebooks/02_statistical_analysis.ipynb) |
| **03_reiot_fairness.ipynb** | Fairness deep-dive by protocol-type groups | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/aliakarma/eagf/blob/main/notebooks/03_reiot_fairness.ipynb) |
| **04_pareto_front.ipynb** | Interactive Pareto front exploration | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/aliakarma/eagf/blob/main/notebooks/04_pareto_front.ipynb) |
| **05_trust_index_sensitivity.ipynb** | Trust Index weight sensitivity analysis | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/aliakarma/eagf/blob/main/notebooks/05_trust_index_sensitivity.ipynb) |

---

## ⚙️ Installation

### Requirements

- Python 3.9+
- pip

### Setup

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

### Edge-IIoTset (Required)

**Dataset**: ML-EdgeIIoT-dataset.csv (real-world IoT anomaly detection)  
**Source**: IEEE Access 2022 (Ferrag et al.)  
**Size**: ~78 MB (157.8K raw rows)  
**Features**: 40 network flow + protocol-specific attributes  
**Labels**: Normal vs. Attack (imbalanced: 23.1K vs. 126.9K)  

**Note**: EAGF uses real Edge-IIoTset data only. No synthetic fallback.

### Setup Instructions

**Option A: Manual Download (Recommended)**

1. Download `ML-EdgeIIoT-dataset.csv` from [IEEE DataPort](https://dx.doi.org/10.21203/rs.3.rs-1433551/v1)
2. Extract and place at:
   ```
   data/real_iot/edge_iiot.csv
   ```

**Option B: Verify Existing Data**

If you already have the dataset:

```bash
# Check file size (~78 MB)
ls -lh data/real_iot/edge_iiot.csv
```

---

## 📂 Output Files & Deliverables

| File | Description | Format |
|------|-------------|--------|
| **results/final_report.txt** | ✨ Main deliverable—aggregated metrics, statistics, validation gates | TXT |
| **results/main_results.csv** | Summary table: baseline, EAGF, Joint DP+Fair across all metrics | CSV |
| **results/pareto_results.csv** | Pareto search results (25 multi-objective runs with trade-off scores) | CSV |
| **figures/pareto_front.png** | 3D visualization: accuracy vs. fairness vs. privacy | PNG |
| **figures/ti_vs_latency.png** | 2D scatter: Trust Index vs. inference latency | PNG |
| **figures/ablation_comparison.png** | Bar chart: pillar ablation study (single vs. multi-pillar) | PNG |
| **[seed-specific]/predictions.json** | Per-sample predictions, confidences, fairness group info | JSON |
| **[seed-specific]/metrics.json** | Detailed metrics for each seed | JSON |

---

## 🚀 Quick Start (Smoke Test)

Validate the entire pipeline in ~5 minutes using a single seed:

```bash
python run_full_pipeline.py \
  --real_dataset edge_iiot \
  --config configs/reiot_real.yaml \
  --seeds 42 \
  --fast
```

**What happens**:
- ✓ Loads Edge-IIoTset and validates data
- ✓ Trains baseline + EAGF + Joint DP+Fair models (1 seed)
- ✓ Computes fairness, privacy, clarity, accountability metrics
- ✓ Generates `results/final_report.txt` with summary
- ✓ Produces figures in `figures/`

**Expected runtime**: ~5 min on CPU (Intel Core i7)

---

## 🏆 Full Experiment (Publication Results)

Reproduce final results with 5 independent seeds (statistical rigor):

```bash
python run_full_pipeline.py \
  --real_dataset edge_iiot \
  --config configs/reiot_real.yaml \
  --seeds 42 43 44 45 46
```

**Output**:
- Mean ± std for all governance metrics
- 95% confidence intervals
- Pareto front visualization (25 multi-objective runs)
- Final report with ablation analysis
- Seed-specific detailed logs

**Expected runtime**: ~10–15 min on CPU

---

## 🔬 Advanced: Pareto Front Search

Explore the full accuracy-fairness-privacy trade-off surface:

```bash
python -c "
from src.training.pareto_trainer import ParetoTrainer
from src.utils.edge_iiot_loader import EdgeIIoTLoader

loader = EdgeIIoTLoader('data/real_iot/edge_iiot.csv')
X_train, X_test, y_train, y_test, groups = loader.load()

trainer = ParetoTrainer(X_train, y_train, groups, seed=42)
trainer.search(n_objectives=3, n_runs=25)  # Explore 25 configurations
trainer.plot_pareto('figures/pareto_custom.png')
"
```

---

## Key Results

**Dataset**: Edge-IIoTset (150K samples, protocol-type protected groups)  
**Baselines**: Unregulated model, Joint DP+Fair  
**Seeds**: 5 independent runs  
**Metrics**: Accuracy, Fairness (FPR Parity), Clarity, Privacy, Accountability, Trust Index

### Summary (Mean ± Std)

| Metric | Baseline | **EAGF** | Δ |
|--------|----------|----------|-----|
| **Accuracy** | 0.6481 ± 0.0251 | **0.6650 ± 0.0079** | +0.0168 (+2.6%) |
| **FPR Parity** | 0.4931 ± 0.0849 | **0.7709 ± 0.0573** | +0.2779 (+56.4%) |
| **Clarity** | 0.6918 ± 0.0432 | **0.7390 ± 0.0548** | +0.0472 (+6.8%) |
| **Privacy** | 0.2475 ± 0.0030 | **0.2482 ± 0.0025** | +0.0007 (+0.3%, preserved) |
| **Accountability** | 0.0000 ± 0.0000 | **0.6667 ± 0.0000** | +0.6667 ✓ |
| **Trust Index** | 0.3581 ± 0.0129 | **0.6062 ± 0.0108** | +0.2481 (+69.3%) |

### Key Findings

- **Fairness via FPR Parity**: EAGF achieves +56.4% improvement in false positive rate fairness across protocol-type groups (web, IoT MQTT, misc). Disparities in false alarm rates reduced from 49.3% to 23% spread.
- **Trust Index**: Composite governance metric improves by +69.3%, indicating strong multi-objective alignment.
- **Privacy Preserved**: Differential privacy (ε=2.4) maintained with negligible change vs. baseline. No privacy regression.
- **Minimal System Overhead**: Inference latency +0.2ms/sample (~11% increase); memory +5.8 MB. Suitable for edge deployment.
- **Calibration Stability**: ECE and Brier scores comparable (within ±0.05), no metric gaming.

---

## 🏗️ Method Overview

### Four-Pillar Governance Architecture

```
┌─────────────────────────────────────────────────────────┐
│                   EAGF Framework                         │
├─────────┬──────────┬──────────┬────────────────────────┤
│ Clarity │ Fairness │ Privacy  │  Accountability        │
├─────────┼──────────┼──────────┼────────────────────────┤
│  SHAP   │ FPRP     │ DP-SGD   │  Audit Logging + Rules │
│  Loss   │ Loss     │ Gradient │  Compliance Coverage   │
└─────────┴──────────┴──────────┴────────────────────────┘
                      ↓
              Trust Index (TI)
         Weighted Aggregation via AHP
```

### Key Components

| Pillar | Metric | Implementation | Config |
|--------|--------|-----------------|--------|
| **Fairness** | FPR Parity | `src/metrics/fairness.py` | `lambda_rp: 0.2` |
| **Privacy** | DP Accounting | `src/metrics/privacy.py` | `dp_epsilon: 2.4` |
| **Clarity** | SHAP Sparsity | `src/metrics/clarity.py` | `lambda_c: 0.05` |
| **Accountability** | Audit Coverage | `src/metrics/accountability.py` | Compliance rules |

---

## 📚 Reproducibility & Validation

### Deterministic Execution

- **Fixed seeds**: 42–46 (5 independent runs)
- **Deterministic pipeline**: Reproducible to ±0.005 variance (NumPy/PyTorch seeds)
- **No hidden preprocessing**: All transformations logged in `audit_logger.py`
- **Hyperparameter justification**: See [configs/reiot_real.yaml](configs/reiot_real.yaml)

### Validation Gates

All experiments must satisfy:
- ✓ FPR Parity EAGF ≥ Baseline + 0.02 (fairness improvement)
- ✓ Privacy EAGF ≥ Baseline (no regression)
- ✓ Accuracy drop ≤ 2% (stability requirement)
- ✓ Trust Index EAGF > Baseline (overall improvement)

---

## 📖 Documentation

| File | Purpose |
|------|---------|
| [docs/metric_definitions.md](docs/metric_definitions.md) | Detailed fairness, privacy, clarity, accountability metrics |
| [docs/regulatory_mapping.md](docs/regulatory_mapping.md) | GDPR, CCPA, ISO alignment |
| [docs/reproducibility.md](docs/reproducibility.md) | Step-by-step reproducibility guide |

---

## 🔗 Related Work

- **Fairness**: Hardt et al. (2016), Moritz et al. (2020)
- **Privacy**: Abadi et al. (2016) DP-SGD, Kairouz et al. (2021) DP survey
- **Explainability**: Lundberg & Lee (2017) SHAP
- **IoT Security**: Ferrag et al. (2022) Edge-IIoTset

---


## 📜 License

MIT License — See [LICENSE](LICENSE) for full terms.

---

## 🤝 Contributing

Contributions welcome! See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

---

## 💬 Support & Issues

For reproducibility help, issues, or questions:

1. **Check diagnostics**: See `results/final_report.txt` for detailed error logs
2. **Verify dataset**: Ensure `data/real_iot/edge_iiot.csv` exists (~78 MB)
3. **Check environment**: 
   ```bash
   python -m pip show scikit-learn fairlearn numpy pandas
   ```
4. **Open issue**: Include OS, Python version, full error trace, and reproducibility steps

---

## 📧 Contact

- **Maintainer**: Ali Akarma
- **Email**: [aliakarma974@gmail.com]
- **Issues**: [GitHub Issues](https://github.com/aliakarma/eagf/issues)

---

**Last Updated**: March 2026 | Python 3.9+ | PyTorch 2.0+
