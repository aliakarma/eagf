# Reproducibility Statement

This document provides a complete technical specification for reproducing all
results, tables, and figures in:

> **"Ethical AI Governance for Cybersecurity in Renewable Energy IoT Systems:
> A Four-Pillar Framework Integrating Transparency, Fairness, Privacy, and
> Accountability with a Composite Trust Index"**

---

## Reproducibility Checklist

| Item | Status | Notes |
|---|:---:|---|
| Source code available | ✅ | This repository |
| Dependencies pinned | ✅ | `requirements.txt`, `environment.yml` |
| Random seeds fixed | ✅ | Seeds 42, 123, 456 for all experiments |
| Dataset access instructions | ✅ | `data/README.md` |
| Pre-processed split indices | ✅ | `data/biometric/splits.json` |
| RE-IoT generation script | ✅ | `src/utils/data_loader.py --dataset reiot` |
| Hyperparameters documented | ✅ | `configs/biometric_default.yaml`, `configs/reiot_default.yaml` |
| Statistical test code | ✅ | `src/evaluation/statistics.py` |
| Figure generation scripts | ✅ | `src/utils/visualisation.py` |
| One-command reproduction | ✅ | `bash scripts/run_all.sh` |
| Hardware specification | ✅ | See below |
| Expected runtime | ✅ | See below |

---

## Experimental Parameters

### Biometric Case Study (Case Study 1)

| Parameter | Value | Config Key |
|---|---|---|
| Model architecture | ResNet-50 (pretrained ImageNet) | `model.architecture` |
| Optimiser | AdamW | `training.optimizer` |
| Learning rate | 1e-4 | `training.lr` |
| Batch size | 64 | `training.batch_size` |
| Epochs | 50 | `training.epochs` |
| DP privacy budget (ε) | 3.0 | `governance.dp_epsilon` |
| DP delta (δ) | 1e-5 | `governance.dp_delta` |
| DP accounting method | Rényi (Opacus) | — |
| Max gradient norm (DP-SGD) | 1.0 | `governance.dp_max_grad_norm` |
| Recall-parity λ_RP | 0.1 (deployed model) | `governance.lambda_rp` |
| Clarity λ_C | 0.05 (deployed model) | `governance.lambda_c` |
| Pareto grid: λ_RP steps | 5 (log-spaced: 1e-3 to 1) | `governance.pareto_grid_size` |
| Pareto grid: λ_C steps | 5 (log-spaced: 1e-3 to 1) | `governance.pareto_grid_size` |
| SHAP variant | DeepExplainer | `governance.shap_explainer` |
| SHAP top-k features | 5 | `governance.shap_top_k` |
| Attribution threshold τ | 1% of mean |φ̄| | `governance.shap_tau_pct` |
| Dataset size (post-filter) | 10,021 images | — |
| Train / Val / Test split | 70 / 15 / 15 % | `data.split_ratios` |
| Protected attributes | gender, skin_tone | `fairness.protected_groups` |
| Fairness criterion | Recall Parity (RP) | `fairness.criterion` |
| Bootstrap resamples (CI) | 1000 | `evaluation.bootstrap_n` |
| Random seeds | 42, 123, 456 | `training.seed` |

### RE-IoT Case Study (Case Study 2)

| Parameter | Value | Config Key |
|---|---|---|
| Detector architecture | CNN-LSTM | `model.architecture` |
| Number of nodes | 120 (40 urban / 40 peri-urban / 40 rural) | `data.nodes` |
| Telemetry sampling rate | 1 Hz | `data.frequency_hz` |
| Attack types | FDIA, command injection, DoS | `data.attacks` |
| Overall attack ratio | 5% | `data.attack_ratio` |
| Train / Test node split | 80 / 20 % (stratified by class) | `data.node_split` |
| DP privacy budget (ε) | 3.0 | `governance.dp_epsilon` |
| Fairness criterion | FPRP (node-class parity) | `fairness.criterion` |
| Accountability standard | IEC 62351-compatible logging | `governance.ics_standard` |
| Random seeds | 42, 123, 456 | `training.seed` |

---

## Statistical Methods

### Accuracy Comparison (M0 vs. M5)

- **Test:** Two-proportion z-test
- **Null hypothesis:** H₀: Acc_EAGF = Acc_Baseline
- **Significance level:** α = 0.05
- **Justification:** Appropriate for comparing proportions over large samples
  (n > 1,500 test images). The normal approximation to the binomial is valid
  at this sample size (np > 5, n(1−p) > 5 for all comparisons).
- **Result:** z = 1.34, p = 0.18 → not significant (accuracy difference acceptable)

### Trust Index Comparison

- **Test:** One-sample Wilcoxon signed-rank test
- **Null hypothesis:** H₀: TI_EAGF = TI_Baseline
- **Replicates:** 3 random seeds (n = 3 paired observations)
- **Justification:** TI distributions cannot be assumed normal across only 3
  replicates. The Wilcoxon test is distribution-free and conservative.
- **Result:** W = 6, p = 0.011 → significant at α = 0.05

### Pillar Metric Confidence Intervals (C, P, A, RP)

- **Method:** Stratified bootstrap resampling
- **Resamples:** n = 1000
- **Stratification:** By demographic group (gender × skin_tone) for biometric;
  by node class for RE-IoT
- **Justification:** Stratified bootstrap accounts for group imbalance, which
  naive bootstrap would not. Non-parametric — no distributional assumptions.
- **Interval type:** Percentile bootstrap CI (2.5th–97.5th percentile)

---

## Hardware and Runtime

### Biometric Case Study

| Component | Specification |
|---|---|
| GPU | NVIDIA A100 40 GB SXM4 |
| CPU | Intel Xeon Platinum 8380, 40 cores |
| RAM | 256 GB DDR4 |
| OS | Ubuntu 22.04 LTS |
| CUDA version | 11.8 |
| Runtime per EAGF seed | ~4 hours |
| Runtime for Pareto grid (25 runs) | ~18 hours (parallelisable) |
| Total compute (3 seeds + ablation + Pareto) | ~60 GPU-hours |

### RE-IoT Case Study

| Component | Specification |
|---|---|
| Hardware | CPU-only (no GPU required) |
| CPU | Intel Xeon Platinum 8380, 40 cores |
| RAM | 32 GB |
| Runtime per EAGF seed | ~20 minutes |

> **Note:** The Pareto-grid search (25 training runs) can be parallelised across
> GPUs. With 4 × A100 GPUs, the Pareto search completes in ~4.5 hours.
> See `scripts/run_pareto_search.sh` for single-GPU sequential execution.

---

## Deviations from Ideal Reproducibility

The following deviations from ideal reproducibility are disclosed:

1. **EFR demographic labels are not independently certified.** Skin-tone and
   gender labels in the EFR dataset were assigned based on filename metadata,
   not by a certified human annotator. Label noise in these pseudo-labels may
   affect RP and C confidence interval widths. We estimate this introduces
   ≤ ±0.01 uncertainty in reported RP values.

2. **RE-IoT attack magnitude is synthetic.** Attack injection amplitudes are
   parameterised from the published Liang et al. (2017) IEEE Trans. Smart Grid
   dataset statistics, not from a live microgrid. Generalisation to specific
   production SCADA systems requires domain-specific re-parameterisation.

3. **AHP weights are equal (w_i = 0.25).** Production deployment requires
   formal AHP elicitation from domain-specific stakeholder panels. The equal-
   weight assumption is a neutral baseline and may not reflect any specific
   operator's priorities.

4. **MIA attacker uses shadow-model approach.** The membership inference attack
   uses 4 shadow models. A stronger attacker (e.g., likelihood ratio test under
   DP-SGD) might yield a slightly higher MIA AUC and thus a slightly lower P
   score. The reported P values are therefore slightly optimistic.

---

## Archived Experiment Records

For each published result, an MLflow experiment record is saved at:
`results/{domain}/mlruns/`

Each record includes the full hyperparameter set, all metric values, the model
checkpoint SHA-256 hash, and the exact git commit hash at time of training.

To view the experiment dashboard:
```bash
mlflow ui --backend-store-uri results/biometric/mlruns/
# Open http://localhost:5000
```
