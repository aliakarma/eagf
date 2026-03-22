# EAGF Metric Definitions

Formal definitions of all four governance pillar metrics and the composite
Trust Index, as presented in the paper (Sections 3.2–3.6).

---

## 1. Transparency: Explanation Clarity (C)

**Definition (per-instance):**

```
ClarityScore(i) = Fidelity(E(i)) / (1 + Size(E(i)))
```

- `Fidelity(E(i)) ∈ [0,1]`: predictive accuracy of a linear surrogate trained
  on the top-k SHAP features for instance i
- `Size(E(i))`: count of features with |SHAP value| ≥ τ · φ̄
  - τ = 0.01 (1% of mean absolute SHAP value φ̄ across the validation set)

**Global clarity:**

```
C = (1/|I|) Σ_{i∈I} ClarityScore(i),   C ∈ [0,1]
```

**Opacity complement:** `O = 1 − C`

**Implementation:** `src/metrics/clarity.py`

---

## 2. Fairness: Recall Parity (RP) and FPRP

### Recall Parity (primary criterion — biometric case study)

```
RP_{A/B} = Recall_A / Recall_B
         = [TP_A / (TP_A + FN_A)] / [TP_B / (TP_B + FN_B)]
```

- Perfect parity: RP = 1
- Disadvantage: RP < 1

**Multi-group generalisation:**

```
RP_gen = min_{g∈G} Recall_g / max_{g∈G} Recall_g
```

### False Positive Rate Parity (primary criterion — RE-IoT case study)

```
FPRP_{A/B} = FPR_A / FPR_B
           = [FP_A / (FP_A + TN_A)] / [FP_B / (FP_B + TN_B)]
```

- Perfect parity: FPRP = 1

### Criterion Selection Rule

| Deployment Context | Primary Error | Chosen Criterion |
|---|---|---|
| Biometric access control | Missed recognition (FN) | RP (Recall Parity) |
| RE-IoT anomaly detection | False alarm (FP) | FPRP |

**Implementation:** `src/metrics/fairness.py`

---

## 3. Privacy (P)

```
P = α · exp(−ε_eff) + (1−α) · (1 − MIA)
```

- `ε_eff`: effective DP budget consumed (Rényi moment accounting, Opacus)
- `MIA ∈ [0,1]`: AUC of the best shadow-model membership inference attacker
  evaluated on the test set. MIA ≈ 0.50 → near-random → strong privacy.
- `α = 0.6`: weight balancing formal (DP) and empirical (MIA) components

**Implementation:** `src/metrics/privacy.py`

---

## 4. Accountability (A)

```
A = (α_audit + α_trace + α_comply) / 3
```

### Sub-components

| Sub-score | Definition | Range |
|---|---|---|
| `α_audit` | Fraction of decisions with a cryptographically signed log entry (model version, input SHA-256 hash, output label, timestamp, operator ID) | [0,1] |
| `α_trace` | Fraction of decisions for which full data lineage (dataset version, preprocessing spec, hyperparameter record) is recoverable within 90 days | [0,1] |
| `α_comply` | Normalised regulatory checklist score. Controls: EU AI Act high-risk requirements; GDPR Art. 22; NIST CSF PR.DS; NIS2 incident reporting; IEC 62351 command logging | [0,1] |

**Design principle:** Accountability is architecturally independent of model
performance. A perfect-accuracy model with no audit logging scores A = 0.

**Implementation:** `src/metrics/accountability.py`, `src/evaluation/audit_logger.py`

---

## 5. Composite Trust Index (TI)

```
TI = w_C · C̃ + w_F · R̃P + w_P · P̃ + w_A · Ã
```

where `X̃ = X_obs / X_ideal` normalises each component to [0,1].

### Ideal Values

| Metric | Ideal Value |
|---|---|
| C | 1.0 |
| RP | 1.0 |
| P | corresponds to ε ≤ 3 and MIA = 0.50 |
| A | 1.0 |

### Weights

| Weight Method | Values | Notes |
|---|---|---|
| Equal (this paper) | w_i = 0.25 for all i | Neutral regulatory baseline |
| AHP (stakeholder-specific) | Derived from 9-point Saaty pairwise comparison matrix | Requires expert elicitation |

**Equal-weight TI reduces to:**

```
TI = (C̃ + R̃P + P̃ + Ã) / 4
```

### Engineering Proxy Caveat

TI is an *engineering proxy* for perceived trustworthiness computed from
objectively measurable system properties. The relationship between TI and
subjective stakeholder trust has not been validated through direct user studies
in a cybersecurity context. Direct user-perception experiments are required
to validate TI as a trust surrogate. This is identified as the highest-priority
future work item.

**Implementation:** `src/metrics/trust_index.py`, `src/utils/ahp.py`
