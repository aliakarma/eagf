# EAGF Regulatory Mapping

This document maps EAGF components to specific regulatory requirements across
five applicable frameworks. It is intended to support compliance officers and
regulatory auditors in assessing EAGF-governed systems.

---

## Summary Table

| Regulation | Requirement | EAGF Component | Evidence (Paper Results) |
|---|---|---|---|
| EU AI Act (High-Risk) | Transparency obligation; human oversight | Clarity (C); α_trace; A | C = 0.88; A = 0.89 |
| GDPR Art. 22 | Meaningful explanation of automated decisions | SHAP top-5 per decision; C metric | C = 0.88 delivers feature-level attribution |
| NIST AI RMF | Govern, Map, Measure, Manage lifecycle | Full EAGF 9-stage lifecycle; TI as KPI | TI = 0.918 (biometric); 0.894 (RE-IoT) |
| NIS2 Directive | Incident reporting; network security | α_comply; IEC 62351 audit log | 100% incident logging in RE-IoT study |
| IEC 62351 | Authenticated command logging; ICS security | α_audit; SHA-256 signed records | α_audit = 1.0 in RE-IoT study |

---

## EU Artificial Intelligence Act (2024)

**Applicability:** AI systems deployed in critical energy infrastructure are
classified as **high-risk AI** under Annex III, Section 2 (AI systems used in
management and operation of critical infrastructure).

| EU AI Act Requirement | Article | EAGF Implementation |
|---|---|---|
| Transparency and provision of information | Art. 13 | Clarity metric (C); SHAP top-5 explanation per decision |
| Human oversight measures | Art. 14 | Stage 9 stakeholder dashboard; TI threshold alerts |
| Accuracy, robustness, and cybersecurity | Art. 15 | Multi-objective training; red-team stress-testing |
| Risk management system | Art. 9 | 9-stage governance lifecycle |
| Data and data governance | Art. 10 | Privacy-preserving data preparation (Stage 2) |
| Technical documentation | Art. 11 | Audit log; model version registry; α_trace |
| Record-keeping | Art. 12 | α_audit: cryptographically signed log entries |
| Post-market monitoring | Art. 72 | Continuous TI monitoring; metric degradation alerts |

**Compliance evidence:** A TI ≥ 0.90 combined with α_comply ≥ 0.85 provides
quantitative evidence supporting EU AI Act conformity assessment for high-risk AI.

---

## General Data Protection Regulation (GDPR, 2016/679)

**Applicability:** Smart-meter energy consumption data at 15-minute granularity
qualifies as personal data under GDPR (can reveal household occupancy, device
usage, lifestyle patterns).

| GDPR Requirement | Article | EAGF Implementation |
|---|---|---|
| Meaningful explanation of automated decisions | Art. 22(3) | SHAP top-5 feature attribution per decision; C metric |
| Data minimisation | Art. 5(1)(c) | Stage 2: anonymisation and data minimisation |
| Privacy by design | Art. 25 | DP-SGD integrated into training (not post-hoc) |
| Data retention limits | Art. 5(1)(e) | α_trace: 90-day audit window; data lineage records |
| Security of processing | Art. 32 | Cryptographic audit logging; MIA stress-testing |

---

## NIST AI Risk Management Framework (NIST AI RMF 1.0, 2023)

The NIST AI RMF organises AI risk management around four functions:
Govern, Map, Measure, and Manage.

| NIST AI RMF Function | Core Activity | EAGF Implementation |
|---|---|---|
| **Govern** | Establish organisational policies for AI risk | Stage 1: governance targets; AHP weight elicitation |
| **Map** | Categorise AI risks in context | Threat taxonomy (Table 3); pillar-to-threat mapping |
| **Measure** | Analyse and assess AI risks | Four pillar metrics; composite TI; ablation study |
| **Manage** | Prioritise and address AI risks | Pareto-guided training; continuous monitoring; Stage 9 reporting |

**Compliance evidence:** The EAGF 9-stage lifecycle directly implements the
Govern → Map → Measure → Manage cycle. TI serves as the unified KPI for the
Measure and Manage functions.

---

## NIS2 Directive (2022/2555)

**Applicability:** Operators of essential services in the energy sector (including
solar-microgrid operators above capacity thresholds) are subject to NIS2 obligations.

| NIS2 Requirement | Article | EAGF Implementation |
|---|---|---|
| Cybersecurity risk management measures | Art. 21 | EAGF multi-objective governance framework |
| Incident reporting (within 24h for significant incidents) | Art. 23 | α_comply: NIS2 incident reporting controls |
| Supply chain security | Art. 21(2)(d) | Model-poisoning detection; data-lineage traceability |
| Use of cryptography | Art. 21(2)(h) | SHA-256 signed audit log entries (α_audit) |
| Network and information system security | Art. 21(2)(a) | DP-SGD; MIA stress-testing; anomaly detection |

---

## IEC 62351 (Industrial Cybersecurity for Power Systems)

**Applicability:** IEC 62351 (Parts 1–14) applies directly to the Modbus/DNP3
communication protocols used in the RE-IoT SCADA architecture.

| IEC 62351 Requirement | Part | EAGF Implementation |
|---|---|---|
| Authentication of users and devices | Part 5, 8 | α_audit: operator identity in every log entry |
| Cryptographic integrity protection | Part 3, 5 | SHA-256 hashing of input features before logging |
| Audit logging of control-plane actions | Part 6 | α_audit: 100% command logging in RE-IoT study |
| Sub-100ms response time for protection | Part 6 | SHAP top-k constraint: mean 22ms per decision |
| Role-based access control for log data | Part 8 | Audit log access restricted to authenticated auditors |

---

## Compliance Score Computation

The α_comply sub-score of the Accountability metric is computed as follows:

1. Instantiate the applicable regulatory checklist (select regulations above
   based on deployment jurisdiction and sector).
2. For each binary control in the checklist, score 1 if satisfied, 0 if not.
3. α_comply = (satisfied controls) / (total controls)

In the RE-IoT case study, the checklist covers 42 binary controls across
EU AI Act, GDPR, NIST CSF PR.DS, NIS2, and IEC 62351. The EAGF model satisfies
37 of 42 controls, yielding α_comply = 0.881.

The five unsatisfied controls require physical infrastructure changes outside
the scope of the AI governance framework (e.g., hardware-level key management
for IEC 62351 Part 8 RBAC), which are flagged for the operator's action plan.
