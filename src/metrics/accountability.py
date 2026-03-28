"""
src/metrics/accountability.py — EAGF Accountability Metric (A)
Paper: Section 3.5, Equation 8
  A = (alpha_audit + alpha_trace + alpha_comply) / 3
"""
import hashlib, json, os, time
import numpy as np
from typing import Optional, Dict
import yaml


def audit_completeness(audit_log_path: str, total_decisions: int) -> float:
    """alpha_audit: fraction of decisions with a signed log entry."""
    if not os.path.exists(audit_log_path):
        return 0.0
    logged = sum(1 for _ in open(audit_log_path))
    return float(min(logged / max(total_decisions, 1), 1.0))


def traceability_score(decisions_with_lineage: int, total_decisions: int) -> float:
    """alpha_trace: data lineage recoverability fraction."""
    if total_decisions == 0:
        return 0.0
    return float(min(decisions_with_lineage / total_decisions, 1.0))


def compliance_score(checklist_path: str,
                     metric_overrides: Optional[Dict[str, bool]] = None,
                     alpha_audit: float = 0.0) -> float:
    """alpha_comply: normalised regulatory checklist score.

    Dynamic overrides are applied for data-driven controls so that
    accountability is earned from real metrics, not static YAML values.

    Args:
        checklist_path: Path to compliance checklist YAML/JSON.
        metric_overrides: Dict mapping control ID to bool, computed from real
            metrics by the caller (e.g., {"mia_stress_test": mia_auc <= 0.60,
            "fairness_monitoring": recall_parity >= 0.95}).
        alpha_audit: Observed audit-log coverage fraction [0, 1] that must be
            computed by the caller from the actual audit log before calling this
            function. Used to override the ``audit_log_completeness`` control.
            Defaults to 0.0 (no audit coverage observed).
    """
    if not os.path.exists(checklist_path):
        return 0.0
    with open(checklist_path) as f:
        if checklist_path.lower().endswith((".yaml", ".yml")):
            checklist = yaml.safe_load(f)
        else:
            checklist = json.load(f)
    controls = checklist.get("controls", [])
    if not controls:
        return 0.0

    # Build effective overrides: audit completeness is always computed dynamically.
    effective_overrides = {"audit_log_completeness": alpha_audit >= 0.95}
    if metric_overrides:
        effective_overrides.update(metric_overrides)

    satisfied = 0
    for c in controls:
        cid = c.get("id", "")
        if cid in effective_overrides:
            satisfied += int(bool(effective_overrides[cid]))
        else:
            satisfied += int(bool(c.get("satisfied", False)))
    return float(satisfied / len(controls))


def accountability_score(alpha_audit: float, alpha_trace: float,
                          alpha_comply: float) -> float:
    """A = (alpha_audit + alpha_trace + alpha_comply) / 3  [Eq. 8]"""
    return float(np.clip((alpha_audit + alpha_trace + alpha_comply) / 3.0, 0.0, 1.0))


def compute_accountability(audit_log_path: str, total_decisions: int,
                            lineage_fraction: Optional[float] = None,
                            checklist_path: Optional[str] = None,
                            model_has_governance: bool = True,
                            metric_overrides: Optional[Dict[str, bool]] = None) -> dict:
    """Full accountability computation.

    Args:
        audit_log_path: Path to JSONL audit log.
        total_decisions: Total decisions made.
        lineage_fraction: Fraction of decisions with data lineage [0,1].
        checklist_path: Path to compliance checklist JSON/YAML.
        model_has_governance: If True, model participates in governance framework.
        metric_overrides: Dict of {control_id: bool} for data-driven overrides.
            Typical keys: "mia_stress_test", "fairness_monitoring".
    """
    # Always measure audit completeness from the actual log file.
    a_audit = audit_completeness(audit_log_path, total_decisions)

    # If lineage is not explicitly provided, use observed audit coverage as proxy.
    if lineage_fraction is None:
        lineage_fraction = a_audit
    a_trace = float(np.clip(lineage_fraction, 0.0, 1.0))

    if checklist_path:
        a_comply = compliance_score(
            checklist_path,
            metric_overrides=metric_overrides,
            alpha_audit=a_audit,
        )
    else:
        a_comply = 0.0

    A = accountability_score(a_audit, a_trace, a_comply)
    return {"accountability": A, "alpha_audit": a_audit,
            "alpha_trace": a_trace, "alpha_comply": a_comply}


def hash_input(data) -> str:
    """SHA-256 hash of input data for audit logging."""
    if hasattr(data, 'tobytes'):
        raw = data.tobytes()
    else:
        raw = str(data).encode()
    return hashlib.sha256(raw).hexdigest()
