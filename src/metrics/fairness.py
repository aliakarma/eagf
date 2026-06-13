"""
src/metrics/fairness.py — EAGF Fairness Metrics: RP and FPRP
Paper: Section 3.3, Equations 3-6
"""
import numpy as np
from typing import Dict, Optional


def _recall(y_true, y_pred, pos_label=1):
    tp = np.sum((y_pred == pos_label) & (y_true == pos_label))
    fn = np.sum((y_pred != pos_label) & (y_true == pos_label))
    denom = tp + fn
    if denom <= 0:
        return np.nan
    return tp / denom


def _fpr(y_true, y_pred, pos_label=1):
    fp = np.sum((y_pred == pos_label) & (y_true != pos_label))
    tn = np.sum((y_pred != pos_label) & (y_true != pos_label))
    denom = fp + tn
    if denom <= 0:
        return np.nan
    return fp / denom


def recall_parity(y_true, y_pred, group_labels, reference_group,
                  pos_label=1) -> Dict:
    """RP_{A/B} = Recall_A / Recall_B  [Eq. 3]"""
    groups = np.unique(group_labels)
    recalls = {}
    for g in groups:
        mask = (group_labels == g)
        if mask.sum() == 0:
            recalls[str(g)] = 0.0
        else:
            recalls[str(g)] = _recall(y_true[mask], y_pred[mask], pos_label)

    recalls = {k: float(v) for k, v in recalls.items() if not np.isnan(v)}
    if len(recalls) < 2:
        return {
            "recall_parity": 1.0,
            "rp_gen": 1.0,
            "rp_pairs": {},
            "per_group_recall": recalls,
        }

    ref_key = str(reference_group)
    ref_recall = recalls.get(ref_key, 1e-12)
    rp_pairs = {}
    for g, r in recalls.items():
        if g != ref_key:
            rp_pairs[f"{g}/{ref_key}"] = r / (ref_recall + 1e-12)

    # Generalised RP: min/max across all groups  [Eq. 4]
    vals = list(recalls.values())
    rp_gen = min(vals) / (max(vals) + 1e-12) if max(vals) > 0 else 1.0

    return {"recall_parity": float(rp_gen), "rp_gen": float(rp_gen),
            "rp_pairs": rp_pairs, "per_group_recall": recalls}


def false_positive_rate_parity(y_true, y_pred, group_labels,
                                reference_group, pos_label=1) -> Dict:
    """FPR parity score: 1 - (max group FPR - min group FPR)."""
    groups = np.unique(group_labels)
    fprs = {}
    for g in groups:
        mask = (group_labels == g)
        if mask.sum() == 0:
            fprs[str(g)] = 0.0
        else:
            fprs[str(g)] = _fpr(y_true[mask], y_pred[mask], pos_label)

    fprs = {k: float(v) for k, v in fprs.items() if not np.isnan(v)}

    ref_key = str(reference_group)
    ref_fpr = fprs.get(ref_key, 1e-12)
    fprp_pairs = {}
    for g, f in fprs.items():
        if g != ref_key:
            fprp_pairs[f"{g}/{ref_key}"] = f / (ref_fpr + 1e-12)

    vals = list(fprs.values())
    if len(vals) < 2:
        disparity = 0.0
        fprp_score = 1.0
    else:
        disparity = max(vals) - min(vals)
        fprp_score = float(np.clip(min(vals) / (max(vals) + 1e-12), 0.0, 1.0))

    return {
        "fpr_parity": fprp_score,
        "fprp": fprp_score,
        "fpr_disparity": float(disparity),
        "fprp_pairs": fprp_pairs,
        "per_group_fpr": fprs,
    }


def select_criterion(deployment_context: str) -> str:
    mapping = {"biometric": "recall_parity", "reiot": "fprp"}
    if deployment_context not in mapping:
        raise ValueError(f"Unknown context: {deployment_context}. "
                         f"Use: {list(mapping.keys())}")
    return mapping[deployment_context]


def compute_fairness(y_true, y_pred, group_labels, reference_group,
                     criterion: str) -> Dict:
    if criterion == "recall_parity":
        return recall_parity(y_true, y_pred, group_labels, reference_group)
    elif criterion == "fprp":
        return false_positive_rate_parity(y_true, y_pred, group_labels, reference_group)
    else:
        raise ValueError(f"Unknown criterion: {criterion}")
