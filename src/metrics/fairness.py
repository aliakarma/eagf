"""
src/metrics/fairness.py — EAGF Fairness Metrics: RP and FPRP
Paper: Section 3.3, Equations 3-6
"""
import numpy as np
from typing import Dict, Optional


def _recall(y_true, y_pred, pos_label=1):
    tp = np.sum((y_pred == pos_label) & (y_true == pos_label))
    fn = np.sum((y_pred != pos_label) & (y_true == pos_label))
    return tp / (tp + fn + 1e-12)


def _fpr(y_true, y_pred, pos_label=1):
    fp = np.sum((y_pred == pos_label) & (y_true != pos_label))
    tn = np.sum((y_pred != pos_label) & (y_true != pos_label))
    return fp / (fp + tn + 1e-12)


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
    """FPRP_{A/B} = FPR_A / FPR_B  [Eq. 6]"""
    groups = np.unique(group_labels)
    fprs = {}
    for g in groups:
        mask = (group_labels == g)
        if mask.sum() == 0:
            fprs[str(g)] = 0.0
        else:
            fprs[str(g)] = _fpr(y_true[mask], y_pred[mask], pos_label)

    ref_key = str(reference_group)
    ref_fpr = fprs.get(ref_key, 1e-12)
    fprp_pairs = {}
    for g, f in fprs.items():
        if g != ref_key:
            fprp_pairs[f"{g}/{ref_key}"] = f / (ref_fpr + 1e-12)

    vals = list(fprs.values())
    # Generalised FPRP: min/max (closer to 1 = more equitable)
    fprp_gen = min(vals) / (max(vals) + 1e-12) if max(vals) > 0 else 1.0

    return {"fprp": float(np.clip(fprp_gen, 0.0, 1.0)),
            "fprp_pairs": fprp_pairs, "per_group_fpr": fprs}


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
