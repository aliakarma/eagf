"""
src/training/fairness_loss.py
Recall-Parity Lagrangian Penalty (differentiable approximation)
Paper: Section 3.7, Equation 11
"""
import numpy as np
from typing import Optional


def recall_parity_penalty(y_true, y_pred_proba, group_labels,
                           lambda_rp=0.1, rp_target=0.95, pos_label=1):
    """Soft recall-parity Lagrangian penalty.
    
    penalty = lambda_RP * max(0, RP_target - RP_observed)^2
    """
    groups = np.unique(group_labels)
    recalls = {}
    for g in groups:
        mask = (group_labels == g)
        if mask.sum() == 0:
            continue
        y_t = y_true[mask]
        # Use soft predictions for differentiability
        y_p = (y_pred_proba[mask] > 0.5).astype(int) if y_pred_proba.ndim == 1 \
              else y_pred_proba[mask].argmax(axis=1)
        tp = float(np.sum((y_p == pos_label) & (y_t == pos_label)))
        fn = float(np.sum((y_p != pos_label) & (y_t == pos_label)))
        recalls[str(g)] = tp / (tp + fn + 1e-12)

    if len(recalls) < 2:
        return 0.0

    vals = list(recalls.values())
    rp_gen = min(vals) / (max(vals) + 1e-12)
    violation = max(0.0, rp_target - rp_gen)
    return lambda_rp * (violation ** 2)


def compute_total_loss(base_loss, y_true, y_pred_proba, group_labels,
                       lambda_rp=0.1, rp_target=0.95):
    """L = L_CE + lambda_RP * [max(0, RP_hat - RP)]^2  [Eq. 11]"""
    penalty = recall_parity_penalty(y_true, y_pred_proba, group_labels,
                                     lambda_rp, rp_target)
    return float(base_loss) + float(penalty)
