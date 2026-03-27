"""
src/training/fairness_loss.py
Training-time objective helpers.

- Fairness penalty is a differentiable approximation used in gradient updates.
- Transparency term is a structural surrogate regularizer; final transparency
    reporting uses post-training clarity metrics.
"""
import numpy as np
import torch
import torch.nn.functional as F


def recall_parity_penalty_torch(y_true, y_pred_proba_pos, group_labels,
                                rp_target=0.95):
    """Differentiable recall-parity penalty on a batch.

    Args:
        y_true: Tensor[int], shape (N,)
        y_pred_proba_pos: Tensor[float], shape (N,), probability of positive class.
        group_labels: Tensor[int], shape (N,)
        rp_target: minimum target ratio min(recall_g)/max(recall_g).

    Returns:
        Scalar torch tensor penalty (without lambda multiplier).
    """
    unique_groups = torch.unique(group_labels)
    recalls = []
    y_true_f = y_true.float()
    for g in unique_groups:
        mask = (group_labels == g)
        if mask.sum() == 0:
            continue
        y_g = y_true_f[mask]
        p_g = y_pred_proba_pos[mask]
        # Soft recall proxy: E[p(y=1)] over true positives in group.
        pos_mass = y_g.sum()
        if pos_mass <= 0:
            continue
        recall_soft = (p_g * y_g).sum() / (pos_mass + 1e-12)
        recalls.append(recall_soft)

    if len(recalls) < 2:
        return torch.zeros((), device=y_true.device)

    recalls_t = torch.stack(recalls)
    rp_gen = recalls_t.min() / (recalls_t.max() + 1e-12)
    return F.relu(torch.tensor(rp_target, device=y_true.device) - rp_gen) ** 2


def clarity_penalty_from_outputs(y_pred_proba, target_confidence=0.80):
    """Structural transparency surrogate from model output confidence.

    Transparency is enforced structurally (e.g., pruning/constraints) and is NOT optimized via gradient descent.
    This term enforces a training-time structural constraint proxy and should
    not be interpreted as direct gradient optimization of the final clarity
    metric reported in evaluation.
    """
    max_conf = y_pred_proba.max(dim=1).values.mean()
    return F.relu(torch.tensor(target_confidence, device=y_pred_proba.device) - max_conf) ** 2


def recall_parity_penalty(y_true, y_pred_proba, group_labels,
                           lambda_rp=0.1, rp_target=0.95, pos_label=1):
    """Numpy compatibility wrapper used in analysis code paths."""
    y_true_t = torch.as_tensor(y_true, dtype=torch.long)
    group_t = torch.as_tensor(group_labels)
    if np.ndim(y_pred_proba) == 2:
        proba_pos_t = torch.as_tensor(y_pred_proba[:, int(pos_label)], dtype=torch.float32)
    else:
        proba_pos_t = torch.as_tensor(y_pred_proba, dtype=torch.float32)
    penalty = recall_parity_penalty_torch(y_true_t, proba_pos_t, group_t, rp_target)
    return float(lambda_rp * penalty.detach().cpu().item())


def compute_total_loss(base_loss, y_true, y_pred_proba, group_labels,
                       lambda_rp=0.1, rp_target=0.95):
    """L = L_CE + lambda_RP * [max(0, RP_hat - RP)]^2  [Eq. 11]"""
    penalty = recall_parity_penalty(y_true, y_pred_proba, group_labels,
                                     lambda_rp, rp_target)
    return float(base_loss) + float(penalty)
