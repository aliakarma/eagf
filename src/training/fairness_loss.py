"""
src/training/fairness_loss.py
Training-time objective helpers.

- Fairness penalty is a differentiable approximation used in gradient updates.
- Transparency term is a structural surrogate regularizer; final transparency
    reporting uses post-training clarity metrics.
"""
import numpy as np
import torch


def recall_parity_penalty_torch(y_true, y_pred_proba_pos, group_labels,
                                rp_target=0.95):
    """Differentiable recall-parity penalty on a batch.

    Args:
        y_true: Tensor[int], shape (N,)
        y_pred_proba_pos: Tensor[float], shape (N,), probability of positive class.
        group_labels: Tensor[int], shape (N,)
        rp_target: target recall-parity ratio min(recall_g)/max(recall_g).

    Returns:
        Scalar torch tensor penalty = |target_rp - current_rp|.
        Gradient is always non-zero whenever current_rp != rp_target.
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
    # Compute soft recall-parity ratio: min_recall / max_recall.
    # Use absolute deviation from target so the gradient is always non-zero
    # whenever the current parity differs from the target.
    current_rp = recalls_t.min() / (recalls_t.max() + 1e-8)
    target_t = torch.tensor(float(rp_target), device=y_true.device, dtype=current_rp.dtype)
    fairness_loss = torch.abs(target_t - current_rp)
    return fairness_loss


def clarity_penalty_from_outputs(y_pred_proba, target_confidence=0.80):
    """Transparency surrogate used during training.

    Clarity is evaluated post-hoc and not directly optimized to avoid
    metric gaming.

    The entropy-based penalty and confidence-floor penalty have been removed
    because they push softmax outputs toward extreme (overconfident) values,
    which artificially inflates model confidence and distorts calibration.
    Such penalties constitute metric gaming: they improve the calibration
    proxy used during training at the expense of true model calibration, ECE,
    and Brier Score measured at evaluation time.

    The only structural regularization that promotes explanation clarity is
    L1 weight sparsity on the input projection, which is applied directly in
    the training loop (see eagf_trainer.py). Sparse input weights reduce the
    effective explanation size, which improves ClarityScore = Fidelity /
    (1 + Size) without manipulating model confidence.

    Args:
        y_pred_proba: Tensor of predicted probabilities, shape (N, C).
        target_confidence: Unused; retained for API backward-compatibility.

    Returns:
        Scalar zero tensor (no gradient through this term).
    """
    _ = target_confidence  # Retained for API compatibility.
    # Clarity is evaluated post-hoc and not directly optimized to avoid
    # metric gaming — return zero so no confidence-inflating gradient flows.
    return torch.zeros((), device=y_pred_proba.device)


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
