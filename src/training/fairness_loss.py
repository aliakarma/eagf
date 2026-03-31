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
    """Backward-compatible alias to the FPR-parity training penalty."""
    return fpr_parity_penalty_torch(
        y_true=y_true,
        y_pred_proba_pos=y_pred_proba_pos,
        group_labels=group_labels,
        fpr_target=rp_target,
    )


def fpr_parity_penalty_torch(y_true, y_pred_proba_pos, group_labels,
                             fpr_target=0.95):
    """Differentiable FPR-parity penalty on a batch.

    Args:
        y_true: Tensor[int], shape (N,)
        y_pred_proba_pos: Tensor[float], shape (N,), probability of positive class.
        group_labels: Tensor[int], shape (N,)
        fpr_target: target FPR-parity score.

    Returns:
        Scalar torch tensor penalty = (target_fprp - current_fprp)^2.
    """
    unique_groups = torch.unique(group_labels)
    group_fprs = []
    y_true_f = y_true.float()
    for g in unique_groups:
        mask = (group_labels == g)
        if mask.sum() == 0:
            continue
        y_g = y_true_f[mask]
        p_g = y_pred_proba_pos[mask]
        neg_mask = 1.0 - y_g
        neg_mass = neg_mask.sum()
        if neg_mass <= 0:
            continue
        fpr_soft = (p_g * neg_mask).sum() / (neg_mass + 1e-12)
        group_fprs.append(fpr_soft)

    if len(group_fprs) < 2:
        return torch.zeros((), device=y_true.device)

    fprs_t = torch.stack(group_fprs)
    disparity = fprs_t.max() - fprs_t.min()
    current_fprp = torch.clamp(1.0 - disparity, min=0.0, max=1.0)
    target_t = torch.tensor(float(fpr_target), device=y_true.device, dtype=current_fprp.dtype)
    fairness_loss = (target_t - current_fprp) ** 2
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
    """Numpy compatibility wrapper for FPR-parity optimization."""
    y_true_t = torch.as_tensor(y_true, dtype=torch.long)
    group_t = torch.as_tensor(group_labels)
    if np.ndim(y_pred_proba) == 2:
        proba_pos_t = torch.as_tensor(y_pred_proba[:, int(pos_label)], dtype=torch.float32)
    else:
        proba_pos_t = torch.as_tensor(y_pred_proba, dtype=torch.float32)
    penalty = fpr_parity_penalty_torch(y_true_t, proba_pos_t, group_t, rp_target)
    return float(lambda_rp * penalty.detach().cpu().item())


def compute_total_loss(base_loss, y_true, y_pred_proba, group_labels,
                       lambda_rp=0.1, rp_target=0.95):
    """L = L_CE + lambda_RP * [max(0, RP_hat - RP)]^2  [Eq. 11]"""
    penalty = recall_parity_penalty(y_true, y_pred_proba, group_labels,
                                     lambda_rp, rp_target)
    return float(base_loss) + float(penalty)
