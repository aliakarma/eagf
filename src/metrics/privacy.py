"""
src/metrics/privacy.py — EAGF Privacy Metric (P)
Paper: Section 3.4, Equation 7
  P = alpha * exp(-epsilon_eff) + (1 - alpha) * (1 - MIA)
"""
import numpy as np
from typing import Optional

ALPHA = 0.6


def privacy_score(epsilon_eff: float, mia_auc: float, alpha: float = ALPHA) -> float:
    """Compute Privacy score P (Eq. 7).
    Args:
        epsilon_eff: Effective DP budget consumed (lower = more private).
        mia_auc: AUC of membership inference attacker (0.5=random=strong privacy).
        alpha: Weight for formal DP component.
    Returns: P in [0, 1].
    """
    formal = alpha * float(np.exp(-epsilon_eff))
    empirical = (1.0 - alpha) * (1.0 - float(mia_auc))
    return float(np.clip(formal + empirical, 0.0, 1.0))


def compute_privacy(epsilon_eff: float = 3.0, mia_auc: float = 0.52,
                    alpha: float = ALPHA) -> dict:
    """Return full privacy metric dictionary."""
    p = privacy_score(epsilon_eff, mia_auc, alpha)
    return {"privacy": p, "epsilon_eff": epsilon_eff,
            "mia_auc": mia_auc, "formal_component": alpha * np.exp(-epsilon_eff),
            "empirical_component": (1.0 - alpha) * (1.0 - mia_auc)}


def simulate_dp_training(base_accuracy: float, epsilon: float) -> float:
    """Estimate accuracy after DP-SGD noise injection.
    Accuracy degradation ~ 1-4% at epsilon=3 (Abadi et al. 2016).
    """
    degradation = 0.025 * (3.0 / max(epsilon, 0.1)) ** 0.5
    return float(np.clip(base_accuracy - degradation, 0.0, 1.0))
