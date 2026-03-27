"""
src/metrics/privacy.py — EAGF Privacy Metric (P)
Paper: Section 3.4, Equation 7
  P = alpha * exp(-epsilon_eff) + (1 - alpha) * (1 - MIA)
"""
import numpy as np
from typing import Optional

ALPHA = 0.6
IDEAL_MIA_AUC = 0.5


def privacy_raw_score(epsilon_eff: float, mia_auc: float, alpha: float = ALPHA) -> float:
    """Compute un-normalized privacy score from Eq. 7."""
    formal = alpha * float(np.exp(-epsilon_eff))
    empirical = (1.0 - alpha) * (1.0 - float(mia_auc))
    return float(np.clip(formal + empirical, 0.0, 1.0))


def privacy_ideal_score(epsilon_eff: float, alpha: float = ALPHA,
             ideal_mia_auc: float = IDEAL_MIA_AUC) -> float:
  """Compute ideal privacy score used for normalization at fixed epsilon."""
  return privacy_raw_score(epsilon_eff, ideal_mia_auc, alpha)


def privacy_score(epsilon_eff: float, mia_auc: float, alpha: float = ALPHA) -> float:
    """Compute normalized Privacy score P.

    Raw score uses Eq. 7 and is normalized by the ideal score at
    the same epsilon with MIA AUC = 0.5.

    Args:
        epsilon_eff: Effective DP budget consumed (lower = more private).
        mia_auc: AUC of membership inference attacker (0.5=random=strong privacy).
        alpha: Weight for formal DP component.
    Returns: P in [0, 1].
    """
    p_raw = privacy_raw_score(epsilon_eff, mia_auc, alpha)
    p_ideal = privacy_ideal_score(epsilon_eff, alpha)
    if p_ideal <= 0.0:
        return 0.0
    return float(np.clip(p_raw / p_ideal, 0.0, 1.0))


def compute_privacy(epsilon_eff: float = 3.0, mia_auc: float = 0.53,
                    alpha: float = ALPHA) -> dict:
    """Return full privacy metric dictionary."""
    p_raw = privacy_raw_score(epsilon_eff, mia_auc, alpha)
    p_ideal = privacy_ideal_score(epsilon_eff, alpha)
    p_normalized = privacy_score(epsilon_eff, mia_auc, alpha)
    return {
        "privacy": p_normalized,
        "privacy_raw": p_raw,
        "privacy_ideal": p_ideal,
        "privacy_normalized": p_normalized,
        "epsilon_eff": epsilon_eff,
        "mia_auc": mia_auc,
        "formal_component": alpha * np.exp(-epsilon_eff),
        "empirical_component": (1.0 - alpha) * (1.0 - mia_auc),
    }


