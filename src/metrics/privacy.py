"""
src/metrics/privacy.py — EAGF Privacy Metric (P)
Paper: Section 3.4, Equation 7
  P = alpha * exp(-epsilon_eff) + (1 - alpha) * (1 - MIA)

Normalization:
  Scores are normalized against the global maximum achievable at eps->0, mia=0.5:
    P_max = alpha * 1 + (1-alpha) * 0.5 = alpha + (1-alpha)*0.5

  When epsilon_eff == inf (no DP), the formal DP component is zero and privacy
  depends only on MIA resistance, which is always lower than a model with real DP.

  MIA AUC is floored at 0.5 (random attacker) since sub-random performance does
  not indicate additional privacy protection.
"""
import numpy as np
from typing import Optional

ALPHA = 0.6
IDEAL_MIA_AUC = 0.5
# Global maximum: achieved at eps->0 (perfect DP) and mia_auc=0.5 (random attacker)
_P_MAX = ALPHA + (1.0 - ALPHA) * 0.5  # = 0.8


def privacy_raw_score(epsilon_eff: float, mia_auc: float, alpha: float = ALPHA) -> float:
    """Compute raw privacy score from Eq. 7.

    MIA AUC is floored at 0.5 (random attacker level). When epsilon_eff is
    infinite (no DP), the formal DP component is excluded.
    """
    mia_auc = max(float(mia_auc), 0.5)  # floor at random attacker level
    empirical = (1.0 - alpha) * (1.0 - mia_auc)
    if not np.isfinite(epsilon_eff):
        # No DP: privacy depends only on MIA resistance
        return float(np.clip(empirical, 0.0, 1.0))
    formal = alpha * float(np.exp(-epsilon_eff))
    return float(np.clip(formal + empirical, 0.0, 1.0))


def privacy_ideal_score(epsilon_eff: float, alpha: float = ALPHA,
             ideal_mia_auc: float = IDEAL_MIA_AUC) -> float:
    """Global-maximum normalization constant.

    Returns P_max = alpha + (1-alpha)*0.5 regardless of epsilon, so that
    scores from different variants (different epsilons) are comparable.
    """
    p_max = alpha + (1.0 - alpha) * ideal_mia_auc
    return float(p_max)


def privacy_score(epsilon_eff: float, mia_auc: float, alpha: float = ALPHA) -> float:
    """Compute normalized Privacy score P in [0, 1].

    Normalizes against global maximum (eps->0, mia=0.5) so that:
    - Baseline (no DP, mia~0.5): P ~ (1-alpha)*0.5 / P_max  (lower bound)
    - DP variant (eps=3, mia=0.5): P includes formal DP term (higher)

    This ensures baseline privacy < EAGF privacy when DP is active.

    Args:
        epsilon_eff: Effective DP budget consumed (lower = more private; inf = no DP).
        mia_auc: AUC of membership inference attacker (0.5=random=best privacy).
        alpha: Weight for formal DP component (default 0.6).
    Returns: P in [0, 1].
    """
    p_raw = privacy_raw_score(epsilon_eff, mia_auc, alpha)
    p_max = privacy_ideal_score(epsilon_eff, alpha)
    if p_max <= 0.0:
        return 0.0
    return float(np.clip(p_raw / p_max, 0.0, 1.0))


def compute_privacy(epsilon_eff: float = 3.0, mia_auc: float = 0.53,
                    alpha: float = ALPHA) -> dict:
    """Return full privacy metric dictionary."""
    mia_auc_floored = max(float(mia_auc), 0.5)
    p_raw = privacy_raw_score(epsilon_eff, mia_auc, alpha)
    p_max = privacy_ideal_score(epsilon_eff, alpha)
    p_normalized = privacy_score(epsilon_eff, mia_auc, alpha)
    return {
        "privacy": p_normalized,
        "privacy_raw": p_raw,
        "privacy_ideal": p_max,
        "privacy_normalized": p_normalized,
        "epsilon_eff": epsilon_eff,
        "mia_auc": mia_auc,
        "formal_component": alpha * (float(np.exp(-epsilon_eff)) if np.isfinite(epsilon_eff) else 0.0),
        "empirical_component": (1.0 - alpha) * (1.0 - mia_auc_floored),
    }


