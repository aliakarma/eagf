"""
src/metrics/trust_index.py — EAGF Composite Trust Index (TI)
Paper: Section 3.6, Equation 9
  TI = w_C*C~ + w_F*RP~ + w_P*P~ + w_A*A~   (X~ = X_obs/X_ideal)
"""
import numpy as np
from typing import Dict, Optional

IDEAL_VALUES = {"clarity": 1.0, "recall_parity": 1.0, "accountability": 1.0}
DEFAULT_WEIGHTS = {"clarity": 0.25, "fairness": 0.25, "privacy": 0.25, "accountability": 0.25}


def compute_ideal_privacy(epsilon=1e-6, mia=0.50, alpha=0.6) -> float:
    from src.metrics.privacy import privacy_score
    return privacy_score(epsilon, mia, alpha)


def normalise(observed: float, ideal: float) -> float:
    if ideal <= 0:
        return 0.0
    return float(np.clip(observed / ideal, 0.0, 1.0))


def trust_index(clarity: float, fairness: float, privacy: float,
                accountability: float, weights: Optional[Dict] = None,
                ideal_privacy: Optional[float] = None) -> Dict:
    """Compute composite Trust Index TI [Eq. 9]."""
    if weights is None:
        weights = DEFAULT_WEIGHTS.copy()
    total_w = sum(weights.values())
    if not np.isclose(total_w, 1.0, atol=1e-4):
        weights = {k: v / total_w for k, v in weights.items()}

    if ideal_privacy is None:
        ideal_privacy = compute_ideal_privacy()

    c_n = normalise(clarity,        IDEAL_VALUES["clarity"])
    f_n = normalise(fairness,       IDEAL_VALUES["recall_parity"])
    p_n = normalise(privacy,        ideal_privacy)
    a_n = normalise(accountability, IDEAL_VALUES["accountability"])

    ti = (weights["clarity"]        * c_n +
          weights["fairness"]       * f_n +
          weights["privacy"]        * p_n +
          weights["accountability"] * a_n)

    return {
        "ti": float(np.clip(ti, 0.0, 1.0)),
        "components": {"clarity_norm": c_n, "fairness_norm": f_n,
                       "privacy_norm": p_n, "accountability_norm": a_n},
        "weights": weights,
        "raw": {"clarity": clarity, "fairness": fairness,
                "privacy": privacy, "accountability": accountability},
    }
