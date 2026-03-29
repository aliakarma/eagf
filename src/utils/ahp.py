"""
src/utils/ahp.py
Analytic Hierarchy Process (AHP) Weight Derivation

Computes Trust Index pillar weights from a pairwise comparison matrix
using the standard AHP procedure (Saaty, 1980).

Paper reference: Section 3.6 (Composite Trust Index, AHP weight derivation)
                 Saaty, T.L. (1980). The Analytic Hierarchy Process.

Default: equal weights (w_i = 0.25) as a neutral regulatory baseline.
For stakeholder-specific weighting, supply a 4×4 pairwise comparison matrix
using Saaty's 1-9 scale, where entry (i,j) represents the relative importance
of pillar i over pillar j.
"""

import numpy as np
from typing import List, Optional


PILLAR_NAMES = ["clarity", "fairness", "privacy", "accountability"]

# Saaty consistency ratio thresholds
CR_THRESHOLD = 0.10  # CR > 0.10 indicates inconsistent judgements

# Random Inconsistency Index for n=4 (Saaty, 1980, Table 3)
RI_4 = 0.90

# --- Dynamic AHP constants ---
# Threshold above which MIA success rate triggers a privacy weight boost.
_MIA_THREAT_THRESHOLD = 0.55
# Multiplicative boost applied when MIA rate exceeds the threshold.
_PRIVACY_BOOST_FACTOR = 1.20   # +20 %
# Threshold above which fairness loss triggers a fairness weight boost.
_FAIRNESS_LOSS_HIGH_THRESHOLD = 0.10
# Additive boost for fairness weight.
_FAIRNESS_BOOST_ADDITIVE = 0.10   # +10 %
# Exponential Moving Average (EMA) smoothing coefficient α ∈ (0, 1].
# Lower values → slower adaptation → less oscillation.
# α = 0.3 provides moderate responsiveness while suppressing noisy updates.
_EMA_ALPHA = 0.30


def equal_weights() -> dict:
    """Return equal AHP weights (w_i = 0.25 for all pillars).

    Returns:
        Dictionary mapping pillar name to weight.
    """
    return {p: 0.25 for p in PILLAR_NAMES}


def ahp_weights(pairwise_matrix: np.ndarray) -> dict:
    """Derive AHP weights from a 4×4 pairwise comparison matrix.

    Standard AHP procedure:
      1. Normalise each column by its sum.
      2. Compute row means as priority weights.
      3. Check consistency ratio (CR = CI / RI).
         CR < 0.10 indicates acceptable consistency.

    Args:
        pairwise_matrix: 4×4 positive reciprocal matrix.
                         Entry (i,j) = importance of pillar i over pillar j.
                         Saaty scale: 1 (equal) to 9 (extreme importance).
                         pairwise_matrix[j,i] = 1 / pairwise_matrix[i,j].

    Returns:
        Dictionary mapping pillar name to normalised weight.

    Raises:
        ValueError: If consistency ratio exceeds CR_THRESHOLD.
    """
    A = np.array(pairwise_matrix, dtype=float)
    n = A.shape[0]

    if A.shape != (n, n):
        raise ValueError(f"Pairwise matrix must be square, got shape {A.shape}.")

    # Step 1: Column-normalised matrix
    col_sums = A.sum(axis=0)
    A_norm = A / col_sums

    # Step 2: Priority vector (row means)
    weights = A_norm.mean(axis=1)

    # Step 3: Consistency check
    lambda_max = float((A @ weights / weights).mean())
    ci = (lambda_max - n) / (n - 1)
    cr = ci / RI_4

    if cr > CR_THRESHOLD:
        raise ValueError(
            f"AHP consistency ratio CR={cr:.3f} exceeds threshold {CR_THRESHOLD}. "
            "Revise pairwise judgements to reduce inconsistency."
        )

    return {PILLAR_NAMES[i]: float(weights[i]) for i in range(n)}


def calculate_dynamic_weights(
    current_privacy_loss: float,
    current_fairness_loss: float,
    mia_attack_success_rate: float,
    previous_weights: Optional[dict] = None,
) -> dict:
    """Compute adaptive AHP pillar weights based on current threat signals.

    Rules (applied sequentially to a copy of the base weights):
      1. If ``mia_attack_success_rate`` > 0.55:
           privacy weight × 1.20  (+20 %)
      2. If ``current_fairness_loss`` > _FAIRNESS_LOSS_HIGH_THRESHOLD:
           fairness weight + 0.10  (+10 pp additive boost)
      3. Weights are re-normalised so that they sum to 1.

    Smoothing:
      The returned weights are an EMA blend of the newly computed weights and
      the ``previous_weights`` (α = 0.30):
          w_out = α * w_new + (1 − α) * w_prev
      This prevents oscillation when threat signals alternate between
      boundary values across epochs.  When ``previous_weights`` is None
      the equal-weight baseline is used as the previous state.

    Args:
        current_privacy_loss:    Scalar privacy loss / MIA AUC estimate for
                                 the current epoch.  Not used directly in the
                                 rules — the signal is ``mia_attack_success_rate``.
        current_fairness_loss:   Scalar fairness penalty from the current epoch
                                 (e.g., recall-parity loss value).
        mia_attack_success_rate: MIA AUC or equivalent attack success rate ∈ [0, 1].
        previous_weights:        Previous epoch's weight dict (for EMA smoothing).
                                 If None, uses equal weights as the prior state.

    Returns:
        Normalised weight dict: {"clarity": w_C, "fairness": w_F,
                                  "privacy": w_P, "accountability": w_A}
        Weights sum to 1.0 and all values are ≥ 0.
    """
    # Start from equal weights as the base.
    w = {k: 0.25 for k in PILLAR_NAMES}

    # Rule 1: MIA threat → boost privacy.
    if mia_attack_success_rate > _MIA_THREAT_THRESHOLD:
        w["privacy"] *= _PRIVACY_BOOST_FACTOR

    # Rule 2: High fairness loss → boost fairness.
    if float(current_fairness_loss) > _FAIRNESS_LOSS_HIGH_THRESHOLD:
        w["fairness"] += _FAIRNESS_BOOST_ADDITIVE

    # Step 3: Normalise so weights sum to 1.
    total = sum(w.values())
    w = {k: v / total for k, v in w.items()}

    # EMA smoothing to prevent oscillation.
    if previous_weights is None:
        previous_weights = equal_weights()

    smoothed = {
        k: _EMA_ALPHA * w[k] + (1.0 - _EMA_ALPHA) * previous_weights.get(k, 0.25)
        for k in PILLAR_NAMES
    }

    # Re-normalise after smoothing (floating-point safety).
    total_s = sum(smoothed.values())
    smoothed = {k: v / total_s for k, v in smoothed.items()}

    return smoothed


def validate_pairwise_matrix(matrix: np.ndarray) -> bool:
    """Validate that a pairwise matrix is positive reciprocal.

    Args:
        matrix: Candidate pairwise comparison matrix.

    Returns:
        True if valid, False otherwise.
    """
    A = np.array(matrix, dtype=float)
    n = A.shape[0]
    for i in range(n):
        for j in range(n):
            if not np.isclose(A[i, j] * A[j, i], 1.0, atol=1e-6):
                return False
            if A[i, j] <= 0:
                return False
    return True
