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
