"""
src/evaluation/calibration.py — Calibration Evaluation Metrics

Implements:
- Expected Calibration Error (ECE)
- Brier Score

These metrics assess how well a model's predicted probabilities reflect
actual outcome frequencies.  A perfectly calibrated model produces ECE = 0.

References:
  Guo et al., "On Calibration of Modern Neural Networks", ICML 2017.
  Brier, G.W., "Verification of Forecasts Expressed in Terms of Probability",
      Monthly Weather Review, 1950.
"""

from __future__ import annotations

import numpy as np


def expected_calibration_error(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 10,
    pos_label: int = 1,
) -> float:
    """Compute the Expected Calibration Error (ECE).

    ECE partitions predictions into *n_bins* equal-width confidence buckets
    and computes the weighted mean absolute difference between the mean
    predicted confidence and the empirical accuracy within each bucket.

    Args:
        y_true: Integer array of true class labels, shape (N,).
        y_prob: 1-D array of predicted probabilities for ``pos_label``,
            shape (N,).  For multi-class models pass the probability of the
            predicted class (i.e. ``proba.max(axis=1)``).
        n_bins: Number of equal-width bins in [0, 1].  Default: 10.
        pos_label: The positive class label.  Default: 1.

    Returns:
        ECE scalar in [0, 1].  Lower is better.
    """
    y_true = np.asarray(y_true, dtype=int)
    y_prob = np.asarray(y_prob, dtype=float)
    if y_prob.ndim != 1:
        raise ValueError("y_prob must be 1-D; pass the probability of the positive class.")

    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = len(y_true)
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (y_prob >= lo) & (y_prob < hi)
        if hi == 1.0:
            mask = mask | (y_prob == 1.0)
        cnt = mask.sum()
        if cnt == 0:
            continue
        mean_conf = float(y_prob[mask].mean())
        mean_acc = float((y_true[mask] == pos_label).mean())
        ece += (cnt / n) * abs(mean_acc - mean_conf)
    return float(ece)


def brier_score(y_true: np.ndarray, y_prob: np.ndarray, pos_label: int = 1) -> float:
    """Compute the Brier Score.

    The Brier Score is the mean squared error between the predicted
    probability for the positive class and the binary outcome indicator.

    Args:
        y_true: Integer array of true class labels, shape (N,).
        y_prob: 1-D array of predicted probabilities for ``pos_label``,
            shape (N,).
        pos_label: The positive class label.  Default: 1.

    Returns:
        Brier Score in [0, 1].  Lower is better; 0 is perfect.
    """
    y_true = np.asarray(y_true, dtype=int)
    y_prob = np.asarray(y_prob, dtype=float)
    if y_prob.ndim != 1:
        raise ValueError("y_prob must be 1-D; pass the probability of the positive class.")

    outcomes = (y_true == pos_label).astype(float)
    return float(np.mean((y_prob - outcomes) ** 2))


def compute_calibration_metrics(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    n_bins: int = 10,
) -> dict:
    """Compute ECE and Brier Score from a probability matrix.

    Args:
        y_true: Integer array of true class labels, shape (N,).
        y_proba: Probability matrix, shape (N, C), where C is the number of
            classes.  For binary classification C=2.
        n_bins: Number of bins for ECE.

    Returns:
        Dictionary with keys:
            - ``ece``: Expected Calibration Error (float).
            - ``brier_score``: Brier Score (float).
            - ``n_samples``: Number of samples evaluated.
            - ``n_bins``: Number of bins used for ECE.
    """
    y_true = np.asarray(y_true, dtype=int)
    y_proba = np.asarray(y_proba, dtype=float)

    if y_proba.ndim == 1:
        # Treat as probability of class 1 directly.
        prob_pos = y_proba
    elif y_proba.shape[1] == 2:
        prob_pos = y_proba[:, 1]
    else:
        # Multi-class ECE (confidence-accuracy ECE as in Guo et al. 2017):
        # Use the maximum predicted probability (confidence in the predicted
        # class) as the confidence estimate.  The correctness indicator
        # (1 if prediction == true class, 0 otherwise) serves as the
        # corresponding accuracy within each confidence bin.  This formulation
        # measures how well the model's top-class confidence reflects actual
        # accuracy — the most widely reported multi-class calibration metric.
        # Multi-class Brier Score: mean sum-of-squared-errors across all classes
        # (reduces to binary Brier for C=2).
        predicted_class = y_proba.argmax(axis=1)
        prob_pos = y_proba[np.arange(len(y_true)), predicted_class]
        outcomes = (np.arange(y_proba.shape[1]) == y_true[:, None]).astype(float)
        bs = float(np.mean(np.sum((y_proba - outcomes) ** 2, axis=1)))
        ece_val = expected_calibration_error(
            (predicted_class == y_true).astype(int),
            prob_pos,
            n_bins=n_bins,
            pos_label=1,
        )
        return {
            "ece": ece_val,
            "brier_score": bs,
            "n_samples": int(len(y_true)),
            "n_bins": n_bins,
        }

    ece_val = expected_calibration_error(y_true, prob_pos, n_bins=n_bins)
    bs_val = brier_score(y_true, prob_pos)
    return {
        "ece": ece_val,
        "brier_score": bs_val,
        "n_samples": int(len(y_true)),
        "n_bins": n_bins,
    }
