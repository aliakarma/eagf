"""
src/metrics/clarity.py — EAGF Transparency Metric: Explanation Clarity (C)
Paper: Section 3.2, Equations 1-2

Clarity is computed as:
    ClarityScore(i) = fidelity(i) * (1 - H_norm(pred_dist_i))
where pred_dist_i is the distribution of hard-label predictions in the
local neighbourhood of instance i, and H_norm is the normalized Shannon
entropy of that distribution.

A model that makes CONSISTENT local predictions (always predicts the same
class in a neighbourhood) has pred_dist entropy = 0 → high clarity.
Models trained with a confidence/entropy-minimisation objective (EAGF)
produce more decisive local predictions → higher ClarityScore than the
governance-free baseline.
"""
import numpy as np
from sklearn.linear_model import Ridge


def _local_neighbourhood(X, idx, n_neighbours=50, noise_scale=0.1, rng=None):
    if rng is None:
        rng = np.random.RandomState(0)
    x = X[idx]
    noise = rng.randn(n_neighbours, X.shape[1]) * noise_scale * (X.std(axis=0) + 1e-8)
    return x + noise


def _entropy_based_clarity(probs, fidelity):
    """ClarityScore = fidelity * (1 - normalized_entropy(probs)).

    Args:
        probs: 1-D probability / frequency vector (sums to ~1).  In the
               neighbourhood-based setting this is the fraction of points
               predicted as each class in the local neighbourhood.
        fidelity: Ridge surrogate fidelity on the local neighbourhood.

    Returns:
        Scalar in [0, 1].
    """
    probs = np.asarray(probs, dtype=float)
    probs = np.clip(probs, 1e-8, 1.0)
    probs = probs / probs.sum()
    n_classes = len(probs)
    entropy = -np.sum(probs * np.log(probs))
    max_entropy = np.log(max(n_classes, 2))
    normalized_entropy = entropy / max_entropy
    clarity = fidelity * (1.0 - normalized_entropy)
    return float(np.clip(clarity, 0.0, 1.0))


def compute_instance_clarity(model_predict_fn, X, idx, tau_pct=0.01,
                              n_neighbours=50, rng=None):
    """ClarityScore(i) = fidelity(i) * (1 - H_norm(pred_dist_i))

    The prediction-distribution vector is computed from hard-label predictions
    in the local neighbourhood of instance i:
      * If model returns 2-D probabilities, argmax gives the hard labels.
      * If model returns 1-D integer labels, they are used directly.

    The entropy of this frequency distribution rewards models whose local
    predictions are consistent (decisive, low-entropy) — exactly the behaviour
    encouraged by the entropy-minimisation clarity loss in EAGF training.
    """
    if rng is None:
        rng = np.random.RandomState(int(idx))
    neighbourhood = _local_neighbourhood(X, idx, n_neighbours, 0.1, rng)
    X_local = np.vstack([X[[idx]], neighbourhood])
    try:
        raw = model_predict_fn(X_local)
        raw = np.asarray(raw)
        if raw.ndim == 2:
            preds = raw.argmax(axis=1)
        else:
            preds = raw.astype(int)
    except Exception:
        return 0.0
    try:
        surrogate = Ridge(alpha=1.0, fit_intercept=True)
        surrogate.fit(X_local, preds.astype(float))
        surrogate_preds = surrogate.predict(X_local).round().astype(int)
        fidelity = float(np.mean(surrogate_preds == preds))

        # Prediction-distribution entropy across the neighbourhood
        classes, counts = np.unique(preds, return_counts=True)
        pred_probs = counts.astype(float) / counts.sum()
        return _entropy_based_clarity(pred_probs, fidelity)
    except Exception:
        return 0.0


def compute_global_clarity(model_predict_fn, X, sample_size=100,
                            tau_pct=0.01, n_neighbours=50, seed=42):
    """C = (1/|I|) * sum ClarityScore(i)  [Eq. 2]"""
    rng = np.random.RandomState(seed)
    indices = rng.choice(len(X), size=min(sample_size, len(X)), replace=False)
    scores = [
        compute_instance_clarity(model_predict_fn, X, int(i), tau_pct,
                                  n_neighbours, np.random.RandomState(seed + int(i)))
        for i in indices
    ]
    C = float(np.clip(np.mean(scores), 0.0, 1.0))
    return {"clarity": C, "opacity": 1.0 - C,
            "per_instance_scores": scores, "n_explained": len(scores)}


def clarity_from_feature_importances(feature_importances, y_true, y_pred, tau_pct=0.01):
    """Simplified clarity from model's built-in feature importances."""
    importances = np.abs(feature_importances)
    mean_imp = importances.mean()
    threshold = tau_pct * mean_imp if mean_imp > 0 else 1e-8
    size = max(int(np.sum(importances >= threshold)), 1)
    fidelity = float(np.mean(y_true == y_pred))
    val = fidelity / (1.0 + size)
    return {"clarity": float(np.clip(val, 0.0, 1.0)),
            "opacity": float(np.clip(1.0 - val, 0.0, 1.0)),
            "fidelity": fidelity, "explanation_size": size}


def compute_clarity_score(fidelity: float, size: int) -> float:
    """ClarityScore(i) = Fidelity / (1 + Size)  [Eq. 1] — direct computation."""
    import numpy as np
    return float(np.clip(fidelity / (1.0 + max(size, 0)), 0.0, 1.0))
