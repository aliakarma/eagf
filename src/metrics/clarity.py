"""
src/metrics/clarity.py — EAGF Transparency Metric: Explanation Clarity (C)
Paper: Section 3.2, Equations 1-2
"""
import numpy as np
from sklearn.linear_model import Ridge
from typing import Optional, Callable


def _local_neighbourhood(X, idx, n_neighbours=50, noise_scale=0.1, rng=None):
    if rng is None:
        rng = np.random.RandomState(0)
    x = X[idx]
    noise = rng.randn(n_neighbours, X.shape[1]) * noise_scale * (X.std(axis=0) + 1e-8)
    return x + noise


def compute_instance_clarity(model_predict_fn, X, idx, tau_pct=0.01,
                              n_neighbours=50, rng=None):
    """ClarityScore(i) = Fidelity(E(i)) / (1 + Size(E(i)))  [Eq. 1]"""
    if rng is None:
        rng = np.random.RandomState(int(idx))
    neighbourhood = _local_neighbourhood(X, idx, n_neighbours, 0.1, rng)
    X_local = np.vstack([X[[idx]], neighbourhood])
    try:
        preds = model_predict_fn(X_local)
        if preds.ndim > 1:
            preds = preds.argmax(axis=1)
    except Exception:
        return 0.0
    try:
        surrogate = Ridge(alpha=1.0, fit_intercept=True)
        surrogate.fit(X_local, preds.astype(float))
        surrogate_preds = surrogate.predict(X_local).round().astype(int)
        fidelity = float(np.mean(surrogate_preds == preds))
        importances = np.abs(surrogate.coef_)
        mean_imp = importances.mean()
        threshold = tau_pct * mean_imp if mean_imp > 0 else 1e-8
        size = max(int(np.sum(importances >= threshold)), 1)
        return float(np.clip(fidelity / (1.0 + size), 0.0, 1.0))
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
