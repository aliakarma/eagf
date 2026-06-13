"""
src/metrics/clarity.py — EAGF Transparency Metric: Explanation Clarity (C)
Paper: Section 3.2, Equations 6-7

Clarity is computed as:
    ClarityScore(i) = Fidelity(E(i)) / (1 + max(0, Size(E(i)) - k))

where:
    Fidelity = predictive accuracy of a linear surrogate trained on top-k
               SHAP features in a local neighbourhood
    Size     = count of features with |phi_j(i)| >= eta * phi_bar
               (eta = 0.01, phi_bar = mean absolute SHAP value)
    k        = 5 (human interpretability budget)

Global clarity C is the mean ClarityScore across a sample of instances.
"""
import numpy as np
from sklearn.linear_model import Ridge

try:
    import shap
    _HAS_SHAP = True
except ImportError:
    shap = None
    _HAS_SHAP = False


def compute_clarity_score(fidelity: float, size: int, k: int = 5) -> float:
    """ClarityScore(i) = Fidelity / (1 + max(0, Size - k))  [Eq. 6]"""
    return float(np.clip(fidelity / (1.0 + max(0, size - k)), 0.0, 1.0))


def _local_neighbourhood(X, idx, n_neighbours=50, noise_scale=0.1, rng=None):
    if rng is None:
        rng = np.random.RandomState(0)
    x = X[idx]
    noise = rng.randn(n_neighbours, X.shape[1]) * noise_scale * (X.std(axis=0) + 1e-8)
    return x + noise


def compute_instance_clarity(model_predict_fn, X, idx, tau_pct=0.01,
                              n_neighbours=50, k=5, rng=None,
                              background_data=None):
    """Compute ClarityScore(i) using SHAP-based feature selection (Eq. 6).

    Steps:
        1. Compute SHAP values for instance i.
        2. Select top-k features by absolute SHAP value.
        3. Size = count of features with |phi_j| >= tau_pct * mean(|phi|).
        4. Generate local neighbourhood around instance i.
        5. Train Ridge surrogate on ONLY the top-k features.
        6. Fidelity = fraction of neighbourhood predictions the surrogate
           replicates correctly.
        7. Return Fidelity / (1 + max(0, Size - k)).

    Falls back to neighbourhood-only approach if SHAP is unavailable.
    """
    if rng is None:
        rng = np.random.RandomState(int(idx))

    x_instance = X[[idx]]

    # --- SHAP-based feature selection ---
    shap_values = None
    if _HAS_SHAP and background_data is not None:
        try:
            bg = background_data
            if len(bg) > 100:
                bg_idx = rng.choice(len(bg), 100, replace=False)
                bg = bg[bg_idx]
            explainer = shap.KernelExplainer(model_predict_fn, bg)
            sv = explainer.shap_values(x_instance, nsamples=100, silent=True)
            if isinstance(sv, list):
                sv = np.abs(np.array(sv)).mean(axis=0)
            shap_values = np.abs(sv).flatten()
        except Exception:
            shap_values = None

    if shap_values is not None and len(shap_values) == X.shape[1]:
        mean_abs_shap = float(np.mean(shap_values))
        threshold = tau_pct * mean_abs_shap if mean_abs_shap > 0 else 1e-8
        size = int(np.sum(shap_values >= threshold))
        top_k_indices = np.argsort(shap_values)[-k:]
    else:
        # Fallback: use Ridge coefficient magnitudes as proxy for feature importance
        neighbourhood = _local_neighbourhood(X, idx, n_neighbours, 0.1, rng)
        X_local = np.vstack([x_instance, neighbourhood])
        try:
            raw = np.asarray(model_predict_fn(X_local))
            preds = raw.argmax(axis=1) if raw.ndim == 2 else raw.astype(int)
        except Exception:
            return 0.0
        try:
            surrogate = Ridge(alpha=1.0, fit_intercept=True)
            surrogate.fit(X_local, preds.astype(float))
            importances = np.abs(surrogate.coef_) if surrogate.coef_.ndim == 1 else np.abs(surrogate.coef_).mean(axis=0)
            mean_imp = float(importances.mean())
            threshold = tau_pct * mean_imp if mean_imp > 0 else 1e-8
            size = int(np.sum(importances >= threshold))
            top_k_indices = np.argsort(importances)[-k:]
        except Exception:
            return 0.0

    # --- Compute fidelity on top-k features ---
    neighbourhood = _local_neighbourhood(X, idx, n_neighbours, 0.1, rng)
    X_local = np.vstack([x_instance, neighbourhood])
    try:
        raw = np.asarray(model_predict_fn(X_local))
        preds = raw.argmax(axis=1) if raw.ndim == 2 else raw.astype(int)
    except Exception:
        return 0.0

    try:
        X_local_topk = X_local[:, top_k_indices]
        surrogate = Ridge(alpha=1.0, fit_intercept=True)
        surrogate.fit(X_local_topk, preds.astype(float))
        surrogate_preds = surrogate.predict(X_local_topk).round().astype(int)
        fidelity = float(np.mean(surrogate_preds == preds))
    except Exception:
        return 0.0

    return compute_clarity_score(fidelity, size, k)


def compute_global_clarity(model_predict_fn, X, sample_size=100,
                            tau_pct=0.01, n_neighbours=50, k=5, seed=42,
                            background_data=None):
    """C = (1/|I|) * sum ClarityScore(i)  [Eq. 7]

    Args:
        model_predict_fn: callable that takes X array and returns predictions.
        X: feature matrix (n_samples, n_features).
        sample_size: number of instances to evaluate clarity on.
        tau_pct: SHAP significance threshold (fraction of mean |phi|).
        n_neighbours: local neighbourhood size per instance.
        k: human interpretability budget (default 5).
        seed: random seed.
        background_data: training data subset for SHAP KernelExplainer.
            If None, falls back to coefficient-based feature selection.
    """
    rng = np.random.RandomState(seed)
    indices = rng.choice(len(X), size=min(sample_size, len(X)), replace=False)
    scores = [
        compute_instance_clarity(model_predict_fn, X, int(i), tau_pct,
                                  n_neighbours, k,
                                  np.random.RandomState(seed + int(i)),
                                  background_data)
        for i in indices
    ]
    C = float(np.clip(np.mean(scores), 0.0, 1.0))
    return {"clarity": C, "opacity": 1.0 - C,
            "per_instance_scores": scores, "n_explained": len(scores)}


def clarity_from_feature_importances(feature_importances, y_true, y_pred,
                                      tau_pct=0.01, k=5):
    """Simplified clarity from model's built-in feature importances."""
    importances = np.abs(feature_importances)
    mean_imp = importances.mean()
    threshold = tau_pct * mean_imp if mean_imp > 0 else 1e-8
    size = max(int(np.sum(importances >= threshold)), 1)
    fidelity = float(np.mean(y_true == y_pred))
    val = compute_clarity_score(fidelity, size, k)
    return {"clarity": val, "opacity": 1.0 - val,
            "fidelity": fidelity, "explanation_size": size}
