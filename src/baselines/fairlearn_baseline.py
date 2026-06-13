"""
src/baselines/fairlearn_baseline.py — Fairlearn Exponentiated Gradient Baseline

Implements Exponentiated Gradient Reduction (Agarwal et al., 2018) as a strong
fairness baseline using the `fairlearn` library.

The baseline:
  1. Trains a logistic-regression estimator under fairness constraints using
     ExponentiatedGradient with EqualizedOdds (or DemographicParity for
     multi-class settings without meaningful positive-label semantics).
  2. Reports the same output metrics format as EAGF train_variant so that
     results can appear in comparison plots and CSVs without modification.

References:
  Agarwal et al., "A Reductions Approach to Fair Classification", ICML 2018.

Usage::

    from src.baselines.fairlearn_baseline import train_fairlearn_baseline
    metrics = train_fairlearn_baseline(variant="fairlearn_eg", config=cfg,
                                       dataset=data, seed=42)
"""

from __future__ import annotations

import os
import time
import warnings
from typing import Any, Dict, Optional

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

try:
    from fairlearn.reductions import (
        DemographicParity,
        EqualizedOdds,
        FalsePositiveRateParity,
        ExponentiatedGradient,
    )
    _HAS_FAIRLEARN = True
except ImportError:  # pragma: no cover
    _HAS_FAIRLEARN = False

try:
    import psutil as _psutil
    _HAS_PSUTIL = True
except ImportError:
    _HAS_PSUTIL = False

from src.evaluation.audit_logger import AuditLogger
from src.evaluation.calibration import compute_calibration_metrics
from src.evaluation.mia_attack import run_shadow_model_attack
from src.metrics.accountability import compute_accountability
from src.metrics.clarity import compute_global_clarity
from src.metrics.fairness import false_positive_rate_parity, recall_parity, select_criterion
from src.metrics.privacy import privacy_score
from src.metrics.trust_index import trust_index
from src.utils.preprocessing import preprocess_biometric, preprocess_reiot

# Assumed CPU power draw for energy estimation (Watts).
_CPU_POWER_WATTS = 65.0


# ---------------------------------------------------------------------------
# Sklearn-compatible adapter so EAGF metric functions work unchanged
# ---------------------------------------------------------------------------

class _FairlearnModelAdapter:
    """Sklearn-like adapter wrapping a trained fairlearn predictor."""

    def __init__(self, predictor, classes: np.ndarray):
        self._predictor = predictor
        self._classes = classes

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.asarray(self._predictor.predict(X), dtype=int)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        n = len(X)
        n_classes = len(self._classes)
        if hasattr(self._predictor, "predict_proba"):
            try:
                return np.asarray(self._predictor.predict_proba(X), dtype=float)
            except Exception:
                pass
        # Fallback: construct one-hot proba from hard predictions.
        preds = self.predict(X)
        proba = np.zeros((n, n_classes), dtype=float)
        for i, p in enumerate(preds):
            cls_idx = np.searchsorted(self._classes, p)
            proba[i, cls_idx] = 1.0
        return proba


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _encode_groups(groups) -> np.ndarray:
    """Map group labels to integer codes."""
    if groups is None:
        return None
    le = LabelEncoder()
    return le.fit_transform([str(g) for g in groups]).astype(int)


def _get_fairness_criterion(config: dict) -> str:
    context = "biometric"
    if "reiot" in str(config.get("data", {}).get("name", "")).lower():
        context = "reiot"
    return select_criterion(context)


def _build_constraint(criterion: str, n_classes: int):
    """Return the appropriate fairlearn constraint for the given criterion."""
    if criterion == "recall_parity" and n_classes == 2:
        return EqualizedOdds()
    if criterion in ("fprp", "fpr_parity") and n_classes == 2:
        try:
            return FalsePositiveRateParity()
        except Exception:
            return EqualizedOdds()
    return DemographicParity()


# ---------------------------------------------------------------------------
# Public training function
# ---------------------------------------------------------------------------

def train_fairlearn_baseline(
    variant: str = "fairlearn_eg",
    config: Optional[Dict[str, Any]] = None,
    dataset: Optional[Dict[str, Any]] = None,
    seed: int = 42,
    output_dir: str = ".",
    return_model: bool = False,
) -> Dict[str, Any]:
    """Train an Exponentiated Gradient model and compute EAGF-compatible metrics.

    Args:
        variant: Name for this run (used in output path naming).
        config: EAGF config dict (same structure as biometric_default.yaml).
        dataset: Dataset dict with keys ``X_train``, ``y_train``, etc.
        seed: Random seed for reproducibility.
        output_dir: Directory for audit logs and per-seed results.
        return_model: If True, include the model adapter in the returned dict.

    Returns:
        Metrics dict compatible with the format returned by
        ``eagf_trainer.train_variant``.

    Raises:
        ImportError: If ``fairlearn`` is not installed.
    """
    if not _HAS_FAIRLEARN:
        raise ImportError(
            "fairlearn is required for this baseline. "
            "Install it with: pip install fairlearn"
        )

    if config is None:
        config = {}
    if dataset is None:
        raise ValueError("dataset must be provided")

    np.random.seed(seed)
    os.makedirs(output_dir, exist_ok=True)
    audit_log_path = os.path.join(output_dir, "audit_log.jsonl")
    device = config.get("training", {}).get("device", "cpu")
    is_reiot = "reiot" in str(config.get("data", {}).get("name", "")).lower()

    # ── Preprocess ───────────────────────────────────────────────────────────
    if is_reiot:
        data = preprocess_reiot(dataset.copy(), seed=seed)
        X_tr_full, y_tr_full = data["X_train"], data["y_train"]
        g_tr_full = data.get("groups_train")
        idx = np.arange(len(y_tr_full))
        idx_train, idx_val = train_test_split(
            idx, test_size=0.2, random_state=seed, stratify=y_tr_full
        )
        X_train, y_train = X_tr_full[idx_train], y_tr_full[idx_train]
        X_val, y_val = X_tr_full[idx_val], y_tr_full[idx_val]
        X_test, y_test = data["X_test"], data["y_test"]
        groups_train = g_tr_full[idx_train] if g_tr_full is not None else None
        groups_test  = data.get("groups_test")
    else:
        data = preprocess_biometric(
            dataset.copy(),
            apply_dp=False,
            epsilon=config.get("governance", {}).get("dp_epsilon", 3.0),
            seed=seed,
        )
        X_train, y_train = data["X_train"], data["y_train"]
        X_val,   y_val   = data.get("X_val", data["X_train"]), data.get("y_val", data["y_train"])
        X_test,  y_test  = data["X_test"], data["y_test"]
        groups_train = data.get("groups_train")
        groups_test  = data.get("groups_test")

    n_classes = int(np.max(y_train)) + 1
    classes = np.arange(n_classes, dtype=int)

    # ── Build and fit ExponentiatedGradient ──────────────────────────────────
    criterion = _get_fairness_criterion(config)
    constraint = _build_constraint(criterion, n_classes)

    # Base estimator: logistic regression with a fixed random state.
    # Use sklearn's default C=1.0 (inverse regularization strength).
    base_estimator = LogisticRegression(
        max_iter=1000,
        random_state=seed,
        solver="lbfgs",
        C=1.0,
    )
    eg = ExponentiatedGradient(
        estimator=base_estimator,
        constraints=constraint,
        max_iter=50,
        nu=1e-6,
    )

    # Fairlearn requires a 1-D sensitive_features array.
    g_encoded = _encode_groups(groups_train)
    if g_encoded is None:
        g_encoded = np.zeros(len(y_train), dtype=int)

    t0 = time.perf_counter()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        eg.fit(X_train, y_train, sensitive_features=g_encoded)
    t_train_elapsed = time.perf_counter() - t0

    model_adapter = _FairlearnModelAdapter(eg, classes)

    # ── Overhead measurements ─────────────────────────────────────────────────
    t_infer_start = time.perf_counter()
    _ = model_adapter.predict_proba(X_test)
    t_infer_elapsed = time.perf_counter() - t_infer_start
    infer_ms = (t_infer_elapsed * 1000.0) / max(len(X_test), 1)

    mem_mb = 0.0
    if _HAS_PSUTIL:
        mem_mb = float(_psutil.Process().memory_info().rss) / (1024.0 ** 2)
    energy_j = t_train_elapsed * _CPU_POWER_WATTS

    # ── Metrics ───────────────────────────────────────────────────────────────
    y_pred_test = model_adapter.predict(X_test)
    acc = float(accuracy_score(y_test, y_pred_test))

    clarity_result = compute_global_clarity(
        model_predict_fn=model_adapter.predict,
        X=X_val,
        sample_size=50,
        seed=seed,
    )
    C = float(np.clip(clarity_result["clarity"], 0.0, 1.0))

    ref_group = "male_light" if criterion == "recall_parity" else "urban"
    # If the configured ref_group is not present in the data, fall back to the
    # most-frequent group to avoid silent failures with real-world datasets.
    if groups_test is not None:
        unique_test_groups = {str(g) for g in groups_test}
        if ref_group not in unique_test_groups:
            from collections import Counter
            ref_group = Counter(str(g) for g in groups_test).most_common(1)[0][0]
    if groups_test is not None and criterion == "fprp":
        fair_result = false_positive_rate_parity(y_test, y_pred_test, groups_test, ref_group)
        Fv = float(fair_result.get("fprp", 0.0))
    elif groups_test is not None and criterion == "recall_parity":
        fair_result = recall_parity(y_test, y_pred_test, groups_test, ref_group)
        Fv = float(fair_result.get("recall_parity", 0.0))
    else:
        Fv = 0.0

    mia_result = run_shadow_model_attack(
        model_adapter, X_train, y_train, X_test, y_test, seed=seed
    )
    mia_auc = float(mia_result["mia_auc"])
    # No formal DP guarantee — privacy score relies entirely on MIA AUC.
    P = privacy_score(epsilon_eff=float("inf"), mia_auc=mia_auc)

    checklist_path = config.get("accountability", {}).get("compliance_checklist")
    acc_result = compute_accountability(
        audit_log_path=audit_log_path,
        total_decisions=len(X_test),
        lineage_fraction=None,
        checklist_path=checklist_path,
        model_has_governance=False,
        metric_overrides={
            "mia_stress_test": mia_auc <= 0.60,
            "fairness_monitoring": Fv >= 0.95,
        },
    )
    A = float(acc_result["accountability"])

    ti_result = trust_index(C, Fv, P, A)

    y_proba_test = model_adapter.predict_proba(X_test)
    calib_result = compute_calibration_metrics(y_test, y_proba_test)

    # Write a minimal audit entry so log infrastructure is exercised.
    logger = AuditLogger(
        audit_log_path,
        model_version=f"fairlearn_eg_{seed}",
        operator_id="system",
    )
    logger.log(
        X_test[:1],
        output_label=int(y_pred_test[0]),
        output_confidence=float(y_proba_test[0].max()),
        inference_time_ms=infer_ms,
        memory_usage_mb=mem_mb,
        energy_overhead_joules=energy_j,
        ece=calib_result["ece"],
        brier_score=calib_result["brier_score"],
    )

    result = {
        "accuracy": acc,
        "recall_parity": Fv,
        "clarity": C,
        "privacy": P,
        "accountability": A,
        "trust_index": float(ti_result["ti"]),
        "mia_auc": mia_auc,
        "epsilon_eff": float("inf"),
        "ece": calib_result["ece"],
        "brier_score": calib_result["brier_score"],
        "inference_time_ms": infer_ms,
        "memory_usage_mb": mem_mb,
        "energy_overhead_joules": energy_j,
        "ti_components": ti_result["components"],
        "audit": acc_result,
    }
    if return_model:
        result["model"] = model_adapter
    return result
