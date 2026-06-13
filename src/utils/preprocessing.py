"""
src/utils/preprocessing.py
Data Preprocessing: Anonymisation and Group-Balance Re-weighting

Paper reference: Section 3.1, Stages 2-3 (Privacy-Preserving Preparation
                 and Bias Assessment/Mitigation in Training Data)
"""

import json
import os
from typing import Dict, List, Optional, Tuple

import numpy as np
from sklearn.preprocessing import StandardScaler


def compute_sample_weights(
    y: np.ndarray,
    groups: Optional[np.ndarray] = None,
    strategy: str = "balanced_group",
) -> np.ndarray:
    """Compute per-sample weights for group-balanced training.

    Addresses Bias Assessment and Mitigation (Stage 3 of EAGF lifecycle).

    Args:
        y: Class labels (n_samples,).
        groups: Protected group labels (n_samples,). If None, uses class balancing.
        strategy: 'balanced_group' | 'balanced_class' | 'uniform'

    Returns:
        weights: (n_samples,) float array of sample weights.
    """
    n = len(y)
    weights = np.ones(n, dtype=np.float64)

    if strategy == "uniform":
        return weights

    if strategy == "balanced_class" or groups is None:
        unique_classes, class_counts = np.unique(y, return_counts=True)
        class_weight = {c: n / (len(unique_classes) * cnt)
                        for c, cnt in zip(unique_classes, class_counts)}
        for i, yi in enumerate(y):
            weights[i] = class_weight[yi]
        return weights

    # balanced_group: weight by (class × group) cell
    if groups is not None:
        unique_combos, combo_counts = np.unique(
            list(zip(y.tolist(), groups.tolist())), axis=0, return_counts=True
        )
        total_combos = len(unique_combos)
        combo_weight = {}
        for combo, cnt in zip(unique_combos, combo_counts):
            key = (combo[0] if hasattr(combo, '__len__') else combo[0], combo[1] if hasattr(combo, '__len__') else combo[1])
            combo_weight[key] = n / (total_combos * cnt)

        for i in range(n):
            key = (str(y[i]), str(groups[i]))
            weights[i] = combo_weight.get(key, 1.0)

    return weights


def apply_feature_noise_dp(
    X: np.ndarray,
    epsilon: float = 3.0,
    delta: float = 1e-5,
    sensitivity: float = 1.0,
    seed: int = 42,
) -> np.ndarray:
    """Apply Gaussian mechanism to feature vectors (data-level DP).

    This is a simplified approximation of DP for pre-training data protection.
    Model-level DP is handled by the trainer via gradient perturbation.

    Args:
        X: Feature matrix (n_samples, n_features).
        epsilon: Privacy budget.
        delta: Privacy delta.
        sensitivity: L2 sensitivity of the features.
        seed: Random seed.

    Returns:
        X_noisy: Privacy-protected feature matrix.
    """
    rng = np.random.RandomState(seed)
    # Gaussian mechanism: sigma = sensitivity * sqrt(2 * ln(1.25/delta)) / epsilon
    sigma = sensitivity * np.sqrt(2.0 * np.log(1.25 / delta)) / epsilon
    noise = rng.randn(*X.shape) * sigma
    return (X + noise).astype(np.float32)


def anonymise(
    X: np.ndarray,
    sensitive_column_indices: Optional[List[int]] = None,
) -> np.ndarray:
    """Remove or mask sensitive feature columns.

    Args:
        X: Feature matrix.
        sensitive_column_indices: Column indices to zero out.

    Returns:
        Anonymised feature matrix.
    """
    X_anon = X.copy()
    if sensitive_column_indices:
        X_anon[:, sensitive_column_indices] = 0.0
    return X_anon


def normalise_features(
    X_train: np.ndarray,
    X_val: Optional[np.ndarray] = None,
    X_test: Optional[np.ndarray] = None,
) -> Tuple:
    """Fit StandardScaler on training data and transform all splits.

    Args:
        X_train: Training features.
        X_val: Validation features (optional).
        X_test: Test features (optional).

    Returns:
        Tuple of (X_train_scaled, X_val_scaled, X_test_scaled, scaler).
    """
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_val_s   = scaler.transform(X_val)   if X_val  is not None else None
    X_test_s  = scaler.transform(X_test)  if X_test is not None else None
    return X_train_s, X_val_s, X_test_s, scaler


def preprocess_biometric(
    dataset: Dict,
    apply_dp: bool = False,
    epsilon: float = 3.0,
    seed: int = 42,
) -> Dict:
    """Full preprocessing pipeline for biometric datasets.

    Applies: normalisation, optional DP noise, sample weight computation.
    For image data (4D arrays), skips StandardScaler since images are
    already ImageNet-normalized by the biometric pipeline.

    Args:
        dataset: Raw dataset dictionary from data_loader.
        apply_dp: Whether to apply DP noise to features.
        epsilon: DP epsilon (used only if apply_dp=True).
        seed: Random seed.

    Returns:
        Preprocessed dataset with added 'sample_weights' key.
    """
    if dataset.get("is_image_data"):
        dataset["sample_weights"] = compute_sample_weights(
            dataset["y_train"],
            groups=dataset.get("groups_train"),
            strategy="balanced_group",
        )
        return dataset

    X_train, X_val, X_test, scaler = normalise_features(
        dataset["X_train"], dataset.get("X_val"), dataset["X_test"]
    )
    dataset["X_train"] = X_train.astype(np.float32)
    dataset["X_test"]  = X_test.astype(np.float32)
    if X_val is not None:
        dataset["X_val"] = X_val.astype(np.float32)
    dataset["scaler"] = scaler

    if apply_dp:
        dataset["X_train"] = apply_feature_noise_dp(
            dataset["X_train"], epsilon=epsilon, seed=seed
        )

    dataset["sample_weights"] = compute_sample_weights(
        dataset["y_train"],
        groups=dataset.get("groups_train"),
        strategy="balanced_group",
    )
    return dataset


def _impute_nan_columns(X: np.ndarray) -> np.ndarray:
    """Replace NaN values with column medians.

    Used to sanitise RE-IoT feature arrays that may contain NaN values
    after :func:`~src.utils.reiot_simulator.inject_network_faults`.
    Columns that are entirely NaN are filled with 0.

    Args:
        X: Float feature matrix (n_samples, n_features).

    Returns:
        Copy of X with all NaN values replaced.
    """
    if not np.isnan(X).any():
        return X
    X_out = X.copy().astype(np.float64)
    col_medians = np.nanmedian(X_out, axis=0)
    col_medians = np.where(np.isnan(col_medians), 0.0, col_medians)
    nan_rows, nan_cols = np.where(np.isnan(X_out))
    X_out[nan_rows, nan_cols] = col_medians[nan_cols]
    return X_out.astype(np.float32)


def preprocess_reiot(dataset: Dict, seed: int = 42) -> Dict:
    """Preprocessing pipeline for RE-IoT datasets."""
    # Impute NaN values (e.g., from inject_network_faults) before scaling.
    for split in ("X_train", "X_test"):
        if split in dataset and np.isnan(dataset[split]).any():
            dataset[split] = _impute_nan_columns(dataset[split])
    scaler = StandardScaler()
    dataset["X_train"] = scaler.fit_transform(dataset["X_train"]).astype(np.float32)
    dataset["X_test"]  = scaler.transform(dataset["X_test"]).astype(np.float32)
    dataset["scaler"]  = scaler

    dataset["sample_weights"] = compute_sample_weights(
        dataset["y_train"],
        groups=dataset.get("groups_train"),
        strategy="balanced_group",
    )
    return dataset
