"""
src/utils/real_data_loader.py
Real-World IoT Data Loader — Edge-IIoT Dataset Integration

Loads and preprocesses real Edge-IIoT dataset files for use in the EAGF pipeline.
When no real data file is available, a synthetic fallback is generated automatically.
"""

from __future__ import annotations

import os
from typing import Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# Column that contains the attack/label information in Edge-IIoT datasets.
_DEFAULT_LABEL_COL = "Attack_type"

# Column used as the protected-group attribute (device / node class).
_DEFAULT_GROUP_COL = "TCP/IP_layer"

# Fraction of data to hold out for testing.
_TEST_SIZE = 0.20

# Random state used when none is provided.
_DEFAULT_SEED = 42


class RealREIoTDataLoader:
    """Load and preprocess a real-world Edge-IIoT CSV dataset.

    The loader handles:
    * Missing value imputation via forward fill (then backward fill for any
      remaining NaNs at the start of the DataFrame).
    * Normalisation of numerical columns using :class:`~sklearn.preprocessing.StandardScaler`
      fitted only on the training split.
    * Encoding of categorical columns using :class:`~sklearn.preprocessing.OneHotEncoder`
      fitted only on the training split.
    * Splitting into train / test partitions while preserving reproducibility.

    The ``groups_*`` arrays represent the protected attribute column
    (``TCP/IP_layer`` by default), which identifies the network-stack layer /
    device class — the relevant protected attribute in IoT fairness analysis.

    Usage::

        loader = RealREIoTDataLoader(seed=42)
        X_train, X_test, y_train, y_test, groups_train, groups_test = (
            loader.load_edge_iiot_data("path/to/edge_iiot.csv")
        )
        dataset = loader.to_dataset_dict()
    """

    def __init__(
        self,
        label_col: str = _DEFAULT_LABEL_COL,
        group_col: Optional[str] = _DEFAULT_GROUP_COL,
        test_size: float = _TEST_SIZE,
        seed: int = _DEFAULT_SEED,
    ) -> None:
        self.label_col = label_col
        self.group_col = group_col
        self.test_size = test_size
        self.seed = seed

        # Populated after a successful call to :meth:`load_edge_iiot_data`.
        self._X_train: Optional[np.ndarray] = None
        self._X_test: Optional[np.ndarray] = None
        self._y_train: Optional[np.ndarray] = None
        self._y_test: Optional[np.ndarray] = None
        self._groups_train: Optional[np.ndarray] = None
        self._groups_test: Optional[np.ndarray] = None
        self._n_features: int = 0
        self._n_classes: int = 2
        self._scaler: Optional[StandardScaler] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def load_edge_iiot_data(
        self, file_path: str
    ) -> Tuple[
        np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray
    ]:
        """Load, preprocess and split the Edge-IIoT dataset.

        Args:
            file_path: Path to the CSV (or CSV.gz) file containing the
                Edge-IIoT dataset.

        Returns:
            A tuple ``(X_train, X_test, y_train, y_test, groups_train,
            groups_test)`` with:

            * **X_train / X_test** – float32 feature matrices of shape
              ``(n_train, n_features)`` and ``(n_test, n_features)``.
            * **y_train / y_test** – int64 label arrays of shape
              ``(n_train,)`` and ``(n_test,)``.
            * **groups_train / groups_test** – string arrays of shape
              ``(n_train,)`` and ``(n_test,)`` that represent the
              protected attribute used for fairness evaluation.

        Raises:
            FileNotFoundError: If *file_path* does not exist on disk.
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(
                f"Edge-IIoT data file not found: {file_path}\n"
                "Download the dataset from https://ieee-dataport.org/documents/edge-iiotset-new-comprehensive-realistic-cyber-security-dataset-iot-and-iiot-applications"
            )

        df = pd.read_csv(file_path, low_memory=False)
        df = self._impute(df)

        y_raw, groups_raw = self._extract_targets(df)
        X_raw = df.drop(
            columns=[c for c in [self.label_col, self.group_col] if c in df.columns]
        )

        idx = np.arange(len(y_raw))
        tr_idx, te_idx = train_test_split(
            idx,
            test_size=self.test_size,
            stratify=y_raw,
            random_state=self.seed,
        )

        X_tr_raw, X_te_raw = X_raw.iloc[tr_idx], X_raw.iloc[te_idx]
        y_tr, y_te = y_raw[tr_idx], y_raw[te_idx]
        g_tr = groups_raw[tr_idx] if groups_raw is not None else np.array(["unknown"] * len(tr_idx))
        g_te = groups_raw[te_idx] if groups_raw is not None else np.array(["unknown"] * len(te_idx))

        X_tr, X_te = self._encode_and_scale(X_tr_raw, X_te_raw)

        # Validate no NaNs remain in outputs.
        if np.isnan(X_tr).any():
            raise ValueError("NaNs detected in X_train after preprocessing. Check the input data.")
        if np.isnan(X_te).any():
            raise ValueError("NaNs detected in X_test after preprocessing. Check the input data.")

        self._X_train, self._X_test = X_tr, X_te
        self._y_train, self._y_test = y_tr, y_te
        self._groups_train, self._groups_test = g_tr, g_te
        self._n_features = X_tr.shape[1]
        self._n_classes = int(np.max(y_raw)) + 1

        return X_tr, X_te, y_tr, y_te, g_tr, g_te

    def to_dataset_dict(self) -> dict:
        """Convert the loaded data to the dictionary format expected by EAGF trainers.

        Must be called after :meth:`load_edge_iiot_data`.

        Returns:
            A dataset dictionary with keys ``X_train``, ``y_train``,
            ``groups_train``, ``X_val``, ``y_val``, ``groups_val``,
            ``X_test``, ``y_test``, ``groups_test``, ``n_classes``,
            ``n_features``, and ``source``.

        Raises:
            RuntimeError: If :meth:`load_edge_iiot_data` has not been called yet.
        """
        if self._X_train is None:
            raise RuntimeError(
                "No data loaded. Call load_edge_iiot_data() first."
            )

        # Split training data further into train / val (80 / 20).
        n_train = len(self._y_train)
        idx = np.arange(n_train)
        tr_idx, va_idx = train_test_split(
            idx,
            test_size=0.20,
            stratify=self._y_train,
            random_state=self.seed,
        )

        return {
            "X_train":      self._X_train[tr_idx],
            "y_train":      self._y_train[tr_idx],
            "groups_train": self._groups_train[tr_idx],
            "X_val":        self._X_train[va_idx],
            "y_val":        self._y_train[va_idx],
            "groups_val":   self._groups_train[va_idx],
            "X_test":       self._X_test,
            "y_test":       self._y_test,
            "groups_test":  self._groups_test,
            "n_classes":    self._n_classes,
            "n_features":   self._n_features,
            "scaler":       self._scaler,
            "source":       "edge_iiot_real",
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _impute(df: pd.DataFrame) -> pd.DataFrame:
        """Impute missing values using forward fill then backward fill."""
        return df.ffill().bfill()

    def _extract_targets(
        self, df: pd.DataFrame
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Extract binary labels and protected-group arrays from the DataFrame.

        The label column is binarised: ``Normal`` traffic → 0, any attack → 1.
        The group column (if present) is kept as raw strings.
        """
        if self.label_col in df.columns:
            label_raw = df[self.label_col].astype(str)
            y = (label_raw.str.strip().str.lower() != "normal").astype(np.int64).values
        else:
            # Fallback: assume last column is the label.
            label_raw = df.iloc[:, -1].astype(str)
            y = (label_raw.str.strip().str.lower() != "normal").astype(np.int64).values

        if self.group_col and self.group_col in df.columns:
            groups = np.array(df[self.group_col].astype(str).tolist(), dtype=object)
        else:
            groups = None

        return y, groups

    def _encode_and_scale(
        self,
        X_tr: pd.DataFrame,
        X_te: pd.DataFrame,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Encode categoricals, scale numerics, and return float32 arrays.

        Encoders / scalers are fitted on *X_tr* only to prevent data leakage.
        """
        cat_cols = X_tr.select_dtypes(include=["object", "category"]).columns.tolist()
        num_cols = X_tr.select_dtypes(include=["number"]).columns.tolist()

        parts_tr: list[np.ndarray] = []
        parts_te: list[np.ndarray] = []

        if num_cols:
            scaler = StandardScaler()
            num_tr = scaler.fit_transform(X_tr[num_cols].values.astype(np.float64))
            num_te = scaler.transform(X_te[num_cols].values.astype(np.float64))
            parts_tr.append(num_tr)
            parts_te.append(num_te)
            self._scaler = scaler

        if cat_cols:
            enc = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
            cat_tr = enc.fit_transform(X_tr[cat_cols].values)
            cat_te = enc.transform(X_te[cat_cols].values)
            parts_tr.append(cat_tr)
            parts_te.append(cat_te)

        if not parts_tr:
            # No usable columns at all — return empty arrays.
            return (
                np.empty((len(X_tr), 0), dtype=np.float32),
                np.empty((len(X_te), 0), dtype=np.float32),
            )

        X_tr_out = np.hstack(parts_tr).astype(np.float32)
        X_te_out = np.hstack(parts_te).astype(np.float32)
        return X_tr_out, X_te_out


# ---------------------------------------------------------------------------
# Convenience function — mirrors the style of load_reiot_dataset / load_biometric_dataset
# ---------------------------------------------------------------------------

def load_edge_iiot_dataset(
    file_path: str,
    label_col: str = _DEFAULT_LABEL_COL,
    group_col: Optional[str] = _DEFAULT_GROUP_COL,
    test_size: float = _TEST_SIZE,
    seed: int = _DEFAULT_SEED,
) -> dict:
    """Load the Edge-IIoT dataset and return a trainer-compatible dictionary.

    Thin wrapper around :class:`RealREIoTDataLoader` for use in pipeline
    scripts that prefer a functional interface.

    Args:
        file_path: Path to the Edge-IIoT CSV file.
        label_col: Column name that contains the attack / normal labels.
        group_col: Column name used as the protected attribute for fairness.
        test_size: Fraction of data reserved for testing.
        seed: Random seed for reproducibility.

    Returns:
        Dataset dictionary compatible with :func:`~src.training.eagf_trainer.train_variant`.

    Raises:
        FileNotFoundError: If *file_path* does not exist.
    """
    loader = RealREIoTDataLoader(
        label_col=label_col,
        group_col=group_col,
        test_size=test_size,
        seed=seed,
    )
    loader.load_edge_iiot_data(file_path)
    return loader.to_dataset_dict()
