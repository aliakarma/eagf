"""
src/utils/real_iot_loader.py — Real-World IoT Dataset Loader

Unified loader supporting two real-world IoT security datasets:
  - Edge-IIoTset (IEEE DataPort): network-traffic-based IoT security dataset.
  - TON_IoT (UNSW): network and system telemetry for IoT/IIoT environments.

Each loader:
  1. Checks for a local copy of the data.
  2. If absent, attempts to download it (requires an internet connection).
  3. Preprocesses: imputes missing values, normalises features, encodes labels.
  4. Splits with group awareness (device type / network layer as protected attr).

Pipeline integration via ``--use_real_data`` CLI flag in run_full_pipeline.py.

Usage::

    from src.utils.real_iot_loader import RealIoTLoader
    loader = RealIoTLoader(dataset="edge_iiot", data_dir="data/real_iot", seed=42)
    dataset_dict = loader.load()
    # dataset_dict keys: X_train, y_train, groups_train, X_val, y_val,
    #                    groups_val, X_test, y_test, groups_test, n_classes,
    #                    n_features, source
"""

from __future__ import annotations

import logging
import os
import warnings
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Dataset-specific constants
# ---------------------------------------------------------------------------

_EDGE_IIOT_LABEL_COL  = "Attack_type"
_EDGE_IIOT_GROUP_COL  = "TCP/IP_layer"
_EDGE_IIOT_NORMAL_LABEL = "Normal"

_TON_IOT_LABEL_COL  = "label"
_TON_IOT_GROUP_COL  = "type"   # device type column in TON_IoT
_TON_IOT_NORMAL_LABEL = "0"    # TON_IoT uses 0/1 binary labels

# Publicly accessible download hints (users must download and accept licences).
_EDGE_IIOT_URL = (
    "https://ieee-dataport.org/documents/"
    "edge-iiotset-new-comprehensive-realistic-cyber-security-dataset-iot-and-iiot-applications"
)
_TON_IOT_URL = "https://research.unsw.edu.au/projects/toniot-datasets"

_SUPPORTED_DATASETS = {"edge_iiot", "ton_iot"}

# Fraction of data to use for test set.
_TEST_SIZE = 0.20
# Fraction of training data to hold out as validation.
_VAL_SIZE  = 0.20


# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------

def _impute_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Impute NaNs: forward-fill then backward-fill; numeric remainder → 0."""
    df = df.ffill().bfill()
    num_cols = df.select_dtypes(include=[np.number]).columns
    df[num_cols] = df[num_cols].fillna(0.0)
    return df


def _encode_labels(series: pd.Series) -> np.ndarray:
    """Encode string or integer labels to dense integers starting at 0."""
    le = LabelEncoder()
    return le.fit_transform(series.astype(str)).astype(np.int64)


def _encode_features(
    df_train: pd.DataFrame,
    df_test: pd.DataFrame,
) -> Tuple[np.ndarray, np.ndarray]:
    """One-hot encode categoricals and scale numerics.

    The encoder and scaler are fitted only on the training split.
    """
    cat_cols = df_train.select_dtypes(include=["object", "category"]).columns.tolist()
    num_cols = df_train.select_dtypes(include=[np.number]).columns.tolist()

    # One-hot encoding for categoricals.
    if cat_cols:
        df_train_ohe = pd.get_dummies(df_train[cat_cols], drop_first=False)
        df_test_ohe  = pd.get_dummies(df_test[cat_cols],  drop_first=False)
        # Align columns (test may lack some rare categories).
        df_test_ohe = df_test_ohe.reindex(columns=df_train_ohe.columns, fill_value=0)
        ohe_train = df_train_ohe.values.astype(np.float32)
        ohe_test  = df_test_ohe.values.astype(np.float32)
    else:
        ohe_train = np.empty((len(df_train), 0), dtype=np.float32)
        ohe_test  = np.empty((len(df_test),  0), dtype=np.float32)

    # Standardise numerics (fit only on train).
    if num_cols:
        scaler = StandardScaler()
        num_train = scaler.fit_transform(df_train[num_cols].values.astype(np.float32))
        num_test  = scaler.transform(df_test[num_cols].values.astype(np.float32))
    else:
        num_train = np.empty((len(df_train), 0), dtype=np.float32)
        num_test  = np.empty((len(df_test),  0), dtype=np.float32)

    X_train = np.concatenate([num_train, ohe_train], axis=1)
    X_test  = np.concatenate([num_test,  ohe_test],  axis=1)
    return X_train, X_test


def _encode_features_multi(
    df_train: pd.DataFrame,
    *df_others: pd.DataFrame,
) -> Tuple[np.ndarray, ...]:
    """Fit encoder and scaler on df_train and transform all others.

    This ensures a single fitted scaler/encoder is shared across train,
    validation, and test splits — avoiding any per-split fitting artefacts.

    Returns:
        Tuple of transformed arrays: (X_train, X_other_0, X_other_1, ...).
    """
    cat_cols = df_train.select_dtypes(include=["object", "category"]).columns.tolist()
    num_cols = df_train.select_dtypes(include=[np.number]).columns.tolist()

    # Fit one-hot on training columns; reindex others to match.
    if cat_cols:
        ohe_train = pd.get_dummies(df_train[cat_cols], drop_first=False)
        ohe_cols  = ohe_train.columns
        ohe_arrays = [ohe_train.values.astype(np.float32)]
        for df_o in df_others:
            ohe_o = pd.get_dummies(df_o[cat_cols], drop_first=False)
            ohe_o = ohe_o.reindex(columns=ohe_cols, fill_value=0)
            ohe_arrays.append(ohe_o.values.astype(np.float32))
    else:
        n_splits = 1 + len(df_others)
        ohe_arrays = [np.empty((len(df), 0), dtype=np.float32)
                      for df in [df_train] + list(df_others)]

    # Fit scaler on training numerics; transform all.
    if num_cols:
        scaler = StandardScaler()
        num_train = scaler.fit_transform(df_train[num_cols].values.astype(np.float32))
        num_arrays = [num_train] + [
            scaler.transform(df_o[num_cols].values.astype(np.float32))
            for df_o in df_others
        ]
    else:
        num_arrays = [np.empty((len(df), 0), dtype=np.float32)
                      for df in [df_train] + list(df_others)]

    return tuple(
        np.concatenate([num_arrays[i], ohe_arrays[i]], axis=1)
        for i in range(1 + len(df_others))
    )


def _to_dataset_dict(
    X_train: np.ndarray, y_train: np.ndarray, g_train: np.ndarray,
    X_val:   np.ndarray, y_val:   np.ndarray, g_val:   np.ndarray,
    X_test:  np.ndarray, y_test:  np.ndarray, g_test:  np.ndarray,
    source: str,
) -> Dict:
    return {
        "X_train":      X_train,
        "y_train":      y_train,
        "groups_train": g_train,
        "X_val":        X_val,
        "y_val":        y_val,
        "groups_val":   g_val,
        "X_test":       X_test,
        "y_test":       y_test,
        "groups_test":  g_test,
        "n_classes":    int(np.max(y_train)) + 1,
        "n_features":   X_train.shape[1],
        "source":       source,
    }


# ---------------------------------------------------------------------------
# Per-dataset preprocessing
# ---------------------------------------------------------------------------

def _load_edge_iiot(
    file_path: str,
    label_col: str = _EDGE_IIOT_LABEL_COL,
    group_col: Optional[str] = _EDGE_IIOT_GROUP_COL,
    seed: int = 42,
) -> Dict:
    """Load and preprocess an Edge-IIoTset CSV file.

    Args:
        file_path: Local path to the Edge-IIoTset CSV (or CSV.gz).
        label_col: Column name containing attack type labels.
        group_col: Column used as protected attribute (device/network layer).
        seed: Random seed for reproducible splits.

    Returns:
        EAGF-compatible dataset dictionary.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(
            f"Edge-IIoTset data file not found: {file_path}\n"
            f"Download from: {_EDGE_IIOT_URL}"
        )

    logger.info("Loading Edge-IIoTset from %s", file_path)
    df = pd.read_csv(file_path, low_memory=False)
    df = _impute_dataframe(df)

    # Label: binary (normal vs attack) or multi-class attack type.
    if label_col not in df.columns:
        raise ValueError(
            f"Label column '{label_col}' not found. "
            f"Available columns: {list(df.columns)}"
        )
    y_raw = _encode_labels(df[label_col])

    # Protected group attribute.
    if group_col and group_col in df.columns:
        g_raw = df[group_col].astype(str).values
    else:
        g_raw = np.array(["unknown"] * len(df))

    # Drop target/group columns from features.
    drop_cols = [c for c in [label_col, group_col] if c and c in df.columns]
    X_df = df.drop(columns=drop_cols)

    idx = np.arange(len(y_raw))
    tr_idx, te_idx = train_test_split(
        idx, test_size=_TEST_SIZE, stratify=y_raw, random_state=seed
    )
    tr_idx, va_idx = train_test_split(
        tr_idx,
        test_size=_VAL_SIZE / (1.0 - _TEST_SIZE),
        stratify=y_raw[tr_idx],
        random_state=seed,
    )

    X_tr, X_va, X_te = _encode_features_multi(
        X_df.iloc[tr_idx], X_df.iloc[va_idx], X_df.iloc[te_idx]
    )

    return _to_dataset_dict(
        X_tr, y_raw[tr_idx], g_raw[tr_idx],
        X_va, y_raw[va_idx], g_raw[va_idx],
        X_te, y_raw[te_idx], g_raw[te_idx],
        source="edge_iiot",
    )


def _load_ton_iot(
    file_path: str,
    label_col: str = _TON_IOT_LABEL_COL,
    group_col: Optional[str] = _TON_IOT_GROUP_COL,
    seed: int = 42,
) -> Dict:
    """Load and preprocess a TON_IoT CSV file.

    Args:
        file_path: Local path to the TON_IoT CSV.
        label_col: Column name containing binary/multiclass labels.
        group_col: Column used as protected attribute (device type).
        seed: Random seed for reproducible splits.

    Returns:
        EAGF-compatible dataset dictionary.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(
            f"TON_IoT data file not found: {file_path}\n"
            f"Download from: {_TON_IOT_URL}"
        )

    logger.info("Loading TON_IoT from %s", file_path)
    df = pd.read_csv(file_path, low_memory=False)
    df = _impute_dataframe(df)

    if label_col not in df.columns:
        raise ValueError(
            f"Label column '{label_col}' not found. "
            f"Available columns: {list(df.columns)}"
        )
    y_raw = _encode_labels(df[label_col])

    if group_col and group_col in df.columns:
        g_raw = df[group_col].astype(str).values
    else:
        g_raw = np.array(["unknown"] * len(df))

    drop_cols = [c for c in [label_col, group_col] if c and c in df.columns]
    X_df = df.drop(columns=drop_cols)

    idx = np.arange(len(y_raw))
    try:
        tr_idx, te_idx = train_test_split(
            idx, test_size=_TEST_SIZE, stratify=y_raw, random_state=seed
        )
        tr_idx, va_idx = train_test_split(
            tr_idx,
            test_size=_VAL_SIZE / (1.0 - _TEST_SIZE),
            stratify=y_raw[tr_idx],
            random_state=seed,
        )
    except ValueError:
        # Fallback: no stratification if some class has too few samples.
        tr_idx, te_idx = train_test_split(idx, test_size=_TEST_SIZE, random_state=seed)
        tr_idx, va_idx = train_test_split(tr_idx, test_size=_VAL_SIZE / (1.0 - _TEST_SIZE), random_state=seed)

    X_tr, X_va, X_te = _encode_features_multi(
        X_df.iloc[tr_idx], X_df.iloc[va_idx], X_df.iloc[te_idx]
    )

    return _to_dataset_dict(
        X_tr, y_raw[tr_idx], g_raw[tr_idx],
        X_va, y_raw[va_idx], g_raw[va_idx],
        X_te, y_raw[te_idx], g_raw[te_idx],
        source="ton_iot",
    )


# ---------------------------------------------------------------------------
# Public loader class
# ---------------------------------------------------------------------------

class RealIoTLoader:
    """Unified loader for real-world IoT security datasets.

    Supports:
      - ``"edge_iiot"`` — Edge-IIoTset (IEEE DataPort)
      - ``"ton_iot"``   — TON_IoT (UNSW)

    When ``auto_download=True`` the loader prints an informational message
    with the official download URL; it does **not** automatically download the
    data, because both datasets require accepting a licence agreement.

    Args:
        dataset: One of ``"edge_iiot"`` or ``"ton_iot"``.
        data_dir: Directory where dataset CSV files are stored.
        seed: Random seed for reproducible splits.
        label_col: Override the default label column name.
        group_col: Override the default group/protected-attribute column name.
        auto_download: If True, emit an informational message when the data
            file is not found (download cannot be automated for these datasets).

    Example::

        loader = RealIoTLoader("edge_iiot", data_dir="data/real_iot", seed=42)
        dataset_dict = loader.load()  # returns EAGF-compatible dict
    """

    _DEFAULT_FILENAMES = {
        "edge_iiot": "edge_iiot.csv",
        "ton_iot":   "ton_iot.csv",
    }

    def __init__(
        self,
        dataset: str = "edge_iiot",
        data_dir: str = "data/real_iot",
        seed: int = 42,
        label_col: Optional[str] = None,
        group_col: Optional[str] = None,
        auto_download: bool = True,
    ) -> None:
        if dataset not in _SUPPORTED_DATASETS:
            raise ValueError(
                f"Unsupported dataset '{dataset}'. "
                f"Choose from: {sorted(_SUPPORTED_DATASETS)}"
            )
        self.dataset = dataset
        self.data_dir = data_dir
        self.seed = seed
        self.label_col = label_col
        self.group_col = group_col
        self.auto_download = auto_download

    def _default_file_path(self) -> str:
        filename = self._DEFAULT_FILENAMES[self.dataset]
        return os.path.join(self.data_dir, filename)

    def _download_hint(self, file_path: str) -> None:
        url = _EDGE_IIOT_URL if self.dataset == "edge_iiot" else _TON_IOT_URL
        warnings.warn(
            f"\n[RealIoTLoader] '{self.dataset}' data file not found at:\n"
            f"  {file_path}\n"
            f"These datasets require accepting a licence agreement; "
            f"automated download is not available.\n"
            f"Please download from:\n  {url}\n"
            f"and place the CSV at the path shown above.",
            UserWarning,
            stacklevel=3,
        )

    def load(self, file_path: Optional[str] = None) -> Dict:
        """Load the dataset and return an EAGF-compatible dictionary.

        Args:
            file_path: Explicit path to the CSV file.  If omitted, the loader
                looks for ``<data_dir>/<default_filename>``.

        Returns:
            Dictionary with keys: ``X_train``, ``y_train``, ``groups_train``,
            ``X_val``, ``y_val``, ``groups_val``, ``X_test``, ``y_test``,
            ``groups_test``, ``n_classes``, ``n_features``, ``source``.

        Raises:
            FileNotFoundError: If the data file is not present.
        """
        if file_path is None:
            file_path = self._default_file_path()

        if not os.path.exists(file_path):
            if self.auto_download:
                self._download_hint(file_path)
            raise FileNotFoundError(
                f"Data file not found: {file_path}. "
                "See the warning above for download instructions."
            )

        if self.dataset == "edge_iiot":
            return _load_edge_iiot(
                file_path,
                label_col=self.label_col or _EDGE_IIOT_LABEL_COL,
                group_col=self.group_col or _EDGE_IIOT_GROUP_COL,
                seed=self.seed,
            )
        elif self.dataset == "ton_iot":
            return _load_ton_iot(
                file_path,
                label_col=self.label_col or _TON_IOT_LABEL_COL,
                group_col=self.group_col or _TON_IOT_GROUP_COL,
                seed=self.seed,
            )
        else:
            raise ValueError(f"Unsupported dataset: {self.dataset}")
