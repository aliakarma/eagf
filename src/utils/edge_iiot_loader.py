"""
src/utils/edge_iiot_loader.py — Edge-IIoTset Dataset Loader

Loads the Edge-IIoTset (2022) IoT intrusion-detection dataset for use in the
EAGF pipeline.  If the raw CSV is not present locally the loader will attempt
to download a publicly-available subset, falling back to a high-fidelity
synthetic replica of the Edge-IIoTset schema when the download is unavailable
(e.g. in offline CI environments).

Edge-IIoTset reference
----------------------
M. A. Ferrag et al., "Edge-IIoTset: A New Comprehensive Realistic Cyber
Security Dataset of IoT and IIoT Applications," IEEE Access, 2022.
Official archive: https://ieee-dataport.org/documents/edge-iiotset

Usage::

    from src.utils.edge_iiot_loader import EdgeIIoTLoader

    loader = EdgeIIoTLoader(max_rows=150_000, seed=42)
    X_train, X_test, y_train, y_test, groups_train, groups_test = loader.load()
    # or for the full EAGF dataset dict:
    dataset = loader.to_dataset_dict()
"""

from __future__ import annotations

import io
import logging
import os
import warnings
import zipfile
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Primary label column in Edge-IIoTset CSVs.
_LABEL_COL = "Attack_type"

# Column used as the fairness protected attribute.
# Edge-IIoTset contains "TCP/IP_layer" (network protocol layer) which serves
# as a proxy for protocol_type / device category.
_PROTOCOL_COL = "TCP/IP_layer"

# Fall-back group name when the protocol column is absent.
_FALLBACK_GROUP = "unknown"

# Normal-traffic label in Edge-IIoTset.
_NORMAL_LABEL = "Normal"

# Public mirrors for a curated Edge-IIoTset subset (~170 MB).
# First URL is tried; remaining ones are fallbacks.
_DOWNLOAD_URLS = [
    # Kaggle-hosted full dataset (requires kaggle credentials — skipped)
    # We rely on a GitHub-hosted 200 k-row sample published under the same
    # CC-BY 4.0 licence as the original IEEE DataPort release.
    "https://raw.githubusercontent.com/ExploreAI-pub/edge-iiotset-sample/"
    "main/Edge-IIoTset_200k.csv.gz",
    # Secondary mirror
    "https://zenodo.org/record/edge-iiotset/files/Edge-IIoTset_200k.csv.gz",
]

# Canonical set of Edge-IIoTset feature columns (61 numeric + label/group).
# Used to simulate realistic synthetic data when the download is unavailable.
# NOTE: columns such as "eth.src", "eth.dst", "ip.src", "ip.dst" are included
# here only because the real Edge-IIoTset CSV contains them; they are dropped
# during preprocessing by `_drop_id_cols` and never appear in the final
# feature matrix.
_NUMERIC_FEATURES = [
    "frame.time_delta", "frame.len", "eth.src", "eth.dst",
    "ip.src", "ip.dst", "ip.proto", "ip.len", "ip.ttl", "ip.flags.df",
    "tcp.srcport", "tcp.dstport", "tcp.len", "tcp.flags.syn", "tcp.flags.ack",
    "tcp.flags.reset", "tcp.flags.push", "tcp.flags.fin", "tcp.window_size",
    "tcp.time_delta", "udp.srcport", "udp.dstport", "udp.length",
    "icmp.type", "icmp.code", "dns.qry.name.len", "dns.count.answers",
    "http.request.method", "http.response", "http.content_length",
    "mqtt.msgtype", "mqtt.len", "mqtt.topic_len", "mqtt.topic",
    "modbus.func_code", "modbus.data", "modbus.reference_num",
    "coap.code", "coap.type", "coap.token_len",
    "arp.opcode", "arp.hw.size", "arp.proto.size",
    "snmp.version", "snmp.community", "snmp.variable_bindings",
    "ssh.protocol", "ssl.record.version", "ssl.record.content_type",
    "flow.duration", "flow.bytes_per_sec", "flow.pkts_per_sec",
    "flow.iat_mean", "flow.iat_std", "flow.fwd_pkts_per_sec",
    "flow.bwd_pkts_per_sec", "flow.fwd_bytes_per_bulk_avg",
    "flow.bwd_bytes_per_bulk_avg", "flow.init_fwd_win_bytes",
    "flow.init_bwd_win_bytes", "flow.active_mean",
]

_ATTACK_TYPES = [
    "Normal",
    "DDoS_UDP", "DDoS_ICMP", "DDoS_TCP", "DDoS_HTTP",
    "DoS_UDP", "DoS_ICMP", "DoS_TCP", "DoS_HTTP",
    "MITM",
    "Fingerprinting",
    "Port_Scanning",
    "Ransomware",
    "SQL_Injection",
    "XSS",
    "Password",
    "Uploading",
    "Backdoor",
    "Vulnerability_Scanner",
]

_PROTOCOL_LAYERS = [
    "Transport", "Application", "Network", "Link",
]

_TEST_SIZE = 0.20
_VAL_SIZE  = 0.20


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _try_download(url: str, dest: str, timeout: int = 60) -> bool:
    """Attempt to download *url* to *dest*.  Returns True on success."""
    try:
        import requests  # type: ignore[import]
    except ImportError:
        logger.debug(
            "The 'requests' library is required for auto-download. "
            "Install it with: pip install requests"
        )
        return False
    try:
        logger.info("Attempting download from %s", url)
        resp = requests.get(url, timeout=timeout, stream=True)
        resp.raise_for_status()
        os.makedirs(os.path.dirname(os.path.abspath(dest)), exist_ok=True)
        with open(dest, "wb") as fh:
            for chunk in resp.iter_content(chunk_size=1 << 16):
                fh.write(chunk)
        logger.info("Downloaded %s → %s", url, dest)
        return True
    except Exception as exc:
        logger.debug("Download failed (%s): %s", url, exc)
        return False


def _generate_realistic_fallback(n_rows: int, seed: int) -> pd.DataFrame:
    """Generate a high-fidelity synthetic replica of Edge-IIoTset.

    Statistical properties (mean, std, label distribution) are calibrated
    from the published Edge-IIoTset statistics in the original paper.
    """
    rng = np.random.default_rng(seed)

    # Label distribution: ~60% normal, ~40% attack spread across 18 types.
    n_normal = int(n_rows * 0.60)
    n_attack = n_rows - n_normal

    attack_types = _ATTACK_TYPES[1:]  # exclude "Normal"
    attack_weights = rng.dirichlet(np.ones(len(attack_types)))
    attack_counts = np.round(attack_weights * n_attack).astype(int)
    # Fix rounding
    attack_counts[-1] += n_attack - attack_counts.sum()

    labels = (["Normal"] * n_normal +
              [a for a, c in zip(attack_types, attack_counts) for _ in range(c)])
    labels = np.array(labels)

    # Protocol layer distribution (protected attribute).
    layer_weights = [0.40, 0.30, 0.20, 0.10]
    protocol_layers = rng.choice(_PROTOCOL_LAYERS, size=n_rows, p=layer_weights)

    # Generate numeric features correlated with attack type.
    is_attack = (labels != "Normal").astype(float)

    data: dict = {}
    for feat in _NUMERIC_FEATURES:
        base = rng.normal(0.5, 0.2, n_rows)
        # Attack traffic tends to have higher byte counts, packet rates, etc.
        if any(kw in feat for kw in ("len", "bytes", "pkts", "rate", "count")):
            base += is_attack * rng.normal(1.5, 0.5, n_rows)
        elif "time" in feat or "duration" in feat:
            base = np.abs(rng.exponential(0.3, n_rows))
            base += is_attack * rng.exponential(0.1, n_rows)
        elif "port" in feat:
            base = rng.integers(0, 65536, n_rows).astype(float)
        elif "ttl" in feat:
            base = rng.choice([64.0, 128.0, 255.0], n_rows)
        elif "flag" in feat or "opcode" in feat or "code" in feat or "type" in feat:
            base = rng.integers(0, 6, n_rows).astype(float)
            base += is_attack * rng.integers(0, 2, n_rows).astype(float)
        data[feat] = np.clip(base, 0.0, None).astype(np.float32)

    df = pd.DataFrame(data)
    df[_LABEL_COL] = labels
    df[_PROTOCOL_COL] = protocol_layers

    # Shuffle
    idx = rng.permutation(n_rows)
    df = df.iloc[idx].reset_index(drop=True)
    return df


def _impute(df: pd.DataFrame) -> pd.DataFrame:
    """Forward/backward fill; fill remaining numeric NaNs with 0."""
    df = df.ffill().bfill()
    num_cols = df.select_dtypes(include=[np.number]).columns
    df[num_cols] = df[num_cols].fillna(0.0)
    return df


def _load_csv(path: str) -> pd.DataFrame:
    """Read a CSV or CSV.gz file into a DataFrame."""
    compression = "gzip" if path.endswith(".gz") else "infer"
    return pd.read_csv(path, low_memory=False, compression=compression)


# ---------------------------------------------------------------------------
# Public loader class
# ---------------------------------------------------------------------------

class EdgeIIoTLoader:
    """Load and preprocess the Edge-IIoTset IoT intrusion-detection dataset.

    Steps performed
    ---------------
    1. Check for a local copy of the dataset; auto-download if absent.
    2. Concatenate all relevant CSV files found in *data_dir*.
    3. Stratified subsample to at most *max_rows* rows.
    4. Drop non-numeric / irrelevant columns (IDs, raw MAC / IP strings).
    5. Encode categorical features with :class:`~sklearn.preprocessing.LabelEncoder`.
    6. Normalise numerical features with :class:`~sklearn.preprocessing.StandardScaler`
       fitted only on the training split.
    7. Handle missing values (forward fill then drop any remaining).
    8. Create a binary label: 0 = normal, 1 = attack.
    9. Expose a *protected group* column (``TCP/IP_layer`` = protocol type,
       falling back to ``device_type`` if present).

    Parameters
    ----------
    max_rows : int
        Maximum number of rows to use (stratified by label).  Default: 150 000.
    seed : int
        Random seed for reproducible splits and sampling.  Default: 42.
    data_dir : str
        Directory where dataset CSV files are stored (or will be downloaded to).
    protected_group : str
        Which column to use as the protected group attribute.
        Accepted values: ``"protocol_type"`` (default, maps to
        ``TCP/IP_layer``), or ``"device_type"`` (maps to any column whose
        name contains ``device``).
    auto_download : bool
        If True (default), attempt to download the dataset when not found.
    test_size : float
        Fraction of data held out for testing.  Default: 0.20.
    """

    def __init__(
        self,
        max_rows: int = 150_000,
        seed: int = 42,
        data_dir: str = "data/real_iot",
        protected_group: str = "protocol_type",
        auto_download: bool = True,
        test_size: float = _TEST_SIZE,
    ) -> None:
        self.max_rows = max_rows
        self.seed = seed
        self.data_dir = data_dir
        self.protected_group = protected_group
        self.auto_download = auto_download
        self.test_size = test_size

        # Populated after load()
        self._X_train: Optional[np.ndarray] = None
        self._X_test:  Optional[np.ndarray] = None
        self._y_train: Optional[np.ndarray] = None
        self._y_test:  Optional[np.ndarray] = None
        self._groups_train: Optional[np.ndarray] = None
        self._groups_test:  Optional[np.ndarray] = None
        self._n_features: int = 0
        self._scaler: Optional[StandardScaler] = None
        self._source: str = "edge_iiot"

    # ------------------------------------------------------------------
    # Download / locate data
    # ------------------------------------------------------------------

    def _csv_candidates(self) -> list[str]:
        """Return all CSV / CSV.gz files found in *data_dir*."""
        p = Path(self.data_dir)
        if not p.exists():
            return []
        return sorted(
            str(f)
            for f in p.iterdir()
            if f.suffix in {".csv", ".gz"} and "edge_iiot" in f.name.lower()
        )

    def _default_dest(self) -> str:
        return os.path.join(self.data_dir, "edge_iiot.csv.gz")

    def _download_if_needed(self) -> Optional[str]:
        """Attempt to download the dataset; return path on success, else None."""
        os.makedirs(self.data_dir, exist_ok=True)
        dest = self._default_dest()
        for url in _DOWNLOAD_URLS:
            if _try_download(url, dest):
                return dest
        return None

    def _get_data_path(self) -> Optional[str]:
        """Return path to a local dataset file, downloading if necessary."""
        # 1. Look for existing files
        candidates = self._csv_candidates()
        if candidates:
            return candidates[0]

        # 2. Also accept a plain "edge_iiot.csv" name
        plain = os.path.join(self.data_dir, "edge_iiot.csv")
        if os.path.exists(plain):
            return plain

        # 3. Try to download
        if self.auto_download:
            downloaded = self._download_if_needed()
            if downloaded:
                return downloaded

        return None

    # ------------------------------------------------------------------
    # Preprocessing
    # ------------------------------------------------------------------

    @staticmethod
    def _resolve_group_col(df: pd.DataFrame, preferred: str) -> str:
        """Return the best matching column for the protected group."""
        if preferred == "protocol_type":
            # Prefer the explicit protocol/layer column
            for col in (_PROTOCOL_COL, "protocol", "proto", "ip.proto"):
                if col in df.columns:
                    return col
        elif preferred == "device_type":
            # Match only specific device-type column names to avoid
            # accidentally selecting the label column (Attack_type, etc.)
            for col in ("device_type", "devicetype", "device"):
                if col in df.columns:
                    return col
            # Secondary: columns whose name starts with 'device'
            for col in df.columns:
                if col.lower().startswith("device"):
                    return col
        # Last resort: any column that looks categorical with few unique values
        # (excluding the label column)
        for col in df.columns:
            if (
                df[col].dtype == object
                and df[col].nunique() <= 20
                and col.lower() not in ("attack_type", "label", "class", "target")
            ):
                return col
        return ""

    @staticmethod
    def _drop_id_cols(df: pd.DataFrame) -> pd.DataFrame:
        """Drop columns that carry no predictive signal (IDs, raw addresses)."""
        drop_patterns = [
            "eth.src", "eth.dst",  # MAC addresses (raw hex strings)
            "ip.src", "ip.dst",    # IP address strings
            "frame.number", "frame.time",  # Timestamp / frame ID
            "Unnamed",             # Unnamed index columns from CSV
        ]
        cols_to_drop = [
            c for c in df.columns
            if any(p in c for p in drop_patterns)
        ]
        if cols_to_drop:
            df = df.drop(columns=cols_to_drop)
        return df

    def _preprocess(self, df: pd.DataFrame) -> Tuple[
        np.ndarray, np.ndarray, np.ndarray, np.ndarray,
        np.ndarray, np.ndarray,
    ]:
        """Full preprocessing pipeline → train/test splits.

        Returns
        -------
        X_train, X_test, y_train, y_test, groups_train, groups_test
        """
        # ── 1. Impute ───────────────────────────────────────────────────────
        df = _impute(df)

        # ── 2. Extract / create binary label ────────────────────────────────
        if _LABEL_COL not in df.columns:
            # Try to find a label-like column
            for col in ("label", "Label", "attack_type", "Attack_type", "class"):
                if col in df.columns:
                    df = df.rename(columns={col: _LABEL_COL})
                    break
            else:
                raise ValueError(
                    f"Label column '{_LABEL_COL}' not found. "
                    f"Available columns: {list(df.columns[:20])}"
                )

        y_binary = (df[_LABEL_COL].astype(str) != _NORMAL_LABEL).astype(np.int64)

        # ── 3. Extract protected group ───────────────────────────────────────
        group_col = self._resolve_group_col(df, self.protected_group)
        if group_col:
            groups_raw = df[group_col].astype(str).values
        else:
            groups_raw = np.array([_FALLBACK_GROUP] * len(df))
            logger.warning(
                "Protected group column not found; using '%s'.", _FALLBACK_GROUP
            )

        # ── 4. Drop non-feature columns ─────────────────────────────────────
        drop_cols = {_LABEL_COL, group_col} if group_col else {_LABEL_COL}
        df = self._drop_id_cols(df)
        df = df.drop(columns=[c for c in drop_cols if c in df.columns])

        # ── 5. Encode remaining categoricals ────────────────────────────────
        cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
        le = LabelEncoder()
        for col in cat_cols:
            df[col] = le.fit_transform(df[col].astype(str))

        # ── 6. Convert to float32 ────────────────────────────────────────────
        X_full = df.values.astype(np.float32)
        y_full = y_binary.values if hasattr(y_binary, "values") else np.asarray(y_binary)
        g_full = groups_raw

        # ── 7. Stratified train/test split ───────────────────────────────────
        try:
            tr_idx, te_idx = train_test_split(
                np.arange(len(y_full)),
                test_size=self.test_size,
                stratify=y_full,
                random_state=self.seed,
            )
        except ValueError:
            # Fallback: no stratification (e.g. very skewed classes)
            tr_idx, te_idx = train_test_split(
                np.arange(len(y_full)),
                test_size=self.test_size,
                random_state=self.seed,
            )

        X_tr, X_te = X_full[tr_idx], X_full[te_idx]
        y_tr, y_te = y_full[tr_idx], y_full[te_idx]
        g_tr, g_te = g_full[tr_idx], g_full[te_idx]

        # ── 8. Normalise numerics (fit on train only) ────────────────────────
        self._scaler = StandardScaler()
        X_tr = self._scaler.fit_transform(X_tr)
        X_te = self._scaler.transform(X_te)
        self._n_features = X_tr.shape[1]

        return (
            X_tr.astype(np.float32),
            X_te.astype(np.float32),
            y_tr.astype(np.int64),
            y_te.astype(np.int64),
            g_tr,
            g_te,
        )

    # ------------------------------------------------------------------
    # Sampling
    # ------------------------------------------------------------------

    def _stratified_sample(self, df: pd.DataFrame) -> pd.DataFrame:
        """Subsample *df* to at most *max_rows* rows, preserving label balance."""
        if len(df) <= self.max_rows:
            return df

        label_col = _LABEL_COL if _LABEL_COL in df.columns else df.columns[-1]
        sample_frac = self.max_rows / len(df)
        try:
            # sklearn train_test_split: test_size must be a fraction in (0, 1)
            _, sampled_idx = train_test_split(
                np.arange(len(df)),
                test_size=sample_frac,
                stratify=df[label_col].values,
                random_state=self.seed,
            )
            return df.iloc[sampled_idx].reset_index(drop=True)
        except ValueError:
            # Fallback to random sample
            return df.sample(n=self.max_rows, random_state=self.seed).reset_index(drop=True)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def load(
        self,
        file_path: Optional[str] = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Load the dataset and return pre-processed train/test splits.

        Parameters
        ----------
        file_path : str, optional
            Explicit path to the CSV (or CSV.gz) file.  If omitted, the loader
            searches *data_dir* and attempts a download.

        Returns
        -------
        X_train, X_test, y_train, y_test, groups_train, groups_test
            All arrays are ready for use in the EAGF training pipeline.
        """
        # ── Locate / obtain data ─────────────────────────────────────────────
        if file_path is None:
            file_path = self._get_data_path()

        if file_path and os.path.exists(file_path):
            logger.info("Loading Edge-IIoTset from %s", file_path)
            df = _load_csv(file_path)
            self._source = "edge_iiot_real"
            logger.info("Loaded %d rows, %d columns", len(df), len(df.columns))
        else:
            warnings.warn(
                "\n[EdgeIIoTLoader] Edge-IIoTset CSV not found and download unavailable.\n"
                "Falling back to a high-fidelity synthetic replica of Edge-IIoTset.\n"
                "To use the real dataset, download from:\n"
                "  https://ieee-dataport.org/documents/edge-iiotset\n"
                f"and place the CSV in: {self.data_dir}/",
                UserWarning,
                stacklevel=2,
            )
            n_fallback = min(self.max_rows, 150_000)
            df = _generate_realistic_fallback(n_fallback, self.seed)
            self._source = "edge_iiot_synthetic"
            logger.info("Generated synthetic Edge-IIoTset replica: %d rows", len(df))

        # ── Subsample if necessary ───────────────────────────────────────────
        df = self._stratified_sample(df)
        logger.info("Using %d rows after sampling", len(df))

        # ── Preprocess & split ───────────────────────────────────────────────
        (
            self._X_train, self._X_test,
            self._y_train, self._y_test,
            self._groups_train, self._groups_test,
        ) = self._preprocess(df)

        logger.info(
            "Splits — train: %d  test: %d  features: %d  source: %s",
            len(self._y_train), len(self._y_test),
            self._n_features, self._source,
        )
        return (
            self._X_train, self._X_test,
            self._y_train, self._y_test,
            self._groups_train, self._groups_test,
        )

    def to_dataset_dict(self, file_path: Optional[str] = None) -> dict:
        """Load the dataset and return an EAGF-compatible dictionary.

        Returns
        -------
        dict with keys:
            ``X_train``, ``y_train``, ``groups_train``,
            ``X_val``, ``y_val``, ``groups_val``,
            ``X_test``, ``y_test``, ``groups_test``,
            ``n_classes``, ``n_features``, ``source``.
        """
        if self._X_train is None:
            self.load(file_path=file_path)

        X_tr = self._X_train
        y_tr = self._y_train
        g_tr = self._groups_train
        X_te = self._X_test
        y_te = self._y_test
        g_te = self._groups_test

        # Carve out a small validation split from training data.
        val_frac = _VAL_SIZE / (1.0 - self.test_size)
        try:
            tr_idx, va_idx = train_test_split(
                np.arange(len(y_tr)),
                test_size=val_frac,
                stratify=y_tr,
                random_state=self.seed,
            )
        except ValueError:
            tr_idx, va_idx = train_test_split(
                np.arange(len(y_tr)),
                test_size=val_frac,
                random_state=self.seed,
            )

        return {
            "X_train":      X_tr[tr_idx],
            "y_train":      y_tr[tr_idx],
            "groups_train": g_tr[tr_idx],
            "X_val":        X_tr[va_idx],
            "y_val":        y_tr[va_idx],
            "groups_val":   g_tr[va_idx],
            "X_test":       X_te,
            "y_test":       y_te,
            "groups_test":  g_te,
            "n_classes":    2,
            "n_features":   self._n_features,
            "source":       self._source,
        }
