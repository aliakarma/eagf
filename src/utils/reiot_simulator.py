"""
src/utils/reiot_simulator.py
Synthetic 5G RE-IoT Telemetry Generator

Generates synthetic power telemetry for 120 virtual IoT nodes:
  - 40 urban  (±2 % load variation, CVSS-rated attack scenarios)
  - 40 peri-urban (±8 %)
  - 40 rural  (±22 %)

Attack types injected at configurable overall ratio:
  - FDIA              (corrupts voltage/current readings)
  - command_injection (inverter switching frequency spike)
  - dos               (gaps in telemetry stream)

Paper reference: Section 5.2 (Case Study 2 — RE-IoT Setup)
"""

import json
import os
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np


# ── Node class profiles (from Liang et al. 2017) ──────────────────────────
NODE_PROFILES = {
    "urban":      {"load_std": 0.02, "base_voltage": 230.0, "base_current": 10.0},
    "periurban":  {"load_std": 0.08, "base_voltage": 220.0, "base_current":  8.0},
    "rural":      {"load_std": 0.22, "base_voltage": 210.0, "base_current":  5.0},
}

FEATURE_NAMES = [
    "voltage", "current", "frequency_deviation",
    "state_of_charge", "inverter_switching_freq",
]

N_FEATURES = len(FEATURE_NAMES)


def _generate_normal_window(
    node_class: str,
    seq_len: int,
    rng: np.random.RandomState,
) -> np.ndarray:
    """Generate one normal telemetry window for a node class."""
    profile = NODE_PROFILES[node_class]
    std = profile["load_std"]

    voltage   = profile["base_voltage"] * (1 + rng.randn(seq_len) * std)
    current   = profile["base_current"] * (1 + rng.randn(seq_len) * std)
    freq_dev  = rng.randn(seq_len) * 0.05
    soc       = np.clip(0.6 + rng.randn(seq_len) * 0.05, 0.0, 1.0)
    inv_freq  = 50.0 + rng.randn(seq_len) * 0.5

    return np.column_stack([voltage, current, freq_dev, soc, inv_freq])


def _inject_fdia(window: np.ndarray, rng: np.random.RandomState) -> np.ndarray:
    """False Data Injection: bias voltage/current readings."""
    w = window.copy()
    bias = rng.uniform(0.15, 0.40)
    start = rng.randint(0, len(w) // 2)
    w[start:, 0] *= (1 + bias)   # voltage
    w[start:, 1] *= (1 - bias)   # current (inverse)
    return w


def _inject_command_injection(window: np.ndarray, rng: np.random.RandomState) -> np.ndarray:
    """Command injection: inverter switching frequency spike."""
    w = window.copy()
    spike_start = rng.randint(len(w) // 3, 2 * len(w) // 3)
    spike_len = rng.randint(3, 10)
    w[spike_start:spike_start + spike_len, 4] *= rng.uniform(3.0, 6.0)
    return w


def _inject_dos(window: np.ndarray, rng: np.random.RandomState) -> np.ndarray:
    """DoS: zero-out telemetry segments (suppressed data)."""
    w = window.copy()
    gap_start = rng.randint(0, len(w) // 2)
    gap_len = rng.randint(5, len(w) // 3)
    w[gap_start:gap_start + gap_len, :] = 0.0
    return w


ATTACK_INJECTORS = {
    "fdia": _inject_fdia,
    "command_injection": _inject_command_injection,
    "dos": _inject_dos,
}


def inject_network_faults(
    data: Dict,
    missing_rate: float = 0.20,
    burst_size: int = 5,
    seed: int = 42,
    apply_to_test: bool = False,
) -> Dict:
    """Inject simulated network faults (burst packet loss) into RE-IoT telemetry.

    Randomly selects ``missing_rate`` fraction of time windows and replaces
    ``burst_size`` consecutive time steps with ``NaN``, simulating real IoT
    packet loss or sensor dropout in the field.

    Args:
        data: Dataset dict from :func:`generate_full_reiot_dataset` or a raw
              ``np.ndarray``.  When a dict is supplied, ``X_train`` (and
              optionally ``X_test``) must have shape
              ``(n_samples, seq_len * n_features)`` — the flattened window
              format used throughout the EAGF pipeline.
        missing_rate: Fraction of windows to corrupt with burst packet loss.
                      Must be in [0, 1].  Default: 0.20.
        burst_size: Number of consecutive time steps to set to NaN per fault
                    event.  Default: 5.
        seed: Random seed for reproducibility.
        apply_to_test: If ``True``, also corrupt ``X_test`` (useful for
                       evaluating model robustness under inference-time faults).

    Returns:
        Modified dataset dict (or array) with NaN values in corrupted windows.
        All other keys are returned unchanged (deep-copied for safety).

    Raises:
        ValueError: If ``missing_rate`` is not in [0, 1].
    """
    if not (0.0 <= missing_rate <= 1.0):
        raise ValueError(
            f"missing_rate must be in [0, 1], got {missing_rate}."
        )

    rng = np.random.RandomState(seed)

    # Accept either a raw array or a dataset dict.
    if isinstance(data, np.ndarray):
        return _apply_burst_faults(data, missing_rate, burst_size, rng)

    result = {
        k: (v.copy() if isinstance(v, np.ndarray) else v)
        for k, v in data.items()
    }
    splits = ["X_train"] + (["X_test"] if apply_to_test else [])
    for split in splits:
        if split not in result or result[split] is None:
            continue
        result[split] = _apply_burst_faults(
            result[split], missing_rate, burst_size, rng
        )
    return result


def _apply_burst_faults(
    X: np.ndarray,
    missing_rate: float,
    burst_size: int,
    rng: np.random.RandomState,
) -> np.ndarray:
    """Apply burst-fault NaN injection to a 2-D feature array.

    Args:
        X: Float array of shape ``(n_samples, n_flat)`` where
           ``n_flat = seq_len * n_features``.
        missing_rate: Fraction of rows to corrupt.
        burst_size: Consecutive time steps to zero out per fault.
        rng: Seeded random state.

    Returns:
        Float32 array with NaN values injected into selected windows.
    """
    n_samples, n_flat = X.shape
    # Infer seq_len; fall back to treating whole row as single time step.
    seq_len = max(n_flat // N_FEATURES, 1)
    effective_burst = min(burst_size, seq_len)

    n_fault = int(np.round(n_samples * missing_rate))
    if n_fault == 0:
        return X.astype(np.float32)

    fault_indices = rng.choice(n_samples, size=n_fault, replace=False)

    # Work in float64 so NaN is representable; reshape to (N, T, F).
    X_out = X.astype(np.float64)
    X_3d = X_out.reshape(n_samples, seq_len, N_FEATURES)

    for idx in fault_indices:
        max_start = max(seq_len - effective_burst, 0)
        t_start = rng.randint(0, max_start + 1)
        X_3d[idx, t_start: t_start + effective_burst, :] = np.nan

    return X_3d.reshape(n_samples, n_flat).astype(np.float32)


def generate_node_dataset(
    node_class: str,
    n_nodes: int,
    n_windows_per_node: int = 100,
    seq_len: int = 60,
    attack_ratio: float = 0.05,
    attacks: Optional[List[str]] = None,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate dataset for a specific node class.

    Args:
        node_class: 'urban', 'periurban', or 'rural'.
        n_nodes: Number of virtual nodes.
        n_windows_per_node: Telemetry windows per node.
        seq_len: Time steps per window.
        attack_ratio: Fraction of windows containing attacks.
        attacks: List of attack type strings.
        seed: Random seed.

    Returns:
        X: (n_samples, seq_len, n_features) feature array
        y: (n_samples,) binary labels (0=normal, 1=attack)
        node_ids: (n_samples,) node index array
    """
    if attacks is None:
        attacks = ["fdia", "command_injection", "dos"]

    rng = np.random.RandomState(seed)
    X_list, y_list, node_list = [], [], []

    for node_id in range(n_nodes):
        for _ in range(n_windows_per_node):
            is_attack = rng.rand() < attack_ratio
            window = _generate_normal_window(node_class, seq_len, rng)

            if is_attack:
                attack_type = attacks[rng.randint(len(attacks))]
                injector = ATTACK_INJECTORS.get(attack_type, _inject_fdia)
                window = injector(window, rng)

            X_list.append(window)
            y_list.append(int(is_attack))
            node_list.append(node_id)

    return (
        np.array(X_list, dtype=np.float32),
        np.array(y_list, dtype=np.int32),
        np.array(node_list, dtype=np.int32),
    )


def generate_full_reiot_dataset(
    n_urban: int = 40,
    n_periurban: int = 40,
    n_rural: int = 40,
    n_windows_per_node: int = 100,
    seq_len: int = 60,
    attack_ratio: float = 0.05,
    attacks: Optional[List[str]] = None,
    node_split: Tuple[float, float] = (0.80, 0.20),
    seed: int = 42,
    output_dir: Optional[str] = None,
) -> Dict:
    """Generate and optionally save the complete RE-IoT dataset.

    Returns:
        Dictionary with keys:
          'X_train', 'y_train', 'groups_train',
          'X_test',  'y_test',  'groups_test',
          'node_classes_train', 'node_classes_test',
          'feature_names', 'attack_ratio', 'metadata'
    """
    if attacks is None:
        attacks = ["fdia", "command_injection", "dos"]

    all_X, all_y, all_groups, all_classes = [], [], [], []
    class_offset = 0

    # Training data imbalance: urban nodes have 3x more samples (deployment bias)
    # This creates the node-class FPR disparity the fairness pillar corrects.
    windows_per_class = {
        "urban":     n_windows_per_node * 3,  # over-represented in training
        "periurban": n_windows_per_node,
        "rural":     n_windows_per_node // 2, # under-represented in training
    }
    for cls_name, n_nodes in [
        ("urban", n_urban),
        ("periurban", n_periurban),
        ("rural", n_rural),
    ]:
        X, y, node_ids = generate_node_dataset(
            node_class=cls_name,
            n_nodes=n_nodes,
            n_windows_per_node=windows_per_class[cls_name],
            seq_len=seq_len,
            attack_ratio=attack_ratio,
            attacks=attacks,
            seed=seed + class_offset,
        )
        all_X.append(X)
        all_y.append(y)
        all_groups.append(node_ids + class_offset)
        all_classes.extend([cls_name] * len(y))
        class_offset += n_nodes

    X_all = np.concatenate(all_X, axis=0)
    y_all = np.concatenate(all_y, axis=0)
    groups_all = np.concatenate(all_groups, axis=0)
    classes_all = np.array(all_classes)

    # Flatten windows: (N, seq_len * n_features) for sklearn compatibility
    N = X_all.shape[0]
    X_flat = X_all.reshape(N, -1)

    # Node-level stratified split
    rng = np.random.RandomState(seed)
    unique_nodes = np.unique(groups_all)
    rng.shuffle(unique_nodes)
    split_idx = int(len(unique_nodes) * node_split[0])
    train_nodes = set(unique_nodes[:split_idx])
    test_nodes  = set(unique_nodes[split_idx:])

    train_mask = np.array([g in train_nodes for g in groups_all])
    test_mask  = ~train_mask

    dataset = {
        "X_train": X_flat[train_mask],
        "y_train": y_all[train_mask],
        "groups_train": classes_all[train_mask],
        "X_test":  X_flat[test_mask],
        "y_test":  y_all[test_mask],
        "groups_test": classes_all[test_mask],
        "feature_names": FEATURE_NAMES,
        "seq_len": seq_len,
        "n_features": N_FEATURES,
        "attack_ratio": attack_ratio,
        "metadata": {
            "n_urban": n_urban,
            "n_periurban": n_periurban,
            "n_rural": n_rural,
            "n_windows_per_node": n_windows_per_node,
            "attacks": attacks,
            "seed": seed,
        },
    }

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        np.save(os.path.join(output_dir, "X_train.npy"), dataset["X_train"])
        np.save(os.path.join(output_dir, "y_train.npy"), dataset["y_train"])
        np.save(os.path.join(output_dir, "X_test.npy"),  dataset["X_test"])
        np.save(os.path.join(output_dir, "y_test.npy"),  dataset["y_test"])
        np.save(os.path.join(output_dir, "groups_train.npy"), dataset["groups_train"].astype(str))
        np.save(os.path.join(output_dir, "groups_test.npy"),  dataset["groups_test"].astype(str))
        with open(os.path.join(output_dir, "metadata.json"), "w") as f:
            json.dump(dataset["metadata"], f, indent=2)
        print(f"RE-IoT dataset saved to: {output_dir}")
        print(f"  Train: {dataset['X_train'].shape[0]} samples")
        print(f"  Test:  {dataset['X_test'].shape[0]} samples")

    return dataset
