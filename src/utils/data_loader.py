"""
src/utils/data_loader.py
EAGF Dataset Loading — with intentional demographic bias in demo data
"""
import argparse, json, os, random
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def generate_demo_biometric(n_samples=2000, n_features=20, n_classes=2, seed=42):
    """Generate demographically biased binary facial-verification dataset.

    Simulates the ACTUAL facial recognition task: binary verification
    (is this person enrolled? 1=yes, 0=no). Groups have different noise
    levels producing intentional recall disparity:
      male_light  (noise=0.8): easiest  -> high recall
      female_dark (noise=2.5): hardest  -> low recall (RP < 1 for baseline)

    This ensures the fairness pillar has measurable work to do and that
    EAGF's recall-parity regularisation produces a visible improvement.

    Args:
        n_samples: Approximate total samples (split evenly across 4 groups).
        n_features: Face embedding dimension (default 20 for speed).
        n_classes: Fixed at 2 (binary: enrolled=1, not-enrolled=0).
        seed: Random seed.

    Returns:
        Dataset dictionary with train/val/test splits and demographic labels.
    """
    rng = np.random.RandomState(seed)
    group_noise = {
        "male_light":   0.8,   # most accurate: luminance-channel advantage
        "male_dark":    1.5,
        "female_light": 1.2,
        "female_dark":  2.5,   # highest noise: luminance-channel disadvantage
    }
    n_per_group = max(n_samples // (len(group_noise) * 2), 50)

    centroid = rng.randn(n_features) * 2.0  # shared enrolled-person centroid

    X_parts, y_parts, g_parts = [], [], []
    for gname, gnoise in group_noise.items():
        # Positive: genuine enrolled-person samples (hard to separate at high noise)
        X_pos = centroid + rng.randn(n_per_group, n_features) * gnoise
        # Negative: impostors (random vectors, easy to separate)
        X_neg = rng.randn(n_per_group, n_features) * 2.5
        X_parts.extend([X_pos.astype(np.float32), X_neg.astype(np.float32)])
        y_parts.extend([np.ones(n_per_group, int), np.zeros(n_per_group, int)])
        g_parts.extend([gname] * n_per_group * 2)

    X_all = np.vstack(X_parts)
    y_all = np.concatenate(y_parts)
    g_all = np.array(g_parts)

    scaler = StandardScaler()
    X_all = scaler.fit_transform(X_all).astype(np.float32)

    idx = np.arange(len(X_all))
    tr_idx, tmp_idx = train_test_split(idx, test_size=0.30, stratify=y_all, random_state=seed)
    va_idx, te_idx  = train_test_split(tmp_idx, test_size=0.50,
                                        stratify=y_all[tmp_idx], random_state=seed)

    return {
        "X_train": X_all[tr_idx], "y_train": y_all[tr_idx], "groups_train": g_all[tr_idx],
        "X_val":   X_all[va_idx], "y_val":   y_all[va_idx], "groups_val":   g_all[va_idx],
        "X_test":  X_all[te_idx], "y_test":  y_all[te_idx], "groups_test":  g_all[te_idx],
        "n_classes": 2, "n_features": n_features, "scaler": scaler,
        "source": "demo_synthetic",
        "splits": {"train": tr_idx.tolist(), "val": va_idx.tolist(), "test": te_idx.tolist()},
    }


def load_efr_dataset(data_root, splits_file=None, seed=42):
    X_train_path = os.path.join(data_root, "X_train.npy")
    if not os.path.exists(X_train_path):
        return None
    d = {}
    for split in ("train", "val", "test"):
        xp = os.path.join(data_root, f"X_{split}.npy")
        yp = os.path.join(data_root, f"y_{split}.npy")
        gp = os.path.join(data_root, f"groups_{split}.npy")
        if not os.path.exists(xp):
            continue
        d[f"X_{split}"] = np.load(xp)
        d[f"y_{split}"] = np.load(yp)
        d[f"groups_{split}"] = np.load(gp) if os.path.exists(gp) else None
    meta = os.path.join(data_root, "metadata.json")
    if os.path.exists(meta):
        with open(meta) as f:
            m = json.load(f)
        d["n_classes"] = m.get("n_classes", int(d["y_train"].max()) + 1)
    else:
        d["n_classes"] = int(d["y_train"].max()) + 1
    d["n_features"] = d["X_train"].shape[1]
    d["source"] = "efr"
    return d


def load_biometric_dataset(data_root="data/biometric/efr_processed",
                            demo=False, n_samples=2000, seed=42):
    if not demo:
        dataset = load_efr_dataset(data_root, seed=seed)
        if dataset is not None:
            print(f"  Loaded EFR dataset from {data_root}")
            return dataset
        print(f"  EFR not found at '{data_root}'. Using synthetic demo data.")
    print(f"  Generating synthetic demo biometric dataset ({n_samples} samples)...")
    return generate_demo_biometric(n_samples=n_samples, seed=seed)


def load_adult_dataset(seed=42, test_size=0.15, val_size=0.15):
    """Load UCI Adult as a real tabular dataset option.

    Returns a dataset dictionary with the same structure expected by trainers.
    """
    adult = fetch_openml("adult", version=2, as_frame=True)
    X_df = adult.data.copy()
    y_raw = adult.target.astype(str)

    # Keep a reproducible protected-group signal for fairness evaluation.
    groups = X_df["sex"].astype(str).values if "sex" in X_df.columns else np.array(["unknown"] * len(X_df))

    X_df = pd.get_dummies(X_df, drop_first=False)
    X = X_df.astype(np.float32).values
    y = (y_raw.values == ">50K").astype(int)

    scaler = StandardScaler()
    X = scaler.fit_transform(X).astype(np.float32)

    idx = np.arange(len(X))
    tr_idx, tmp_idx = train_test_split(idx, test_size=(test_size + val_size), stratify=y, random_state=seed)
    val_ratio_in_tmp = val_size / (test_size + val_size)
    va_idx, te_idx = train_test_split(
        tmp_idx,
        test_size=(1.0 - val_ratio_in_tmp),
        stratify=y[tmp_idx],
        random_state=seed,
    )

    return {
        "X_train": X[tr_idx],
        "y_train": y[tr_idx],
        "groups_train": groups[tr_idx],
        "X_val": X[va_idx],
        "y_val": y[va_idx],
        "groups_val": groups[va_idx],
        "X_test": X[te_idx],
        "y_test": y[te_idx],
        "groups_test": groups[te_idx],
        "n_classes": 2,
        "n_features": X.shape[1],
        "scaler": scaler,
        "source": "uci_adult",
        "splits": {"train": tr_idx.tolist(), "val": va_idx.tolist(), "test": te_idx.tolist()},
    }


def load_reiot_dataset(data_root="data/reiot", n_urban=40, n_periurban=40,
                       n_rural=40, n_windows_per_node=100,
                       attack_ratio=0.05, seed=42, force_regenerate=False):
    X_path = os.path.join(data_root, "X_train.npy")
    if os.path.exists(X_path) and not force_regenerate:
        print(f"  Loading RE-IoT dataset from {data_root}...")
        d = {}
        for k in ("X_train","y_train","X_test","y_test"):
            d[k] = np.load(os.path.join(data_root, f"{k}.npy"))
        for k in ("groups_train","groups_test"):
            p = os.path.join(data_root, f"{k}.npy")
            d[k] = np.load(p, allow_pickle=True) if os.path.exists(p) else None
        meta = os.path.join(data_root, "metadata.json")
        if os.path.exists(meta):
            with open(meta) as f:
                d["metadata"] = json.load(f)
        return d
    print("  Generating RE-IoT synthetic dataset...")
    from src.utils.reiot_simulator import generate_full_reiot_dataset
    return generate_full_reiot_dataset(
        n_urban=n_urban, n_periurban=n_periurban, n_rural=n_rural,
        n_windows_per_node=n_windows_per_node,
        attack_ratio=attack_ratio, seed=seed, output_dir=data_root)


def parse_args():
    p = argparse.ArgumentParser(description="EAGF Data Loader")
    p.add_argument("--dataset", choices=["efr","reiot","demo"], required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--nodes", type=int, default=120)
    p.add_argument("--attack-ratio", type=float, default=0.05)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--n-samples", type=int, default=2000)
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output, exist_ok=True)
    if args.dataset in ("efr","demo"):
        dataset = load_biometric_dataset(
            data_root=args.output, demo=(args.dataset=="demo"),
            n_samples=args.n_samples, seed=args.seed)
        if "splits" in dataset:
            with open(os.path.join(args.output, "splits.json"), "w") as f:
                json.dump(dataset["splits"], f)
        print(f"  Train:{len(dataset['y_train'])} Val:{len(dataset['y_val'])} "
              f"Test:{len(dataset['y_test'])}")
    else:
        per = max(1, args.nodes // 3)
        dataset = load_reiot_dataset(
            data_root=args.output, n_urban=per, n_periurban=per, n_rural=per,
            attack_ratio=args.attack_ratio, seed=args.seed, force_regenerate=True)
        print(f"  Train:{len(dataset['y_train'])} Test:{len(dataset['y_test'])}")


if __name__ == "__main__":
    main()
