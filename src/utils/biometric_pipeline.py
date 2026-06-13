"""
src/utils/biometric_pipeline.py — Image Data Pipeline for CS1 Biometric Case Study

Handles loading and generating image-format data (3×224×224) for ResNet-50.
Real EFR dataset: loads from preprocessed numpy arrays.
Demo/CI mode: generates synthetic image-like tensors with demographic bias.

Paper: Section 5.1 — EFR dataset (10,021 images, 158 classes)
"""

import json
import os
from typing import Dict, Optional

import numpy as np
from sklearn.model_selection import train_test_split

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(3, 1, 1)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(3, 1, 1)


def normalize_imagenet(images: np.ndarray) -> np.ndarray:
    """Apply ImageNet channel-wise normalization to (N, 3, H, W) arrays."""
    return ((images - IMAGENET_MEAN) / IMAGENET_STD).astype(np.float32)


def generate_demo_biometric_images(
    n_samples: int = 400,
    n_classes: int = 10,
    seed: int = 42,
) -> Dict:
    """Generate synthetic image-like tensors for CS1 demo/CI mode.

    Produces (N, 3, 224, 224) float32 arrays with controlled demographic bias
    so the fairness pillar has measurable work to do. Uses low-rank spatial
    patterns per class to keep images distinguishable without real faces.

    Args:
        n_samples: Total samples (split across 4 demographic groups).
        n_classes: Number of identity classes.
        seed: Random seed.

    Returns:
        Dataset dict with 4D image arrays and demographic labels.
    """
    rng = np.random.RandomState(seed)

    group_names = ["male_light", "male_dark", "female_light", "female_dark"]
    n_per_group = max(n_samples // len(group_names), 10)
    total = n_per_group * len(group_names)
    n_classes = min(n_classes, total // 3)
    n_classes = max(n_classes, 2)

    prototypes = []
    for _ in range(n_classes):
        u = rng.randn(3, 16, 16).astype(np.float32) * 0.15
        proto = np.repeat(np.repeat(u, 14, axis=1), 14, axis=2)
        prototypes.append(proto)

    group_noise = {
        "male_light": 0.02,
        "male_dark": 0.025,
        "female_light": 0.03,
        "female_dark": 0.05,
    }
    group_shift = {
        "male_light": 0.0,
        "male_dark": 0.01,
        "female_light": -0.01,
        "female_dark": 0.02,
    }

    X_parts, y_parts, g_parts = [], [], []
    sample_idx = 0
    for gname in group_names:
        for _ in range(n_per_group):
            c = sample_idx % n_classes
            sample_idx += 1
            img = prototypes[c].copy()
            img += group_shift[gname]
            img += rng.randn(3, 224, 224).astype(np.float32) * group_noise[gname]
            img = np.clip(img + 0.5, 0.0, 1.0)
            X_parts.append(img)
            y_parts.append(c)
            g_parts.append(gname)

    X_all = np.stack(X_parts).astype(np.float32)
    y_all = np.array(y_parts, dtype=int)
    g_all = np.array(g_parts)

    flip_mask = rng.rand(len(y_all)) < 0.05
    y_all[flip_mask] = rng.randint(0, n_classes, size=flip_mask.sum())

    idx = np.arange(len(X_all))
    _, class_counts = np.unique(y_all, return_counts=True)
    can_stratify = int(class_counts.min()) >= 3
    tr_idx, tmp_idx = train_test_split(
        idx, test_size=0.30,
        stratify=y_all if can_stratify else None,
        random_state=seed,
    )
    _, tmp_counts = np.unique(y_all[tmp_idx], return_counts=True)
    can_stratify_tmp = int(tmp_counts.min()) >= 2
    va_idx, te_idx = train_test_split(
        tmp_idx, test_size=0.50,
        stratify=y_all[tmp_idx] if can_stratify_tmp else None,
        random_state=seed,
    )

    X_all = normalize_imagenet(X_all)

    return {
        "X_train": X_all[tr_idx], "y_train": y_all[tr_idx], "groups_train": g_all[tr_idx],
        "X_val": X_all[va_idx], "y_val": y_all[va_idx], "groups_val": g_all[va_idx],
        "X_test": X_all[te_idx], "y_test": y_all[te_idx], "groups_test": g_all[te_idx],
        "n_classes": n_classes,
        "n_features": 3 * 224 * 224,
        "image_shape": (3, 224, 224),
        "source": "demo_synthetic_images",
        "is_image_data": True,
    }


def load_efr_image_dataset(data_root: str, seed: int = 42) -> Optional[Dict]:
    """Load real EFR face dataset from preprocessed numpy arrays.

    Expected directory structure::

        data_root/
            X_train.npy     # (N, 3, 224, 224) float32, pixel range [0,1]
            y_train.npy     # (N,) int
            groups_train.npy
            X_val.npy
            y_val.npy
            groups_val.npy
            X_test.npy
            y_test.npy
            groups_test.npy
            metadata.json   # {"n_classes": 158, ...}

    Returns:
        Dataset dict or None if not found.
    """
    X_path = os.path.join(data_root, "X_train.npy")
    if not os.path.exists(X_path):
        return None

    d: Dict = {}
    for split in ("train", "val", "test"):
        xp = os.path.join(data_root, f"X_{split}.npy")
        yp = os.path.join(data_root, f"y_{split}.npy")
        gp = os.path.join(data_root, f"groups_{split}.npy")
        if not os.path.exists(xp):
            continue
        d[f"X_{split}"] = np.load(xp).astype(np.float32)
        d[f"y_{split}"] = np.load(yp)
        d[f"groups_{split}"] = (
            np.load(gp, allow_pickle=True) if os.path.exists(gp) else None
        )

    meta_path = os.path.join(data_root, "metadata.json")
    if os.path.exists(meta_path):
        with open(meta_path) as f:
            m = json.load(f)
        d["n_classes"] = m.get("n_classes", int(d["y_train"].max()) + 1)
    else:
        d["n_classes"] = int(d["y_train"].max()) + 1

    d["n_features"] = 3 * 224 * 224
    d["image_shape"] = (3, 224, 224)
    d["source"] = "efr_images"
    d["is_image_data"] = True

    if "X_train" in d and d["X_train"].ndim == 4:
        if d["X_train"].min() >= -0.1 and d["X_train"].max() <= 1.1:
            for split in ("train", "val", "test"):
                key = f"X_{split}"
                if key in d and d[key] is not None:
                    d[key] = normalize_imagenet(d[key])

    return d


def load_biometric_image_dataset(
    data_root: str = "data/biometric/efr_processed",
    demo: bool = False,
    n_samples: int = 400,
    n_classes: int = 10,
    seed: int = 42,
) -> Dict:
    """Load biometric image dataset, falling back to synthetic demo."""
    if not demo:
        dataset = load_efr_image_dataset(data_root, seed=seed)
        if dataset is not None:
            print(f"  Loaded EFR image dataset from {data_root}")
            return dataset
        print(f"  EFR images not found at '{data_root}'. "
              f"Using synthetic image demo data.")

    print(f"  Generating synthetic demo biometric images "
          f"({n_samples} samples, {n_classes} classes)...")
    return generate_demo_biometric_images(
        n_samples=n_samples, n_classes=n_classes, seed=seed,
    )
