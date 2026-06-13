"""Minimal external-style baseline: reweighting + DP-SGD.

This module provides a lightweight baseline pipeline that:
1) Applies fairness preprocessing via group/class reweighting.
2) Trains with DP-SGD using the existing EAGF trainer internals.
3) Reports the same metric set as EAGF.
"""

import argparse
import json
import os

import numpy as np

from src.training.eagf_trainer import (
    ModelAdapter,
    _compute_all_metrics,
    _train_torch_model,
    load_config,
    set_seed,
)
from src.utils.data_loader import load_adult_dataset, load_biometric_dataset, load_reiot_dataset
from src.utils.preprocessing import compute_sample_weights, preprocess_biometric, preprocess_reiot


def _resample_with_weights(X, y, groups, sample_weights, seed):
    """Resample training rows according to preprocessing weights."""
    rng = np.random.RandomState(seed)
    probs = np.asarray(sample_weights, dtype=np.float64)
    probs = probs / probs.sum()
    idx = rng.choice(len(y), size=len(y), replace=True, p=probs)
    X_rs = X[idx]
    y_rs = y[idx]
    g_rs = groups[idx] if groups is not None else None
    return X_rs, y_rs, g_rs


def _load_dataset_from_config(config, seed=42, demo=False):
    name = str(config.get("data", {}).get("name", "")).lower()
    if "reiot" in name:
        data_root = config.get("data", {}).get("root", "data/reiot")
        return load_reiot_dataset(data_root=data_root, seed=seed)
    if "adult" in name:
        return load_adult_dataset(seed=seed)
    data_root = config.get("data", {}).get("root", "data/biometric/efr_processed")
    return load_biometric_dataset(data_root=data_root, demo=demo, seed=seed)


def run_baseline(config, seed=42, device="cpu", output_dir="results/baseline_dp", demo=False,
                  use_dp=False):
    """Run AIF360-style reweighting baseline.

    Args:
        use_dp: If False (default), trains without DP-SGD (pure reweighting
            for single-pillar fairness comparison as in paper Table 17).
            If True, adds DP-SGD on top of reweighting.
    """
    set_seed(seed)
    os.makedirs(output_dir, exist_ok=True)

    dataset = _load_dataset_from_config(config, seed=seed, demo=demo)
    is_reiot = "reiot" in str(config.get("data", {}).get("name", "")).lower()

    if is_reiot:
        data = preprocess_reiot(dataset.copy(), seed=seed)
        X_train, y_train = data["X_train"], data["y_train"]
        X_test, y_test = data["X_test"], data["y_test"]
        groups_train = data.get("groups_train")
        groups_test = data.get("groups_test")
        # Keep compatibility with metric code expecting a validation split.
        X_val, y_val = X_train, y_train
        groups_val = groups_train
    else:
        data = preprocess_biometric(
            dataset.copy(),
            apply_dp=False,
            epsilon=config.get("governance", {}).get("dp_epsilon", 3.0),
            seed=seed,
        )
        X_train, y_train = data["X_train"], data["y_train"]
        X_val, y_val = data.get("X_val", X_train), data.get("y_val", y_train)
        X_test, y_test = data["X_test"], data["y_test"]
        groups_train = data.get("groups_train")
        groups_val = data.get("groups_val", groups_train)
        groups_test = data.get("groups_test")

    # Fairness preprocessing (simple reweighting + weighted resampling).
    sample_weights = compute_sample_weights(y_train, groups=groups_train, strategy="balanced_group")
    X_train_rw, y_train_rw, groups_train_rw = _resample_with_weights(
        X_train, y_train, groups_train, sample_weights, seed=seed
    )

    train_variant_name = "privacy" if use_dp else "baseline"
    config = json.loads(json.dumps(config))
    config.setdefault("training", {})
    config["training"]["device"] = device

    model, epsilon_eff, _overhead = _train_torch_model(
        variant=train_variant_name,
        config=config,
        X_train=X_train_rw,
        y_train=y_train_rw,
        groups_train=groups_train_rw,
        X_val=X_val,
        y_val=y_val,
        device=device,
        seed=seed,
    )
    adapter = ModelAdapter(model, device=device)

    metrics = _compute_all_metrics(
        model_adapter=adapter,
        X_train=X_train_rw,
        y_train=y_train_rw,
        X_val=X_val,
        y_val=y_val,
        groups_val=groups_val,
        X_test=X_test,
        y_test=y_test,
        groups_test=groups_test,
        config=config,
        variant="baseline_dp_reweight",
        audit_log_path=os.path.join(output_dir, "audit_log.jsonl"),
        epsilon_eff=epsilon_eff,
        seed=seed,
        accountability_enabled=False,
    )

    out_path = os.path.join(output_dir, "results.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({k: float(v) if isinstance(v, (float, np.floating)) else v for k, v in metrics.items()}, f, indent=2)

    return metrics


def parse_args():
    p = argparse.ArgumentParser(description="Run external baseline: reweighting + DP-SGD")
    p.add_argument("--config", required=True)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="cpu")
    p.add_argument("--output", required=True)
    p.add_argument("--demo", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    config = load_config(args.config)
    metrics = run_baseline(
        config=config,
        seed=args.seed,
        device=args.device,
        output_dir=args.output,
        demo=args.demo,
    )
    print("Baseline run complete.")
    for key in ("accuracy", "recall_parity", "privacy", "clarity", "accountability", "trust_index"):
        if key in metrics:
            print(f"{key}: {float(metrics[key]):.4f}")


if __name__ == "__main__":
    main()
