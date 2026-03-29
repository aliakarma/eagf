#!/usr/bin/env python3
"""
Three-stage EAGF sweep utility.

Runs EAGF-only training across a seed list for three preset governance profiles,
then selects the best stage using weighted metrics with threshold penalties.

Usage:
    python scripts/sweep_three_stage.py
    python scripts/sweep_three_stage.py --seeds 42 43 44 45 46 47 48 49 50 51 --epochs 80
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import statistics
import subprocess
import sys
import time
from copy import deepcopy

import yaml


DEFAULT_SEEDS = [42, 43, 44, 45, 46, 47, 48, 49, 50, 51]
TARGET_METRICS = ["accuracy", "recall_parity", "clarity", "privacy"]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Three-stage EAGF sweep")
    p.add_argument("--config", default="configs/biometric_default.yaml")
    p.add_argument("--seeds", nargs="+", type=int, default=DEFAULT_SEEDS)
    p.add_argument("--epochs", type=int, default=None,
                   help="Optional training epochs override for all stages")
    p.add_argument("--device", default="cpu")
    p.add_argument("--output-root", default="results/sweeps/three_stage")
    p.add_argument("--write-best-config", default="configs/biometric_tuned_auto.yaml")
    p.add_argument("--w-acc", type=float, default=0.25)
    p.add_argument("--w-rp", type=float, default=0.25)
    p.add_argument("--w-clarity", type=float, default=0.25)
    p.add_argument("--w-privacy", type=float, default=0.25)
    return p.parse_args()


def load_yaml(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def save_yaml(path: str, obj: dict) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(obj, f, sort_keys=False)


def stage_profiles(base_cfg: dict) -> list[dict]:
    """Return three practical tuning profiles.

    The profiles provide different trade-offs:
    - privacy_first: tighter epsilon, moderate fairness/clarity penalties
    - fairness_first: stronger fairness penalty, slightly looser epsilon
    - balanced: strong but balanced penalties and privacy budget
    """
    return [
        {
            "name": "privacy_first",
            "overrides": {
                "governance": {
                    "dp_epsilon": 2.2,
                    "dp_max_grad_norm": 0.8,
                    "lambda_rp": 0.12,
                    "lambda_c": 0.14,
                    "shap_sample_size": max(200, int(base_cfg.get("governance", {}).get("shap_sample_size", 200))),
                }
            },
        },
        {
            "name": "fairness_first",
            "overrides": {
                "governance": {
                    "dp_epsilon": 3.2,
                    "dp_max_grad_norm": 1.0,
                    "lambda_rp": 0.24,
                    "lambda_c": 0.12,
                    "shap_sample_size": max(220, int(base_cfg.get("governance", {}).get("shap_sample_size", 200))),
                }
            },
        },
        {
            "name": "balanced",
            "overrides": {
                "governance": {
                    "dp_epsilon": 2.6,
                    "dp_max_grad_norm": 0.9,
                    "lambda_rp": 0.20,
                    "lambda_c": 0.20,
                    "shap_sample_size": max(260, int(base_cfg.get("governance", {}).get("shap_sample_size", 200))),
                }
            },
        },
    ]


def deep_update(dst: dict, src: dict) -> dict:
    for k, v in src.items():
        if isinstance(v, dict) and isinstance(dst.get(k), dict):
            deep_update(dst[k], v)
        else:
            dst[k] = v
    return dst


def run_seed(config_path: str, seed: int, device: str, out_dir: str) -> None:
    cmd = [
        sys.executable,
        "-m",
        "src.training.eagf_trainer",
        "--config",
        config_path,
        "--model",
        "eagf",
        "--seed",
        str(seed),
        "--device",
        device,
        "--output",
        out_dir,
    ]
    print("[RUN]", " ".join(cmd))
    proc = subprocess.run(cmd, check=False)
    if proc.returncode != 0:
        raise RuntimeError(f"Seed {seed} failed with exit code {proc.returncode}")


def read_result_json(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def summarize_stage(stage_dir: str, seeds: list[int]) -> dict:
    rows = []
    for seed in seeds:
        rp = os.path.join(stage_dir, "eagf", f"seed_{seed}", "results.json")
        if not os.path.exists(rp):
            continue
        data = read_result_json(rp)
        data["seed"] = seed
        rows.append(data)

    if not rows:
        raise RuntimeError(f"No results found under {stage_dir}")

    summary = {
        "n_seeds": len(rows),
        "metrics": {},
    }
    for metric in TARGET_METRICS + ["trust_index"]:
        vals = [float(r.get(metric, 0.0)) for r in rows]
        summary["metrics"][metric] = {
            "mean": round(float(statistics.fmean(vals)), 6),
            "std": round(float(statistics.pstdev(vals)), 6) if len(vals) > 1 else 0.0,
        }

    return summary


def stage_score(summary: dict, thresholds: dict, weights: dict) -> tuple[float, dict]:
    penalties = {}
    score = 0.0
    for m in TARGET_METRICS:
        mean_v = float(summary["metrics"][m]["mean"])
        score += weights[m] * mean_v
        target = float(thresholds[m])
        shortfall = max(0.0, target - mean_v)
        penalties[m] = shortfall
        score -= (shortfall * 1.5)
    return score, penalties


def write_csv(path: str, rows: list[dict], headers: list[str]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=headers)
        w.writeheader()
        w.writerows(rows)


def main() -> int:
    args = parse_args()
    t0 = time.time()

    base_cfg = load_yaml(args.config)
    if args.epochs is not None:
        base_cfg.setdefault("training", {})["epochs"] = int(args.epochs)

    thr_cfg = base_cfg.get("thresholds", {})
    thresholds = {
        "accuracy": float(thr_cfg.get("min_accuracy", 0.0)),
        "recall_parity": float(thr_cfg.get("min_recall_parity", 0.95)),
        "clarity": float(thr_cfg.get("min_clarity", 0.80)),
        "privacy": float(thr_cfg.get("min_privacy", 0.80)),
    }

    weights = {
        "accuracy": float(args.w_acc),
        "recall_parity": float(args.w_rp),
        "clarity": float(args.w_clarity),
        "privacy": float(args.w_privacy),
    }

    total_w = sum(weights.values())
    if total_w <= 0:
        raise ValueError("Metric weights must sum to a positive value")
    for k in weights:
        weights[k] = weights[k] / total_w

    run_id = time.strftime("%Y%m%d_%H%M%S")
    run_root = os.path.join(args.output_root, run_id)
    cfg_root = os.path.join(run_root, "configs")
    os.makedirs(cfg_root, exist_ok=True)

    profiles = stage_profiles(base_cfg)
    stage_rows = []
    all_details = {}

    print("=" * 72)
    print("3-stage EAGF sweep")
    print(f"seeds: {args.seeds}")
    print(f"device: {args.device}")
    print(f"output: {run_root}")
    print("=" * 72)

    for idx, profile in enumerate(profiles, start=1):
        stage_name = profile["name"]
        print(f"\n[Stage {idx}/3] {stage_name}")
        stage_cfg = deep_update(deepcopy(base_cfg), profile["overrides"])
        stage_cfg_path = os.path.join(cfg_root, f"{stage_name}.yaml")
        save_yaml(stage_cfg_path, stage_cfg)

        stage_dir = os.path.join(run_root, stage_name)
        for seed in args.seeds:
            out_dir = os.path.join(stage_dir, "eagf", f"seed_{seed}")
            os.makedirs(out_dir, exist_ok=True)
            run_seed(stage_cfg_path, seed, args.device, out_dir)

        summary = summarize_stage(stage_dir, args.seeds)
        score, penalties = stage_score(summary, thresholds, weights)
        summary["score"] = round(score, 6)
        summary["penalties"] = penalties
        all_details[stage_name] = {
            "config_path": stage_cfg_path,
            "summary": summary,
            "governance": stage_cfg.get("governance", {}),
        }

        row = {
            "stage": stage_name,
            "score": round(score, 6),
            "accuracy_mean": summary["metrics"]["accuracy"]["mean"],
            "recall_parity_mean": summary["metrics"]["recall_parity"]["mean"],
            "clarity_mean": summary["metrics"]["clarity"]["mean"],
            "privacy_mean": summary["metrics"]["privacy"]["mean"],
            "trust_index_mean": summary["metrics"]["trust_index"]["mean"],
            "n_seeds": summary["n_seeds"],
        }
        stage_rows.append(row)

    stage_rows.sort(key=lambda r: float(r["score"]), reverse=True)
    best = stage_rows[0]
    best_name = str(best["stage"])
    best_cfg_path = all_details[best_name]["config_path"]
    best_cfg = load_yaml(best_cfg_path)

    save_yaml(args.write_best_config, best_cfg)

    leaderboard_csv = os.path.join(run_root, "leaderboard.csv")
    write_csv(
        leaderboard_csv,
        stage_rows,
        headers=[
            "stage",
            "score",
            "accuracy_mean",
            "recall_parity_mean",
            "clarity_mean",
            "privacy_mean",
            "trust_index_mean",
            "n_seeds",
        ],
    )

    summary_json = os.path.join(run_root, "summary.json")
    with open(summary_json, "w", encoding="utf-8") as f:
        json.dump(
            {
                "run_root": run_root,
                "seeds": args.seeds,
                "weights": weights,
                "thresholds": thresholds,
                "best_stage": best_name,
                "best_stage_config": args.write_best_config,
                "leaderboard": stage_rows,
                "details": all_details,
            },
            f,
            indent=2,
        )

    print("\n" + "=" * 72)
    print("Sweep complete")
    print(f"Best stage: {best_name}")
    print(f"Leaderboard: {leaderboard_csv}")
    print(f"Summary JSON: {summary_json}")
    print(f"Best config written: {args.write_best_config}")
    print(f"Elapsed: {time.time() - t0:.1f}s")
    print("=" * 72)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
