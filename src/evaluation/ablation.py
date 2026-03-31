"""
src/evaluation/ablation.py — Six-Variant Ablation Study Runner + Aggregator
Paper: Section 5.1.2, Table 4
"""
import argparse, json, os
from pathlib import Path
from typing import Dict, List
import numpy as np

try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False

MODEL_LABELS = {
    "baseline":       "M0: Baseline (no governance)",
    "transparency":   "M1: +Transparency only",
    "fairness":       "M2: +Fairness only",
    "privacy":        "M3: +Privacy only",
    "accountability": "M4: +Accountability only",
    "eagf":           "M5: EAGF (all pillars, joint)",
}
METRICS = ["accuracy", "recall_parity", "clarity", "privacy", "accountability", "trust_index"]


def bootstrap_ci(values, n_resamples=500, ci=0.95):
    values = np.array(values, dtype=float)
    resampled = [np.mean(np.random.choice(values, len(values), replace=True))
                 for _ in range(n_resamples)]
    alpha = (1 - ci) / 2
    return {"mean": float(np.mean(values)),
            "lower": float(np.percentile(resampled, 100 * alpha)),
            "upper": float(np.percentile(resampled, 100 * (1 - alpha)))}


def load_model_results(results_dir, model, seeds):
    metrics_by_seed = {m: [] for m in METRICS}
    for seed in seeds:
        result_path = Path(results_dir) / model / f"seed_{seed}" / "results.json"
        if not result_path.exists():
            raise FileNotFoundError(f"Missing: {result_path}")
        with open(result_path) as f:
            res = json.load(f)
        for m in METRICS:
            metrics_by_seed[m].append(float(res.get(m, 0.0)))
    return {m: np.array(v) for m, v in metrics_by_seed.items()}


def aggregate_ablation(results_dir, models, seeds, output_path):
    rows = []
    for model in models:
        try:
            metrics = load_model_results(results_dir, model, seeds)
        except FileNotFoundError as e:
            print(f"  Warning: {e}")
            continue
        row = {"model_id": model, "model_label": MODEL_LABELS.get(model, model)}
        for metric in METRICS:
            ci = bootstrap_ci(metrics[metric])
            row[f"{metric}_mean"]      = round(ci["mean"], 4)
            row[f"{metric}_ci_lower"]  = round(ci["lower"], 4)
            row[f"{metric}_ci_upper"]  = round(ci["upper"], 4)
        rows.append(row)

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

    if HAS_PANDAS:
        import pandas as pd
        df = pd.DataFrame(rows)
        df.to_csv(output_path, index=False, float_format="%.4f")
    else:
        import csv as _csv_abl
        if rows:
            headers = list(rows[0].keys())
            with open(output_path, "w", newline="") as f:
                writer = _csv_abl.DictWriter(f, fieldnames=headers)
                writer.writeheader()
                writer.writerows(rows)

    print(f"Ablation summary -> {output_path}")
    return rows


def run_full_ablation(config_path, seeds, output_dir, demo=True):
    """Run all six ablation variants end-to-end."""
    import yaml
    from src.training.eagf_trainer import train_variant
    from src.utils.data_loader import load_biometric_dataset

    with open(config_path) as f:
        config = yaml.safe_load(f)

    data_root = config.get("data", {}).get("root", "data/biometric")
    dataset   = load_biometric_dataset(data_root=data_root, demo=demo,
                                        n_samples=1500, seed=seeds[0])

    models = list(MODEL_LABELS.keys())
    for model in models:
        for seed in seeds:
            variant_dir = os.path.join(output_dir, model, f"seed_{seed}")
            print(f"\n  [{model} | seed={seed}]")
            train_variant(model, config, dataset.copy(), seed=seed,
                          output_dir=variant_dir)

    summary_path = os.path.join(output_dir, "ablation_summary.csv")
    aggregate_ablation(output_dir, models, seeds, summary_path)
    return summary_path


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--results-dir", required=True)
    p.add_argument("--models", nargs="+", default=list(MODEL_LABELS.keys()))
    p.add_argument("--seeds",  nargs="+", type=int, default=[42, 123, 456])
    p.add_argument("--output", required=True)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    aggregate_ablation(args.results_dir, args.models, args.seeds, args.output)
