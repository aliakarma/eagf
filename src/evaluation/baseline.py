"""
src/evaluation/baseline.py — Baseline Comparator + Result Aggregator
Aggregates per-seed results into main_results.csv / node_class_results.csv
"""
import argparse, json, os
import numpy as np

try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False

METRICS = ["accuracy", "recall_parity", "clarity", "privacy", "accountability", "trust_index",
           "ece", "brier_score",
           "inference_time_ms", "memory_usage_mb", "energy_overhead_joules"]


def _write_csv(rows, path):
    import csv as _cw
    import os
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    if HAS_PANDAS:
        import pandas as pd
        pd.DataFrame(rows).to_csv(path, index=False, float_format="%.4f")
    else:
        if rows:
            headers = list(rows[0].keys())
            with open(path, "w", newline="") as f:
                writer = _cw.DictWriter(f, fieldnames=headers)
                writer.writeheader()
                writer.writerows(rows)


def aggregate_results(results_dir, seeds, output_path, fairness_criterion="recall_parity"):
    """Aggregate baseline, EAGF, and any optional strong-baseline results across seeds."""
    # Discover which variants have results present.
    candidate_variants = ["baseline", "eagf", "joint_dp_fair"]
    rows = []
    for variant in candidate_variants:
        vals = {m: [] for m in METRICS}
        for seed in seeds:
            rp = os.path.join(results_dir, variant, f"seed_{seed}", "results.json")
            if not os.path.exists(rp):
                rp = os.path.join(results_dir, f"seed_{seed}", "results.json")
            if not os.path.exists(rp):
                continue
            with open(rp) as f:
                res = json.load(f)
            for m in METRICS:
                vals[m].append(float(res.get(m, 0.0)))
        if not vals["accuracy"]:
            continue
        row = {"model": variant}
        for m in METRICS:
            v = np.array(vals[m])
            row[f"{m}_mean"] = round(float(v.mean()), 4)
            row[f"{m}_std"]  = round(float(v.std()),  4)
        rows.append(row)

    _write_csv(rows, output_path)
    print(f"Results → {output_path}")
    return rows


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--results-dir", required=True)
    p.add_argument("--seeds", nargs="+", type=int, default=[42, 123, 456])
    p.add_argument("--output", required=True)
    p.add_argument("--fairness-criterion", default="recall_parity")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    aggregate_results(args.results_dir, args.seeds, args.output,
                      args.fairness_criterion)
