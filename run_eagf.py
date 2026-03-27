#!/usr/bin/env python3
"""
run_eagf.py — EAGF One-Command Q1 Benchmark Pipeline

Runs the full Q1-style evaluation suite:
1) Multi-seed baselines + ablations with confidence intervals
2) Pareto lambda sweep and trade-off plots
3) Differential privacy trade-off sweep
4) Clarity validation against SHAP concentration
5) Artifact manifest for reproducibility

Usage:
    # Full run (default 10 seeds)
    python run_eagf.py

    # Fast smoke run (1 seed)
    python run_eagf.py --fast

    # With UCI Adult dataset
    python run_eagf.py --data-root adult

Output:
    results/tables/*.csv
    results/plots/*.png
    results/logs/*.json
"""

import argparse
import json
import os
import sys
import time
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import yaml


def parse_args():
    p = argparse.ArgumentParser(description="EAGF Q1 Benchmark Pipeline")
    p.add_argument("--fast",       action="store_true",
                   help="Fast mode: 1 seed, 20 epochs (for testing)")
    p.add_argument("--seeds",      nargs="+", type=int,
                   default=[42, 43, 44, 45, 46, 47, 48, 49, 50, 51])
    p.add_argument("--epochs",     type=int,  default=50)
    p.add_argument("--output",     default="results")
    p.add_argument("--data-root",  default=None,
                   help="Dataset root. Use 'adult' for UCI Adult, or path for EFR; synthetic demo if not set")
    return p.parse_args()


def banner(msg):
    w = 66
    print("\n" + "=" * w)
    print(f"  {msg}")
    print("=" * w)


def run_biometric_ablation(config, dataset, seeds, output_dir):
    """Run 6-variant ablation across all seeds."""
    from src.training.eagf_trainer import train_variant
    from src.evaluation.ablation import aggregate_ablation

    VARIANTS = ["baseline", "transparency", "fairness",
                "privacy", "accountability", "eagf"]

    banner("STEP 2 — Biometric Ablation Study (M0–M5)")
    for variant in VARIANTS:
        for seed in seeds:
            vdir = os.path.join(output_dir, "ablation", variant, f"seed_{seed}")
            train_variant(variant, config, dataset.copy(),
                          seed=seed, output_dir=vdir)

    summary_path = os.path.join(output_dir, "ablation", "ablation_summary.csv")
    aggregate_ablation(
        results_dir=os.path.join(output_dir, "ablation"),
        models=VARIANTS, seeds=seeds,
        output_path=summary_path,
    )
    return summary_path


def run_biometric_main(config, dataset, seeds, output_dir):
    """Run baseline vs EAGF for main results table."""
    from src.training.eagf_trainer import train_variant
    from src.evaluation.baseline import aggregate_results
    from src.evaluation.statistics import run_all_tests

    banner("STEP 3 — Biometric Main Results (Baseline vs. EAGF)")
    for variant in ("baseline", "eagf"):
        for seed in seeds:
            vdir = os.path.join(output_dir, variant, f"seed_{seed}")
            train_variant(variant, config, dataset.copy(),
                          seed=seed, output_dir=vdir)

    main_csv = os.path.join(output_dir, "main_results.csv")
    aggregate_results(output_dir, seeds, main_csv)

    stats_path = os.path.join(output_dir, "statistical_tests.json")
    if len(seeds) >= 3:
        run_all_tests(
            baseline_dir=os.path.join(output_dir, "baseline"),
            eagf_dir=os.path.join(output_dir, "eagf"),
            seeds=seeds,
            output_path=stats_path,
        )
    else:
        print(f"  Note: Wilcoxon test requires ≥3 seeds; skipping with {len(seeds)} seed(s).")
        print(f"  Run with --seeds 42 123 456 for full statistical analysis.")
        import json as _json
        with open(stats_path, "w") as _f:
            _json.dump({"note": f"Insufficient seeds ({len(seeds)}) for Wilcoxon. Use --seeds 42 123 456."}, _f)
    return main_csv


def run_pareto(config, dataset, seed, output_dir):
    """Run 5×5 Pareto grid search."""
    from src.training.pareto_trainer import run_pareto_search

    banner("STEP 4 — Pareto-Front Hyperparameter Search")
    return run_pareto_search(
        config=config,
        lambda_rp_range=(1e-3, 1.0),
        lambda_c_range=(1e-3, 1.0),
        n_steps=5,
        seed=seed,
        device="cpu",
        output_dir=os.path.join(output_dir, "pareto"),
        dataset=dataset,
    )


def run_reiot_experiment(output_dir, seeds):
    """Run RE-IoT case study."""
    from src.utils.data_loader import load_reiot_dataset
    from src.utils.preprocessing import preprocess_reiot
    from src.training.eagf_trainer import train_variant
    from src.evaluation.baseline import aggregate_results

    banner("STEP 5 — RE-IoT Anomaly Detection Case Study")

    with open("configs/reiot_default.yaml") as f:
        config = yaml.safe_load(f)

    reiot_data = load_reiot_dataset(
        data_root="data/reiot", n_urban=20, n_periurban=20, n_rural=20,
        n_windows_per_node=50, seed=seeds[0],
    )
    config["data"]["name"] = "reiot"

    for variant in ("baseline", "eagf"):
        for seed in seeds:
            vdir = os.path.join(output_dir, "reiot", variant, f"seed_{seed}")
            train_variant(variant, config, reiot_data.copy(),
                          seed=seed, output_dir=vdir)

    node_csv = os.path.join(output_dir, "reiot", "node_class_results.csv")
    aggregate_results(
        os.path.join(output_dir, "reiot"), seeds, node_csv,
        fairness_criterion="fprp",
    )
    return node_csv


def generate_figures(output_dir):
    """Generate all paper figures."""
    from src.utils.visualisation import (
        plot_ablation_bar, plot_pareto_front, plot_fprp_bar,
    )

    banner("STEP 6 — Figure Generation")
    os.makedirs("figures", exist_ok=True)

    ablation_csv = os.path.join(output_dir, "biometric", "ablation", "ablation_summary.csv")
    if os.path.exists(ablation_csv):
        plot_ablation_bar(ablation_csv, "figures/figure3.png")

    pareto_json = os.path.join(output_dir, "biometric", "pareto", "pareto_results.json")
    if os.path.exists(pareto_json):
        plot_pareto_front(pareto_json, "figures/pareto_front.png")

    reiot_csv = os.path.join(output_dir, "reiot", "node_class_results.csv")
    if os.path.exists(reiot_csv):
        plot_fprp_bar(reiot_csv, "figures/reiot_fprp.png")


def print_summary(output_dir):
    """Print final summary table to console."""
    banner("RESULTS SUMMARY")
    ablation_csv = os.path.join(output_dir, "biometric", "ablation", "ablation_summary.csv")
    if not os.path.exists(ablation_csv):
        return

    print(f"\n{'Model':<25} {'Acc':>7} {'RP':>7} {'C':>7} {'P':>7} {'A':>7} {'TI':>7}")
    print("-" * 68)
    import csv as _csv_mod
    with open(ablation_csv, newline='') as _f:
        _abl_rows = list(_csv_mod.DictReader(_f))
    for vals in _abl_rows:
        label = vals.get("model_label", vals.get("model_id", "?"))
        try:
            print(f"{label:<25} "
                  f"{float(vals.get('accuracy_mean',0)):7.3f} "
                  f"{float(vals.get('recall_parity_mean',0)):7.3f} "
                  f"{float(vals.get('clarity_mean',0)):7.3f} "
                  f"{float(vals.get('privacy_mean',0)):7.3f} "
                  f"{float(vals.get('accountability_mean',0)):7.3f} "
                  f"{float(vals.get('trust_index_mean',0)):7.3f}")
        except Exception:
            pass

    main_csv = os.path.join(output_dir, "biometric", "main_results.csv")
    if os.path.exists(main_csv):
        print(f"\n  Main results → {main_csv}")

    stats_json = os.path.join(output_dir, "biometric", "statistical_tests.json")
    if os.path.exists(stats_json):
        with open(stats_json) as f:
            stats = json.load(f)
        if "note" in stats:
            print(f"\n  Statistical tests: {stats['note']}")
        else:
            n_rep = stats.get("trust_index", {}).get("n_replicates", 3)
            acc_test = stats.get("accuracy", {}).get("ztest", {})
            ti_stat  = stats.get("trust_index", {})
            ti_test  = ti_stat.get("wilcoxon") or ti_stat.get("paired_ttest") or {}
            test_name = "Wilcoxon" if "wilcoxon" in ti_stat else "Paired t-test"
            print(f"\n  Statistical tests ({n_rep} seeds):")
            print(f"    Accuracy (z-test):        p={acc_test.get('p_value',1.0):.4f} "
                  f"({'not significant' if acc_test.get('p_value',1)>0.05 else 'SIGNIFICANT ✓'})")
            print(f"    Trust Index ({test_name}): p={ti_test.get('p_value',1.0):.4f} "
                  f"({'SIGNIFICANT ✓' if ti_test.get('p_value',1)<0.05 else 'not significant'})")
            if "note" in ti_test:
                print(f"    Note: {ti_test['note']}")
            # Show absolute TI improvement
            ti_delta = stats.get("trust_index", {}).get("delta_mean", 0)
            b_ci = stats.get("trust_index", {}).get("baseline", {})
            e_ci = stats.get("trust_index", {}).get("eagf", {})
            if b_ci and e_ci:
                print(f"    TI improvement: {b_ci.get('mean',0):.3f} → {e_ci.get('mean',0):.3f} "
                      f"(Δ={ti_delta:+.3f})")


def main():
    args = parse_args()

    if args.fast:
        args.seeds  = [42]
        args.epochs = 20
        print("Fast mode: 1 seed, 20 epochs")

    t_start = time.time()
    banner("EAGF: Ethical AI Governance Framework — Q1 Benchmark Pipeline")
    print(f"  Seeds:  {args.seeds}")
    print(f"  Epochs: {args.epochs}")
    print(f"  Output: {args.output}")

    # ── Step 1: Configuration & Data ──────────────────────────────────────
    banner("STEP 1 — Configuration and Data")
    with open("configs/biometric_default.yaml") as f:
        config = yaml.safe_load(f)
    config["training"]["epochs"] = args.epochs

    from src.utils.data_loader import load_adult_dataset, load_biometric_dataset
    if args.data_root == "adult":
        print("  UCI Adult dataset...")
        dataset = load_adult_dataset(seed=args.seeds[0])
        config.setdefault("data", {})
        config["data"]["name"] = "adult"
    else:
        demo = (args.data_root is None)
        data_root = args.data_root or "data/biometric/efr_processed"
        print(f"  {'Demo synthetic' if demo else 'EFR'} dataset...")
        dataset = load_biometric_dataset(
            data_root=data_root, demo=demo, n_samples=1600, seed=args.seeds[0],
        )
    print(f"  Train:{len(dataset['y_train'])} "
          f"Val:{len(dataset['y_val'])} Test:{len(dataset['y_test'])}")

    os.makedirs(args.output, exist_ok=True)
    if "splits" in dataset:
        with open(os.path.join("data", "biometric", "splits.json"), "w") as f:
            json.dump(dataset["splits"], f)

    # ── Q1 benchmark suite ────────────────────────────────────────────────
    banner("STEP 2 — Q1 Benchmark Suite")
    from src.evaluation.benchmark_suite import run_q1_benchmark_suite

    manifest = run_q1_benchmark_suite(
        config=config,
        dataset=dataset,
        seeds=args.seeds,
        output_root=args.output,
    )

    elapsed = time.time() - t_start
    banner(f"DONE  —  total time {elapsed/60:.1f} min")
    print("\n  Q1 artifacts:")
    print(f"    Tables: {len(manifest['tables'])} CSV files in {args.output}/tables")
    print(f"    Plots:  {len(manifest['plots'])} PNG files in {args.output}/plots")
    print(f"    Logs:   {len(manifest['logs'])} JSON files in {args.output}/logs")
    print(f"    Manifest: {args.output}/logs/artifact_manifest.json")
    print()


if __name__ == "__main__":
    main()
