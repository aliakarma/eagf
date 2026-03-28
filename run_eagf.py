#!/usr/bin/env python3
"""
run_eagf.py — EAGF One-Command Full Experiment Pipeline

Runs the complete EAGF experiment: ablation study, RE-IoT case study,
statistical tests, figure generation, and summary report.

Usage:
    # Full run (uses synthetic demo data — no downloads needed)
    python run_eagf.py

    # Fast demo (1 seed, fewer epochs — runs in ~2 minutes)
    python run_eagf.py --fast

    # Full paper results (3 seeds, 50 epochs — runs in ~20 minutes)
    python run_eagf.py --seeds 42 123 456 --epochs 50

    # With real EFR dataset
    python run_eagf.py --data-root data/biometric/efr_processed

Output:
    results/biometric/ablation/ablation_summary.csv   (Table 4)
    results/biometric/main_results.csv                (Table 5)
    results/reiot/node_class_results.csv              (Table 6)
    figures/figure3.png                               (Figure 3)
    figures/pareto_front.png
    figures/reiot_fprp.png
    results/summary_report.md
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
    p = argparse.ArgumentParser(description="EAGF Full Experiment Pipeline")
    p.add_argument("--fast",       action="store_true",
                   help="Fast mode: 1 seed, 20 epochs (for testing)")
    p.add_argument("--seeds",      nargs="+", type=int, default=[42, 123, 456])
    p.add_argument("--epochs",     type=int,  default=50)
    p.add_argument("--output",     default="results")
    p.add_argument("--data-root",  default=None,
                   help="EFR dataset root; uses synthetic demo if not set")
    p.add_argument("--skip-reiot", action="store_true")
    p.add_argument("--skip-pareto", action="store_true",
                   help="Skip 25-run Pareto grid (saves ~5x time)")
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
    banner("EAGF: Ethical AI Governance Framework — Full Pipeline")
    print(f"  Seeds:  {args.seeds}")
    print(f"  Epochs: {args.epochs}")
    print(f"  Output: {args.output}")

    # ── Step 1: Configuration & Data ──────────────────────────────────────
    banner("STEP 1 — Configuration and Data")
    with open("configs/biometric_default.yaml") as f:
        config = yaml.safe_load(f)
    config["training"]["epochs"] = args.epochs

    from src.utils.data_loader import load_biometric_dataset
    demo = (args.data_root is None)
    data_root = args.data_root or "data/biometric/efr_processed"
    print(f"  {'Demo synthetic' if demo else 'EFR'} dataset...")
    dataset = load_biometric_dataset(
        data_root=data_root, demo=demo, n_samples=1600, seed=args.seeds[0],
    )
    print(f"  Train:{len(dataset['y_train'])} "
          f"Val:{len(dataset['y_val'])} Test:{len(dataset['y_test'])}")

    bio_out = os.path.join(args.output, "biometric")
    os.makedirs(bio_out, exist_ok=True)
    if "splits" in dataset:
        with open(os.path.join("data", "biometric", "splits.json"), "w") as f:
            json.dump(dataset["splits"], f)

    # ── Steps 2-3: Biometric experiments ──────────────────────────────────
    run_biometric_ablation(config, dataset, args.seeds, bio_out)
    run_biometric_main(config, dataset, args.seeds, bio_out)

    # ── Step 4: Pareto search (optional) ─────────────────────────────────
    if not args.skip_pareto:
        run_pareto(config, dataset, args.seeds[0], bio_out)
    else:
        print("\n  Skipping Pareto search (--skip-pareto).")

    # ── Step 5: RE-IoT ────────────────────────────────────────────────────
    if not args.skip_reiot:
        run_reiot_experiment(args.output, args.seeds)
    else:
        print("\n  Skipping RE-IoT (--skip-reiot).")

    # ── Step 6: Figures ───────────────────────────────────────────────────
    generate_figures(args.output)

    # ── Step 7: Report ────────────────────────────────────────────────────
    from src.evaluation.report_generator import generate_report
    report_path = os.path.join(args.output, "summary_report.md")
    generate_report(bio_out, os.path.join(args.output, "reiot"), report_path)

    # ── Final summary ─────────────────────────────────────────────────────
    print_summary(args.output)

    elapsed = time.time() - t_start
    banner(f"DONE  —  total time {elapsed/60:.1f} min")
    print(f"\n  Key outputs:")
    print(f"    {args.output}/biometric/ablation/ablation_summary.csv  (Table 4)")
    print(f"    {args.output}/biometric/main_results.csv               (Table 5)")
    if not args.skip_reiot:
        print(f"    {args.output}/reiot/node_class_results.csv         (Table 6)")
    if not args.skip_pareto:
        print(f"    figures/figure3.png                                (Figure 3)")
    print()


if __name__ == "__main__":
    main()
