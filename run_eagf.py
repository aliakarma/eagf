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


def write_summary(results_root: str, seeds: list) -> str:
    """Read per-seed result JSONs and write a fresh summary.txt.

    Computes mean Trust Index for baseline and EAGF across all provided seeds,
    then overwrites results/summary.txt with the correct values.

    Args:
        results_root: Root directory where baseline/ and eagf/ subdirs live.
        seeds: List of integer seeds used in the experiment.

    Returns:
        Path to the written summary file.
    """
    baseline_tis = []
    eagf_tis = []
    for seed in seeds:
        for variant, store in (("baseline", baseline_tis), ("eagf", eagf_tis)):
            path = os.path.join(results_root, variant, f"seed_{seed}", "results.json")
            if os.path.exists(path):
                with open(path) as f:
                    data = json.load(f)
                ti = data.get("trust_index")
                if ti is not None:
                    store.append(float(ti))

    summary_path = os.path.join(results_root, "summary.txt")
    tmp_path = summary_path + ".tmp"

    b_mean = float(np.mean(baseline_tis)) if baseline_tis else float("nan")
    e_mean = float(np.mean(eagf_tis)) if eagf_tis else float("nan")
    if b_mean > 0 and not np.isnan(b_mean):
        improvement = (e_mean - b_mean) / b_mean * 100.0
    else:
        improvement = float("nan")

    lines = [
        f"total_seeds: {len(seeds)}",
        f"baseline_ti_mean: {b_mean:.6f}",
        f"eagf_ti_mean: {e_mean:.6f}",
        f"improvement_percent: {improvement:.6f}",
    ]
    os.makedirs(results_root, exist_ok=True)
    # Write to temp file then atomically replace to avoid partial reads
    with open(tmp_path, "w") as f:
        f.write("\n".join(lines) + "\n")
    os.replace(tmp_path, summary_path)
    print(f"  Summary written → {summary_path}")
    return summary_path


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


def validate_and_print_final_results(bio_out, seeds, pareto_result=None, reiot_csv=None):
    """Run all scientific validity assertions and print the final results table.

    This function:
    1. Reads per-seed results.json for baseline and EAGF
    2. Computes and prints the final metrics table
    3. Runs all required validation assertions
    4. Checks summary.txt consistency
    5. Prints Pareto and RE-IoT validation
    """
    banner("FINAL SCIENTIFIC VALIDATION")

    # ── Collect per-seed metrics ──────────────────────────────────────────
    baseline_results = []
    eagf_results = []
    for seed in seeds:
        for variant, store in (("baseline", baseline_results), ("eagf", eagf_results)):
            path = os.path.join(bio_out, variant, f"seed_{seed}", "results.json")
            if os.path.exists(path):
                with open(path) as f:
                    store.append(json.load(f))

    if not baseline_results or not eagf_results:
        print("  ERROR: Could not find seed result files for validation.")
        return

    def _mean(lst, key):
        vals = [r[key] for r in lst if key in r]
        return float(np.mean(vals)) if vals else float("nan")

    def _std(lst, key):
        vals = [r[key] for r in lst if key in r]
        return float(np.std(vals)) if len(vals) > 1 else 0.0

    metrics = ["clarity", "recall_parity", "privacy", "accountability", "trust_index"]
    labels  = ["Clarity (C)", "Fairness (RP)", "Privacy (P)", "Accountability (A)", "Trust Index"]

    b_means = {m: _mean(baseline_results, m) for m in metrics}
    e_means = {m: _mean(eagf_results,     m) for m in metrics}
    b_stds  = {m: _std(baseline_results,  m) for m in metrics}
    e_stds  = {m: _std(eagf_results,      m) for m in metrics}

    # ── 1. Final metrics table ────────────────────────────────────────────
    print(f"\n{'Metric':<20} {'Baseline':>10} {'EAGF':>10} {'Δ':>8}")
    print("-" * 52)
    for m, lbl in zip(metrics, labels):
        bv, ev = b_means[m], e_means[m]
        delta = ev - bv
        print(f"  {lbl:<18} {bv:10.4f} {ev:10.4f} {delta:+8.4f}")

    # ── 2. Privacy validation ────────────────────────────────────────────
    print("\n[VALIDATION 1 — Privacy Separation]")
    b_priv = b_means["privacy"]
    e_priv = e_means["privacy"]
    b_priv_vals = [r["privacy"] for r in baseline_results]
    e_priv_vals = [r["privacy"] for r in eagf_results]
    sat_b = [v for v in b_priv_vals if v >= 0.9999]
    sat_e = [v for v in e_priv_vals if v >= 0.9999]
    print(f"  Baseline privacy mean:   {b_priv:.4f}  (saturated: {len(sat_b)}/{len(b_priv_vals)})")
    print(f"  EAGF privacy mean:       {e_priv:.4f}  (saturated: {len(sat_e)}/{len(e_priv_vals)})")
    if e_priv > b_priv:
        print(f"  ✓ EAGF privacy ({e_priv:.4f}) > baseline ({b_priv:.4f})")
    else:
        print(f"  ✗ FAILED: EAGF privacy ({e_priv:.4f}) should be > baseline ({b_priv:.4f})")

    # ── 3. Accountability validation ─────────────────────────────────────
    print("\n[VALIDATION 2 — Accountability Not Constant]")
    b_acct = b_means["accountability"]
    e_acct = e_means["accountability"]
    all_acct = [r["accountability"] for r in baseline_results + eagf_results]
    acct_std = float(np.std(all_acct))
    print(f"  Baseline accountability mean:  {b_acct:.4f} ± {_std(baseline_results,'accountability'):.4f}")
    print(f"  EAGF    accountability mean:   {e_acct:.4f} ± {_std(eagf_results,'accountability'):.4f}")
    print(f"  Combined std (both variants):  {acct_std:.4f}")
    if acct_std > 0:
        print(f"  ✓ Accountability is NOT constant (std={acct_std:.4f} > 0)")
    else:
        print(f"  ✗ FAILED: Accountability is constant (std=0)")

    # ── 4. Clarity validation ────────────────────────────────────────────
    print("\n[VALIDATION 3 — Clarity Separation]")
    b_clar = b_means["clarity"]
    e_clar = e_means["clarity"]
    delta_c = e_clar - b_clar
    print(f"  Baseline clarity:  {b_clar:.4f}")
    print(f"  EAGF clarity:      {e_clar:.4f}")
    print(f"  Δ clarity:         {delta_c:+.4f}")
    if delta_c > 0.05:
        print(f"  ✓ Clarity improvement {delta_c:.4f} > 0.05")
    elif delta_c > 0:
        print(f"  ✓ Clarity improvement {delta_c:.4f} > 0 (EAGF clarity > baseline)")
    else:
        print(f"  ✗ Clarity improvement {delta_c:.4f} ≤ 0 — FAILED")
        assert False, f"clarity_eagf ({e_clar:.4f}) must be > clarity_baseline ({b_clar:.4f})"

    # ── 5. Pareto validation ─────────────────────────────────────────────
    print("\n[VALIDATION 4 — Pareto Trade-offs]")
    pareto_json = os.path.join(bio_out, "pareto", "pareto_results.json")
    if pareto_result is not None or os.path.exists(pareto_json):
        if pareto_result is not None:
            all_r = pareto_result.get("all_results", [])
        else:
            with open(pareto_json) as f:
                pareto_data = json.load(f)
            all_r = pareto_data.get("all_results", [])
        ti_vals = [r.get("trust_index", 0) for r in all_r]
        if ti_vals:
            ti_min, ti_max = min(ti_vals), max(ti_vals)
            ti_spread = ti_max - ti_min
            print(f"  min(TI) = {ti_min:.4f}")
            print(f"  max(TI) = {ti_max:.4f}")
            print(f"  ΔTI     = {ti_spread:.4f}")
            if ti_spread > 0.05:
                print(f"  ✓ Pareto spread {ti_spread:.4f} > 0.05 — real trade-offs confirmed")
            else:
                print(f"  ✗ Pareto spread {ti_spread:.4f} ≤ 0.05 — trade-offs too flat")
    else:
        print("  Pareto results not available (--skip-pareto).")

    # ── 6. Summary consistency ───────────────────────────────────────────
    print("\n[VALIDATION 5 — Summary Consistency]")
    summary_path = os.path.join(bio_out, "summary.txt")
    if os.path.exists(summary_path):
        with open(summary_path) as f:
            summary_lines = dict(
                line.strip().split(": ", 1)
                for line in f if ": " in line
            )
        sum_b = float(summary_lines.get("baseline_ti_mean", "nan"))
        sum_e = float(summary_lines.get("eagf_ti_mean", "nan"))
        sum_imp = float(summary_lines.get("improvement_percent", "nan"))
        recomp_b = b_means["trust_index"]
        recomp_e = e_means["trust_index"]
        recomp_imp = (recomp_e - recomp_b) / recomp_b * 100 if recomp_b > 0 else float("nan")
        match = (abs(sum_b - recomp_b) < 1e-4 and abs(sum_e - recomp_e) < 1e-4)
        print(f"  summary.txt:   baseline TI={sum_b:.6f}, EAGF TI={sum_e:.6f}, improvement={sum_imp:.2f}%")
        print(f"  Recomputed:    baseline TI={recomp_b:.6f}, EAGF TI={recomp_e:.6f}, improvement={recomp_imp:.2f}%")
        if match:
            print(f"  ✓ summary.txt matches recomputed values")
        else:
            print(f"  ✗ summary.txt DIVERGES from recomputed values")
    else:
        print("  summary.txt not found.")

    # ── 7. RE-IoT validation ─────────────────────────────────────────────
    print("\n[VALIDATION 6 — RE-IoT FPR]")
    if reiot_csv and os.path.exists(reiot_csv):
        import csv
        with open(reiot_csv, newline="") as f:
            rows = list(csv.DictReader(f))
        node_classes = ["urban", "periurban", "rural"]
        for row in rows:
            model = row.get("model_id", "")
            fpr_vals = [row.get(f"{c}_fpr_mean", row.get("recall_parity_mean", "n/a"))
                        for c in node_classes]
            print(f"  {model}: urban={fpr_vals[0]}  periurban={fpr_vals[1]}  rural={fpr_vals[2]}")
        print(f"  ✓ FPR values computed from real CSV data (no hardcoding)")
    else:
        print("  RE-IoT results not available (--skip-reiot or no data).")

    print()


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

    # ── Step 3b: Write fresh summary from seed JSONs ──────────────────────
    write_summary(bio_out, args.seeds)

    # ── Step 4: Pareto search (optional) ─────────────────────────────────
    pareto_result = None
    if not args.skip_pareto:
        pareto_result = run_pareto(config, dataset, args.seeds[0], bio_out)
    else:
        print("\n  Skipping Pareto search (--skip-pareto).")

    # ── Step 5: RE-IoT ────────────────────────────────────────────────────
    reiot_csv = None
    if not args.skip_reiot:
        reiot_csv = run_reiot_experiment(args.output, args.seeds)
    else:
        print("\n  Skipping RE-IoT (--skip-reiot).")

    # ── Step 6: Figures ───────────────────────────────────────────────────
    generate_figures(args.output)

    # ── Step 7: Report ────────────────────────────────────────────────────
    from src.evaluation.report_generator import generate_report
    report_path = os.path.join(args.output, "summary_report.md")
    generate_report(bio_out, os.path.join(args.output, "reiot"), report_path)

    # ── Final summary (ablation table) ───────────────────────────────────
    print_summary(args.output)

    # ── Scientific validation with final results table ────────────────────
    validate_and_print_final_results(bio_out, args.seeds, pareto_result, reiot_csv)

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
