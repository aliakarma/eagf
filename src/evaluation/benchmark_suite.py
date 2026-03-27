"""Q1-level benchmarking suite for EAGF experiments."""

import json
import importlib
import os
from copy import deepcopy

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from src.metrics.clarity import compute_instance_clarity
from src.training.eagf_trainer import train_variant


CORE_METHODS = [
    "standard",
    "fairness_threshold",
    "dp_only",
    "full",
]

ABLATION_METHODS = [
    "full",
    "ablate_no_fairness",
    "ablate_no_dp",
    "ablate_no_clarity",
    "ablate_no_accountability",
]

METRICS = ["accuracy", "recall_parity", "privacy", "clarity", "accountability", "trust_index"]


def _ensure_dirs(root):
    tables = os.path.join(root, "tables")
    plots = os.path.join(root, "plots")
    logs = os.path.join(root, "logs")
    os.makedirs(tables, exist_ok=True)
    os.makedirs(plots, exist_ok=True)
    os.makedirs(logs, exist_ok=True)
    return tables, plots, logs


def _ci_tdist(values, confidence=0.95):
    arr = np.array(values, dtype=float)
    n = len(arr)
    mean = float(np.mean(arr))
    std = float(np.std(arr, ddof=1)) if n > 1 else 0.0
    if n <= 1:
        return {"mean": mean, "std": std, "ci_low": mean, "ci_high": mean}
    se = std / np.sqrt(n)
    t_crit = stats.t.ppf((1.0 + confidence) / 2.0, df=n - 1)
    margin = float(t_crit * se)
    return {"mean": mean, "std": std, "ci_low": mean - margin, "ci_high": mean + margin}


def _format_ci(mean, low, high):
    return f"{mean:.4f} [{low:.4f}, {high:.4f}]"


def run_multi_seed_methods(config, dataset, seeds, methods, output_root):
    tables_dir, _, logs_dir = _ensure_dirs(output_root)

    raw_rows = []
    for method in methods:
        for seed in seeds:
            run_dir = os.path.join(output_root, "runs", method, f"seed_{seed}")
            metrics = train_variant(method, deepcopy(config), dataset.copy(), seed=seed, output_dir=run_dir)
            row = {"method": method, "seed": seed}
            for m in METRICS + ["mia_auc", "epsilon_eff"]:
                row[m] = float(metrics[m])
            raw_rows.append(row)

    raw_df = pd.DataFrame(raw_rows)
    raw_csv = os.path.join(tables_dir, "raw_multi_seed_results.csv")
    raw_df.to_csv(raw_csv, index=False)

    summary_rows = []
    for method, grp in raw_df.groupby("method"):
        row = {"Method": method}
        for m in METRICS:
            ci = _ci_tdist(grp[m].values)
            row[f"{m}_mean"] = ci["mean"]
            row[f"{m}_std"] = ci["std"]
            row[f"{m}_ci_low"] = ci["ci_low"]
            row[f"{m}_ci_high"] = ci["ci_high"]
            row[f"{m}_mean_ci"] = _format_ci(ci["mean"], ci["ci_low"], ci["ci_high"])
        summary_rows.append(row)

    summary_df = pd.DataFrame(summary_rows).sort_values("Method")
    summary_csv = os.path.join(tables_dir, "method_summary_with_ci.csv")
    summary_df.to_csv(summary_csv, index=False)

    with open(os.path.join(logs_dir, "multi_seed_summary.json"), "w") as f:
        json.dump(summary_rows, f, indent=2)

    print("\nQ1 Summary Table")
    printable = summary_df[[
        "Method",
        "accuracy_mean_ci",
        "recall_parity_mean_ci",
        "privacy_mean_ci",
        "clarity_mean_ci",
        "accountability_mean_ci",
        "trust_index_mean_ci",
    ]]
    print(printable.to_string(index=False))

    return raw_df, summary_df


def run_pareto_lambda_sweep(config, dataset, seed, output_root):
    tables_dir, plots_dir, logs_dir = _ensure_dirs(output_root)

    lambda_values = [0.0, 0.25, 0.5, 0.75, 1.0]
    rows = []
    for lambda_rp in lambda_values:
        for lambda_c in lambda_values:
            cfg = deepcopy(config)
            cfg.setdefault("governance", {})
            cfg["governance"]["lambda_rp"] = float(lambda_rp)
            cfg["governance"]["lambda_c"] = float(lambda_c)
            run_dir = os.path.join(output_root, "pareto_runs", f"lrp_{lambda_rp:.2f}_lc_{lambda_c:.2f}")
            metrics = train_variant("full", cfg, dataset.copy(), seed=seed, output_dir=run_dir)
            rows.append({
                "lambda_rp": float(lambda_rp),
                "lambda_c": float(lambda_c),
                "accuracy": float(metrics["accuracy"]),
                "recall_parity": float(metrics["recall_parity"]),
                "epsilon_eff": float(metrics["epsilon_eff"]),
                "mia_auc": float(metrics["mia_auc"]),
                "clarity": float(metrics["clarity"]),
                "accountability": float(metrics["accountability"]),
                "trust_index": float(metrics["trust_index"]),
                "privacy": float(metrics["privacy"]),
            })

    pareto_df = pd.DataFrame(rows)
    pareto_csv = os.path.join(tables_dir, "pareto_lambda_sweep.csv")
    pareto_df.to_csv(pareto_csv, index=False)
    with open(os.path.join(logs_dir, "pareto_lambda_sweep.json"), "w") as f:
        json.dump(rows, f, indent=2)

    # Plot 1: Accuracy vs Fairness (RP)
    plt.figure(figsize=(7, 5))
    plt.scatter(pareto_df["accuracy"], pareto_df["recall_parity"], c=pareto_df["trust_index"], cmap="viridis")
    plt.xlabel("Accuracy")
    plt.ylabel("Fairness (Recall Parity)")
    plt.title("Accuracy vs Fairness")
    plt.colorbar(label="Trust Index")
    plt.grid(alpha=0.2)
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "accuracy_vs_fairness.png"), dpi=220)
    plt.close()

    # Plot 2: Accuracy vs Privacy (epsilon)
    plt.figure(figsize=(7, 5))
    plt.scatter(pareto_df["accuracy"], pareto_df["epsilon_eff"], c=pareto_df["mia_auc"], cmap="plasma")
    plt.xlabel("Accuracy")
    plt.ylabel("Privacy Budget (epsilon)")
    plt.title("Accuracy vs Privacy Budget")
    plt.colorbar(label="MIA AUC")
    plt.grid(alpha=0.2)
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "accuracy_vs_epsilon.png"), dpi=220)
    plt.close()

    # Plot 3: Trust Index vs lambda_rp (averaged over lambda_c)
    lrp_curve = pareto_df.groupby("lambda_rp", as_index=False)["trust_index"].mean()
    plt.figure(figsize=(7, 5))
    plt.plot(lrp_curve["lambda_rp"], lrp_curve["trust_index"], marker="o")
    plt.xlabel("lambda_rp")
    plt.ylabel("Trust Index")
    plt.title("Trust Index vs lambda_rp")
    plt.grid(alpha=0.2)
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "ti_vs_lambda_rp.png"), dpi=220)
    plt.close()

    # Plot 4: Trust Index vs lambda_c (averaged over lambda_rp)
    lc_curve = pareto_df.groupby("lambda_c", as_index=False)["trust_index"].mean()
    plt.figure(figsize=(7, 5))
    plt.plot(lc_curve["lambda_c"], lc_curve["trust_index"], marker="o")
    plt.xlabel("lambda_c")
    plt.ylabel("Trust Index")
    plt.title("Trust Index vs lambda_c")
    plt.grid(alpha=0.2)
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "ti_vs_lambda_c.png"), dpi=220)
    plt.close()

    return pareto_df


def run_dp_tradeoff(config, dataset, seeds, output_root):
    tables_dir, plots_dir, logs_dir = _ensure_dirs(output_root)

    eps_values = [1.0, 3.0, 5.0, 10.0]
    rows = []
    for eps in eps_values:
        for seed in seeds:
            cfg = deepcopy(config)
            cfg.setdefault("governance", {})
            cfg["governance"]["dp_epsilon"] = float(eps)
            run_dir = os.path.join(output_root, "dp_sweep", f"eps_{eps:.1f}", f"seed_{seed}")
            metrics = train_variant("dp_only", cfg, dataset.copy(), seed=seed, output_dir=run_dir)
            rows.append({
                "epsilon_target": float(eps),
                "seed": int(seed),
                "epsilon_eff": float(metrics["epsilon_eff"]),
                "accuracy": float(metrics["accuracy"]),
                "mia_auc": float(metrics["mia_auc"]),
                "privacy": float(metrics["privacy"]),
            })

    dp_df = pd.DataFrame(rows)
    dp_df.to_csv(os.path.join(tables_dir, "dp_tradeoff.csv"), index=False)
    with open(os.path.join(logs_dir, "dp_tradeoff.json"), "w") as f:
        json.dump(rows, f, indent=2)

    grouped = dp_df.groupby("epsilon_target", as_index=False).mean(numeric_only=True)

    plt.figure(figsize=(7, 5))
    plt.plot(grouped["epsilon_target"], grouped["accuracy"], marker="o")
    plt.xlabel("Target epsilon")
    plt.ylabel("Accuracy")
    plt.title("Accuracy vs epsilon")
    plt.grid(alpha=0.2)
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "dp_accuracy_vs_epsilon.png"), dpi=220)
    plt.close()

    plt.figure(figsize=(7, 5))
    plt.plot(grouped["epsilon_target"], grouped["mia_auc"], marker="o")
    plt.xlabel("Target epsilon")
    plt.ylabel("MIA AUC")
    plt.title("MIA AUC vs epsilon")
    plt.grid(alpha=0.2)
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "dp_mia_vs_epsilon.png"), dpi=220)
    plt.close()

    return dp_df


def run_clarity_validation(config, dataset, seed, output_root):
    _, _, logs_dir = _ensure_dirs(output_root)

    metrics, model = train_variant(
        "full",
        deepcopy(config),
        dataset.copy(),
        seed=seed,
        output_dir=os.path.join(output_root, "clarity_validation", "model"),
        return_model=True,
    )

    X = dataset["X_test"]
    n_eval = min(40, len(X))
    X_eval = X[:n_eval]
    clarity_scores = np.array([
        compute_instance_clarity(model.predict, X_eval, i, n_neighbours=30, rng=np.random.RandomState(seed + i))
        for i in range(n_eval)
    ])

    try:
        shap = importlib.import_module("shap")
    except Exception as exc:
        raise RuntimeError("SHAP is required for clarity validation but is not installed") from exc

    background = dataset["X_train"][:20]
    explainer = shap.KernelExplainer(model.predict_proba, background)
    shap_values = explainer.shap_values(X_eval, nsamples=60)
    if isinstance(shap_values, list):
        sv = np.array(shap_values[1])
    else:
        sv = np.array(shap_values)
        if sv.ndim == 3:
            sv = sv[:, :, 1]

    abs_sv = np.abs(sv)
    topk = np.sort(abs_sv, axis=1)[:, -5:]
    concentration = topk.sum(axis=1) / (abs_sv.sum(axis=1) + 1e-12)

    pearson = stats.pearsonr(clarity_scores, concentration)
    spearman = stats.spearmanr(clarity_scores, concentration)

    result = {
        "n_eval": int(n_eval),
        "pearson_r": float(pearson.statistic),
        "pearson_p": float(pearson.pvalue),
        "spearman_rho": float(spearman.correlation),
        "spearman_p": float(spearman.pvalue),
        "model_clarity": float(metrics["clarity"]),
    }

    with open(os.path.join(logs_dir, "clarity_validation.json"), "w") as f:
        json.dump(result, f, indent=2)
    return result


def run_q1_benchmark_suite(config, dataset, seeds, output_root):
    tables_dir, plots_dir, logs_dir = _ensure_dirs(output_root)

    raw_df, summary_df = run_multi_seed_methods(
        config=config,
        dataset=dataset,
        seeds=seeds,
        methods=CORE_METHODS + ABLATION_METHODS,
        output_root=output_root,
    )

    pareto_df = run_pareto_lambda_sweep(
        config=config,
        dataset=dataset,
        seed=seeds[0],
        output_root=output_root,
    )

    dp_df = run_dp_tradeoff(
        config=config,
        dataset=dataset,
        seeds=seeds,
        output_root=output_root,
    )

    clarity_result = run_clarity_validation(
        config=config,
        dataset=dataset,
        seed=seeds[0],
        output_root=output_root,
    )

    manifest = {
        "tables": sorted([f for f in os.listdir(tables_dir) if f.endswith(".csv")]),
        "plots": sorted([f for f in os.listdir(plots_dir) if f.endswith(".png")]),
        "logs": sorted([f for f in os.listdir(logs_dir) if f.endswith(".json")]),
        "n_raw_rows": int(len(raw_df)),
        "n_pareto_rows": int(len(pareto_df)),
        "n_dp_rows": int(len(dp_df)),
        "clarity_validation": clarity_result,
    }
    with open(os.path.join(logs_dir, "artifact_manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)

    return manifest
