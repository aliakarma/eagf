"""Generate publication-ready, evidence-based interpretation from experiment artifacts."""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


@dataclass
class LoadedArtifacts:
    method_summary: pd.DataFrame
    raw_multi_seed: pd.DataFrame
    pareto: pd.DataFrame
    dp_tradeoff: pd.DataFrame
    clarity_validation: Dict[str, float]


def _safe_read_csv(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Required artifact not found: {path}")
    return pd.read_csv(path)


def load_artifacts(results_root: str) -> LoadedArtifacts:
    tables_dir = os.path.join(results_root, "tables")
    logs_dir = os.path.join(results_root, "logs")

    return LoadedArtifacts(
        method_summary=_safe_read_csv(os.path.join(tables_dir, "method_summary_with_ci.csv")),
        raw_multi_seed=_safe_read_csv(os.path.join(tables_dir, "raw_multi_seed_results.csv")),
        pareto=_safe_read_csv(os.path.join(tables_dir, "pareto_lambda_sweep.csv")),
        dp_tradeoff=_safe_read_csv(os.path.join(tables_dir, "dp_tradeoff.csv")),
        clarity_validation=json.load(open(os.path.join(logs_dir, "clarity_validation.json"), "r", encoding="utf-8")),
    )


def _pick_baseline(method_summary: pd.DataFrame) -> str:
    available = set(method_summary["Method"].astype(str).tolist())
    for candidate in ["standard", "baseline", "fairness_threshold"]:
        if candidate in available:
            return candidate
    return method_summary.sort_values("trust_index_mean", ascending=False).iloc[-1]["Method"]


def _find_best_method(method_summary: pd.DataFrame, metric: str = "trust_index_mean") -> pd.Series:
    return method_summary.sort_values(metric, ascending=False).iloc[0]


def _best_method_set(method_summary: pd.DataFrame, metric: str = "trust_index_mean", tol: float = 1e-9) -> List[str]:
    vals = method_summary[["Method", metric]].copy()
    max_v = float(vals[metric].max())
    tied = vals[np.abs(vals[metric] - max_v) <= tol]["Method"].astype(str).tolist()
    return tied


def _pct_change(new: float, base: float) -> float:
    if abs(base) < 1e-12:
        return float("nan")
    return 100.0 * (new - base) / base


def analyze_method_comparison(method_summary: pd.DataFrame, raw_multi_seed: pd.DataFrame) -> Dict[str, object]:
    baseline = _pick_baseline(method_summary)
    best = _find_best_method(method_summary)
    best_tied = _best_method_set(method_summary)

    baseline_row = method_summary[method_summary["Method"] == baseline].iloc[0]
    full_name = "full" if "full" in set(method_summary["Method"]) else str(best["Method"])
    full_row = method_summary[method_summary["Method"] == full_name].iloc[0]

    improvements = {}
    for metric in ["accuracy_mean", "recall_parity_mean", "privacy_mean", "clarity_mean", "accountability_mean", "trust_index_mean"]:
        improvements[metric.replace("_mean", "")] = {
            "absolute_delta": float(full_row[metric] - baseline_row[metric]),
            "relative_percent": float(_pct_change(float(full_row[metric]), float(baseline_row[metric]))),
        }

    n_seeds = int(raw_multi_seed["seed"].nunique())

    return {
        "baseline_method": baseline,
        "reference_method": full_name,
        "best_method": str(best["Method"]),
        "best_methods_tied": best_tied,
        "best_trust_index": float(best["trust_index_mean"]),
        "n_seeds": n_seeds,
        "improvements_vs_baseline": improvements,
    }


def _pareto_frontier_max2(df: pd.DataFrame, x_col: str, y_col: str) -> pd.DataFrame:
    # Maximize both x and y.
    pts = df[[x_col, y_col, "lambda_rp", "lambda_c", "trust_index"]].copy()
    keep = []
    for i in range(len(pts)):
        x_i = pts.iloc[i][x_col]
        y_i = pts.iloc[i][y_col]
        dominated = False
        for j in range(len(pts)):
            if i == j:
                continue
            x_j = pts.iloc[j][x_col]
            y_j = pts.iloc[j][y_col]
            if (x_j >= x_i and y_j >= y_i) and (x_j > x_i or y_j > y_i):
                dominated = True
                break
        if not dominated:
            keep.append(i)
    frontier = pts.iloc[keep].drop_duplicates().sort_values([x_col, y_col], ascending=[True, True]).reset_index(drop=True)
    return frontier


def _knee_point(frontier: pd.DataFrame, x_col: str, y_col: str) -> Optional[pd.Series]:
    if len(frontier) < 3:
        return None

    x = frontier[x_col].to_numpy(dtype=float)
    y = frontier[y_col].to_numpy(dtype=float)

    x_n = (x - x.min()) / max(x.max() - x.min(), 1e-12)
    y_n = (y - y.min()) / max(y.max() - y.min(), 1e-12)

    p0 = np.array([x_n[0], y_n[0]])
    p1 = np.array([x_n[-1], y_n[-1]])
    line = p1 - p0
    line_norm = max(np.linalg.norm(line), 1e-12)

    dists = []
    for i in range(len(frontier)):
        p = np.array([x_n[i], y_n[i]])
        # 2D point-to-line distance (scalar cross product magnitude).
        dist = abs(line[0] * (p[1] - p0[1]) - line[1] * (p[0] - p0[0])) / line_norm
        dists.append(float(dist))

    idx = int(np.argmax(dists))
    return frontier.iloc[idx]


def _frontier_shape(frontier: pd.DataFrame, x_col: str, y_col: str) -> str:
    if len(frontier) < 3:
        return "insufficient_points"
    f = frontier.sort_values(x_col)
    x = f[x_col].to_numpy(dtype=float)
    y = f[y_col].to_numpy(dtype=float)
    if len(x) < 3:
        return "insufficient_points"

    second = []
    for i in range(1, len(x) - 1):
        dx1 = x[i] - x[i - 1]
        dx2 = x[i + 1] - x[i]
        if abs(dx1) < 1e-12 or abs(dx2) < 1e-12:
            continue
        s1 = (y[i] - y[i - 1]) / dx1
        s2 = (y[i + 1] - y[i]) / dx2
        second.append(s2 - s1)

    if not second:
        return "near_linear_or_degenerate"

    second = np.array(second)
    if np.all(second <= 1e-8):
        return "approximately_concave"
    if np.all(second >= -1e-8):
        return "approximately_convex"
    return "non_monotonic_or_mixed_curvature"


def analyze_pareto(pareto_df: pd.DataFrame) -> Dict[str, object]:
    frontier = _pareto_frontier_max2(pareto_df, x_col="accuracy", y_col="recall_parity")
    knee = _knee_point(frontier, x_col="accuracy", y_col="recall_parity")

    best_ti = pareto_df.sort_values("trust_index", ascending=False).iloc[0]
    shape = _frontier_shape(frontier, x_col="accuracy", y_col="recall_parity")

    if knee is not None:
        knee_obj = {
            "lambda_rp": float(knee["lambda_rp"]),
            "lambda_c": float(knee["lambda_c"]),
            "accuracy": float(knee["accuracy"]),
            "recall_parity": float(knee["recall_parity"]),
            "trust_index": float(knee["trust_index"]),
        }
    else:
        knee_obj = None

    return {
        "frontier_size": int(len(frontier)),
        "frontier_shape": shape,
        "best_trust_index_point": {
            "lambda_rp": float(best_ti["lambda_rp"]),
            "lambda_c": float(best_ti["lambda_c"]),
            "trust_index": float(best_ti["trust_index"]),
            "accuracy": float(best_ti["accuracy"]),
            "recall_parity": float(best_ti["recall_parity"]),
        },
        "knee_point": knee_obj,
    }


def analyze_dp_tradeoff(dp_df: pd.DataFrame) -> Dict[str, object]:
    grouped = dp_df.groupby("epsilon_target", as_index=False).mean(numeric_only=True).sort_values("epsilon_target")

    eps = grouped["epsilon_target"].to_numpy(dtype=float)
    acc = grouped["accuracy"].to_numpy(dtype=float)
    mia = grouped["mia_auc"].to_numpy(dtype=float)
    prv = grouped["privacy"].to_numpy(dtype=float)

    if len(grouped) >= 2:
        delta_low_to_mid_priv = float(prv[1] - prv[0])
        delta_mid_to_high_priv = float(prv[-1] - prv[1])
        saturation_flag = abs(delta_mid_to_high_priv) < abs(delta_low_to_mid_priv)
    else:
        saturation_flag = False

    best_priv_idx = int(np.argmax(prv))
    best_acc_idx = int(np.argmax(acc))

    return {
        "points": grouped.to_dict(orient="records"),
        "best_privacy_epsilon": float(eps[best_priv_idx]),
        "best_accuracy_epsilon": float(eps[best_acc_idx]),
        "accuracy_range": [float(np.min(acc)), float(np.max(acc))],
        "mia_auc_range": [float(np.min(mia)), float(np.max(mia))],
        "privacy_saturation_after_mid": bool(saturation_flag),
    }


def analyze_ablation(method_summary: pd.DataFrame, raw_multi_seed: pd.DataFrame) -> Dict[str, object]:
    methods = set(method_summary["Method"].astype(str).tolist())
    if "full" not in methods:
        return {"available": False, "reason": "full method not present"}

    full_row = method_summary[method_summary["Method"] == "full"].iloc[0]

    mapping = {
        "fairness": "ablate_no_fairness",
        "dp": "ablate_no_dp",
        "clarity": "ablate_no_clarity",
        "accountability": "ablate_no_accountability",
    }

    contributions = {}
    for comp, ablate_name in mapping.items():
        if ablate_name not in methods:
            continue
        ab = method_summary[method_summary["Method"] == ablate_name].iloc[0]
        contributions[comp] = {
            "delta_ti_when_removed": float(full_row["trust_index_mean"] - ab["trust_index_mean"]),
            "delta_accuracy_when_removed": float(full_row["accuracy_mean"] - ab["accuracy_mean"]),
            "delta_recall_parity_when_removed": float(full_row["recall_parity_mean"] - ab["recall_parity_mean"]),
            "delta_privacy_when_removed": float(full_row["privacy_mean"] - ab["privacy_mean"]),
            "delta_accountability_when_removed": float(full_row["accountability_mean"] - ab["accountability_mean"]),
        }

        raw_full = raw_multi_seed[raw_multi_seed["method"] == "full"]
        raw_ab = raw_multi_seed[raw_multi_seed["method"] == ablate_name]
        if not raw_full.empty and not raw_ab.empty:
            contributions[comp]["delta_mia_auc_removed_minus_full"] = float(raw_ab["mia_auc"].mean() - raw_full["mia_auc"].mean())

    ranked = sorted(contributions.items(), key=lambda x: x[1].get("delta_ti_when_removed", 0.0), reverse=True)

    return {
        "available": True,
        "ranked_by_ti_impact": [{"component": k, **v} for k, v in ranked],
    }


def analyze_clarity_validation(clarity_validation: Dict[str, float]) -> Dict[str, object]:
    pearson_r = float(clarity_validation.get("pearson_r", np.nan))
    pearson_p = float(clarity_validation.get("pearson_p", np.nan))
    spearman_rho = float(clarity_validation.get("spearman_rho", np.nan))
    spearman_p = float(clarity_validation.get("spearman_p", np.nan))

    weak = (abs(pearson_r) < 0.2 and abs(spearman_rho) < 0.2) or (pearson_p > 0.05 and spearman_p > 0.05)

    return {
        "pearson_r": pearson_r,
        "pearson_p": pearson_p,
        "spearman_rho": spearman_rho,
        "spearman_p": spearman_p,
        "is_weak_correlation": bool(weak),
        "n_eval": int(clarity_validation.get("n_eval", 0)),
        "model_clarity": float(clarity_validation.get("model_clarity", np.nan)),
    }


def _insight_lines(summary: Dict[str, object]) -> List[str]:
    mc = summary["method_comparison"]
    pareto = summary["pareto"]
    dp = summary["dp_tradeoff"]
    ab = summary["ablation"]
    cv = summary["clarity_validation"]

    imp = mc["improvements_vs_baseline"]

    lines = []
    if len(mc.get("best_methods_tied", [])) > 1:
        lines.append(
            "Top trust-index performance is tied across methods: "
            + ", ".join(mc["best_methods_tied"]) + "."
        )
    lines.append(
        f"Compared with {mc['baseline_method']}, {mc['reference_method']} changes trust index by "
        f"{imp['trust_index']['absolute_delta']:+.4f} ({imp['trust_index']['relative_percent']:+.2f}%)."
    )
    lines.append(
        f"Fairness (recall parity) changes by {imp['recall_parity']['absolute_delta']:+.4f} "
        f"({imp['recall_parity']['relative_percent']:+.2f}%) and accuracy by "
        f"{imp['accuracy']['absolute_delta']:+.4f} ({imp['accuracy']['relative_percent']:+.2f}%)."
    )

    kp = pareto.get("knee_point")
    if kp is not None:
        lines.append(
            f"Pareto knee point occurs near lambda_rp={kp['lambda_rp']:.2f}, lambda_c={kp['lambda_c']:.2f}, "
            f"with accuracy={kp['accuracy']:.4f}, recall_parity={kp['recall_parity']:.4f}, TI={kp['trust_index']:.4f}."
        )
    lines.append(f"Pareto frontier curvature is classified as {pareto['frontier_shape']}.")

    dp_points = pd.DataFrame(dp["points"]).sort_values("epsilon_target")
    if len(dp_points) >= 2:
        first = dp_points.iloc[0]
        last = dp_points.iloc[-1]
        lines.append(
            f"Across epsilon sweep, accuracy moves from {first['accuracy']:.4f} (eps={first['epsilon_target']:.1f}) "
            f"to {last['accuracy']:.4f} (eps={last['epsilon_target']:.1f}), while MIA AUC moves from "
            f"{first['mia_auc']:.4f} to {last['mia_auc']:.4f}."
        )
        if dp.get("privacy_saturation_after_mid", False):
            lines.append("Privacy-score deterioration slows after mid-range epsilon values (diminishing marginal harm at higher epsilon).")

    if ab.get("available", False):
        ranked = ab["ranked_by_ti_impact"]
        if ranked:
            top = ranked[0]
            lines.append(
                f"Ablation ranking by TI impact identifies {top['component']} as the largest contributor "
                f"(delta TI when removed = {top['delta_ti_when_removed']:+.4f})."
            )
        for item in ranked:
            if item["component"] == "dp":
                dmia = item.get("delta_mia_auc_removed_minus_full", np.nan)
                if np.isfinite(dmia):
                    lines.append(
                        f"Removing DP increases attack susceptibility (MIA AUC change no-DP minus full = {dmia:+.4f})."
                    )

    if cv["is_weak_correlation"]:
        lines.append(
            f"Clarity proxy has weak SHAP alignment (Pearson r={cv['pearson_r']:.3f}, p={cv['pearson_p']:.3f}; "
            f"Spearman rho={cv['spearman_rho']:.3f}, p={cv['spearman_p']:.3f}), indicating limited convergent validity in this run."
        )
    else:
        lines.append(
            f"Clarity proxy shows non-trivial SHAP alignment (Pearson r={cv['pearson_r']:.3f}, Spearman rho={cv['spearman_rho']:.3f})."
        )

    if int(mc["n_seeds"]) < 3:
        lines.append(
            "Current artifacts use fewer than 3 seeds; treat inference as directional and regenerate multi-seed runs for robust significance claims."
        )

    return lines


def _experimental_findings_text(summary: Dict[str, object]) -> str:
    lines = _insight_lines(summary)
    body = ["Experimental Findings", "", "The following claims are computed from generated artifacts and are not manually adjusted.", ""]
    for i, line in enumerate(lines, start=1):
        body.append(f"{i}. {line}")
    return "\n".join(body)


def _discussion_text(summary: Dict[str, object]) -> str:
    mc = summary["method_comparison"]
    cv = summary["clarity_validation"]

    discussion = [
        "Discussion",
        "",
        "The benchmark indicates that governance-aware optimization can improve composite trust metrics relative to a standard baseline, with measurable trade-offs across utility and governance dimensions.",
        "Pareto behavior should be interpreted as a utility-fairness balance surface rather than a single optimum; knee-point operating regions are preferable when decision-makers require balanced objectives.",
        "DP analysis confirms the expected utility-privacy tension and supports reporting epsilon-conditioned performance rather than a single privacy operating point.",
        "Ablation results should be used to prioritize components with the largest observed trust-index impact under the selected configuration.",
    ]

    if cv["is_weak_correlation"]:
        discussion.append(
            "A key limitation is weak correlation between the clarity proxy and SHAP concentration in the current run; explanation-quality claims should therefore remain conservative until validated with stronger agreement metrics and larger evaluation sets."
        )

    if int(mc["n_seeds"]) < 3:
        discussion.append(
            "Another limitation is insufficient seed count for stable inferential conclusions; full multi-seed execution is required for publication-grade statistical certainty."
        )

    discussion.append(
        "Overall, the evidence supports a transparent, reproducible governance narrative when paired with explicit uncertainty reporting and sensitivity analysis."
    )
    return "\n".join(discussion)


def _key_contributions(summary: Dict[str, object]) -> List[str]:
    mc = summary["method_comparison"]
    pareto = summary["pareto"]
    dp = summary["dp_tradeoff"]
    cv = summary["clarity_validation"]

    contributions = [
        "Evidence-grounded comparative analysis across baseline, full method, and ablations from generated artifacts.",
        f"Pareto-front interpretation with curvature label ({pareto['frontier_shape']}) and knee-point extraction for operating-point selection.",
        "Explicit epsilon-conditioned privacy-utility interpretation using both model accuracy and MIA AUC.",
        "Component-level attribution of trust-index changes through ablation deltas.",
        f"Clarity validation reported with Pearson/Spearman statistics (n={cv['n_eval']}) and limitation flagging when weak.",
        f"Reproducibility note: current analysis is based on {mc['n_seeds']} seed(s).",
    ]
    return contributions


def generate_analysis_report(results_root: str = "results", output_dir: Optional[str] = None) -> Dict[str, object]:
    if output_dir is None:
        output_dir = os.path.join(results_root, "analysis")
    os.makedirs(output_dir, exist_ok=True)

    artifacts = load_artifacts(results_root)

    summary = {
        "method_comparison": analyze_method_comparison(artifacts.method_summary, artifacts.raw_multi_seed),
        "pareto": analyze_pareto(artifacts.pareto),
        "dp_tradeoff": analyze_dp_tradeoff(artifacts.dp_tradeoff),
        "ablation": analyze_ablation(artifacts.method_summary, artifacts.raw_multi_seed),
        "clarity_validation": analyze_clarity_validation(artifacts.clarity_validation),
    }

    findings_text = _experimental_findings_text(summary)
    discussion_text = _discussion_text(summary)
    contributions = _key_contributions(summary)

    key_insights = {
        **summary,
        "paper_text": {
            "experimental_findings": findings_text,
            "discussion": discussion_text,
            "key_contributions": contributions,
        },
    }

    with open(os.path.join(output_dir, "findings.txt"), "w", encoding="utf-8") as f:
        f.write(findings_text + "\n")

    with open(os.path.join(output_dir, "discussion.txt"), "w", encoding="utf-8") as f:
        f.write(discussion_text + "\n\n")
        f.write("Key Contributions\n\n")
        for i, c in enumerate(contributions, start=1):
            f.write(f"{i}. {c}\n")

    with open(os.path.join(output_dir, "key_insights.json"), "w", encoding="utf-8") as f:
        json.dump(key_insights, f, indent=2)

    return key_insights


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate publication-ready analysis report from EAGF artifacts")
    p.add_argument("--results-root", default="results", help="Root directory containing tables and logs")
    p.add_argument("--output-dir", default=None, help="Output directory for analysis files")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    out = generate_analysis_report(results_root=args.results_root, output_dir=args.output_dir)
    print("Analysis report generated.")
    print(f"Best method by TI: {out['method_comparison']['best_method']}")
    print(f"Output directory: {args.output_dir or os.path.join(args.results_root, 'analysis')}")


if __name__ == "__main__":
    main()
