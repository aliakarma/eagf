"""
src/evaluation/statistics.py — Statistical Significance Tests
Paper: Section 5.1.3 (two-proportion z-test, Wilcoxon, bootstrap)
"""
import json, os
import numpy as np
from scipy import stats


def two_proportion_ztest(acc_a, acc_b, n_a, n_b):
    """Two-proportion z-test for accuracy comparison."""
    p_pool = (acc_a * n_a + acc_b * n_b) / (n_a + n_b)
    se = np.sqrt(p_pool * (1 - p_pool) * (1/n_a + 1/n_b))
    if se == 0:
        return {"z": 0.0, "p_value": 1.0}
    z = (acc_a - acc_b) / se
    p = 2 * (1 - stats.norm.cdf(abs(z)))
    return {"z": float(z), "p_value": float(p), "significant": bool(p < 0.05)}


def wilcoxon_test(x, y=None, alternative="two-sided", method="auto"):
    """Wilcoxon signed-rank test (exact method for small n).

    For n=10 (CS1), exact Wilcoxon gives a minimum p of 0.002.
    """
    if y is None:
        y = np.zeros_like(x)
    x, y = np.array(x), np.array(y)
    diffs = x - y
    n = len(diffs)
    if n < 2:
        return {"statistic": 0.0, "p_value": 1.0,
                "note": "Inconclusive: need >= 2 replicates for significance testing",
                "significant": False}
    if n == 2:
        try:
            stat, p = stats.ttest_rel(x, y)
            return {"statistic": float(stat), "p_value": float(p),
                    "test": "paired_ttest_n2", "significant": bool(p < 0.05)}
        except Exception as e:
            return {"statistic": 0.0, "p_value": 1.0, "error": str(e)}
    try:
        wil_method = method if method != "auto" else ("exact" if n <= 25 else "approx")
        stat, p = stats.wilcoxon(diffs, alternative=alternative, method=wil_method)
        return {"statistic": float(stat), "p_value": float(p),
                "test": f"wilcoxon_{wil_method}", "n": n,
                "significant": bool(p < 0.05)}
    except Exception as e:
        try:
            stat, p = stats.wilcoxon(diffs, alternative=alternative)
            return {"statistic": float(stat), "p_value": float(p),
                    "significant": bool(p < 0.05)}
        except Exception as e2:
            return {"statistic": 0.0, "p_value": 1.0, "error": str(e2)}


def paired_ttest(x, y):
    """Paired t-test with degrees of freedom reporting (for CS2, n=5)."""
    x, y = np.array(x), np.array(y)
    n = len(x)
    if n < 2:
        return {"statistic": 0.0, "p_value": 1.0, "df": 0, "significant": False}
    try:
        stat, p = stats.ttest_rel(x, y)
        return {"statistic": float(stat), "p_value": float(p),
                "df": n - 1, "test": f"paired_ttest_t({n-1})",
                "significant": bool(p < 0.05)}
    except Exception as e:
        return {"statistic": 0.0, "p_value": 1.0, "df": n - 1, "error": str(e)}


def bootstrap_ci(values, n_resamples=1000, ci=0.95):
    values = np.array(values)
    resampled = [np.mean(np.random.choice(values, len(values), replace=True))
                 for _ in range(n_resamples)]
    alpha = (1 - ci) / 2
    return {
        "mean": float(np.mean(values)),
        "std": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
        "ci_lower": float(np.percentile(resampled, 100 * alpha)),
        "ci_upper": float(np.percentile(resampled, 100 * (1 - alpha))),
    }


def run_all_tests(baseline_dir, eagf_dir, seeds, output_path, n_test=1500,
                   case_study="cs2"):
    """Run all statistical tests comparing baseline vs EAGF."""
    METRICS = ["accuracy", "recall_parity", "clarity", "privacy",
               "accountability", "trust_index"]
    baseline_vals = {m: [] for m in METRICS}
    eagf_vals     = {m: [] for m in METRICS}

    # Enforce paired-seed comparisons: only seeds available in BOTH arms are used.
    paired_seeds = []
    for seed in seeds:
        b_path = os.path.join(baseline_dir, f"seed_{seed}", "results.json")
        e_path = os.path.join(eagf_dir, f"seed_{seed}", "results.json")
        if not (os.path.exists(b_path) and os.path.exists(e_path)):
            continue

        with open(b_path) as f:
            b_res = json.load(f)
        with open(e_path) as f:
            e_res = json.load(f)

        for m in METRICS:
            baseline_vals[m].append(float(b_res.get(m, 0.0)))
            eagf_vals[m].append(float(e_res.get(m, 0.0)))
        paired_seeds.append(int(seed))

    results = {}
    for m in METRICS:
        if not baseline_vals[m] or not eagf_vals[m]:
            continue
        b, e = np.array(baseline_vals[m]), np.array(eagf_vals[m])
        results[m] = {
            "baseline":   bootstrap_ci(b),
            "eagf":       bootstrap_ci(e),
            "delta_mean": float(np.mean(e) - np.mean(b)),
            "n_replicates": len(e),
            "paired_seeds": paired_seeds,
        }
        if case_study == "cs1":
            # CS1 (n=10): exact Wilcoxon signed-rank
            results[m]["wilcoxon"] = wilcoxon_test(e, b, method="exact")
        elif case_study == "cs2":
            # CS2 (n=5): paired t-test
            results[m]["paired_ttest"] = paired_ttest(e, b)
        else:
            if len(e) >= 10:
                results[m]["wilcoxon"] = wilcoxon_test(e, b, method="exact")
            elif len(e) >= 5:
                results[m]["paired_ttest"] = paired_ttest(e, b)
            else:
                results[m]["paired_ttest"] = paired_ttest(e, b)

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Statistical tests -> {output_path}")
    return results


def parse_args():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--baseline-dir", required=True)
    p.add_argument("--eagf-dir",     required=True)
    p.add_argument("--seeds", nargs="+", type=int,
                   default=[42, 123, 456, 789, 101, 202, 303, 404, 505, 606])
    p.add_argument("--output", required=True)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_all_tests(args.baseline_dir, args.eagf_dir, args.seeds, args.output)
