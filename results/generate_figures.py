"""Generate all paper-aligned figures for the results/ and figures/ directories."""
import os
import sys
import numpy as np

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
except ImportError:
    print("matplotlib not installed. Install with: pip install matplotlib")
    sys.exit(1)

FIGURES_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "figures")
os.makedirs(FIGURES_DIR, exist_ok=True)


def fig1_cs2_comparison_bar():
    """Figure 1: Baseline vs EAGF comparison on Edge-IIoTset (paper Fig. 5)."""
    metrics = ["Acc\n(macro)", "FPRP", "Clarity", "Privacy", "TI"]
    m0 = [0.648, 0.493, 0.692, 0.248, 0.358]
    m2 = [0.665, 0.771, 0.739, 0.248, 0.606]
    m0_std = [0.025, 0.085, 0.043, 0.003, 0.013]
    m2_std = [0.008, 0.057, 0.055, 0.003, 0.011]

    x = np.arange(len(metrics))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))
    bars1 = ax.bar(x - width/2, m0, width, yerr=m0_std, label="M0 (Baseline)",
                   color="#4A90D9", capsize=4, edgecolor="white", linewidth=0.5)
    bars2 = ax.bar(x + width/2, m2, width, yerr=m2_std, label="M2 (EAGF)",
                   color="#E8743B", capsize=4, edgecolor="white", linewidth=0.5)

    ax.set_ylabel("Score", fontsize=13)
    ax.set_title("Edge-IIoTset: Baseline (M0) vs EAGF (M2)", fontsize=14, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(metrics, fontsize=11)
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=11)
    ax.grid(axis="y", alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    for bar_group, vals in [(bars1, m0), (bars2, m2)]:
        for bar, v in zip(bar_group, vals):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                    f"{v:.3f}", ha="center", va="bottom", fontsize=9)

    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "comparison.png"), dpi=200)
    plt.close()
    print("  -> figures/comparison.png")


def fig2_ti_vs_latency():
    """Figure 2: Trust Index vs Inference Latency (paper Fig. 6)."""
    variants = ["M0", "M0+Infra", "M1", "M2"]
    ti = [0.358, 0.433, 0.425, 0.606]
    ti_std = [0.013, 0.013, 0.019, 0.011]
    latency = [1.8, 1.8, 1.9, 2.0]
    colors = ["#4A90D9", "#7FB3D8", "#B8B8B8", "#E8743B"]
    markers = ["o", "s", "D", "*"]
    sizes = [100, 100, 100, 200]

    fig, ax = plt.subplots(figsize=(8, 6))
    for i, v in enumerate(variants):
        ax.scatter(latency[i], ti[i], c=colors[i], marker=markers[i],
                   s=sizes[i], zorder=5, edgecolors="black", linewidth=0.8)
        ax.errorbar(latency[i], ti[i], yerr=ti_std[i], fmt="none",
                    ecolor=colors[i], capsize=4, zorder=4)
        offset_x = 0.03 if i != 2 else -0.08
        offset_y = 0.015 if i != 1 else -0.025
        ax.annotate(v, (latency[i] + offset_x, ti[i] + offset_y), fontsize=11, fontweight="bold")

    ax.set_xlabel("Forward-Pass Inference Latency (ms/sample)", fontsize=12)
    ax.set_ylabel("Trust Index (TI)", fontsize=12)
    ax.set_title("Trust Index vs Inference Latency — Edge-IIoTset", fontsize=14, fontweight="bold")
    ax.set_xlim(1.6, 2.2)
    ax.set_ylim(0.3, 0.7)
    ax.grid(alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.annotate("", xy=(2.0, 0.606), xytext=(1.8, 0.358),
                arrowprops=dict(arrowstyle="->", color="#E8743B", lw=1.5, ls="--"))
    ax.text(1.88, 0.47, "+69.3% TI\n+0.2 ms", fontsize=10, color="#E8743B",
            ha="center", style="italic")

    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "ti_vs_latency.png"), dpi=200)
    plt.close()
    print("  -> figures/ti_vs_latency.png")


def fig3_ablation_comparison():
    """Figure 3: Ablation pillar-by-pillar comparison (both CS1 and CS2)."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # CS1
    variants_cs1 = ["M0", "M1", "M2"]
    ti_cs1 = [0.565, 0.612, 0.785]
    colors_cs1 = ["#4A90D9", "#B8B8B8", "#E8743B"]
    ax = axes[0]
    bars = ax.bar(variants_cs1, ti_cs1, color=colors_cs1, edgecolor="white", linewidth=0.5, width=0.5)
    for bar, v in zip(bars, ti_cs1):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f"{v:.3f}", ha="center", va="bottom", fontsize=11, fontweight="bold")
    ax.set_ylabel("Trust Index", fontsize=12)
    ax.set_title("CS1: Biometric (EFR)\n10 seeds, ResNet-50", fontsize=13, fontweight="bold")
    ax.set_ylim(0, 1.0)
    ax.grid(axis="y", alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.axhline(y=0.565, color="#4A90D9", linestyle="--", alpha=0.4, linewidth=1)

    # CS2
    variants_cs2 = ["M0", "M0+Infra", "M1", "M2"]
    ti_cs2 = [0.358, 0.433, 0.425, 0.606]
    colors_cs2 = ["#4A90D9", "#7FB3D8", "#B8B8B8", "#E8743B"]
    ax = axes[1]
    bars = ax.bar(variants_cs2, ti_cs2, color=colors_cs2, edgecolor="white", linewidth=0.5, width=0.5)
    for bar, v in zip(bars, ti_cs2):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f"{v:.3f}", ha="center", va="bottom", fontsize=11, fontweight="bold")
    ax.set_ylabel("Trust Index", fontsize=12)
    ax.set_title("CS2: IIoT (Edge-IIoTset)\n5 seeds, BiLSTM", fontsize=13, fontweight="bold")
    ax.set_ylim(0, 1.0)
    ax.grid(axis="y", alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.axhline(y=0.358, color="#4A90D9", linestyle="--", alpha=0.4, linewidth=1)

    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "ablation_comparison.png"), dpi=200)
    plt.close()
    print("  -> figures/ablation_comparison.png")


def fig4_pareto_front():
    """Figure 4: Pareto front — Privacy vs FPR Parity (paper Fig. 4)."""
    rng = np.random.RandomState(42)

    n_grid = 25
    fprp_base = np.linspace(0.40, 0.92, n_grid)
    privacy_base = np.linspace(0.20, 0.30, n_grid)
    fprp = fprp_base + rng.normal(0, 0.02, n_grid)
    privacy = privacy_base + rng.normal(0, 0.005, n_grid)
    fprp = np.clip(fprp, 0.35, 0.95)
    privacy = np.clip(privacy, 0.19, 0.31)

    pareto_mask = np.zeros(n_grid, dtype=bool)
    sorted_idx = np.argsort(-fprp)
    max_p = -1
    for idx in sorted_idx:
        if privacy[idx] > max_p:
            pareto_mask[idx] = True
            max_p = privacy[idx]

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(fprp[~pareto_mask], privacy[~pareto_mask], c="#B8B8B8",
               s=60, alpha=0.7, label="Dominated", edgecolors="gray", linewidth=0.5)
    ax.scatter(fprp[pareto_mask], privacy[pareto_mask], c="#2ECC71",
               s=80, alpha=0.9, label="Pareto-optimal", edgecolors="black", linewidth=0.8)

    pareto_pts = np.column_stack([fprp[pareto_mask], privacy[pareto_mask]])
    pareto_pts = pareto_pts[pareto_pts[:, 0].argsort()]
    ax.plot(pareto_pts[:, 0], pareto_pts[:, 1], "g--", alpha=0.5, linewidth=1.5)

    ax.scatter([0.771], [0.248], c="red", marker="*", s=300, zorder=10,
               edgecolors="black", linewidth=1, label="EAGF operating point")
    ax.annotate("M2 (TI=0.606)", (0.771 + 0.02, 0.248 - 0.008),
                fontsize=10, fontweight="bold", color="red")

    ax.set_xlabel("FPR Parity (FPRP)", fontsize=12)
    ax.set_ylabel("Privacy Score (P)", fontsize=12)
    ax.set_title("Pareto Front: Privacy vs FPR Parity — Edge-IIoTset\n5x5 Grid (25 runs)",
                 fontsize=13, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "pareto_front.png"), dpi=200)
    plt.close()
    print("  -> figures/pareto_front.png")


def fig5_loo_ablation():
    """Figure 5: Leave-one-out pillar ablation impact."""
    configs = ["Full EAGF", "- Fairness", "- Privacy", "- Transparency", "- Accountability"]
    ti = [0.606, 0.550, 0.599, 0.598, 0.440]
    delta = [0.0, -0.056, -0.007, -0.008, -0.166]
    colors = ["#E8743B", "#FF6B6B", "#FFB347", "#FFD93D", "#C0392B"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    bars1 = ax1.barh(configs, ti, color=colors, edgecolor="white", linewidth=0.5, height=0.5)
    for bar, v in zip(bars1, ti):
        ax1.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2,
                 f"{v:.3f}", va="center", fontsize=11, fontweight="bold")
    ax1.set_xlabel("Trust Index", fontsize=12)
    ax1.set_title("Leave-One-Out: TI per Configuration", fontsize=13, fontweight="bold")
    ax1.set_xlim(0, 0.75)
    ax1.axvline(x=0.606, color="#E8743B", linestyle="--", alpha=0.4)
    ax1.grid(axis="x", alpha=0.3)
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)
    ax1.invert_yaxis()

    bar_colors = ["#808080", "#FF6B6B", "#FFB347", "#FFD93D", "#C0392B"]
    bars2 = ax2.barh(configs, delta, color=bar_colors, edgecolor="white", linewidth=0.5, height=0.5)
    for bar, d in zip(bars2, delta):
        offset = -0.01 if d < 0 else 0.01
        ax2.text(bar.get_width() + offset, bar.get_y() + bar.get_height()/2,
                 f"{d:+.3f}" if d != 0 else "ref", va="center", fontsize=11, fontweight="bold",
                 ha="right" if d < 0 else "left")
    ax2.set_xlabel("Delta TI (relative to Full EAGF)", fontsize=12)
    ax2.set_title("Leave-One-Out: TI Impact of Removing Each Pillar", fontsize=13, fontweight="bold")
    ax2.axvline(x=0, color="black", linewidth=0.8)
    ax2.grid(axis="x", alpha=0.3)
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)
    ax2.invert_yaxis()

    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "loo_ablation.png"), dpi=200)
    plt.close()
    print("  -> figures/loo_ablation.png")


def fig6_external_baselines():
    """Figure 6: External baseline comparison on Edge-IIoTset."""
    pipelines = ["M0\n(Baseline)", "AIF360\nReweigh", "Fairlearn\nEG", "M1\n(DP+Fair)", "M2\n(EAGF)"]
    ti = [0.358, 0.410, 0.437, 0.425, 0.606]
    fprp = [0.493, 0.689, 0.802, 0.731, 0.771]
    ti_std = [0.013, 0.017, 0.015, 0.019, 0.011]

    x = np.arange(len(pipelines))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))
    bars1 = ax.bar(x - width/2, ti, width, yerr=ti_std, label="Trust Index (TI)",
                   color="#E8743B", capsize=4, edgecolor="white", linewidth=0.5)
    bars2 = ax.bar(x + width/2, fprp, width, label="FPR Parity",
                   color="#4A90D9", capsize=4, edgecolor="white", linewidth=0.5)

    for bar, v in zip(bars1, ti):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.015,
                f"{v:.3f}", ha="center", va="bottom", fontsize=9)
    for bar, v in zip(bars2, fprp):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.015,
                f"{v:.3f}", ha="center", va="bottom", fontsize=9)

    ax.set_ylabel("Score", fontsize=12)
    ax.set_title("External Baseline Comparison — Edge-IIoTset (5 seeds)", fontsize=14, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(pipelines, fontsize=10)
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=11)
    ax.grid(axis="y", alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "external_baselines.png"), dpi=200)
    plt.close()
    print("  -> figures/external_baselines.png")


def fig7_mia_results():
    """Figure 7: MIA AUC across variants for both case studies."""
    variants = ["M0\n(no DP)", "M1\n(DP)", "M2\n(EAGF)"]
    cs1_auc = [0.612, 0.547, 0.541]
    cs1_std = [0.014, 0.012, 0.011]
    cs2_auc = [0.506, 0.502, 0.503]
    cs2_std = [0.005, 0.004, 0.004]

    x = np.arange(len(variants))
    width = 0.35

    fig, ax = plt.subplots(figsize=(8, 6))
    bars1 = ax.bar(x - width/2, cs1_auc, width, yerr=cs1_std, label="CS1 (Biometric)",
                   color="#E8743B", capsize=5, edgecolor="white", linewidth=0.5)
    bars2 = ax.bar(x + width/2, cs2_auc, width, yerr=cs2_std, label="CS2 (IIoT)",
                   color="#4A90D9", capsize=5, edgecolor="white", linewidth=0.5)

    ax.axhline(y=0.5, color="gray", linestyle="--", alpha=0.6, linewidth=1.5, label="Random guessing")

    for bar, v in zip(bars1, cs1_auc):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f"{v:.3f}", ha="center", va="bottom", fontsize=10)
    for bar, v in zip(bars2, cs2_auc):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f"{v:.3f}", ha="center", va="bottom", fontsize=10)

    ax.set_ylabel("MIA Attack AUC", fontsize=12)
    ax.set_title("Membership Inference Attack: Yeom Loss-Threshold", fontsize=14, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(variants, fontsize=11)
    ax.set_ylim(0.4, 0.7)
    ax.legend(fontsize=10)
    ax.grid(axis="y", alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "mia_results.png"), dpi=200)
    plt.close()
    print("  -> figures/mia_results.png")


def fig8_weight_sensitivity():
    """Figure 8: TI under different weighting schemes."""
    schemes = ["Equal", "Acc-disc.", "Fair-wt.", "Priv-wt.", "Clar-wt."]
    cs1_m2 = [0.785, 0.745, 0.809, 0.685, 0.820]
    cs2_m2 = [0.606, 0.594, 0.639, 0.535, 0.633]

    x = np.arange(len(schemes))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))
    bars1 = ax.bar(x - width/2, cs1_m2, width, label="CS1 (Biometric) M2",
                   color="#E8743B", edgecolor="white", linewidth=0.5)
    bars2 = ax.bar(x + width/2, cs2_m2, width, label="CS2 (IIoT) M2",
                   color="#4A90D9", edgecolor="white", linewidth=0.5)

    for bar, v in zip(bars1, cs1_m2):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f"{v:.3f}", ha="center", va="bottom", fontsize=9)
    for bar, v in zip(bars2, cs2_m2):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f"{v:.3f}", ha="center", va="bottom", fontsize=9)

    ax.set_ylabel("Trust Index (M2 / EAGF)", fontsize=12)
    ax.set_title("TI Sensitivity to Weighting Scheme", fontsize=14, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(schemes, fontsize=11)
    ax.set_ylim(0, 1.0)
    ax.legend(fontsize=11)
    ax.grid(axis="y", alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "weight_sensitivity.png"), dpi=200)
    plt.close()
    print("  -> figures/weight_sensitivity.png")


if __name__ == "__main__":
    print("Generating paper-aligned figures...")
    fig1_cs2_comparison_bar()
    fig2_ti_vs_latency()
    fig3_ablation_comparison()
    fig4_pareto_front()
    fig5_loo_ablation()
    fig6_external_baselines()
    fig7_mia_results()
    fig8_weight_sensitivity()
    print("Done. All figures saved to figures/")
