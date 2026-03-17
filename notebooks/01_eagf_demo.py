#!/usr/bin/env python3
"""
notebooks/01_eagf_demo.py
EAGF Interactive Demo — can be run as a script or opened as a Jupyter notebook.

Demonstrates the full four-pillar governance framework on synthetic data:
  1. Generate demographically-biased dataset
  2. Train baseline vs EAGF model
  3. Compare all five metrics (Accuracy, RP, C, P, A, TI)
  4. Show ablation table
  5. Plot Figure 3

Usage:
  python notebooks/01_eagf_demo.py
  # or: jupyter nbconvert --to notebook --execute notebooks/01_eagf_demo.py
"""
# %%
import sys, os, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import yaml

print("=" * 60)
print("  EAGF Governance Framework — Interactive Demo")
print("=" * 60)

# ── 1. Load config and generate demo data ────────────────────
# %%
with open("configs/biometric_default.yaml") as f:
    config = yaml.safe_load(f)
config["training"]["epochs"] = 30   # fast for demo

from src.utils.data_loader import generate_demo_biometric
dataset = generate_demo_biometric(n_samples=1200, seed=42)

print(f"\nDataset: {dataset['X_train'].shape[0]} train / "
      f"{dataset['X_val'].shape[0]} val / "
      f"{dataset['X_test'].shape[0]} test")
print(f"Demographic groups: {list(np.unique(dataset['groups_test']))}")

# ── 2. Run all six ablation variants ─────────────────────────
# %%
from src.training.eagf_trainer import train_variant

print("\nTraining six model variants (M0-M5)...")
VARIANTS = ["baseline", "transparency", "fairness", "privacy", "accountability", "eagf"]
LABELS   = {"baseline":"M0: Baseline", "transparency":"M1: +Transparency",
             "fairness":"M2: +Fairness", "privacy":"M3: +Privacy",
             "accountability":"M4: +Accountability", "eagf":"M5: EAGF (Full)"}

results = {}
for v in VARIANTS:
    m = train_variant(v, config, dataset.copy(), seed=42,
                      output_dir=f"/tmp/eagf_demo/{v}/seed_42")
    results[v] = m

# ── 3. Print ablation table ───────────────────────────────────
# %%
print(f"\n{'Model':<25} {'Acc':>6} {'RP':>6} {'C':>6} {'P':>6} {'A':>6} {'TI':>6}")
print("-" * 62)
for v in VARIANTS:
    m = results[v]
    print(f"{LABELS[v]:<25} "
          f"{m['accuracy']:6.3f} "
          f"{m['recall_parity']:6.3f} "
          f"{m['clarity']:6.3f} "
          f"{m['privacy']:6.3f} "
          f"{m['accountability']:6.3f} "
          f"{m['trust_index']:6.3f}")

# ── 4. Key findings ───────────────────────────────────────────
# %%
ti_vals = {v: results[v]["trust_index"] for v in VARIANTS}
single_max = max(ti_vals[v] for v in ["transparency","fairness","privacy","accountability"])

print(f"\n  Key findings:")
print(f"    M5 TI = {ti_vals['eagf']:.3f}  vs  max single-pillar TI = {single_max:.3f}")
print(f"    Joint governance {'IS' if ti_vals['eagf'] > single_max else 'IS NOT'} necessary ✓")
print(f"    M3 (+Privacy only) RP = {results['privacy']['recall_parity']:.3f}  "
      f"vs  M0 baseline RP = {results['baseline']['recall_parity']:.3f}")
if results["privacy"]["recall_parity"] < results["baseline"]["recall_parity"]:
    print("    Privacy-only DEGRADES fairness ✓ (Bagdasaryan-Shmatikov coupling confirmed)")

# ── 5. Plot Figure 3 ─────────────────────────────────────────
# %%
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    metrics = ["accuracy", "recall_parity", "clarity", "privacy", "trust_index"]
    mlabels = ["Accuracy", "Recall Parity", "Clarity (C)", "Privacy (P)", "Trust Index (TI)"]
    x = np.arange(len(metrics))
    w = 0.35

    fig, ax = plt.subplots(figsize=(11, 5))
    b_v = [results["baseline"][m] for m in metrics]
    e_v = [results["eagf"][m]     for m in metrics]

    bb = ax.bar(x - w/2, b_v, w, label="Baseline (M0)", color="#F08080", edgecolor="white")
    be = ax.bar(x + w/2, e_v, w, label="EAGF (M5)",     color="#3CB371", edgecolor="white")

    for bar in list(bb)+list(be):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.01,
                f"{bar.get_height():.2f}", ha="center", va="bottom", fontsize=9)

    ax.set_xticks(x); ax.set_xticklabels(mlabels)
    ax.set_ylim(0, 1.12); ax.set_ylabel("Score")
    ax.legend(); ax.grid(axis="y", alpha=0.25)
    ax.spines[["top","right"]].set_visible(False)
    ax.set_title("EAGF Demo: Baseline vs. Full Framework")
    plt.tight_layout()
    os.makedirs("figures", exist_ok=True)
    plt.savefig("figures/demo_comparison.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n  Figure saved → figures/demo_comparison.png")
except Exception as e:
    print(f"\n  (Figure generation skipped: {e})")

print("\nDemo complete. Run 'python run_eagf.py --fast' for the full pipeline.")
