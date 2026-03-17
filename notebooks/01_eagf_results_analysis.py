"""
notebooks/01_eagf_results_analysis.py
EAGF Results Analysis — run as a script or open in Jupyter as a notebook

Produces:
  - Ablation table (Table 4 equivalent)
  - Cross-pillar trade-off visualisation
  - Trust Index component breakdown
  - Statistical test summary

Usage:
  python notebooks/01_eagf_results_analysis.py
  # Or in Jupyter: jupyter nbconvert --to notebook --execute this_file.py
"""
import sys, os, json, warnings
warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

# ── 0. Run fast experiment if results don't exist ─────────────────────────
ABLATION_CSV = 'results/biometric/ablation/ablation_summary.csv'
if not os.path.exists(ABLATION_CSV):
    print("Results not found. Running fast experiment first...")
    os.system('python -W ignore run_eagf.py --fast --skip-pareto')

# ── 1. Load results ────────────────────────────────────────────────────────
import csv

def read_csv(path):
    with open(path, newline='') as f:
        return list(csv.DictReader(f))

rows = read_csv(ABLATION_CSV)

# ── 2. Print ablation table ────────────────────────────────────────────────
print("\n" + "="*75)
print("  EAGF Ablation Study — Table 4")
print("="*75)
print(f"{'Model':<28} {'Acc':>7} {'RP':>7} {'C':>7} {'P':>7} {'A':>7} {'TI':>7}")
print("-"*75)

for r in rows:
    label = r.get('model_label', r.get('model_id', '?'))[:28]
    # CI strings for TI
    ti_lo = float(r.get('trust_index_ci_lower', r.get('trust_index_mean', 0)))
    ti_hi = float(r.get('trust_index_ci_upper', r.get('trust_index_mean', 0)))
    print(f"{label:<28} "
          f"{float(r['accuracy_mean']):7.3f} "
          f"{float(r['recall_parity_mean']):7.3f} "
          f"{float(r['clarity_mean']):7.3f} "
          f"{float(r['privacy_mean']):7.3f} "
          f"{float(r['accountability_mean']):7.3f} "
          f"{float(r['trust_index_mean']):7.3f}")

# ── 3. Key findings ─────────────────────────────────────────────────────────
print("\n" + "="*75)
print("  Key Findings")
print("="*75)

model_map = {r.get('model_id', r.get('model_label','')): r for r in rows}
ti_vals = {k: float(v.get('trust_index_mean', 0)) for k, v in model_map.items()}

baseline_ti = ti_vals.get('baseline', 0)
eagf_ti     = ti_vals.get('eagf', 0)
single_max  = max(ti_vals.get(v, 0) for v in
                  ['transparency', 'fairness', 'privacy', 'accountability'])
privacy_rp  = float(model_map.get('privacy', {}).get('recall_parity_mean', 1))
baseline_rp = float(model_map.get('baseline', {}).get('recall_parity_mean', 1))

print(f"\n  Finding 1: M5 EAGF TI ({eagf_ti:.3f}) > max single-pillar TI ({single_max:.3f})")
print(f"             Joint governance is {'CONFIRMED' if eagf_ti > single_max else 'NOT confirmed'} necessary.")

if privacy_rp < baseline_rp:
    print(f"\n  Finding 2: Privacy-only (M3) DEGRADES Recall Parity")
    print(f"             Baseline RP={baseline_rp:.3f} → M3 RP={privacy_rp:.3f} (Δ={privacy_rp-baseline_rp:+.3f})")
    print(f"             Confirms Bagdasaryan-Shmatikov privacy-fairness coupling.")
else:
    print(f"\n  Finding 2: Privacy-only RP={privacy_rp:.3f} (baseline={baseline_rp:.3f})")

ti_gain = (eagf_ti - baseline_ti) / baseline_ti * 100
print(f"\n  Finding 3: Trust Index improvement: {baseline_ti:.3f} → {eagf_ti:.3f}"
      f" (+{ti_gain:.1f}% relative)")

# ── 4. Load and print statistical tests ────────────────────────────────────
stats_path = 'results/biometric/statistical_tests.json'
if os.path.exists(stats_path):
    print("\n" + "="*75)
    print("  Statistical Tests")
    print("="*75)
    with open(stats_path) as f:
        stats = json.load(f)
    for metric in ['accuracy', 'trust_index', 'clarity', 'privacy', 'accountability']:
        s = stats.get(metric, {})
        b_mean = s.get('baseline', {}).get('mean', 0)
        e_mean = s.get('eagf', {}).get('mean', 0)
        test = s.get('ztest', s.get('wilcoxon', {}))
        p    = test.get('p_value', 1.0)
        sig  = '**' if p < 0.01 else ('*' if p < 0.05 else 'ns')
        print(f"  {metric:<20}: baseline={b_mean:.3f}  eagf={e_mean:.3f}  "
              f"Δ={e_mean-b_mean:+.3f}  p={p:.4f} [{sig}]")

# ── 5. Visualisation: TI component breakdown ───────────────────────────────
print("\n  Generating component breakdown chart...")
model_names = [r.get('model_label', r.get('model_id','?'))[:18] for r in rows]
C_vals = [float(r['clarity_mean']) for r in rows]
RP_vals= [float(r['recall_parity_mean']) for r in rows]
P_vals = [float(r['privacy_mean']) for r in rows]
A_vals = [float(r['accountability_mean']) for r in rows]

x = np.arange(len(rows))
w = 0.2
fig, ax = plt.subplots(figsize=(14, 5))
bars = {}
for i, (vals, label, color) in enumerate([
    (C_vals,  'Clarity (C)',        '#4C72B0'),
    (RP_vals, 'Recall Parity (RP)', '#DD8452'),
    (P_vals,  'Privacy (P)',        '#55A868'),
    (A_vals,  'Accountability (A)', '#C44E52'),
]):
    ax.bar(x + (i-1.5)*w, vals, w, label=label, color=color, alpha=0.85)

ax.set_xticks(x)
ax.set_xticklabels([n[:16] for n in model_names], rotation=15, ha='right', fontsize=8)
ax.set_ylabel('Score', fontsize=11)
ax.set_title('EAGF Pillar Breakdown Across Model Variants', fontsize=12, fontweight='bold')
ax.legend(loc='upper left', fontsize=9)
ax.set_ylim(0, 1.15)
ax.grid(axis='y', alpha=0.2)
ax.spines[['top','right']].set_visible(False)
plt.tight_layout()
os.makedirs('figures', exist_ok=True)
plt.savefig('figures/pillar_breakdown.png', dpi=200, bbox_inches='tight')
plt.close()
print("  Saved → figures/pillar_breakdown.png")

# ── 6. Cross-pillar trade-off: Privacy vs Fairness ─────────────────────────
print("\n  Generating privacy-fairness trade-off plot...")
fig, ax = plt.subplots(figsize=(7, 5))
colours = ['#808080','#4C72B0','#DD8452','#C44E52','#8172B3','#2CA02C']
for i, r in enumerate(rows):
    lbl = r.get('model_label', r.get('model_id','?'))
    mid = lbl.find(': ') + 2
    short = lbl[mid:][:18] if mid > 1 else lbl[:18]
    p  = float(r['privacy_mean'])
    rp = float(r['recall_parity_mean'])
    ti = float(r['trust_index_mean'])
    ax.scatter(p, rp, s=ti*400, color=colours[i], zorder=3, alpha=0.9,
               label=f"M{i}: {short}")
    ax.annotate(f"M{i}", (p, rp), textcoords='offset points',
                xytext=(5,5), fontsize=8)

ax.set_xlabel('Privacy (P)', fontsize=11)
ax.set_ylabel('Recall Parity (RP)', fontsize=11)
ax.set_title('Privacy vs. Fairness Trade-off\n(bubble size ∝ Trust Index)', fontsize=11)
ax.legend(loc='lower right', fontsize=8, framealpha=0.7)
ax.grid(alpha=0.2)
ax.spines[['top','right']].set_visible(False)
plt.tight_layout()
plt.savefig('figures/privacy_fairness_tradeoff.png', dpi=200, bbox_inches='tight')
plt.close()
print("  Saved → figures/privacy_fairness_tradeoff.png")

print("\n  Analysis complete.")
print("  Output figures: figures/pillar_breakdown.png")
print("                  figures/privacy_fairness_tradeoff.png")
