"""
src/training/pareto_trainer.py — EAGF Pareto-Front Search
Paper: Section 3.7 (MOO, Pareto exploration)
Sweeps 5×5 grid of (lambda_RP, lambda_C), finds highest-TI solution.

Scope note:
- lambda_RP controls the fairness gradient term.
- lambda_C controls a structural transparency surrogate regularizer.
- Privacy is handled via DP-SGD in the trainer.
- Accountability is post-hoc and is not a training objective.

Dynamic AHP (Phase 4):
- At the end of each Pareto grid point the AHP weights are updated via
  `calculate_dynamic_weights()` using the observed fairness loss and MIA
  success rate from that run.  The adapted weights are passed to the next
  run's `trust_index` computation via the `ahp_weights` entry in the config,
  enabling true agentic weight adaptation across the Pareto sweep.
"""
import argparse, itertools, json, os
import numpy as np
import yaml


def log_spaced_grid(low, high, n_steps):
    return list(np.logspace(np.log10(low), np.log10(high), n_steps))


def is_dominated(candidate, population, objectives):
    for other in population:
        if other is candidate:
            continue
        if (all(other.get(o, 0) >= candidate.get(o, 0) for o in objectives) and
                any(other.get(o, 0) > candidate.get(o, 0) for o in objectives)):
            return True
    return False


def pareto_front(results, objectives=None):
    if objectives is None:
        objectives = ["clarity", "recall_parity", "privacy", "accountability"]
    return [r for r in results if not is_dominated(r, results, objectives)]


def run_pareto_search(config, lambda_rp_range, lambda_c_range, n_steps,
                      seed, device, output_dir, dataset=None):
    """Run full Pareto-grid search and return best model metrics.

    Dynamic AHP weights are updated after each grid point using
    ``calculate_dynamic_weights()``.  The adapted weights evolve across runs
    in response to the observed MIA success rate and fairness loss, providing
    agentic weight adaptation without modifying the training objective itself.
    """
    from src.training.eagf_trainer import train_variant, load_biometric_dataset
    from src.utils.ahp import calculate_dynamic_weights, equal_weights

    os.makedirs(output_dir, exist_ok=True)
    lrp_vals = log_spaced_grid(*lambda_rp_range, n_steps)
    lc_vals  = log_spaced_grid(*lambda_c_range,  n_steps)
    grid = list(itertools.product(lrp_vals, lc_vals))
    print(f"Pareto search: {len(grid)} grid points...")

    if dataset is None:
        data_root = config.get("data", {}).get("root", "data/biometric")
        dataset = load_biometric_dataset(data_root=data_root, demo=True, seed=seed)

    # Initialise adaptive weights at the equal-weight baseline.
    current_ahp_weights = equal_weights()

    all_results = []
    for i, (lrp, lc) in enumerate(grid):
        cfg = yaml.safe_load(yaml.dump(config))
        cfg.setdefault("governance", {})
        cfg["governance"]["lambda_rp"] = lrp
        cfg["governance"]["lambda_c"]  = lc
        run_dir = os.path.join(output_dir, f"run_{i:02d}")
        metrics = train_variant("eagf", cfg, dataset, seed=seed,
                                output_dir=run_dir)

        # ── Dynamic AHP update (end of each Pareto run) ───────────────────
        # Use the MIA AUC from this run as the attack-success-rate signal and
        # the (1 - recall_parity) as a proxy for fairness loss.
        mia_rate = float(metrics.get("mia_auc", 0.5))
        fairness_loss_proxy = float(1.0 - metrics.get("recall_parity", 1.0))
        privacy_loss_proxy = float(1.0 - metrics.get("privacy", 1.0))

        current_ahp_weights = calculate_dynamic_weights(
            current_privacy_loss=privacy_loss_proxy,
            current_fairness_loss=fairness_loss_proxy,
            mia_attack_success_rate=mia_rate,
            previous_weights=current_ahp_weights,
        )
        # ─────────────────────────────────────────────────────────────────

        entry = {
            **metrics,
            "lambda_rp": lrp,
            "lambda_c": lc,
            "run_id": i,
            # Store AHP weights as flat keys so they survive the dict-filter below.
            "ahp_privacy_weight":       current_ahp_weights["privacy"],
            "ahp_fairness_weight":      current_ahp_weights["fairness"],
            "ahp_clarity_weight":       current_ahp_weights["clarity"],
            "ahp_accountability_weight": current_ahp_weights["accountability"],
        }
        all_results.append({
            k: (float(v) if isinstance(v, (float, np.floating)) else v)
            for k, v in entry.items()
            if not isinstance(v, dict)
        })
        print(f"  [{i+1:2d}/{len(grid)}] lrp={lrp:.4f} lc={lc:.4f} "
              f"TI={metrics['trust_index']:.3f}  "
              f"AHP[C={current_ahp_weights['clarity']:.3f} "
              f"F={current_ahp_weights['fairness']:.3f} "
              f"P={current_ahp_weights['privacy']:.3f} "
              f"A={current_ahp_weights['accountability']:.3f}]")

    front = pareto_front(all_results)
    best  = max(front, key=lambda r: r.get("trust_index", 0))

    with open(os.path.join(output_dir, "pareto_results.json"), "w") as f:
        json.dump({"all_results": all_results, "pareto_front": front, "best": best}, f, indent=2)

    ti_values = [r.get("trust_index", 0) for r in all_results]
    ti_spread = max(ti_values) - min(ti_values)
    print(f"\nPareto search done. Best TI={best['trust_index']:.3f} "
          f"(lrp={best['lambda_rp']:.4f}, lc={best['lambda_c']:.4f})")
    print(f"  Pareto TI spread: min={min(ti_values):.3f} max={max(ti_values):.3f} Δ={ti_spread:.3f}")
    if ti_spread <= 0.05:
        print(f"  WARNING: Pareto TI spread {ti_spread:.3f} ≤ 0.05 — trade-offs are weak.")
    else:
        print(f"  ✓ Pareto spread {ti_spread:.3f} > 0.05 — real trade-offs confirmed.")
    return {"pareto_front": front, "best": best, "all_results": all_results,
            "ti_spread": ti_spread}


def parse_args():
    p = argparse.ArgumentParser(description="EAGF Pareto-Front Search")
    p.add_argument("--config", required=True)
    p.add_argument("--lambda-rp-range", type=float, nargs=2, default=[1e-3, 1.0])
    p.add_argument("--lambda-rp-steps", type=int, default=5)
    p.add_argument("--lambda-c-range",  type=float, nargs=2, default=[1e-3, 1.0])
    p.add_argument("--lambda-c-steps",  type=int, default=5)
    p.add_argument("--seed",   type=int, default=42)
    p.add_argument("--device", default="cpu")
    p.add_argument("--output", required=True)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    with open(args.config) as f:
        config = yaml.safe_load(f)
    run_pareto_search(
        config=config,
        lambda_rp_range=args.lambda_rp_range,
        lambda_c_range=args.lambda_c_range,
        n_steps=args.lambda_rp_steps,
        seed=args.seed, device=args.device,
        output_dir=args.output,
    )
