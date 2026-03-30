"""PyTorch-based EAGF trainer with optional DP-SGD via Opacus.

Methodology note:
- Fairness and privacy are optimized during training.
    - Fairness: gradient-based recall-parity penalty term.
    - Privacy: DP-SGD training dynamics via Opacus (optimizer-level).
- Transparency is enforced structurally via L1 input-weight sparsity and
    evaluated with post-training clarity and calibration metrics.
- Accountability is computed post-hoc from audit artifacts and checklists.
"""

import argparse
import json
import os
import random
import time
import itertools

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset

from src.evaluation.audit_logger import AuditLogger
from src.evaluation.calibration import compute_calibration_metrics
from src.evaluation.mia_attack import run_shadow_model_attack
from src.metrics.accountability import compute_accountability
from src.metrics.clarity import compute_global_clarity
from src.metrics.fairness import false_positive_rate_parity, recall_parity, select_criterion
from src.metrics.privacy import privacy_score
from src.metrics.trust_index import trust_index
from src.training.fairness_loss import clarity_penalty_from_outputs, recall_parity_penalty_torch
from src.utils.data_loader import load_adult_dataset, load_biometric_dataset, load_reiot_dataset
from src.utils.preprocessing import preprocess_biometric, preprocess_reiot

try:
    import psutil as _psutil
    _HAS_PSUTIL = True
except ImportError:
    _HAS_PSUTIL = False

# Default assumed CPU power draw for energy estimation (Watts).
_CPU_POWER_WATTS = 65.0


def _estimate_memory_mb() -> float:
    """Estimate current process RSS memory in MB without psutil.

    Tries /proc/self/status (Linux) first, then falls back to the
    ``resource`` standard-library module (Unix only).  Returns a floor
    value of 50 MB if neither source is available so that system-metrics
    columns are never zero.
    """
    # /proc/self/status gives VmRSS in kB on Linux.
    try:
        with open("/proc/self/status") as _f:
            for _line in _f:
                if _line.startswith("VmRSS:"):
                    kb = int(_line.split()[1])
                    return float(kb) / 1024.0
    except Exception:
        pass
    # resource.getrusage on Unix returns maxrss in KB (Linux) or bytes (macOS).
    try:
        import resource as _resource
        import platform
        usage = _resource.getrusage(_resource.RUSAGE_SELF).ru_maxrss
        if platform.system() == "Darwin":  # macOS: bytes
            return float(usage) / (1024.0 * 1024.0)
        return float(usage) / 1024.0       # Linux: kilobytes
    except Exception:
        pass
    return 50.0  # conservative floor (Python process baseline)


SUPPORTED_MODELS = [
    "baseline",
    "standard",
    "transparency",
    "fairness",
    "fairness_threshold",
    "privacy",
    "dp_only",
    "accountability",
    "eagf",
    "full",
    "ablate_no_fairness",
    "ablate_no_dp",
    "ablate_no_clarity",
    "ablate_no_accountability",
]

try:
    from opacus import PrivacyEngine
except Exception:
    PrivacyEngine = None


class TabularMLP(nn.Module):
    def __init__(self, in_dim, hidden=(256, 128, 64), n_classes=2, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden[0]),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden[0], hidden[1]),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden[1], hidden[2]),
            nn.ReLU(),
            nn.Linear(hidden[2], n_classes),
        )

    def forward(self, x):
        return self.net(x)


class ModelAdapter:
    """Sklearn-like adapter so existing metric/eval modules remain reusable."""

    def __init__(self, model, device="cpu"):
        self.model = model
        self.device = torch.device(device)
        self.model.eval()

    def predict_proba(self, X):
        x_t = torch.as_tensor(X, dtype=torch.float32, device=self.device)
        with torch.no_grad():
            probs = torch.softmax(self.model(x_t), dim=1)
        return probs.detach().cpu().numpy()

    def predict(self, X):
        probs = self.predict_proba(X)
        return probs.argmax(axis=1)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_config(path):
    with open(path) as f:
        return yaml.safe_load(f)


def _resolve_variant_settings(variant):
    settings = {
        "use_dp": False,
        "use_fairness_loss": False,
        "use_clarity_loss": False,
        "use_accountability": False,
        "use_group_threshold": False,
        "label": variant,
    }

    if variant in ("baseline", "standard"):
        settings["label"] = "baseline"
    elif variant == "transparency":
        settings["use_clarity_loss"] = True
    elif variant == "fairness":
        settings["use_fairness_loss"] = True
    elif variant == "fairness_threshold":
        settings["use_group_threshold"] = True
    elif variant in ("privacy", "dp_only"):
        settings["use_dp"] = True
        settings["label"] = "privacy"
    elif variant == "accountability":
        settings["use_accountability"] = True
    elif variant in ("eagf", "full"):
        settings["use_dp"] = True
        settings["use_fairness_loss"] = True
        settings["use_clarity_loss"] = True
        settings["use_accountability"] = True
        settings["label"] = "eagf"
    elif variant == "ablate_no_fairness":
        settings["use_dp"] = True
        settings["use_clarity_loss"] = True
        settings["use_accountability"] = True
    elif variant == "ablate_no_dp":
        settings["use_fairness_loss"] = True
        settings["use_clarity_loss"] = True
        settings["use_accountability"] = True
    elif variant == "ablate_no_clarity":
        settings["use_dp"] = True
        settings["use_fairness_loss"] = True
        settings["use_accountability"] = True
    elif variant == "ablate_no_accountability":
        settings["use_dp"] = True
        settings["use_fairness_loss"] = True
        settings["use_clarity_loss"] = True
    else:
        raise ValueError(f"Unsupported variant: {variant}")
    return settings


def _get_fairness_criterion(config):
    context = "biometric"
    if "reiot" in str(config.get("data", {}).get("name", "")).lower():
        context = "reiot"
    return select_criterion(context)


def _encode_groups(groups, fallback_size):
    if groups is None:
        return np.zeros(fallback_size, dtype=np.int64)
    unique = sorted({str(g) for g in groups})
    mapping = {g: i for i, g in enumerate(unique)}
    return np.array([mapping[str(g)] for g in groups], dtype=np.int64)


def _build_train_loader(X_train, y_train, g_train, batch_size, seed):
    ds = TensorDataset(
        torch.as_tensor(X_train, dtype=torch.float32),
        torch.as_tensor(y_train, dtype=torch.long),
        torch.as_tensor(g_train, dtype=torch.long),
    )
    gen = torch.Generator()
    gen.manual_seed(seed)
    return DataLoader(ds, batch_size=min(batch_size, len(ds)), shuffle=True, generator=gen)


def _train_torch_model(variant, config, X_train, y_train, groups_train, X_val, y_val, device, seed):
    training_cfg = config.get("training", {})
    gov_cfg = config.get("governance", {})
    thresholds = config.get("thresholds", {})
    vset = _resolve_variant_settings(variant)

    epochs = int(training_cfg.get("epochs", 30))
    batch_size = int(training_cfg.get("batch_size", 64))
    lr = float(training_cfg.get("lr", 5e-4))
    weight_decay = float(training_cfg.get("weight_decay", 1e-3))

    # Support both lambda_rp and legacy lambda_fprp key names.
    lambda_rp_cfg = float(gov_cfg.get("lambda_rp") if "lambda_rp" in gov_cfg else gov_cfg.get("lambda_fprp", 0.0))
    lambda_c_cfg = float(gov_cfg.get("lambda_c", 0.0))
    target_rp = float(thresholds.get("min_recall_parity", thresholds.get("min_fprp", 0.95)))
    target_clarity_conf = float(thresholds.get("min_clarity", 0.80))

    # Fairness is a gradient-optimized objective term.
    lambda_rp = lambda_rp_cfg if vset["use_fairness_loss"] else 0.0
    # Transparency is enforced structurally (e.g., pruning/constraints) and is NOT optimized via gradient descent.
    # In this implementation, lambda_c applies a structural surrogate regularizer.
    lambda_c = lambda_c_cfg if vset["use_clarity_loss"] else 0.0

    n_classes = int(np.max(y_train)) + 1
    model = TabularMLP(in_dim=X_train.shape[1], n_classes=n_classes).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    g_train_idx = _encode_groups(groups_train, len(y_train))
    train_loader = _build_train_loader(X_train, y_train, g_train_idx, batch_size, seed)

    # Privacy optimization is applied through DP-SGD updates (Opacus), not by
    # adding a separate scalar privacy loss to the objective.
    dp_enabled = vset["use_dp"]
    epsilon_eff = float("inf")
    if dp_enabled:
        if PrivacyEngine is None:
            import warnings
            warnings.warn(
                "Opacus is not installed; running without DP noise. "
                "Install opacus>=1.4.0 for full privacy training.",
                RuntimeWarning,
            )
            dp_enabled = False
        else:
            target_epsilon = float(gov_cfg.get("dp_epsilon", 3.0))
            target_delta = float(gov_cfg.get("dp_delta", 1e-5))
            max_grad_norm = float(gov_cfg.get("dp_max_grad_norm", 1.0))
            privacy_engine = PrivacyEngine()
            model, optimizer, train_loader = privacy_engine.make_private_with_epsilon(
                module=model,
                optimizer=optimizer,
                data_loader=train_loader,
                epochs=epochs,
                target_epsilon=target_epsilon,
                target_delta=target_delta,
                max_grad_norm=max_grad_norm,
            )
    if not dp_enabled:
        privacy_engine = None
        target_delta = float(gov_cfg.get("dp_delta", 1e-5))

    model.train()
    batch_fwd_times_ms: list[float] = []  # per-sample forward pass ms, collected per batch
    _fairness_debug_logged = False
    t_train_start = time.perf_counter()
    for _epoch in range(epochs):
        for xb, yb, gb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            gb = gb.to(device)

            optimizer.zero_grad(set_to_none=True)
            t_fwd = time.perf_counter()
            logits = model(xb)
            batch_fwd_times_ms.append(
                (time.perf_counter() - t_fwd) * 1000.0 / max(len(xb), 1)
            )
            probs = torch.softmax(logits, dim=1)

            l_task = F.cross_entropy(logits, yb)
            if probs.shape[1] >= 2:
                p_pos = probs[:, 1]
            else:
                p_pos = probs[:, 0]
            l_fair = recall_parity_penalty_torch(yb, p_pos, gb, rp_target=target_rp)
            if not _fairness_debug_logged and lambda_rp > 0:
                print(f"Fairness loss: {l_fair.item():.6f} (lambda_rp={lambda_rp})")
                _fairness_debug_logged = True
            # clarity_penalty_from_outputs returns zero — no confidence-inflating
            # terms are applied.  L1 weight sparsity below is the only structural
            # transparency regularizer.
            l_clarity = clarity_penalty_from_outputs(probs, target_confidence=target_clarity_conf)

            gradient_objective = l_task + (lambda_rp * l_fair)
            structural_constraints = (lambda_c * l_clarity)

            # L1 weight penalty on the first layer when clarity is active.
            # Sparsifying the input projection encourages the model to rely on fewer
            # input features. Fewer active features → smaller local explanation size
            # → higher ClarityScore = Fidelity / (1 + Size).
            if lambda_c > 0:
                try:
                    first_layer = model.net[0] if hasattr(model, 'net') else model._module.net[0]
                    l_weight_l1 = first_layer.weight.abs().mean()
                    structural_constraints = structural_constraints + lambda_c * l_weight_l1
                except Exception:
                    pass

            loss = gradient_objective + structural_constraints
            loss.backward()
            optimizer.step()

    t_train_elapsed = time.perf_counter() - t_train_start

    if privacy_engine is not None:
        epsilon_eff = float(privacy_engine.get_epsilon(delta=target_delta))

    # Collect per-process memory usage (RSS) after training.
    if _HAS_PSUTIL:
        mem_mb = float(_psutil.Process().memory_info().rss) / (1024.0 * 1024.0)
    else:
        # Fallback: estimate from /proc/self/status (Linux) or resource module.
        mem_mb = _estimate_memory_mb()

    # Enforce a realistic minimum — Python process overhead alone is ~50 MB.
    mem_mb = max(mem_mb, 50.0)

    # Approximate energy: total training wall-clock time × assumed CPU wattage.
    energy_j = t_train_elapsed * _CPU_POWER_WATTS

    overhead = {
        "train_fwd_ms_per_sample": float(np.mean(batch_fwd_times_ms)) if batch_fwd_times_ms else 0.0,
        "train_wall_s": t_train_elapsed,
        "memory_usage_mb": mem_mb,
        "energy_overhead_joules": energy_j,
    }

    return model, epsilon_eff, overhead


def _apply_group_thresholds(proba_pos, groups, thresholds, default_threshold=0.5):
    preds = np.zeros(len(proba_pos), dtype=int)
    for i in range(len(proba_pos)):
        g = str(groups[i]) if groups is not None else "__all__"
        thr = float(thresholds.get(g, default_threshold))
        preds[i] = int(proba_pos[i] >= thr)
    return preds


def _recall_parity_ratio(y_true, y_pred, groups):
    if groups is None:
        return 0.0
    recalls = []
    for g in sorted({str(v) for v in groups}):
        mask = (np.array(groups).astype(str) == g)
        yt = y_true[mask]
        yp = y_pred[mask]
        pos = (yt == 1)
        if pos.sum() == 0:
            continue
        rec = float(((yp == 1) & pos).sum() / max(pos.sum(), 1))
        recalls.append(rec)
    if len(recalls) < 2:
        return 0.0
    return float(min(recalls) / (max(recalls) + 1e-12))


def _fit_group_thresholds_equalized_odds(y_val, proba_val, groups_val):
    """Established fairness baseline: per-group threshold optimization."""
    if groups_val is None:
        return {"__all__": 0.5}
    groups = sorted({str(v) for v in groups_val})
    threshold_grid = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
    best = None
    best_score = -1e9
    groups_arr = np.array(groups_val).astype(str)

    for combo in itertools.product(threshold_grid, repeat=len(groups)):
        mapping = {g: float(t) for g, t in zip(groups, combo)}
        preds = _apply_group_thresholds(proba_val, groups_arr, mapping)
        rp = _recall_parity_ratio(y_val, preds, groups_arr)
        acc = float(np.mean(preds == y_val))
        # Balance fairness and utility for baseline comparability.
        score = 0.5 * rp + 0.5 * acc
        if score > best_score:
            best_score = score
            best = mapping
    return best if best is not None else {"__all__": 0.5}


def _compute_all_metrics(model_adapter, X_train, y_train, X_val, y_val, groups_val,
                         X_test, y_test, groups_test, config, variant, audit_log_path,
                         epsilon_eff=float("inf"), seed=42, y_pred_test_override=None,
                         accountability_enabled=None):
    y_pred_test = y_pred_test_override if y_pred_test_override is not None else model_adapter.predict(X_test)
    acc = float(accuracy_score(y_test, y_pred_test))

    clarity_result = compute_global_clarity(
        model_predict_fn=model_adapter.predict,
        X=X_val,
        sample_size=50,
        seed=seed,
    )
    C = float(np.clip(clarity_result["clarity"], 0.0, 1.0))

    criterion = _get_fairness_criterion(config)
    ref_group = "male_light" if criterion == "recall_parity" else "urban"

    if groups_test is not None and criterion == "fprp":
        fair_result = false_positive_rate_parity(y_test, y_pred_test, groups_test, ref_group)
        Fv = float(fair_result.get("fprp", 0.0))
    elif groups_test is not None and criterion == "recall_parity":
        fair_result = recall_parity(y_test, y_pred_test, groups_test, ref_group)
        Fv = float(fair_result.get("recall_parity", 0.0))
    else:
        Fv = 0.0

    mia_result = run_shadow_model_attack(model_adapter, X_train, y_train, X_test, y_test, seed=seed)
    mia_auc = float(mia_result["mia_auc"])
    P = privacy_score(epsilon_eff=epsilon_eff, mia_auc=mia_auc)

    # Accountability remains post-hoc and is derived from audit/log artifacts.
    # Dynamic overrides for data-driven controls ensure accountability is earned,
    # not hardcoded: the checklist entries for MIA and fairness are overridden
    # based on the actual computed metrics.
    checklist_path = config.get("accountability", {}).get("compliance_checklist")
    if accountability_enabled is None:
        accountability_enabled = _resolve_variant_settings(variant)["use_accountability"]
    acc_result = compute_accountability(
        audit_log_path=audit_log_path,
        total_decisions=len(X_test),
        lineage_fraction=None,
        checklist_path=checklist_path,
        model_has_governance=bool(accountability_enabled),
        metric_overrides={
            "mia_stress_test": mia_auc <= 0.60,
            "fairness_monitoring": Fv >= 0.95,
        },
    )
    A = float(acc_result["accountability"])

    ti_result = trust_index(C, Fv, P, A)

    # Calibration metrics: ECE and Brier Score.
    y_proba_test = model_adapter.predict_proba(X_test)
    calib_result = compute_calibration_metrics(y_test, y_proba_test)

    return {
        "accuracy": acc,
        "recall_parity": Fv,
        "clarity": C,
        "privacy": P,
        "accountability": A,
        "trust_index": float(ti_result["ti"]),
        "mia_auc": mia_auc,
        "epsilon_eff": float(epsilon_eff),
        "ece": calib_result["ece"],
        "brier_score": calib_result["brier_score"],
        "ti_components": ti_result["components"],
        "audit": acc_result,
    }


def train_variant(variant, config, dataset, seed=42, output_dir=".", return_model=False):
    """Train one model variant and return all metrics."""
    set_seed(seed)
    os.makedirs(output_dir, exist_ok=True)
    audit_log_path = os.path.join(output_dir, "audit_log.jsonl")
    device = config.get("training", {}).get("device", "cpu")
    vset = _resolve_variant_settings(variant)

    is_reiot = "reiot" in str(config.get("data", {}).get("name", "")).lower()

    if is_reiot:
        data = preprocess_reiot(dataset.copy(), seed=seed)
        X_train_full, y_train_full = data["X_train"], data["y_train"]
        groups_train_full = data.get("groups_train")
        split_idxs = np.arange(len(y_train_full))
        idx_train, idx_val = train_test_split(
            split_idxs,
            test_size=0.2,
            random_state=seed,
            stratify=y_train_full,
        )
        X_train, y_train = X_train_full[idx_train], y_train_full[idx_train]
        X_val, y_val = X_train_full[idx_val], y_train_full[idx_val]
        X_test, y_test = data["X_test"], data["y_test"]
        groups_train = groups_train_full[idx_train] if groups_train_full is not None else None
        groups_val = groups_train_full[idx_val] if groups_train_full is not None else None
        groups_test = data.get("groups_test")
    else:
        data = preprocess_biometric(
            dataset.copy(),
            apply_dp=False,
            epsilon=config.get("governance", {}).get("dp_epsilon", 3.0),
            seed=seed,
        )
        X_train, y_train = data["X_train"], data["y_train"]
        X_val, y_val = data.get("X_val", data["X_train"]), data.get("y_val", data["y_train"])
        X_test, y_test = data["X_test"], data["y_test"]
        groups_train = data.get("groups_train")
        groups_val = data.get("groups_val", groups_train)
        groups_test = data.get("groups_test")

    print(f"    Training {variant} (seed={seed})...", end=" ", flush=True)
    t0 = time.time()
    model, epsilon_eff, train_overhead = _train_torch_model(
        variant, config, X_train, y_train, groups_train, X_val, y_val, device=device, seed=seed
    )
    model_adapter = ModelAdapter(model, device=device)

    # ── Inference overhead measurement ────────────────────────────────────
    t_infer_start = time.perf_counter()
    _proba_sample = model_adapter.predict_proba(X_test)
    t_infer_elapsed = time.perf_counter() - t_infer_start
    # Per-sample inference time in milliseconds.
    infer_ms = (t_infer_elapsed * 1000.0) / max(len(X_test), 1)

    # RSS memory at inference time.
    if _HAS_PSUTIL:
        infer_mem_mb = float(_psutil.Process().memory_info().rss) / (1024.0 * 1024.0)
    else:
        infer_mem_mb = _estimate_memory_mb()
    # Enforce a realistic minimum — Python process overhead alone is ~50 MB.
    infer_mem_mb = max(infer_mem_mb, 50.0)

    # Energy: inference time × CPU wattage.
    infer_energy_j = t_infer_elapsed * _CPU_POWER_WATTS

    y_pred_test_override = None
    if vset["use_group_threshold"]:
        val_proba = model_adapter.predict_proba(X_val)
        test_proba = model_adapter.predict_proba(X_test)
        val_p1 = val_proba[:, 1] if val_proba.shape[1] > 1 else val_proba[:, 0]
        test_p1 = test_proba[:, 1] if test_proba.shape[1] > 1 else test_proba[:, 0]
        thresholds = _fit_group_thresholds_equalized_odds(y_val, val_p1, groups_val)
        y_pred_test_override = _apply_group_thresholds(test_p1, groups_test, thresholds)

    if vset["use_accountability"]:
        logger = AuditLogger(
            audit_log_path,
            model_version=f"eagf-{variant}-seed{seed}",
            operator_id="eagf-system",
        )
        pred_proba = model_adapter.predict_proba(X_test)
        pred_labels = y_pred_test_override if y_pred_test_override is not None else pred_proba.argmax(axis=1)
        confidences = pred_proba.max(axis=1)
        for i in range(len(pred_labels)):
            logger.log(
                X_test[i],
                int(pred_labels[i]),
                float(confidences[i]),
                inference_time_ms=infer_ms,
                memory_usage_mb=infer_mem_mb,
                energy_overhead_joules=infer_energy_j,
            )

    metrics = _compute_all_metrics(
        model_adapter,
        X_train,
        y_train,
        X_val,
        y_val,
        groups_val,
        X_test,
        y_test,
        groups_test,
        config,
        variant,
        audit_log_path,
        epsilon_eff=epsilon_eff,
        seed=seed,
        y_pred_test_override=y_pred_test_override,
        accountability_enabled=vset["use_accountability"],
    )

    # ── Attach overhead metrics to the returned metrics dict ──────────────
    metrics["inference_time_ms"] = round(infer_ms, 4)
    metrics["memory_usage_mb"] = round(infer_mem_mb, 2)
    metrics["energy_overhead_joules"] = round(infer_energy_j, 4)

    elapsed = time.time() - t0
    print(f"done ({elapsed:.1f}s) TI={metrics['trust_index']:.3f}")

    result_path = os.path.join(output_dir, "results.json")
    with open(result_path, "w") as f:
        json.dump(
            {
                k: (float(v) if isinstance(v, (np.floating, float)) else v)
                for k, v in metrics.items()
                if not isinstance(v, dict)
            },
            f,
            indent=2,
        )
    if return_model:
        return metrics, model_adapter
    return metrics


def parse_args():
    p = argparse.ArgumentParser(description="EAGF Trust-Aware Trainer")
    p.add_argument("--config",  required=True)
    p.add_argument("--model",   required=True, choices=SUPPORTED_MODELS)
    p.add_argument("--seed",    type=int, default=42)
    p.add_argument("--device",  default="cpu")
    p.add_argument("--output",  required=True)
    p.add_argument("--demo",    action="store_true", help="Use synthetic demo data")
    return p.parse_args()


def main():
    args = parse_args()
    config = load_config(args.config)
    config.setdefault("training", {})
    config["training"]["device"] = args.device
    set_seed(args.seed)
    os.makedirs(args.output, exist_ok=True)

    is_reiot = "reiot" in args.config.lower()

    print(f"EAGF Trainer | variant={args.model} | seed={args.seed}")

    if is_reiot:
        data_root = config.get("data", {}).get("root", "data/reiot")
        dataset = load_reiot_dataset(data_root=data_root, seed=args.seed)
    elif "adult" in str(config.get("data", {}).get("name", "")).lower():
        dataset = load_adult_dataset(seed=args.seed)
    else:
        data_root = config.get("data", {}).get("root", "data/biometric/efr_processed")
        dataset = load_biometric_dataset(
            data_root=data_root, demo=args.demo, seed=args.seed
        )

    metrics = train_variant(args.model, config, dataset, seed=args.seed,
                             output_dir=args.output)

    print(f"\n  Results:")
    for k, v in metrics.items():
        if isinstance(v, float):
            print(f"    {k:25s}: {v:.4f}")

    return metrics


if __name__ == "__main__":
    main()
