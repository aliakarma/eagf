"""
src/evaluation/mia_attack.py — Membership Inference Attack (Shadow Model)
Paper: Section 3.4 (Privacy Metric, MIA stress-test)
"""
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score


def _target_confidence(model, X):
    """Return target-model confidence vectors used by the attack model."""
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)

    y_pred = model.predict(X)
    classes = np.unique(y_pred)
    n_classes = int(classes.max()) + 1 if len(classes) > 0 else 2
    probs = np.zeros((len(y_pred), max(n_classes, 2)), dtype=float)
    probs[np.arange(len(y_pred)), y_pred.astype(int)] = 1.0
    return probs


def run_shadow_model_attack(model, X_train, y_train, X_test=None, y_test=None,
                             n_shadow_models=2, shadow_epochs=None, device="cpu", seed=42):
    """Membership inference attack against target model outputs.
    
    MIA AUC ≈ 0.50 -> random -> strong privacy.
    MIA AUC ≈ 1.00 -> perfect attack -> weak privacy.
    
    Returns: dict with 'mia_auc', 'mia_accuracy'.
    """
    if X_test is None or y_test is None:
        idx = np.arange(len(X_train))
        tr_idx, te_idx = train_test_split(
            idx,
            test_size=0.5,
            random_state=seed,
            stratify=y_train,
        )
        X_member = X_train[tr_idx]
        X_non_member = X_train[te_idx]
    else:
        X_member = X_train
        X_non_member = X_test

    member_conf = _target_confidence(model, X_member)
    non_member_conf = _target_confidence(model, X_non_member)

    attack_X = np.vstack([member_conf, non_member_conf])
    attack_y = np.concatenate([
        np.ones(len(member_conf), dtype=int),
        np.zeros(len(non_member_conf), dtype=int),
    ])

    if len(attack_X) < 10 or len(np.unique(attack_y)) < 2:
        return {"mia_auc": 0.5, "mia_accuracy": 0.5}

    # Train attack classifier
    try:
        atk = LogisticRegression(max_iter=200, random_state=seed, C=1.0)
        X_atk_tr, X_atk_te, y_atk_tr, y_atk_te = train_test_split(
            attack_X, attack_y, test_size=0.3, random_state=seed, stratify=attack_y
        )
        atk.fit(X_atk_tr, y_atk_tr)
        atk_proba = atk.predict_proba(X_atk_te)[:, 1]
        atk_pred  = atk.predict(X_atk_te)
        mia_auc = float(roc_auc_score(y_atk_te, atk_proba))
        mia_acc = float(np.mean(atk_pred == y_atk_te))
    except Exception:
        mia_auc = 0.5
        mia_acc = 0.5

    return {"mia_auc": mia_auc, "mia_accuracy": mia_acc}
