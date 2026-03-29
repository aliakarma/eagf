"""tests/test_metrics.py — Unit tests for all four pillar metrics and TI."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pytest


# ── Clarity ────────────────────────────────────────────────────────────────
class TestClarity:
    def test_clarity_score_range(self):
        from src.metrics.clarity import compute_instance_clarity
        X = np.random.randn(100, 20).astype(np.float32)
        dummy_model = lambda x: np.zeros(len(x), dtype=int)
        score = compute_instance_clarity(dummy_model, X, 0, n_neighbours=20)
        assert 0.0 <= score <= 1.0, f"Clarity score {score} out of [0,1]"

    def test_global_clarity_returns_dict(self):
        from src.metrics.clarity import compute_global_clarity
        X = np.random.randn(50, 10).astype(np.float32)
        dummy_model = lambda x: np.random.randint(0, 2, len(x))
        result = compute_global_clarity(dummy_model, X, sample_size=10, seed=42)
        assert "clarity" in result
        assert "opacity" in result
        assert abs(result["clarity"] + result["opacity"] - 1.0) < 1e-6

    def test_clarity_from_importances(self):
        from src.metrics.clarity import clarity_from_feature_importances
        importances = np.array([0.5, 0.3, 0.1, 0.05, 0.05])
        y_true = np.array([0, 1, 0, 1, 1])
        y_pred = np.array([0, 1, 0, 1, 0])
        result = clarity_from_feature_importances(importances, y_true, y_pred)
        assert 0.0 <= result["clarity"] <= 1.0
        assert result["explanation_size"] >= 1

    def test_high_fidelity_sparsity_gives_high_score(self):
        from src.metrics.clarity import compute_clarity_score
        # High fidelity=0.95, size=2 -> 0.95/3 ≈ 0.317
        score = compute_clarity_score(0.95, 2)
        assert score == pytest.approx(0.95 / 3, abs=1e-6)

    def test_zero_fidelity_gives_zero(self):
        from src.metrics.clarity import compute_clarity_score
        assert compute_clarity_score(0.0, 5) == 0.0


# ── Fairness ────────────────────────────────────────────────────────────────
class TestFairness:
    def _make_data(self):
        rng = np.random.RandomState(42)
        y_true  = rng.randint(0, 2, 200)
        y_pred  = y_true.copy()
        # Make group B (dark) harder: flip 20% of positives
        groups  = np.array(["light"] * 100 + ["dark"] * 100)
        dark_pos = np.where((groups == "dark") & (y_true == 1))[0]
        flip = rng.choice(dark_pos, size=int(0.2 * len(dark_pos)), replace=False)
        y_pred[flip] = 1 - y_pred[flip]
        return y_true, y_pred, groups

    def test_recall_parity_perfect(self):
        from src.metrics.fairness import recall_parity
        y = np.array([0, 1, 0, 1])
        g = np.array(["A", "A", "B", "B"])
        r = recall_parity(y, y, g, "A")  # perfect predictions
        assert r["recall_parity"] == pytest.approx(1.0, abs=0.01)

    def test_recall_parity_biased(self):
        from src.metrics.fairness import recall_parity
        y_true, y_pred, groups = self._make_data()
        r = recall_parity(y_true, y_pred, groups, "light")
        assert r["recall_parity"] < 1.0, "Biased data should give RP < 1"

    def test_fprp_range(self):
        from src.metrics.fairness import false_positive_rate_parity
        rng = np.random.RandomState(0)
        y_true = rng.randint(0, 2, 300)
        y_pred = rng.randint(0, 2, 300)
        groups = np.array(["urban"] * 100 + ["rural"] * 100 + ["periurban"] * 100)
        r = false_positive_rate_parity(y_true, y_pred, groups, "urban")
        assert 0.0 <= r["fprp"] <= 2.0

    def test_select_criterion(self):
        from src.metrics.fairness import select_criterion
        assert select_criterion("biometric") == "recall_parity"
        assert select_criterion("reiot") == "fprp"
        with pytest.raises(ValueError):
            select_criterion("unknown_context")


# ── Privacy ─────────────────────────────────────────────────────────────────
class TestPrivacy:
    def test_privacy_score_range(self):
        from src.metrics.privacy import privacy_score
        for eps in [0.5, 1.0, 3.0, 8.0]:
            for mia in [0.50, 0.60, 0.75]:
                p = privacy_score(eps, mia)
                assert 0.0 <= p <= 1.0, f"P={p} out of range for eps={eps}, mia={mia}"

    def test_lower_epsilon_higher_privacy(self):
        from src.metrics.privacy import privacy_score
        p_tight = privacy_score(epsilon_eff=1.0, mia_auc=0.52)
        p_loose = privacy_score(epsilon_eff=8.0, mia_auc=0.52)
        assert p_tight > p_loose

    def test_lower_mia_higher_privacy(self):
        from src.metrics.privacy import privacy_score
        p_strong = privacy_score(epsilon_eff=3.0, mia_auc=0.50)
        p_weak   = privacy_score(epsilon_eff=3.0, mia_auc=0.90)
        assert p_strong > p_weak

    def test_compute_privacy_dict(self):
        from src.metrics.privacy import compute_privacy
        r = compute_privacy(epsilon_eff=3.0, mia_auc=0.52)
        assert "privacy" in r
        assert "epsilon_eff" in r
        assert "mia_auc" in r


# ── Accountability ──────────────────────────────────────────────────────────
class TestAccountability:
    def test_accountability_score_formula(self):
        from src.metrics.accountability import accountability_score
        a = accountability_score(1.0, 1.0, 1.0)
        assert a == pytest.approx(1.0)
        b = accountability_score(0.0, 0.0, 0.0)
        assert b == pytest.approx(0.0)
        c = accountability_score(0.9, 0.8, 0.7)
        assert c == pytest.approx((0.9 + 0.8 + 0.7) / 3)

    def test_compute_accountability_no_governance(self):
        from src.metrics.accountability import compute_accountability
        r = compute_accountability("/tmp/nonexistent_log.jsonl", 100,
                                    model_has_governance=False)
        assert r["accountability"] < 0.5

    def test_compute_accountability_with_governance(self):
        from src.metrics.accountability import compute_accountability
        r = compute_accountability("/tmp/nonexistent_log.jsonl", 100,
                                    model_has_governance=True)
        assert r["accountability"] > 0.7

    def test_hash_input(self):
        from src.metrics.accountability import hash_input
        x = np.array([1.0, 2.0, 3.0])
        h = hash_input(x)
        assert len(h) == 64  # SHA-256 hex
        assert h == hash_input(x)   # deterministic


# ── Trust Index ─────────────────────────────────────────────────────────────
class TestTrustIndex:
    def test_perfect_ti_equals_one(self):
        from src.metrics.trust_index import trust_index
        from src.metrics.privacy import privacy_score
        ideal_p = privacy_score(3.0, 0.50)
        r = trust_index(clarity=1.0, fairness=1.0, privacy=ideal_p, accountability=1.0)
        assert r["ti"] == pytest.approx(1.0, abs=0.01)

    def test_zero_ti(self):
        from src.metrics.trust_index import trust_index
        r = trust_index(clarity=0.0, fairness=0.0, privacy=0.0, accountability=0.0)
        assert r["ti"] == pytest.approx(0.0, abs=0.01)

    def test_ti_increases_with_pillars(self):
        from src.metrics.trust_index import trust_index
        r_base  = trust_index(0.55, 0.93, 0.12, 0.23)
        r_eagf  = trust_index(0.88, 0.99, 0.90, 0.89)
        assert r_eagf["ti"] > r_base["ti"]

    def test_ti_range(self):
        from src.metrics.trust_index import trust_index
        rng = np.random.RandomState(0)
        for _ in range(50):
            c, f, p, a = rng.rand(4)
            r = trust_index(c, f, p, a)
            assert 0.0 <= r["ti"] <= 1.0

    def test_weights_sum_enforced(self):
        from src.metrics.trust_index import trust_index
        # Weights that don't sum to 1 should be normalised
        w = {"clarity": 1.0, "fairness": 1.0, "privacy": 1.0, "accountability": 1.0}
        r = trust_index(0.8, 0.9, 0.7, 0.8, weights=w)
        assert 0.0 <= r["ti"] <= 1.0


# ── Integration test ─────────────────────────────────────────────────────────
class TestIntegration:
    def test_full_pipeline_runs(self):
        """End-to-end test: data -> train baseline + eagf -> metrics -> TI."""
        import yaml
        import warnings
        warnings.filterwarnings("ignore")

        with open("configs/biometric_default.yaml") as f:
            config = yaml.safe_load(f)
        config["training"]["epochs"] = 10

        from src.utils.data_loader import generate_demo_biometric
        from src.training.eagf_trainer import train_variant

        dataset = generate_demo_biometric(n_samples=300, seed=42)

        m_base = train_variant("baseline", config, dataset.copy(),
                               seed=42, output_dir="/tmp/test_integration/baseline/seed_42")
        m_eagf = train_variant("eagf",     config, dataset.copy(),
                               seed=42, output_dir="/tmp/test_integration/eagf/seed_42")

        assert 0.0 <= m_base["trust_index"] <= 1.0
        assert 0.0 <= m_eagf["trust_index"] <= 1.0
        assert m_eagf["trust_index"] > m_base["trust_index"], \
            f"EAGF TI ({m_eagf['trust_index']:.3f}) should exceed Baseline TI ({m_base['trust_index']:.3f})"

    def test_ablation_key_finding(self):
        """No single-pillar model should beat EAGF on TI."""
        import yaml
        import warnings
        warnings.filterwarnings("ignore")

        with open("configs/biometric_default.yaml") as f:
            config = yaml.safe_load(f)
        config["training"]["epochs"] = 15

        from src.utils.data_loader import generate_demo_biometric
        from src.training.eagf_trainer import train_variant

        dataset = generate_demo_biometric(n_samples=400, seed=42)
        TIs = {}
        for v in ["baseline","transparency","fairness","privacy","accountability","eagf"]:
            m = train_variant(v, config, dataset.copy(),
                              seed=42, output_dir=f"/tmp/test_ablation/{v}/seed_42")
            TIs[v] = m["trust_index"]

        single_max = max(TIs[v] for v in ["transparency","fairness","privacy","accountability"])
        assert TIs["eagf"] > single_max, \
            f"EAGF TI ({TIs['eagf']:.3f}) must exceed max single-pillar TI ({single_max:.3f})"

    def test_reiot_simulation(self):
        """RE-IoT simulator produces correct shapes and attack ratio."""
        from src.utils.reiot_simulator import generate_full_reiot_dataset
        d = generate_full_reiot_dataset(n_urban=3, n_periurban=3, n_rural=3,
                                         n_windows_per_node=20, seed=42)
        assert d["X_train"].ndim == 2
        assert d["y_train"].ndim == 1
        attack_rate = d["y_train"].mean()
        assert 0.02 <= attack_rate <= 0.15, f"Attack rate {attack_rate:.3f} out of expected range"


# ── Dynamic AHP ────────────────────────────────────────────────────────────
class TestDynamicAHP:
    """Unit tests for calculate_dynamic_weights in src/utils/ahp.py."""

    def _w(self, **kwargs):
        from src.utils.ahp import calculate_dynamic_weights
        return calculate_dynamic_weights(**kwargs)

    def test_weights_sum_to_one_baseline(self):
        """Weights must always normalise to 1.0."""
        w = self._w(current_privacy_loss=0.0,
                    current_fairness_loss=0.0,
                    mia_attack_success_rate=0.5)
        assert abs(sum(w.values()) - 1.0) < 1e-9

    def test_mia_above_threshold_increases_privacy(self):
        """MIA > 0.55 must increase privacy weight relative to no-threat case."""
        w_safe  = self._w(current_privacy_loss=0.1,
                          current_fairness_loss=0.0,
                          mia_attack_success_rate=0.50)
        w_threat = self._w(current_privacy_loss=0.1,
                           current_fairness_loss=0.0,
                           mia_attack_success_rate=0.60)
        assert w_threat["privacy"] > w_safe["privacy"], (
            f"privacy weight should rise under MIA threat: "
            f"{w_threat['privacy']:.4f} vs {w_safe['privacy']:.4f}"
        )

    def test_high_fairness_loss_increases_fairness(self):
        """High fairness loss must increase fairness weight."""
        w_low  = self._w(current_privacy_loss=0.1,
                         current_fairness_loss=0.05,
                         mia_attack_success_rate=0.50)
        w_high = self._w(current_privacy_loss=0.1,
                         current_fairness_loss=0.20,
                         mia_attack_success_rate=0.50)
        assert w_high["fairness"] > w_low["fairness"], (
            f"fairness weight should rise with high fairness loss: "
            f"{w_high['fairness']:.4f} vs {w_low['fairness']:.4f}"
        )

    def test_weights_sum_to_one_under_both_threats(self):
        """Weights still sum to 1 when both privacy and fairness rules fire."""
        w = self._w(current_privacy_loss=0.5,
                    current_fairness_loss=0.5,
                    mia_attack_success_rate=0.70)
        assert abs(sum(w.values()) - 1.0) < 1e-9

    def test_ema_smoothing_reduces_jump(self):
        """EMA smoothing must dampen abrupt weight swings."""
        from src.utils.ahp import equal_weights
        prev = equal_weights()
        # Apply a maximum-threat update.
        w_new = self._w(current_privacy_loss=1.0,
                        current_fairness_loss=1.0,
                        mia_attack_success_rate=1.0,
                        previous_weights=prev)
        # EMA α = 0.3, so the change must be ≤ 0.3 × range.
        for k in prev:
            delta = abs(w_new[k] - prev[k])
            assert delta <= 0.30 + 1e-9, (
                f"Weight '{k}' changed by {delta:.4f} > 0.30 — smoothing failed"
            )

    def test_reproducibility_across_seeds(self):
        """Dynamic weights are deterministic (no random state)."""
        w1 = self._w(current_privacy_loss=0.3,
                     current_fairness_loss=0.2,
                     mia_attack_success_rate=0.65)
        w2 = self._w(current_privacy_loss=0.3,
                     current_fairness_loss=0.2,
                     mia_attack_success_rate=0.65)
        for k in w1:
            assert abs(w1[k] - w2[k]) < 1e-12, f"Weight '{k}' not deterministic"

    def test_all_pillar_names_present(self):
        """Returned dict must contain all four EAGF pillar keys."""
        from src.utils.ahp import PILLAR_NAMES
        w = self._w(current_privacy_loss=0.1,
                    current_fairness_loss=0.1,
                    mia_attack_success_rate=0.50)
        for name in PILLAR_NAMES:
            assert name in w, f"Pillar '{name}' missing from dynamic weights"
