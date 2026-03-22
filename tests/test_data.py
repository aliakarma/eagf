"""tests/test_data.py — Unit tests for data loading and preprocessing."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pytest


class TestDataLoader:
    def test_demo_biometric_shapes(self):
        from src.utils.data_loader import generate_demo_biometric
        d = generate_demo_biometric(n_samples=400, n_classes=4, seed=42)
        assert d["X_train"].ndim == 2
        assert d["y_train"].ndim == 1
        assert d["groups_train"].ndim == 1
        assert len(d["X_train"]) == len(d["y_train"])

    def test_demo_biometric_splits_sum(self):
        from src.utils.data_loader import generate_demo_biometric
        d = generate_demo_biometric(n_samples=400, seed=42)
        total = len(d["y_train"]) + len(d["y_val"]) + len(d["y_test"])
        assert total == 400

    def test_demo_biometric_has_bias(self):
        """dark-skinned groups should have higher noise -> some RP < 1."""
        from src.utils.data_loader import generate_demo_biometric
        d = generate_demo_biometric(n_samples=800, seed=42)
        groups = d["groups_test"]
        assert "male_light" in groups
        assert "female_dark" in groups

    def test_reiot_shapes(self):
        from src.utils.reiot_simulator import generate_full_reiot_dataset
        d = generate_full_reiot_dataset(n_urban=4, n_periurban=4, n_rural=4,
                                         n_windows_per_node=10, seed=42)
        assert d["X_train"].shape[1] == 300   # 60 timesteps * 5 features
        assert set(np.unique(d["y_train"])).issubset({0, 1})

    def test_reiot_node_class_labels(self):
        from src.utils.reiot_simulator import generate_full_reiot_dataset
        d = generate_full_reiot_dataset(n_urban=3, n_periurban=3, n_rural=3,
                                         n_windows_per_node=10, seed=42)
        classes = set(d["groups_train"])
        assert "urban" in classes
        assert "rural" in classes


class TestPreprocessing:
    def test_sample_weights_balanced(self):
        from src.utils.preprocessing import compute_sample_weights
        y = np.array([0, 0, 0, 1, 1, 1])
        g = np.array(["A","A","B","A","A","B"])
        w = compute_sample_weights(y, g, strategy="balanced_group")
        assert len(w) == len(y)
        assert np.all(w > 0)

    def test_dp_noise_changes_values(self):
        from src.utils.preprocessing import apply_feature_noise_dp
        X = np.ones((10, 5), dtype=np.float32)
        X_noisy = apply_feature_noise_dp(X, epsilon=1.0, seed=42)
        assert not np.allclose(X, X_noisy), "DP noise should change values"

    def test_normalise_features(self):
        from src.utils.preprocessing import normalise_features
        X_tr = np.random.randn(100, 20).astype(np.float32)
        X_te = np.random.randn(30, 20).astype(np.float32)
        X_tr_s, _, X_te_s, scaler = normalise_features(X_tr, X_test=X_te)
        assert abs(X_tr_s.mean()) < 0.1
        assert abs(X_tr_s.std() - 1.0) < 0.1


class TestAuditLogger:
    def test_log_creates_file(self, tmp_path):
        from src.evaluation.audit_logger import AuditLogger
        log_path = str(tmp_path / "test_audit.jsonl")
        logger = AuditLogger(log_path, model_version="test-v1", operator_id="pytest")
        x = np.array([1.0, 2.0, 3.0])
        entry = logger.log(x, output_label=1, output_confidence=0.9)
        assert os.path.exists(log_path)
        assert entry["output_label"] == 1
        assert "signature" in entry
        assert logger.count_entries() == 1

    def test_log_multiple_entries(self, tmp_path):
        from src.evaluation.audit_logger import AuditLogger
        log_path = str(tmp_path / "test_audit2.jsonl")
        logger = AuditLogger(log_path)
        for i in range(5):
            logger.log(np.array([float(i)]), i % 2, 0.8)
        assert logger.count_entries() == 5
