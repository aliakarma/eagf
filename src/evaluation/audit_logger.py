"""
src/evaluation/audit_logger.py — Cryptographic Audit Logger
Paper: Section 3.5 (Accountability alpha_audit sub-score)
"""
import hashlib, json, os, time
from pathlib import Path
from typing import Optional

import numpy as np

try:
    import psutil as _psutil
    _HAS_PSUTIL = True
except ImportError:
    _HAS_PSUTIL = False


class AuditLogger:
    """Append-only audit log writer.
    Writes one JSONL entry per AI decision with SHA-256 input hash.
    """
    def __init__(self, log_path: str, sign_key_path: Optional[str] = None,
                 model_version: str = "unknown", operator_id: str = "system"):
        self.log_path = Path(log_path)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self.model_version = model_version
        self.operator_id = operator_id

    def log(self, input_data, output_label: int, output_confidence: float,
            inference_time_ms: Optional[float] = None,
            memory_usage_mb: Optional[float] = None,
            energy_overhead_joules: Optional[float] = None) -> dict:
        """Write one signed audit entry.

        Args:
            input_data: Raw input to the model (numpy array or any object).
            output_label: Predicted class label.
            output_confidence: Prediction confidence in [0, 1].
            inference_time_ms: Time taken for the forward pass in milliseconds.
            memory_usage_mb: RSS memory usage of the process in megabytes.
            energy_overhead_joules: Estimated energy cost (time_s * CPU watts).
        """
        if hasattr(input_data, 'tobytes'):
            raw = input_data.tobytes()
        else:
            raw = str(input_data).encode()
        input_hash = hashlib.sha256(raw).hexdigest()

        entry = {
            "model_version":    self.model_version,
            "input_hash":       input_hash,
            "output_label":     int(output_label),
            "output_confidence": float(output_confidence),
            "timestamp":        time.time(),
            "operator_id":      self.operator_id,
        }

        # System overhead fields — included only when explicitly provided.
        if inference_time_ms is not None:
            entry["inference_time_ms"] = float(inference_time_ms)
        if memory_usage_mb is not None:
            entry["memory_usage_mb"] = float(memory_usage_mb)
        if energy_overhead_joules is not None:
            entry["energy_overhead_joules"] = float(energy_overhead_joules)

        # Simple HMAC-style signature (SHA-256 of entry content)
        content = json.dumps({k: v for k, v in entry.items()}, sort_keys=True)
        entry["signature"] = hashlib.sha256(content.encode()).hexdigest()

        with open(self.log_path, "a") as f:
            f.write(json.dumps(entry) + "\n")
        return entry

    def count_entries(self) -> int:
        if not self.log_path.exists():
            return 0
        return sum(1 for _ in open(self.log_path))
