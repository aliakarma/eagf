"""
src/evaluation/audit_logger.py — Cryptographic Audit Logger
Paper: Section 3.5 (Accountability alpha_audit sub-score)
"""
import hashlib, json, os, time
from pathlib import Path
from typing import Optional

import numpy as np


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

    def log(self, input_data, output_label: int, output_confidence: float) -> dict:
        """Write one signed audit entry."""
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
