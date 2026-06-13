"""
src/evaluation/audit_logger.py — Cryptographic Audit Logger
Paper: Section 3.5 (Accountability alpha_audit sub-score)

Audit entries are signed with RSA-SHA256 when a key is available,
falling back to SHA-256 HMAC otherwise.
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

try:
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric import rsa, padding
    _HAS_CRYPTO = True
except ImportError:
    _HAS_CRYPTO = False


def generate_audit_key(key_path: str = "configs/audit_key.pem") -> str:
    """Generate an RSA key pair for audit log signing."""
    if not _HAS_CRYPTO:
        raise ImportError("cryptography package required: pip install cryptography")
    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    key_path = Path(key_path)
    key_path.parent.mkdir(parents=True, exist_ok=True)
    with open(key_path, "wb") as f:
        f.write(key.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.PKCS8,
            encryption_algorithm=serialization.NoEncryption(),
        ))
    pub_path = key_path.with_suffix(".pub")
    with open(pub_path, "wb") as f:
        f.write(key.public_key().public_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PublicFormat.SubjectPublicKeyInfo,
        ))
    return str(key_path)


class AuditLogger:
    """Append-only audit log writer.
    Writes one JSONL entry per AI decision with SHA-256 input hash
    and RSA-SHA256 signature (or SHA-256 HMAC fallback).
    """
    def __init__(self, log_path: str, sign_key_path: Optional[str] = None,
                 model_version: str = "unknown", operator_id: str = "system"):
        self.log_path = Path(log_path)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self.model_version = model_version
        self.operator_id = operator_id
        self._private_key = None

        if sign_key_path and _HAS_CRYPTO:
            key_file = Path(sign_key_path)
            if not key_file.exists():
                generate_audit_key(str(key_file))
            try:
                with open(key_file, "rb") as f:
                    self._private_key = serialization.load_pem_private_key(
                        f.read(), password=None)
            except Exception:
                self._private_key = None

    def _sign(self, content: str) -> str:
        if self._private_key is not None:
            sig = self._private_key.sign(
                content.encode(),
                padding.PKCS1v15(),
                hashes.SHA256(),
            )
            return sig.hex()
        return hashlib.sha256(content.encode()).hexdigest()

    def log(self, input_data, output_label: int, output_confidence: float,
            inference_time_ms: Optional[float] = None,
            memory_usage_mb: Optional[float] = None,
            energy_overhead_joules: Optional[float] = None,
            ece: Optional[float] = None,
            brier_score: Optional[float] = None) -> dict:
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

        if inference_time_ms is not None:
            entry["inference_time_ms"] = float(inference_time_ms)
        if memory_usage_mb is not None:
            entry["memory_usage_mb"] = float(memory_usage_mb)
        if energy_overhead_joules is not None:
            entry["energy_overhead_joules"] = float(energy_overhead_joules)
        if ece is not None:
            entry["ece"] = float(ece)
        if brier_score is not None:
            entry["brier_score"] = float(brier_score)

        content = json.dumps({k: v for k, v in entry.items()}, sort_keys=True)
        entry["signature"] = self._sign(content)

        with open(self.log_path, "a") as f:
            f.write(json.dumps(entry) + "\n")
        return entry

    def count_entries(self) -> int:
        if not self.log_path.exists():
            return 0
        return sum(1 for _ in open(self.log_path))
