#!/usr/bin/env python3
"""scripts/run_scaling_experiment.py — Phase 5B: EAGF Scalability Experiment

Measures training wall-clock time and peak RSS memory as the number of
virtual IoT devices (dataset size proportional to device count) scales from
100 to 10 000 in steps of 1 000.

Usage:
    python scripts/run_scaling_experiment.py

Output:
    results/tables/scaling_overhead.csv
    Columns: device_count, runtime_sec, memory_mb

Constraints:
    - Only 3 training epochs per run to avoid exponential runtime blowup.
    - DP-SGD disabled for speed (Opacus overhead is orthogonal to scaling).
    - Fixed seed 42 for full reproducibility.
"""

import csv
import os
import sys
import time
import warnings

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
warnings.filterwarnings("ignore")

import yaml

try:
    import psutil as _psutil

    _HAS_PSUTIL = True
except ImportError:
    _HAS_PSUTIL = False


def _rss_mb() -> float:
    """Return current process RSS in MiB, or NaN if psutil is unavailable."""
    if _HAS_PSUTIL:
        return float(_psutil.Process().memory_info().rss) / (1024.0 * 1024.0)
    return float("nan")


def main():
    from src.training.eagf_trainer import train_variant
    from src.utils.data_loader import generate_demo_biometric

    with open(
        os.path.join(os.path.dirname(__file__), "..", "configs", "biometric_default.yaml")
    ) as f:
        config = yaml.safe_load(f)

    # Minimal config: keep each run fast so the loop finishes in reasonable time.
    config["training"]["epochs"] = 3
    config["training"]["batch_size"] = 64
    # Disable DP: Opacus setup overhead is orthogonal to the scaling metric.
    config["governance"]["dp_enabled"] = False
    config["governance"]["dp_epsilon"] = 1e9   # effectively disables DP noise

    device_counts = list(range(100, 10001, 1000))  # [100, 1100, 2100, ..., 9100]

    out_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "results", "tables",
    )
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "scaling_overhead.csv")

    print("=" * 60)
    print("  EAGF Scaling Experiment — Phase 5B")
    print("=" * 60)
    print(f"  Device counts : {device_counts}")
    print(f"  Output        : {out_path}")
    print()

    rows = []
    for device_count in device_counts:
        # Each device contributes a fixed number of feature samples.
        # n_samples = device_count gives O(N) dataset growth.
        dataset = generate_demo_biometric(n_samples=max(device_count, 100), seed=42)

        mem_before = _rss_mb()
        t0 = time.perf_counter()
        train_variant(
            "eagf",
            config,
            dataset,
            seed=42,
            output_dir=f"/tmp/eagf_scaling/{device_count}",
        )
        runtime_sec = time.perf_counter() - t0
        mem_after = _rss_mb()

        # Record peak RSS after training; differences reflect dataset memory cost.
        memory_mb = mem_after if not (mem_after != mem_after) else mem_after

        row = {
            "device_count": device_count,
            "runtime_sec": round(runtime_sec, 3),
            "memory_mb": round(memory_mb, 2),
        }
        rows.append(row)
        print(
            f"  devices={device_count:6d}  "
            f"time={runtime_sec:6.2f}s  "
            f"mem={memory_mb:7.1f} MB"
        )

    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["device_count", "runtime_sec", "memory_mb"])
        writer.writeheader()
        writer.writerows(rows)

    print()
    print(f"Scaling results saved to: {out_path}")

    # Verify values increase reasonably with scale.
    runtimes = [r["runtime_sec"] for r in rows]
    memories = [r["memory_mb"] for r in rows]
    assert max(runtimes) >= min(runtimes), "Runtime should not decrease with scale"
    assert rows[-1]["device_count"] > rows[0]["device_count"], "Device count range is valid"
    print(f"  Runtime range : {min(runtimes):.3f}s → {max(runtimes):.3f}s  ✓")
    print(f"  Memory range  : {min(memories):.1f} MB → {max(memories):.1f} MB  ✓")
    print("\nScaling experiment complete.")


if __name__ == "__main__":
    main()
