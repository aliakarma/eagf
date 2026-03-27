"""Verify privacy metric values for epsilon=3.0 and MIA=0.53."""

from src.metrics.privacy import compute_privacy


def main() -> None:
    metrics = compute_privacy(epsilon_eff=3.0, mia_auc=0.53)
    print(f"P_raw={metrics['privacy_raw']:.5f}")
    print(f"P_ideal={metrics['privacy_ideal']:.5f}")
    print(f"P_normalized={metrics['privacy_normalized']:.4f}")


if __name__ == "__main__":
    main()
