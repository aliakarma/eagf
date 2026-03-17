# Contributing to EAGF

Thank you for your interest in contributing to the Ethical AI Governance Framework.

## Getting Started

```bash
git clone https://github.com/your-org/eagf.git
cd eagf
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
```

## Running Tests

```bash
# Standalone runner (no pytest needed)
python -W ignore tests/run_tests.py

# With pytest (if installed)
pytest tests/ -v -W ignore
```

All 63 tests must pass before submitting a pull request.

## Code Style

- Format with `black src/ tests/` before committing
- All public functions require docstrings with `Args:` and `Returns:`
- New metrics must include unit tests in `tests/test_metrics.py`

## Adding a New Governance Pillar Metric

1. Implement in `src/metrics/your_metric.py` following the existing pattern
2. Return a dictionary with at minimum `{"your_metric": float_value}`
3. Add the metric to `trust_index.py` weights and normalisation
4. Write ≥ 5 unit tests covering: range, monotonicity, edge cases
5. Update `docs/metric_definitions.md` with formal definition and equation

## Adding a New Ablation Variant

1. Add variant key to `SUPPORTED_MODELS` in `src/training/eagf_trainer.py`
2. Handle it in `train_variant()` and `_compute_all_metrics()`
3. Add variant label to `MODEL_LABELS` in `src/evaluation/ablation.py`

## Reporting Issues

Open an issue with:
- Python version and OS
- Full traceback
- Minimal reproducible example
- Expected vs actual output

## Paper Citation

If you build on this work, please cite the paper (see `CITATION.cff`).
