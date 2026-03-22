"""
tests/conftest.py — pytest configuration for EAGF test suite.
Adds project root to sys.path and suppresses convergence warnings.
"""
import sys
import os
import warnings

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Suppress sklearn convergence warnings in tests (low epoch counts by design)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", message=".*ConvergenceWarning.*")
warnings.filterwarnings("ignore", message=".*Maximum iterations.*")
