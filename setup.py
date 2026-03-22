"""setup.py — EAGF package. Allows: pip install -e ."""
from setuptools import setup, find_packages

with open("README.md", encoding="utf-8") as f:
    long_description = f.read()

setup(
    name="eagf",
    version="1.0.0",
    description=(
        "Ethical AI Governance Framework: Four-Pillar System for "
        "Transparency, Fairness, Privacy, and Accountability in RE-IoT Cybersecurity"
    ),
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/aliakarma/eagf",
    license="MIT",
    packages=find_packages(include=["src", "src.*"]),
    python_requires=">=3.9",
    install_requires=[
        "numpy>=1.24.0",
        "pandas>=2.0.0",
        "scipy>=1.10.0",
        "scikit-learn>=1.3.0",
        "statsmodels>=0.14.0",
        "matplotlib>=3.7.0",
        "pyyaml>=6.0",
        "tqdm>=4.66.0",
    ],
    extras_require={
        "full": ["shap>=0.42.0", "torch>=2.0.0", "opacus>=1.4.0"],
        "dev":  ["pytest>=7.4.0", "black>=23.7.0", "isort>=5.12.0"],
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Security",
    ],
    keywords="trustworthy-ai ethical-ai fairness privacy accountability cybersecurity iot",
)
