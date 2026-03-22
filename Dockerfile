FROM python:3.10-slim

LABEL maintainer="salman.jan@aou.org.bh"
LABEL description="EAGF: Ethical AI Governance Framework"
LABEL version="1.0.0"

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies (subset that installs without network issues)
COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install --no-cache-dir \
        numpy \
        pandas \
        scipy \
        scikit-learn \
        matplotlib \
        pyyaml \
        tqdm \
        statsmodels && \
    rm -rf /root/.cache/pip

# Copy project files
COPY . .

# Create output directories
RUN mkdir -p results figures data/biometric data/reiot

# Set Python path
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

# Default: run fast demo (overridable via docker run command)
CMD ["python", "-W", "ignore", "run_eagf.py", "--fast", "--skip-pareto"]

# ── Usage ──────────────────────────────────────────────────────────────────
# Build:
#   docker build -t eagf:1.0 .
#
# Fast demo (~3 min):
#   docker run --rm -v $(pwd)/results:/app/results -v $(pwd)/figures:/app/figures eagf:1.0
#
# Full paper results (~20 min, 3 seeds):
#   docker run --rm -v $(pwd)/results:/app/results -v $(pwd)/figures:/app/figures \
#     eagf:1.0 python -W ignore run_eagf.py --seeds 42 123 456 --epochs 50
#
# Tests only:
#   docker run --rm eagf:1.0 python -W ignore tests/run_tests.py
