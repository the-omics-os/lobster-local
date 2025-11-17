# Multi-stage Dockerfile for Lobster AI CLI
# Optimized for bioinformatics workloads with CLI-first design
# Supports both interactive chat and automation (query mode)

# ==============================================================================
# Stage 1: Base image with system dependencies
# ==============================================================================
FROM --platform=linux/amd64 python:3.11-slim AS base

# Install system dependencies required for bioinformatics packages
# Includes compilers, HDF5, XML parsers, BLAS/LAPACK for numerical computing
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    gfortran \
    git \
    curl \
    build-essential \
    libhdf5-dev \
    libxml2-dev \
    libxslt-dev \
    libffi-dev \
    libssl-dev \
    libcurl4-openssl-dev \
    pkg-config \
    libblas-dev \
    liblapack-dev \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# ==============================================================================
# Stage 2: Builder stage for Python dependencies
# ==============================================================================
FROM base AS builder

WORKDIR /app

# Copy dependency definitions
COPY pyproject.toml ./
COPY README.md ./

# Create virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Upgrade pip and install build tools inside the venv
RUN pip install --no-cache-dir --upgrade pip setuptools wheel packaging build

# Install PyTorch CPU-only FIRST (avoids 3GB CUDA packages)
# This must happen before installing the main package to avoid pulling CUDA version
RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu

# Install application + dependencies inside venv
COPY . .
RUN pip install --no-cache-dir .

# ==============================================================================
# Stage 3: Final runtime image
# ==============================================================================
FROM base AS runtime

# Create non-root user for security
RUN useradd -m -u 1000 -s /bin/bash lobsteruser

WORKDIR /app

# Copy virtual environment from builder
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy application code
COPY --chown=lobsteruser:lobsteruser . .

# Create necessary directories and set permissions
# - .lobster_workspace: Persistent workspace for analysis sessions
# - data/cache: GEO cache and other downloaded data
# - .geo_cache: Legacy cache location (for backward compatibility)
RUN mkdir -p \
    /app/.lobster_workspace/data \
    /app/.lobster_workspace/plots \
    /app/.lobster_workspace/exports \
    /app/data/cache \
    /app/.geo_cache \
    && chown -R lobsteruser:lobsteruser /app

# Switch to non-root user
USER lobsteruser

# Environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app \
    LOBSTER_WORKSPACE_DIR=/app/.lobster_workspace \
    LOBSTER_CACHE_DIR=/app/data/cache

# Labels for metadata
LABEL maintainer="Omics-OS <info@omics-os.com>" \
      description="Lobster AI - Multi-Agent Bioinformatics CLI" \
      version="2.5.0" \
      org.opencontainers.image.source="https://github.com/the-omics-os/lobster-local" \
      org.opencontainers.image.documentation="https://github.com/the-omics-os/lobster-local/wiki"

# Default: Interactive CLI mode
# Can be overridden with: docker run ... lobster query "..."
ENTRYPOINT ["lobster"]
CMD ["chat --reasoning"]
