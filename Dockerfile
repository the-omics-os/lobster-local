# Multi-stage Dockerfile for Lobster AI Streamlit App
# Optimized for AWS App Runner deployment

# Stage 1: Base image with system dependencies
FROM python:3.13-slim as base

# Install system dependencies required for bioinformatics packages
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

# Stage 2: Builder stage for Python dependencies
FROM base as builder

# Set working directory
WORKDIR /app

# Copy dependency definitions
COPY pyproject.toml ./

# Create virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Upgrade pip and install build tools *inside the venv*
RUN pip install --no-cache-dir --upgrade pip setuptools wheel packaging build

# Install app + dependencies inside venv
COPY . .
RUN pip install --no-cache-dir .

# Stage 3: Final runtime image
FROM base as runtime

# Create non-root user for security
RUN useradd -m -u 1000 -s /bin/bash lobsteruser && \
    mkdir -p /home/lobsteruser/.streamlit

# Set working directory
WORKDIR /app

# Copy virtual environment from builder
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy application code
COPY --chown=lobsteruser:lobsteruser . .

# Create necessary directories and set permissions
RUN mkdir -p /app/.lobster_workspace/data \
             /app/.lobster_workspace/plots \
             /app/.lobster_workspace/exports \
             /app/data/cache \
    && chown -R lobsteruser:lobsteruser /app /home/lobsteruser

# Create Streamlit config
RUN echo '[server]\n\
headless = true\n\
port = 8501\n\
address = "0.0.0.0"\n\
enableCORS = false\n\
enableXsrfProtection = false\n\
\n\
[browser]\n\
gatherUsageStats = false\n\
\n\
[theme]\n\
base = "light"\n\
primaryColor = "#ff4b4b"\n\
backgroundColor = "#ffffff"\n\
secondaryBackgroundColor = "#f0f2f6"\n\
textColor = "#262730"' > /home/lobsteruser/.streamlit/config.toml \
    && chown lobsteruser:lobsteruser /home/lobsteruser/.streamlit/config.toml

# Switch to non-root user
USER lobsteruser

# Environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app

# Expose port for App Runner
EXPOSE 8501

# Health check (modern Streamlit uses /healthz)
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD curl -f http://localhost:8501/healthz || exit 1

# Default command to run Streamlit
CMD ["streamlit", "run", "lobster/streamlit_app.py"]

# Labels for metadata
LABEL maintainer="Homara AI <support@homara.ai>"
LABEL description="Lobster AI - Multi-Agent Bioinformatics Streamlit App"
LABEL version="2.0.0"