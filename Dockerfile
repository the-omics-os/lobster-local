# Multi-stage build for Lobster AI
# Optimized for size and security

# Stage 1: Base image with system dependencies
FROM python:3.11-slim as base

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    git \
    curl \
    build-essential \
    libhdf5-dev \
    libxml2-dev \
    libxslt-dev \
    libffi-dev \
    libssl-dev \
    libcurl4-openssl-dev \
    && rm -rf /var/lib/apt/lists/*

# Stage 2: Builder stage for Python dependencies
FROM base as builder

# Set working directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --user --no-cache-dir --upgrade pip setuptools wheel && \
    pip install --user --no-cache-dir -r requirements.txt

# Stage 3: Final runtime image
FROM base as runtime

# Create non-root user
RUN useradd -m -u 1000 -s /bin/bash lobster

# Set working directory
WORKDIR /app

# Copy Python dependencies from builder
COPY --from=builder /root/.local /home/lobster/.local

# Copy application code
COPY . .

# Install Lobster AI in development mode
RUN pip install -e .

# Create necessary directories
RUN mkdir -p /home/lobster/.lobster /app/data/exports && \
    chown -R lobster:lobster /app /home/lobster/.lobster

# Switch to non-root user
USER lobster

# Set environment variables
ENV PATH=/home/lobster/.local/bin:$PATH
ENV PYTHONUNBUFFERED=1
ENV LOBSTER_HOME=/home/lobster/.lobster

# Default workspace
VOLUME ["/home/lobster/.lobster"]

# Health check
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
    CMD lobster --version || exit 1

# Default command
ENTRYPOINT ["lobster"]
CMD ["chat"]

# Labels
LABEL maintainer="Homara AI <support@homara.ai>"
LABEL description="Lobster AI - Multi-Agent Bioinformatics Analysis System"
LABEL version="1.0.0"
