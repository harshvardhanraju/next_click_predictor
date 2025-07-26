# Multi-stage build for optimized image size (using slim instead of alpine for better ML support)
FROM python:3.11-slim as builder

# Install only essential build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Create virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Upgrade pip and install wheel
RUN pip install --no-cache-dir --upgrade pip wheel

# Copy requirements
COPY requirements.txt .

# Install packages with fallback to source builds
RUN pip install --no-cache-dir \
    --prefer-binary \
    -r requirements.txt && \
    # Aggressive cleanup to reduce size
    find /opt/venv -name "*.pyc" -exec rm -f {} + && \
    find /opt/venv -name "__pycache__" -exec rm -rf {} + && \
    find /opt/venv -name "*.so" -exec strip {} + 2>/dev/null || true && \
    find /opt/venv -path "*/tests/*" -exec rm -rf {} + && \
    find /opt/venv -path "*/test/*" -exec rm -rf {} + && \
    find /opt/venv -name "*.pyx" -delete && \
    find /opt/venv -name "*.pxd" -delete && \
    find /opt/venv -path "*/docs/*" -exec rm -rf {} + && \
    find /opt/venv -path "*/examples/*" -exec rm -rf {} + && \
    find /opt/venv -name "*.md" -delete && \
    find /opt/venv -name "*.txt" -delete && \
    find /opt/venv -path "*/include/*" -exec rm -rf {} + && \
    find /opt/venv -name "*.h" -delete && \
    find /opt/venv -name "*.c" -delete

# Production stage - minimal runtime
FROM python:3.11-slim

# Install only runtime dependencies for OpenCV and ML libraries
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Copy the cleaned virtual environment
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Set working directory
WORKDIR /app

# Copy only essential application files
COPY src/ ./src/

# Create necessary directories
RUN mkdir -p data logs

# Set optimized environment variables
ENV PYTHONPATH=/app \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONOPTIMIZE=2 \
    PIP_NO_CACHE_DIR=1

# Security: use non-root user
RUN useradd --create-home --shell /bin/bash --uid 1001 appuser && \
    chown -R appuser:appuser /app
USER appuser

# Expose port
EXPOSE 8000

# Efficient health check
HEALTHCHECK --interval=60s --timeout=10s --start-period=30s --retries=2 \
    CMD python -c "import socket; s=socket.socket(); s.settimeout(5); s.connect(('localhost', 8000)); s.close()" || exit 1

# Run application
CMD ["python", "-m", "uvicorn", "src.web_service:app", "--host", "0.0.0.0", "--port", "8000"]