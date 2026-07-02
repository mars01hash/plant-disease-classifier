# Multi-stage build for production ML application
# Stage 1: Builder (install dependencies)
# Stage 2: Runtime (minimal image for deployment)

# ============================================================================
# STAGE 1: BUILDER
# ============================================================================
FROM python:3.11-slim as builder

WORKDIR /app

# Install system dependencies (needed for some Python packages)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libopencv-dev \
    python3-opencv \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .

# Install Python dependencies in a virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# ============================================================================
# STAGE 2: RUNTIME
# ============================================================================
FROM python:3.11-slim

WORKDIR /app

# Install runtime dependencies only (much smaller image)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libopencv-core4.5 \
    libopencv-imgproc4.5 \
    && rm -rf /var/lib/apt/lists/*

# Copy virtual environment from builder
COPY --from=builder /opt/venv /opt/venv

# Set up environment
ENV PATH="/opt/venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# Create non-root user (security best practice)
RUN useradd -m -u 1000 mluser

# Copy application code
COPY config.yaml .
COPY src/ ./src/
COPY models/ ./models/

# Set permissions
RUN chown -R mluser:mluser /app

# Switch to non-root user
USER mluser

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:8000/health')" || exit 1

# Expose port
EXPOSE 8000

# Start API server
CMD ["python", "src/api.py"]

# ============================================================================
# BUILD INSTRUCTIONS:
# docker build -t plant-classifier:latest .
# 
# RUN INSTRUCTIONS:
# docker run -p 8000:8000 plant-classifier:latest
# 
# With GPU support:
# docker run --gpus all -p 8000:8000 plant-classifier:latest
# ============================================================================
