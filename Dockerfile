FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HFI_CONFIG=/app/config/config.yaml

WORKDIR /app

# CPU-only PyTorch first (the default PyPI wheel pulls in several GB of CUDA libraries).
RUN pip install --index-url https://download.pytorch.org/whl/cpu "torch>=2.1.0"

# Package metadata + sources (pyproject.toml reads README.md and the package version)
COPY pyproject.toml README.md LICENSE ./
COPY src/ src/
RUN pip install .

# Runtime assets
COPY config/ config/
COPY neo4j/ neo4j/

# Run as an unprivileged user
RUN useradd --create-home --uid 10001 appuser && chown -R appuser /app
USER appuser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=3)"

CMD ["uvicorn", "hybrid_fraud.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
