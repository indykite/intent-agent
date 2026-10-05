# Inspired by
# https://sourcery.ai/blog/python-docker/
# https://fastapi.tiangolo.com/deployment/docker/#dockerfile
# Supports multi-arch
FROM python:3.13-trixie@sha256:874a8d051a420a2a02c833e7b45f15b67a360b7052dc2074a54387657f16fc3c AS python-deps

# Setup env
ENV LANG=C.UTF-8 \
    LC_ALL=C.UTF-8 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONFAULTHANDLER=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

COPY pyproject.toml uv.lock ./

# hadolint ignore=DL3013
RUN pip install --no-cache-dir uv && \
    uv lock --check && \
    uv sync && \
    rm pyproject.toml uv.lock

COPY indykite ./indykite

#
### Production image
FROM gcr.io/distroless/python3-debian13@sha256:0e48fa084c225991c3f4e133306cd64fdca9ed028215a2d8a67eb798c857a981 AS runtime

# Setup env
ENV LANG=C.UTF-8 \
    LC_ALL=C.UTF-8 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONFAULTHANDLER=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/app/.venv/bin:$PATH" \
    PYTHONPATH="/app/.venv/lib/python3.13/site-packages"

WORKDIR /app

# Copy application, virtual env and empty dirs for models from python-deps stage
COPY --from=python-deps --chown=nonroot /app ./

# set the uid as an integer for compatibility with runAsNonRoot in Kubernetes
USER 65532

# distroless has no shell and no curl; python3 is the only interpreter guaranteed present
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD ["python3", "-c", "import sys,urllib.request; sys.exit(0 if urllib.request.urlopen('http://localhost:8001/health', timeout=5).status == 200 else 1)"]

# Run the application
ENTRYPOINT ["python3", "-m", "indykite.intent_agent"]
