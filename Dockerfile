# Inspired by
# https://sourcery.ai/blog/python-docker/
# https://fastapi.tiangolo.com/deployment/docker/#dockerfile
# Supports multi-arch
FROM python:3.13-trixie@sha256:82c46c08c991d3d3ff10476ac5e386c2c28f27bbd3985a02c5e39fe99ab272eb AS python-deps

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
FROM gcr.io/distroless/python3-debian13@sha256:e866e2f9fbaa19f63e579cf2f61b7f76cd8a74c43a62a0af95d48678d9878436 AS runtime

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
