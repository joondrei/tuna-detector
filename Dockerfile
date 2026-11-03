FROM python:3.12-slim

COPY --from=ghcr.io/astral-sh/uv:0.13.1 /uv /uvx /bin/

ENV UV_PROJECT_ENVIRONMENT=/opt/venv \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    PATH="/opt/venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    OMP_NUM_THREADS=1 \
    MKL_NUM_THREADS=1

WORKDIR /app

COPY pyproject.toml uv.lock ./
RUN uv sync --locked --no-install-project --all-extras

COPY configs ./configs
COPY models ./models
COPY pipeline ./pipeline
COPY scripts ./scripts
RUN uv sync --locked --all-extras

RUN useradd --create-home --uid 1000 runner && chown -R runner /app
USER runner

CMD ["python", "scripts/tune_trial.py"]
