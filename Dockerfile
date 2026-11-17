# One image for both services: the MLflow tracking server and the prediction API.
# Sharing it keeps the MLflow client and server on exactly the same version.
FROM ghcr.io/astral-sh/uv:0.13.5 AS uv
FROM python:3.13.17-slim-bookworm

RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --create-home --uid 1000 app

WORKDIR /app

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    PATH="/app/.venv/bin:$PATH" \
    MLFLOW_DISABLE_AGENT_HINT=1

COPY --from=uv /uv /uvx /bin/

# Dependencies first: this layer is reused until uv.lock changes.
COPY pyproject.toml uv.lock ./
RUN uv sync --locked --no-dev --no-install-project --no-cache

# Then the code. Models are not copied in: the API pulls the production model
# from the MLflow registry at startup.
COPY src ./src
COPY app ./app
RUN uv sync --locked --no-dev --no-cache

USER app
EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
