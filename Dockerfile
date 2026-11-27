ARG PYTHON_VERSION=3.12

FROM python:${PYTHON_VERSION}-slim AS builder

COPY --from=ghcr.io/astral-sh/uv:0.13.10 /uv /bin/uv

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_NO_CACHE=1 \
    UV_PYTHON_DOWNLOADS=never \
    UV_PYTHON=/usr/local/bin/python3

WORKDIR /app

COPY pyproject.toml uv.lock ./
RUN uv sync --locked --no-dev --no-install-project

COPY README.md ./
COPY src ./src
RUN uv sync --locked --no-dev --no-editable


FROM python:${PYTHON_VERSION}-slim AS runtime

ARG UID=1000
ARG GID=1000

RUN groupadd --gid "${GID}" app \
    && useradd --uid "${UID}" --gid app --no-create-home --shell /usr/sbin/nologin app

COPY --from=builder /app/.venv /app/.venv
COPY scripts /app/scripts

ENV PATH="/app/.venv/bin:${PATH}" \
    PYTHONUNBUFFERED=1 \
    STOCK_PLANNER_DATA_DIR=/app/data

WORKDIR /app
USER app

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=3s --start-period=10s --retries=3 \
    CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=2)"]

CMD ["uvicorn", "stock_planner.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
