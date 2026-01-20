FROM python:3.11-slim

RUN apt-get update && \
    apt-get install --no-install-recommends -y build-essential gcc && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

COPY uv.lock uv.lock
COPY pyproject.toml pyproject.toml
COPY README.md README.md
COPY sentiment_api_starter.py sentiment_api_starter.py
WORKDIR /
RUN uv sync --locked --no-cache


EXPOSE 8080

CMD ["uv", "run", "uvicorn", "sentiment_api_starter:app", "--port", "8080", "--host", "0.0.0.0", "--workers", "1"]
