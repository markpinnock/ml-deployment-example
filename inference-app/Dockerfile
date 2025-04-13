FROM python:3.12-slim
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /app

COPY pyproject.toml .
RUN uv sync --no-cache

COPY . /app

CMD ["uv", "run", "fastapi", "run", "src/app/app.py", "--port", "80"]
