FROM python:3.12-slim
# Lambda Web Adapter: a Lambda extension that turns invocations into HTTP requests to :8080.
# Outside Lambda it is never started, so the same image runs in compose.
COPY --from=public.ecr.aws/awsguru/aws-lambda-adapter:1.1.0 /lambda-adapter /opt/extensions/lambda-adapter
COPY --from=ghcr.io/astral-sh/uv:0.9.21 /uv /usr/local/bin/uv
WORKDIR /app
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy UV_PROJECT_ENVIRONMENT=/app/.venv \
    PATH=/app/.venv/bin:$PATH PORT=8080
COPY pyproject.toml uv.lock ./
COPY packages packages
RUN uv sync --frozen --no-dev
COPY migrations migrations
EXPOSE 8080
CMD ["uvicorn", "--factory", "vigilex_api.app:create_app", "--host", "0.0.0.0", "--port", "8080"]
