# Control-plane image: the same image runs the API and the worker (docker-compose.yml picks the role).
# GPU work never runs here; the worker starts Modal sandboxes, which need Modal credentials
# (~/.modal.toml, mounted read-only by compose, never baked into the image).
# Same Python as the host .venv, so behaviour matches `uv run`.
FROM python:3.14-slim

COPY --from=ghcr.io/astral-sh/uv:0.8 /uv /usr/local/bin/uv
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy PYTHONUNBUFFERED=1 PATH="/app/.venv/bin:$PATH"
WORKDIR /app

# Dependencies first (cached until pyproject.toml / uv.lock change), then the package itself.
COPY pyproject.toml uv.lock README.md ./
RUN uv sync --frozen --no-dev --no-install-project
COPY fes ./fes
COPY experiments ./experiments
RUN uv sync --frozen --no-dev

# results/ is bind-mounted from the host (trial results, scheduler records).
EXPOSE 8060
ENTRYPOINT ["fes-control-plane"]
CMD ["--role", "all", "--host", "0.0.0.0"]
