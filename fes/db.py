"""Postgres connections for the control plane (the job queue / events and the artifact catalog).

The database runs in Docker (docker-compose.yml: `docker compose up -d`). Connect with
DATABASE_URL; the default matches the compose file. Connections are autocommit: each statement
commits on its own, and multi-statement updates use `with conn.transaction():`.
"""

import os
import re

import psycopg

DATABASE_URL = os.environ.get("DATABASE_URL", "postgresql://fes:fes@127.0.0.1:5432/fes")


def connect(url: str | None = None, readonly: bool = False) -> psycopg.Connection:
    """A new autocommit connection (one per Store / Catalog; each guards it with its own lock)."""
    try:
        conn = psycopg.connect(url or DATABASE_URL, autocommit=True)
    except psycopg.OperationalError as e:
        raise SystemExit(f"cannot reach Postgres at {redacted(url or DATABASE_URL)}: {e}\n"
                         "start it with: docker compose up -d") from None
    if readonly:
        conn.execute("SET SESSION CHARACTERISTICS AS TRANSACTION READ ONLY")
    return conn


def create_schema(conn: psycopg.Connection, ddl: list[str]) -> None:
    """Run CREATE ... IF NOT EXISTS statements under an advisory lock, so two processes starting
    together (API and worker) don't race on creating the same tables."""
    with conn.transaction():
        conn.execute("SELECT pg_advisory_xact_lock(724001)")
        for stmt in ddl:
            conn.execute(stmt)


def redacted(url: str) -> str:
    """The URL with its password hidden, for logs."""
    return re.sub(r"(//[^:/@]+):[^@]*@", r"\1:***@", url)
