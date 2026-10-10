"""Minimal versioned-SQL migration runner (plain versioned SQL files).

Applies db/migrations/*.sql in filename order, tracking applied files in schema_migrations.
Called automatically at app startup, or by hand: python -m app.db
"""

import os
from pathlib import Path

import psycopg

from .normalise import NORMALISER_VERSION

CODE_VERSION = os.environ.get("CODE_VERSION", "dev")
DATABASE_URL = os.environ.get(
    "DATABASE_URL", "postgresql://postgres:local@localhost:5432/ocr_pipeline"
)
MIGRATIONS_DIR = Path(__file__).resolve().parents[1] / "db" / "migrations"


def migrate(conn: psycopg.Connection) -> list[str]:
    """Apply pending migrations; returns the names applied. Safe to call repeatedly."""
    # Transaction-scoped lock, held until the caller commits: concurrent workers starting
    # together serialise here instead of racing to apply the same file.
    conn.execute("SELECT pg_advisory_xact_lock(7283461)")
    with conn.cursor() as cur:
        cur.execute(
            "CREATE TABLE IF NOT EXISTS schema_migrations ("
            "  name text PRIMARY KEY,"
            "  applied_at timestamptz NOT NULL DEFAULT now()"
            ")"
        )
        applied = {row[0] for row in cur.execute("SELECT name FROM schema_migrations").fetchall()}
        ran = []
        for path in sorted(MIGRATIONS_DIR.glob("*.sql")):
            if path.name in applied:
                continue
            # One transaction per file: a failed migration leaves no half-applied DDL.
            with conn.transaction(), conn.cursor() as cur:
                cur.execute(path.read_text())
                cur.execute(
                    "INSERT INTO schema_migrations (name) VALUES (%s)", (path.name,)
                )
            ran.append(path.name)
    return ran


def new_run(conn: psycopg.Connection, embedding_model: str | None = None) -> str:
    """Lineage root: one pipeline_runs row stamped with code, normaliser and embedding model versions."""
    return conn.execute(
        """INSERT INTO pipeline_runs (code_version, normaliser_version, embedding_model, finished_at)
           VALUES (%s, %s, %s, now()) RETURNING id""",
        (CODE_VERSION, NORMALISER_VERSION, embedding_model),
    ).fetchone()[0]


if __name__ == "__main__":
    with psycopg.connect(DATABASE_URL) as conn:
        applied = migrate(conn)
    print("\n".join(applied) if applied else "up to date")
