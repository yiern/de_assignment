"""Shared fixture: a throwaway database with migrations applied, one per test session.

Requires the compose database (tests run inside the api container, where DATABASE_URL
points at db:5432).
"""

import uuid

import pytest
import psycopg
from psycopg.conninfo import conninfo_to_dict, make_conninfo

from app.db import DATABASE_URL, migrate


@pytest.fixture(scope="session")
def db_url():
    admin = psycopg.connect(DATABASE_URL, dbname="postgres", autocommit=True)
    name = "ocr_test_" + uuid.uuid4().hex[:8]
    admin.execute(f"CREATE DATABASE {name}")
    try:
        url = make_conninfo(DATABASE_URL, **{**conninfo_to_dict(DATABASE_URL), "dbname": name})
        with psycopg.connect(url) as conn:
            migrate(conn)
        yield url
    finally:
        admin.execute(f"DROP DATABASE {name} WITH (FORCE)")
        admin.close()
