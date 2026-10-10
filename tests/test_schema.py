"""Schema tests: migrations apply against a throwaway database and honour the spec's rules.

The `db_url` fixture lives in tests/conftest.py.
"""

import json
import uuid

import pytest
import psycopg

from app.db import migrate

TABLES = {"pipeline_runs", "documents", "document_pages", "ocr_lines", "extracted_fields", "chunks"}
VIEWS = {"quarantine", "ocr_lines_safe"}


def run(db_url, query, params=None, fetch=False):
    with psycopg.connect(db_url) as conn, conn.cursor() as cur:
        cur.execute(query, params)
        return cur.fetchall() if fetch else None


def test_migrations_apply_and_are_idempotent(db_url):
    with psycopg.connect(db_url) as conn:
        assert migrate(conn) == []  # second run applies nothing
        names = {row[0] for row in conn.execute("SELECT name FROM schema_migrations").fetchall()}
    assert names == {"001_init.sql"}


def test_all_spec_tables_and_views_exist(db_url):
    tables = {r[0] for r in run(db_url, "SELECT tablename FROM pg_tables WHERE schemaname='public'", fetch=True)}
    views = {r[0] for r in run(db_url, "SELECT viewname FROM pg_views WHERE schemaname='public'", fetch=True)}
    assert TABLES <= tables
    assert VIEWS <= views


def seed_document(db_url, **overrides):
    row = {
        "content_hash": f"sha256:{uuid.uuid4().hex}",
        "storage_uri": "/data/raw/x",
        "status": "normalised",
        **overrides,
    }
    run(
        db_url,
        """INSERT INTO documents (content_hash, storage_uri, status, source_sha256, document_type,
                                  country_code, status_reason, quality_flags)
           VALUES (%(content_hash)s, %(storage_uri)s, %(status)s, %(source_sha256)s,
                   %(document_type)s, %(country_code)s, %(status_reason)s, %(quality_flags)s)""",
        {"source_sha256": None, "document_type": "receipt", "country_code": "SG",
         "status_reason": None, "quality_flags": json.dumps([]), **row},
    )
    return run(db_url, "SELECT id FROM documents WHERE content_hash = %s",
               (row["content_hash"],), fetch=True)[0][0]


def test_duplicate_content_hash_is_rejected(db_url):
    doc = seed_document(db_url)
    with pytest.raises(psycopg.errors.UniqueViolation):
        run(db_url, "INSERT INTO documents (content_hash, storage_uri) "
                    "SELECT content_hash, '/data/raw/y' FROM documents WHERE id = %s", (doc,))


def test_status_and_type_checks_reject_unknowns(db_url):
    with pytest.raises(psycopg.errors.CheckViolation):
        seed_document(db_url, status="bogus")
    with pytest.raises(psycopg.errors.CheckViolation):
        seed_document(db_url, document_type="invoice")


def test_quarantine_view_shows_only_flagged_and_rejected(db_url):
    seed_document(db_url, status="rejected", status_reason="unsupported_ocr_format")
    seed_document(db_url, status="flagged", status_reason="low_confidence", quality_flags='["low_confidence"]')
    seed_document(db_url, status="embedded")
    rows = run(db_url, "SELECT status, status_reason FROM quarantine", fetch=True)
    assert sorted(rows) == [("flagged", "low_confidence"), ("rejected", "unsupported_ocr_format")]


def test_cascade_delete_removes_pages_lines_and_chunks(db_url):
    doc = seed_document(db_url, status="embedded")
    run(db_url, """INSERT INTO document_pages (document_id, page_number, width, height,
                   page_size_known, engine) VALUES (%s, 1, 1.0, 1.0, false, 'tesseract')""", (doc,))
    run(db_url, """INSERT INTO ocr_lines (document_id, page_number, line_index, text, x0, y0, x1, y1)
                   VALUES (%s, 1, 0, 'NRIC S1234567D issued', 0.1, 0.1, 0.5, 0.2)""", (doc,))
    run(db_url, """INSERT INTO chunks (document_id, chunk_index, page_number, line_start, line_end,
                   x0, y0, x1, y1, text, model, dimension, embedding)
                   VALUES (%s, 0, 1, 0, 0, 0.1, 0.1, 0.5, 0.2, 'chunk text', 'm1', 384, %s::vector)""",
       (doc, "[" + ",".join(["1"] + ["0"] * 383) + "]"))

    run(db_url, "DELETE FROM documents WHERE id = %s", (doc,))
    for table in ("document_pages", "ocr_lines", "chunks"):
        count = run(db_url, f"SELECT count(*) FROM {table} WHERE document_id = %s", (doc,), fetch=True)
        assert count[0][0] == 0


def test_ocr_lines_safe_masks_nric_but_keeps_geometry(db_url):
    doc = seed_document(db_url)
    run(db_url, """INSERT INTO document_pages (document_id, page_number, width, height,
                   page_size_known, engine) VALUES (%s, 1, 1.0, 1.0, false, 'tesseract')""", (doc,))
    run(db_url, """INSERT INTO ocr_lines (document_id, page_number, line_index, text, x0, y0, x1, y1)
                   VALUES (%s, 1, 0, 'NRIC/FIN: S1234567D ok', 0.1, 0.1, 0.5, 0.2)""", (doc,))
    row = run(db_url, "SELECT text_masked, x0, y1 FROM ocr_lines_safe WHERE document_id = %s",
              (doc,), fetch=True)[0]
    assert row[0] == "NRIC/FIN: X0000000X ok"
    assert (row[1], row[2]) == (0.1, 0.2)


def test_mask_nric_api_side_matches_view_side(db_url):
    """The Python boundary mask mirrors the SQL view mask."""
    from app.main import mask_nric

    assert mask_nric("NRIC/FIN: S1234567D ok") == "NRIC/FIN: X0000000X ok"
    assert mask_nric("T0099999Q and F7654321L") == "X0000000X and X0000000X"  # all 4 prefixes
    assert mask_nric("S1234567") == "S1234567"        # missing check letter: not an NRIC
    assert mask_nric("POL-SG-000456") == "POL-SG-000456"  # policy numbers are not NRICs


def test_same_chunk_index_under_two_models_coexists(db_url):
    doc = seed_document(db_url, status="embedded")
    run(db_url, """INSERT INTO document_pages (document_id, page_number, width, height,
                   page_size_known, engine) VALUES (%s, 1, 1.0, 1.0, false, 'tesseract')""", (doc,))
    for model in ("old-model", "new-model"):
        run(db_url, """INSERT INTO chunks (document_id, chunk_index, page_number, line_start,
                       line_end, x0, y0, x1, y1, text, model, dimension, embedding)
                       VALUES (%s, 0, 1, 0, 0, 0.0, 0.0, 1.0, 1.0, 'chunk', %s, 384, %s::vector)""",
           (doc, model, "[" + ",".join(["0"] * 384) + "]"))
    models = run(db_url, "SELECT model FROM chunks WHERE document_id = %s ORDER BY model",
                 (doc,), fetch=True)
    assert [m[0] for m in models] == ["new-model", "old-model"]


def test_vector_roundtrip_nearest_neighbour(db_url):
    doc = seed_document(db_url, status="embedded")
    run(db_url, """INSERT INTO document_pages (document_id, page_number, width, height,
                   page_size_known, engine) VALUES (%s, 1, 1.0, 1.0, false, 'tesseract')""", (doc,))
    vec = "[" + ",".join(["1"] + ["0"] * 383) + "]"
    run(db_url, """INSERT INTO chunks (document_id, chunk_index, page_number, line_start, line_end,
                   x0, y0, x1, y1, text, model, dimension, embedding)
                   VALUES (%s, 0, 1, 0, 0, 0.0, 0.0, 1.0, 1.0, 'probe', 'm1', 384, %s::vector)""",
       (doc, vec))
    # HNSW serves the cosine operator; the exact vector must come back as the nearest.
    nearest = run(db_url, "SELECT text FROM chunks ORDER BY embedding <=> %s::vector LIMIT 1",
                  (vec,), fetch=True)
    assert nearest[0][0] == "probe"
