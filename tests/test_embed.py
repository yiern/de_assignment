"""Chunking unit tests (pure functions) + embed/search tests against the throwaway DB.

The embedding model is stubbed with a deterministic vector so no model download happens
in tests.
"""

import uuid

import pytest
import psycopg

from app import embed


def line(index, text, y0, y1=None, x0=0.1):
    return {"line_index": index, "text": text,
            "bbox": (x0, y0, 0.5, y0 + 0.01 if y1 is None else y1)}


def tight_lines(n):
    """n lines 0.01 apart vertically — one visual block, gaps well below any threshold."""
    return [line(i, f"line {i}", y0=0.1 + i * 0.02) for i in range(n)]


def test_side_by_side_lines_never_split():
    # value starts left of the label's bottom edge (same row): gap <= 0
    lines = [line(0, "Total:", 0.4, y1=0.42), line(1, "S$93.20", 0.395)]
    assert len(embed.chunk_page(lines)) == 1


def test_gap_boundary_starts_new_chunk():
    lines = tight_lines(3) + [line(3, "TOTAL", 0.5)]  # 0.07 gap vs 0.01 within-block
    groups = embed.chunk_page(lines)
    assert [[part["line_index"] for part in group] for group in groups] == [[0, 1, 2], [3]]


def test_cap_splits_even_without_gaps():
    groups = embed.chunk_page(tight_lines(embed.MAX_LINES + 3))
    assert all(len(group) <= embed.MAX_LINES for group in groups)
    assert sum(len(group) for group in groups) == embed.MAX_LINES + 3


def test_chunk_document_unions_bbox_and_text():
    chunks = embed.chunk_document({1: [line(0, "alpha", 0.1), line(1, "beta", 0.12)]})
    assert len(chunks) == 1
    assert chunks[0]["text"] == "alpha\nbeta"
    assert (chunks[0]["line_start"], chunks[0]["line_end"]) == (0, 1)
    assert chunks[0]["y1"] == 0.13


class FakeEmbedder:
    """Deterministic stand-in: text -> 384-dim one-hot vector keyed by character sum."""

    def embed(self, texts):
        for text in texts:
            vec = [0.0] * 384
            vec[sum(ord(c) for c in text) % 384] = 1.0
            yield vec


def insert_document(conn, doc_id, document_type, lines, country_code="SG"):
    """documents + one page + its ocr_lines rows; `lines` are line() dicts."""
    conn.execute(
        """INSERT INTO documents (id, content_hash, storage_uri, status, document_type,
               country_code) VALUES (%s, %s, %s, 'loaded', %s, %s)""",
        (doc_id, f"sha256:{doc_id}", f"/raw/{doc_id}", document_type, country_code),
    )
    conn.execute(
        """INSERT INTO document_pages (document_id, page_number, width, height,
               page_size_known, engine) VALUES (%s, 1, 1.0, 1.0, true, 'tesseract')""",
        (doc_id,),
    )
    with conn.cursor() as cur:
        cur.executemany(
            """INSERT INTO ocr_lines (document_id, page_number, line_index, text,
                   x0, y0, x1, y1) VALUES (%s, 1, %s, %s, %s, %s, %s, %s)""",
            [(doc_id, ln["line_index"], ln["text"], *ln["bbox"]) for ln in lines],
        )


@pytest.fixture(scope="module")
def seeded_db(db_url):
    """One document, one page, 3 lines in 2 blocks; model stubbed for the whole module."""
    original, embed.embedder = embed.embedder, (lambda model=None: FakeEmbedder())
    conn = psycopg.connect(db_url)
    conn.autocommit = True
    doc_id = "11111111-1111-1111-1111-111111111111"
    insert_document(conn, doc_id, "receipt", tight_lines(2) + [line(2, "TOTAL S$93.20", 0.5)])
    yield conn, uuid.UUID(doc_id)
    conn.close()
    embed.embedder = original


def test_embed_is_idempotent_per_model_and_flips_status(seeded_db):
    conn, doc_id = seeded_db
    assert embed.embed_document(conn, doc_id) == 2
    assert embed.embed_document(conn, doc_id) == 2  # same model: replace, not duplicate
    count = conn.execute("SELECT count(*) FROM chunks WHERE document_id = %s",
                         (doc_id,)).fetchone()[0]
    assert count == 2
    status = conn.execute("SELECT status FROM documents WHERE id = %s", (doc_id,)).fetchone()[0]
    assert status == "embedded"


def test_two_models_coexist(seeded_db):
    conn, doc_id = seeded_db
    embed.embed_document(conn, doc_id, model="other-model")
    models = conn.execute(
        "SELECT model, count(*) FROM chunks WHERE document_id = %s GROUP BY model ORDER BY model",
        (doc_id,)).fetchall()
    assert models == [(embed.MODEL, 2), ("other-model", 2)]  # ORDER BY model: 'B' < 'o'


def test_search_nearest_and_model_filter(seeded_db):
    conn, doc_id = seeded_db
    probe = list(FakeEmbedder().embed(["TOTAL S$93.20"]))[0]
    rows = embed.search(conn, probe, top_k=5)
    assert {str(r[1]) for r in rows} == {str(doc_id)}          # one document's chunks only
    assert rows[0][5].endswith("TOTAL S$93.20")                # exact vector ranks first
    assert all(r[10] == embed.MODEL for r in rows)             # other model filtered out


def test_search_metadata_filters(seeded_db):
    conn, doc_id = seeded_db
    probe = list(FakeEmbedder().embed(["TOTAL S$93.20"]))[0]
    assert embed.search(conn, probe, document_type="receipt")[0][2] == "receipt"
    assert embed.search(conn, probe, document_type="referral_letter") == []
    assert embed.search(conn, probe, country_code="VN") == []


def seed_document(conn, doc_id, document_type, model, texts):
    """One chunk per line (0.04 gaps clear the split threshold), embedded under `model`."""
    insert_document(conn, doc_id, document_type,
                    [line(i, text, y0=0.1 + i * 0.05) for i, text in enumerate(texts)])
    embed.embed_document(conn, doc_id, model=model)


def test_filtered_search_does_not_shrink_the_result_set(seeded_db):
    """The contract expects top_k *matching* chunks. HNSW only scans the unfiltered
    neighbourhood, so search() sets hnsw.iterative_scan whenever a filter is present;
    without it a selective filter silently returns fewer hits than exist."""
    conn, _ = seeded_db
    model = "filter-model"
    referral, receipt = uuid.uuid4(), uuid.uuid4()
    try:
        seed_document(conn, referral, "referral_letter", model, [f"referral {i}" for i in range(4)])
        seed_document(conn, receipt, "receipt", model, [f"total {i}" for i in range(2)])
        probe = list(FakeEmbedder().embed(["referral 0"]))[0]

        hits = embed.search(conn, probe, model=model, document_type="referral_letter", top_k=10)
        assert len(hits) == 4                                       # every matching chunk
        assert {str(row[1]) for row in hits} == {str(referral)}

        # same model without the filter sees both documents: the filter is what shrank it
        assert len(embed.search(conn, probe, model=model, top_k=10)) == 6
    finally:
        conn.execute("DELETE FROM documents WHERE id = ANY(%s)", ([referral, receipt],))
