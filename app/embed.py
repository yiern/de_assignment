"""Chunk + embed stage and the search query behind POST /search.

Chunking: consecutive OCR lines on a page are grouped at vertical gaps — a gap (y0 of the
next line minus y1 of the current) larger than 1.5x the page's median line height starts a
new chunk (blank space relative to text size is what marks a section); a line/character cap
splits pathologically dense runs. Side-by-side label/value lines share a row (gap <= 0) and
can never be split. Chunks are contiguous line ranges, so each carries its union bbox and
exact lineage into ocr_lines.

Embedding: fastembed (ONNX, CPU, no API key) with BAAI/bge-small-en-v1.5, 384 dimensions —
matching the chunks.embedding vector(384) typmod. Rows are keyed per model, so re-embedding
under a new model coexists with the old one; re-embedding under the same model
replaces it (delete + insert in one transaction).
"""

import importlib.metadata
import json
import logging
import statistics

from .db import new_run

log = logging.getLogger(__name__)

DIMENSION = 384
MODEL = "BAAI/bge-small-en-v1.5"
MAX_LINES = 8     # fallback cap: split dense runs that never hit a gap
MAX_CHARS = 500

_embedders: dict = {}


def embedder(model: str = MODEL):
    """Load each model once per process; imported lazily so tests can stub it."""
    if model not in _embedders:
        # import here, not at module top: importing app.embed must stay cheap and
        # dependency-free (tests stub the model instead of downloading it)
        from fastembed import TextEmbedding

        _embedders[model] = TextEmbedding(model)
    return _embedders[model]


def model_version() -> str | None:
    """fastembed exposes no per-model version; the library version pins the ONNX export."""
    try:
        return importlib.metadata.version("fastembed")
    except importlib.metadata.PackageNotFoundError:
        return None


def chunk_page(lines: list[dict]) -> list[list[dict]]:
    """Group one page's lines (sorted by reading order) into contiguous chunks."""
    # reading order: top-to-bottom, then left-to-right within a visual row
    lines = sorted(lines, key=lambda line: (line["bbox"][1], line["bbox"][0]))
    if not lines:
        return []
    # a section break is blank space noticeably taller than this page's own text
    threshold = 1.5 * statistics.median(ln["bbox"][3] - ln["bbox"][1] for ln in lines)
    groups, current = [], [lines[0]]
    for line in lines[1:]:
        # blank strip since the last line kept; negative means the same visual row
        gap = line["bbox"][1] - current[-1]["bbox"][3]
        chars = sum(len(part["text"]) for part in current)
        # break on a real gap, or when a gapless run hits the fallback caps
        if gap > threshold or len(current) >= MAX_LINES or chars + len(line["text"]) > MAX_CHARS:
            groups.append(current)
            current = []
        current.append(line)
    groups.append(current)  # the final group never sees a "next" gap to close it
    return groups


def chunk_document(lines_by_page: dict[int, list[dict]]) -> list[dict]:
    """Chunk rows ready for the chunks table (minus embedding)."""
    chunks = []
    # page-respecting: a chunk never spans two pages
    for page_number in sorted(lines_by_page):
        for group in chunk_page(lines_by_page[page_number]):
            chunks.append({
                "page_number": page_number,
                # lineage: the chunk IS this contiguous range of ocr_lines
                "line_start": group[0]["line_index"],
                "line_end": group[-1]["line_index"],
                # union bbox: the corner bracket around everything in the group
                "x0": min(line["bbox"][0] for line in group),
                "y0": min(line["bbox"][1] for line in group),
                "x1": max(line["bbox"][2] for line in group),
                "y1": max(line["bbox"][3] for line in group),
                "text": "\n".join(line["text"] for line in group),
            })
    return chunks


def embed_document(conn, document_id: str, model: str | None = None, run_id: str | None = None):
    """Chunk + embed one loaded document; idempotent per (document, model). Returns chunk count."""
    model = model or MODEL
    # this stage reads only from ocr_lines: it can re-run without the zone files
    rows = conn.execute(
        """SELECT page_number, line_index, text, x0, y0, x1, y1
           FROM ocr_lines WHERE document_id = %s
           ORDER BY page_number, line_index""",
        (document_id,),
    ).fetchall()
    by_page: dict[int, list[dict]] = {}
    for page, index, text, x0, y0, x1, y1 in rows:
        by_page.setdefault(page, []).append(
            {"line_index": index, "text": text, "bbox": (x0, y0, x1, y1)}
        )
    # Returns a dictionary per chunk. See function for full struct
    chunks = chunk_document(by_page)
    for index, chunk in enumerate(chunks):
        chunk["chunk_index"] = index

    # vectors line up with chunks by position: the model preserves input order.
    vectors = (
        [[float(x) for x in vec] for vec in embedder(model).embed([c["text"] for c in chunks])]
        if chunks
        else []
    )
    if any(len(vec) != DIMENSION for vec in vectors):
        raise ValueError(
            f"model {model!r} does not produce {DIMENSION}-d vectors; ALTER chunks.embedding "
            "to the new dimension before embedding with it"
        )
    run_id = run_id or new_run(conn, embedding_model=model)
    version = model_version()
    # crash mid-way leaves the previous embedding set intact
    with conn.transaction():
        # Same model -> replace; another model -> coexist (UNIQUE(document_id, chunk_index, model)).
        conn.execute("DELETE FROM chunks WHERE document_id = %s AND model = %s",
                     (document_id, model))
        with conn.cursor() as cur:
            cur.executemany(
                """INSERT INTO chunks (document_id, chunk_index, page_number, line_start,
                       line_end, x0, y0, x1, y1, text, model, model_version, dimension,
                       embedding, run_id)
                   VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s::vector, %s)""",
                [(document_id, chunk["chunk_index"], chunk["page_number"],
                  chunk["line_start"], chunk["line_end"],
                  chunk["x0"], chunk["y0"], chunk["x1"], chunk["y1"],
                  chunk["text"], model, version, len(vector), json.dumps(vector), run_id)
                 for chunk, vector in zip(chunks, vectors)],
            )
        conn.execute(
            "UPDATE documents SET status = 'embedded', embedded_at = now() WHERE id = %s",
            (document_id,),
        )
    return len(chunks)


def embed_loaded(conn, model: str | None = None, run_id: str | None = None) -> tuple[int, int]:
    """Embed every loaded/embedded document (re-embed stage); returns (embedded, failed).

    Each document commits or rolls back on its own, so one failure never stops the batch.
    """
    model = model or MODEL
    # one run row shared by the whole batch: these chunks were produced together
    run_id = run_id or new_run(conn, embedding_model=model)
    conn.commit()
    # includes 'embedded' documents: re-embedding under --model is the migration path
    docs = conn.execute(
        "SELECT id FROM documents WHERE status IN ('loaded', 'embedded') ORDER BY received_at"
    ).fetchall()
    done = failed = 0
    for (doc_id,) in docs:
        try:
            embed_document(conn, doc_id, model, run_id)
            conn.commit()
            done += 1
        except Exception as exc:  # message omitted: it may carry document content
            conn.rollback()
            failed += 1
            log.error("embed failed for %s: %s", doc_id, type(exc).__name__)
    return done, failed


def search(conn, query_vector: list[float], model: str | None = None,
           document_type: str | None = None, country_code: str | None = None,
           top_k: int = 5) -> list[tuple]:
    """Search: nearest chunks under cosine, filtered by document metadata."""
    vec = json.dumps(query_vector)
    # fixed core; optional filters are appended as AND fragments with params in
    # lockstep (placeholders only, never string-interpolated values)
    sql = """SELECT c.id, c.document_id, d.document_type, c.page_number,
                    1 - (c.embedding <=> %s::vector) AS score, c.text,
                    c.x0, c.y0, c.x1, c.y1, c.model
             FROM chunks c JOIN documents d ON d.id = c.document_id
             WHERE c.model = %s"""
    params: list = [vec, model or MODEL]
    if document_type:
        sql += " AND d.document_type = %s"
        params.append(document_type)
    if country_code:
        sql += " AND d.country_code = %s"
        params.append(country_code)
    sql += " ORDER BY c.embedding <=> %s::vector LIMIT %s"
    params += [vec, top_k]
    # 1 - distance: pgvector's <=> is cosine DISTANCE, the contract wants similarity
    #
    # The HNSW scan runs before the filter, so a filtered query can come back with fewer
    # than top_k chunks while more matches exist. iterative_scan keeps walking until top_k
    # survive the filter. SET LOCAL inside a transaction, so the setting never leaks onto
    # the pooled connection. Ceiling: hnsw.max_scan_tuples (20000 by default) can still stop
    # the walk early on a very selective filter - partial indexes per type/country then.
    if document_type or country_code:
        with conn.transaction():
            conn.execute("SET LOCAL hnsw.iterative_scan = relaxed_order")
            return conn.execute(sql, params).fetchall()
    return conn.execute(sql, params).fetchall()
