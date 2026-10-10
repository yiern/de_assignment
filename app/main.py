import os
import re
import uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, Form, Request, Response, UploadFile
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from psycopg.rows import dict_row
from psycopg_pool import ConnectionPool

from . import embed
from .db import DATABASE_URL, migrate
from .pipeline import PipelineError, document_id, process

MAX_UPLOAD_BYTES = int(os.environ.get("MAX_UPLOAD_BYTES", 10 * 1024 * 1024))

pool: ConnectionPool | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global pool
    pool = ConnectionPool(DATABASE_URL, open=True)
    with pool.connection() as conn:
        migrate(conn)  # pgvector extension and schema come from db/migrations
    yield
    pool.close()


app = FastAPI(title="OCR Ingestion Pipeline", lifespan=lifespan)


@app.exception_handler(RequestValidationError)
async def invalid_request(request: Request, exc: RequestValidationError):
    return JSONResponse(status_code=422, content={"error": "invalid_request"})


@app.exception_handler(Exception)
async def unhandled(request: Request, exc: Exception):
    # body is fixed: never echo exception text, it can carry document content
    return JSONResponse(status_code=500, content={"error": "internal_server_error"})


# The API boundary masks the same identifier pattern the ocr_lines_safe
# view masks in SQL. Search results need the chunk text for context but
# never need direct identifiers. Keep both regexes in sync; tests pin each surface.
_NRIC_RE = re.compile(r"[STFG][0-9]{7}[A-Z]")


def mask_nric(text: str) -> str:
    return _NRIC_RE.sub("X0000000X", text)


@app.get("/health")
def health():
    with pool.connection() as conn:
        ver = conn.execute(
            "SELECT version(), extversion FROM pg_extension WHERE extname = 'vector'"
        ).fetchone()
    return {"status": "ok", "postgres": ver[0].split()[1], "pgvector": ver[1]}


@app.post("/documents")
def create_document(
    response: Response,
    file: UploadFile | None = File(None),
    country_code: str | None = Form(None),
):
    if file is None:
        return JSONResponse(status_code=400, content={"error": "file_missing"})
    data = file.file.read(MAX_UPLOAD_BYTES + 1)
    if len(data) > MAX_UPLOAD_BYTES:
        return JSONResponse(status_code=413, content={"error": "file_too_large"})
    try:
        with pool.connection() as conn:
            result = process(conn, data, file.filename, file.content_type, country_code)
    except PipelineError as exc:  # 422s are already quarantined by process()
        body = {"error": exc.code}
        if exc.status_code == 422:  # GET /documents/{document_id} shows the reason
            body |= {"document_id": document_id(exc.content_hash),
                     "content_hash": f"sha256:{exc.content_hash}"}
        return JSONResponse(status_code=exc.status_code, content=body)
    response.status_code = 200 if result["duplicate"] else 201
    return result


@app.get("/documents/{doc_id}")
def get_document(doc_id: str):
    """Quality visibility via API: status, status_reason, quality flags and field verdicts."""
    not_found = JSONResponse(status_code=404, content={"error": "not_found"})
    try:
        doc_uuid = uuid.UUID(doc_id)
    except ValueError:
        return not_found
    with pool.connection() as conn, conn.cursor(row_factory=dict_row) as cur:
        doc = cur.execute(
            """SELECT id AS document_id, status, status_reason, quality_flags, document_type,
                      country_code, received_at, normalised_at, loaded_at, embedded_at
               FROM documents WHERE id = %s""", (doc_uuid,),
        ).fetchone()
        if doc is None:
            return not_found
        fields = cur.execute(
            """SELECT field_name, raw_value, value, validation_status
               FROM extracted_fields WHERE document_id = %s ORDER BY field_name""",
            (doc_uuid,),
        ).fetchall()
    return {**doc, "document_id": str(doc["document_id"]), "fields": fields}


class SearchBody(BaseModel):
    query: str
    top_k: int = Field(default=5, ge=1, le=50)
    filters: dict[str, str] = Field(default_factory=dict)  # document_type, country_code
    model: str = embed.MODEL  # which embedding model's chunks to search


SEARCH_FILTERS = {"document_type", "country_code"}


@app.post("/search")
def search(body: SearchBody):
    # Contract response. bbox is the chunk's union bbox, normalised ratios 0-1, origin top-left.
    if not body.query.strip():
        return JSONResponse(status_code=400, content={"error": "empty_query"})
    unknown = sorted(set(body.filters) - SEARCH_FILTERS)
    if unknown:
        # A typo'd key would otherwise be ignored and the query answered over the whole
        # corpus: a wrong answer that looks like a right one.
        return JSONResponse(status_code=400,
                            content={"error": "unknown_filter", "parameter": unknown[0]})
    query_vec = [float(x) for x in next(embed.embedder(body.model).embed([body.query]))]
    with pool.connection() as conn:
        rows = embed.search(
            conn, query_vec, model=body.model,
            document_type=body.filters.get("document_type"),
            country_code=body.filters.get("country_code"),
            top_k=body.top_k,
        )
    return {"results": [
        {"chunk_id": str(r[0]), "document_id": str(r[1]), "document_type": r[2],
         "page": r[3], "score": round(float(r[4]), 4), "text": mask_nric(r[5]),
         "bbox": [r[6], r[7], r[8], r[9]], "embedding_model": r[10]}
        for r in rows
    ]}
