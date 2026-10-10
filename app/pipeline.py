"""
upload -> raw zone -> normalise -> normalised zone -> relational load -> chunk/embed.

Each stage stays independently re-runnable: the raw zone keeps the original bytes, the
normalised zone keeps the parsed output, and the database keeps the OCR lines. Stages commit
separately, so a failure after load leaves the document at status 'loaded' (re-runnable via
`reembed`) rather than losing it. Failures are quarantined, never crashed on. PDF/image
uploads are OCRed locally (`app/ocr.py`) and enter the pipeline as synthetic 'rapidocr'
engine exports.
"""

import hashlib
import json
import logging
import os
import re
import time
import uuid
from contextlib import contextmanager
from pathlib import Path

import psycopg

from . import embed, ocr
from .db import new_run
from .normalise import NormaliseError, normalise
from .quality import extract_fields, get_page_flags

log = logging.getLogger(__name__)

# Zone locations on disk (compose mounts the data volume at /data). Tests rebind the two
# constants directly via monkeypatch.setattr.
DATA_DIR = Path(os.environ.get("DATA_DIR", "/data"))
RAW_ZONE = DATA_DIR / "raw"
NORMALISED_ZONE = DATA_DIR / "normalised"

# The contract's 400 for a non-JSON upload; anything else that fails to parse is 422.
JSON_MEDIA_TYPES = {"application/json", "text/json", "application/octet-stream", ""}

# Bonus path: PDF/images are OCRed locally (app/ocr.py). Exact-set membership: anything
# else (.txt, video/...) stays a 400 invalid_mime_type.
OCR_MEDIA_TYPES = {"application/pdf", "image/png", "image/jpeg", "image/tiff",
                   "image/bmp", "image/webp"}

# Markers are stored space-free: RapidOCR frequently drops inter-word spaces
# ("Total Amount Paid:" -> "TotalAmountPaid:"), so classify() squashes text the same way.
_TYPE_MARKERS = {
    "medical_certificate": ("MEDICALCERTIFICATE",),
    "referral_letter": ("REFERRALLETTER",),
    "receipt": ("TAXINVOICE", "OFFICIALRECEIPT", "RECEIPT"),
}


class PipelineError(Exception):
    """`code` is the API error body, `status_code` the HTTP status for it.

    `content_hash` is set once the bytes are known, so the caller can quarantine them.
    """

    def __init__(self, code, status_code, content_hash=None):
        super().__init__(code)
        self.code = code
        self.status_code = status_code
        self.content_hash = content_hash


def classify(document):
    """Placeholder document-type classifier: marker text anywhere on the page."""
    text = re.sub(
        r"\s+", "",
        " ".join(line["text"] for page in document["pages"] for line in page["lines"]),
    ).upper()
    for document_type, markers in _TYPE_MARKERS.items():
        if any(marker in text for marker in markers):
            return document_type
    return None


def document_id(content_hash: str) -> str:
    """Deterministic id: same bytes -> same id, so re-ingest cannot fork the document."""
    return str(uuid.uuid5(uuid.NAMESPACE_URL, f"sha256:{content_hash}"))


def existing_status(conn, content_hash: str) -> str | None:
    """Pipeline status for these exact bytes, or None if they never loaded. A rejected
    quarantine placeholder counts as never loaded. One definition of "already ingested?"
    shared by the load stage's guard and the pipeline's resume point."""
    row = conn.execute(
        "SELECT status FROM documents WHERE content_hash = %s AND status <> 'rejected'",
        (f"sha256:{content_hash}",),
    ).fetchone()
    return row[0] if row else None


def resolve_country(form_value, envelope_value):
    """Form field, then the envelope, then SG. A supplied-but-invalid value (not two letters)
    is skipped and flagged rather than rejected. Returns (country_code, flags)."""
    flags = []
    for origin, value in (("form", form_value), ("envelope", envelope_value)):
        if value is None or str(value).strip() == "":
            continue  # not supplied
        code = str(value).strip().upper()
        if re.fullmatch(r"[A-Z]{2}", code):
            return code, flags
        flags.append(f"country_code_invalid:{origin}")
    return "SG", flags


def check_media_type(content_type):
    media_type = (content_type or "").split(";")[0].strip().lower()
    if media_type not in JSON_MEDIA_TYPES | OCR_MEDIA_TYPES:
        raise PipelineError("invalid_mime_type", 400)


def _write_atomic(path: Path, data: bytes):
    """Write-then-rename so a crash never leaves a truncated file that looks complete."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    tmp.write_bytes(data)
    tmp.replace(path)


def land_raw(data: bytes, content_hash: str) -> None:
    """Raw zone: persist the uploaded bytes untouched, named by their SHA-256."""
    raw_path = RAW_ZONE / content_hash
    if not raw_path.exists():
        _write_atomic(raw_path, data)  # byte for byte; the file name IS the content hash


def read_envelope(data: bytes, content_hash: str, filename=None) -> dict:
    """Uploaded bytes -> OCR engine envelope: parse JSON, or OCR a PDF/image (bonus path)."""
    mime = ocr.sniff(data)
    if mime:  # PDF/image -> local OCR -> synthetic engine export
        try:
            envelope = ocr.build_envelope(data, filename=filename, mime_type=mime)
        except ocr.DocumentTooLarge as exc:
            raise PipelineError("file_too_large", 413, content_hash) from exc
        except ocr.UnreadableDocument as exc:
            raise PipelineError("unreadable_file", 422, content_hash) from exc
        # keep the synthetic export next to the raw bytes: re-normalisation from the
        # raw zone stays a pure normalise() call (no re-OCR needed)
        _write_atomic(
            RAW_ZONE / f"{content_hash}.ocr.json",
            json.dumps(envelope, indent=2).encode("utf-8"),
        )
    else:
        try:
            envelope = json.loads(data)
        except (UnicodeDecodeError, json.JSONDecodeError, RecursionError) as exc:
            raise PipelineError("unreadable_file", 422, content_hash) from exc
    return envelope


def normalise_to_zone(envelope: dict, content_hash: str, country_code=None) -> Path:
    """Normalise an envelope into the canonical form and write it to the normalised zone."""
    try:
        document = normalise(envelope, content_sha256=content_hash)
    except NormaliseError as exc:
        # A recognised-but-broken engine payload is unreadable_file, anything else is
        # unsupported_ocr_format; the code comes from the error class.
        raise PipelineError(exc.code, 422, content_hash) from exc

    document["source"]["country_code"], document["ingest_flags"] = resolve_country(
        country_code, document["source"].get("country_code")
    )
    normalised_path = NORMALISED_ZONE / f"{content_hash}.normalised.json"
    _write_atomic(normalised_path, json.dumps(document, indent=2).encode("utf-8"))
    return normalised_path


# ---------------------------------------------------------------- relational load


def _insert_document(conn, doc, doc_id, doc_type, flags, run_id):
    source = doc["source"]
    raw_path = RAW_ZONE / source["content_sha256"]
    try:
        size = raw_path.stat().st_size
    except OSError:
        size = None  # raw zone not mounted; keep loading, size stays NULL
    # a rejected quarantine placeholder (same bytes, once 422'd) is replaced
    conn.execute("DELETE FROM documents WHERE id = %s", (doc_id,))
    conn.execute(
        """INSERT INTO documents (id, content_hash, source_sha256, original_filename,
               mime_type, size_bytes, storage_uri, document_type, country_code, status,
               quality_flags, run_id, normalised_at, normaliser_version, loaded_at)
           VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, 'loaded', %s, %s, %s, %s, now())""",
        (doc_id, f"sha256:{source['content_sha256']}", source.get("source_sha256"),
         source.get("original_filename"), source.get("mime_type"), size, str(raw_path),
         doc_type, source.get("country_code", "SG"), json.dumps(flags), run_id,
         doc.get("normalised_at"), doc.get("normaliser_version")),
    )


def _insert_page(conn, doc, doc_id, page):
    native = page.get("native_size") or {}
    conn.execute(
        """INSERT INTO document_pages (document_id, page_number, width, height,
               native_width, native_height, native_unit, page_size_known,
               page_size_source, engine, engine_version, processed_at, mean_confidence)
           VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)""",
        (doc_id, page["page_number"], page["size"]["width"], page["size"]["height"],
         native.get("width"), native.get("height"), native.get("unit"),
         page["page_size_known"], page.get("page_size_source"), doc["ocr"]["engine"],
         doc["ocr"].get("engine_version"), doc["ocr"].get("processed_at"),
         page.get("mean_confidence")),
    )
    with conn.cursor() as cur:
        cur.executemany(
            """INSERT INTO ocr_lines (document_id, page_number, line_index, text,
                   x0, y0, x1, y1, confidence, confidence_source)
               VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)""",
            [(doc_id, page["page_number"], ln["line_index"], ln["text"],
              *ln["bbox"], ln.get("confidence"), ln.get("confidence_source"))
             for ln in page["lines"]],
        )


def _insert_fields(conn, doc_id, fields, run_id):
    with conn.cursor() as cur:
        cur.executemany(
            """INSERT INTO extracted_fields (document_id, field_name, raw_value, value,
                   validation_status, run_id) VALUES (%s, %s, %s, %s, %s, %s)""",
            [(doc_id, f["field_name"], f["raw_value"], f["value"], f["validation_status"],
              run_id) for f in fields],
        )


def load_document(conn: psycopg.Connection, path: Path, run_id: str) -> bool:
    """Load one normalised file. Returns False when the document already exists."""
    doc = json.loads(path.read_text(encoding="utf-8"))
    content_hash = doc["source"]["content_sha256"]
    if existing_status(conn, content_hash):
        return False

    doc_type = classify(doc)
    if doc_type is None:
        raise PipelineError("unsupported_document_type", 422, content_hash)
    by_page = {page["page_number"]: page["lines"] for page in doc["pages"]}
    flags = doc.get("ingest_flags", []) + get_page_flags(by_page)
    fields = extract_fields(by_page, doc_type)

    doc_id = document_id(content_hash)
    with conn.transaction():  # the document lands completely or not at all
        _insert_document(conn, doc, doc_id, doc_type, flags, run_id)
        for page in doc["pages"]:
            _insert_page(conn, doc, doc_id, page)
        _insert_fields(conn, doc_id, fields, run_id)
    return True


def load_all(conn: psycopg.Connection, run_id: str | None = None) -> tuple[int, int, int]:
    """Load every normalised file; returns (loaded, skipped, failed). One bad file never
    stops the batch: each document commits or rolls back on its own."""
    run_id = run_id or new_run(conn)
    loaded = skipped = failed = 0
    for path in sorted(NORMALISED_ZONE.glob("*.normalised.json")):
        try:
            if load_document(conn, path, run_id):
                loaded += 1
            else:
                skipped += 1
            conn.commit()
        except Exception as exc:  # message omitted: it may carry document content
            conn.rollback()
            failed += 1
            log.error("load failed for %s: %s", path.name, type(exc).__name__)
    return loaded, skipped, failed


# ------------------------------------------------------------------------ orchestration


def quarantine(conn, content_hash: str, filename, reason: str):
    """Record a failed file as a 'rejected' document (the quarantine view reads these).

    An existing non-rejected row is never touched; a later successful load replaces this one.
    """
    try:
        conn.execute(
            """INSERT INTO documents (id, content_hash, original_filename, storage_uri,
                   status, status_reason) VALUES (%s, %s, %s, %s, 'rejected', %s)
               ON CONFLICT (content_hash) DO UPDATE SET status_reason = EXCLUDED.status_reason
               WHERE documents.status = 'rejected'""",
            (document_id(content_hash), f"sha256:{content_hash}", filename,
             str(RAW_ZONE / content_hash), reason),
        )
        conn.commit()
    except Exception as exc:  # recording must never mask the original failure
        conn.rollback()
        log.error("quarantine recording failed: %s", type(exc).__name__)


@contextmanager
def _timed(timings: dict, stage: str):
    start = time.perf_counter()
    yield
    timings[stage] = round((time.perf_counter() - start) * 1000)


def _result(conn, content_hash: str, model: str, duplicate: bool, timings: dict) -> dict:
    doc_id, doc_type, country, status, flags, pages, chunks = conn.execute(
        """SELECT d.id, d.document_type, d.country_code, d.status, d.quality_flags,
                  (SELECT count(*) FROM document_pages WHERE document_id = d.id),
                  (SELECT count(*) FROM chunks WHERE document_id = d.id AND model = %s)
           FROM documents d WHERE d.content_hash = %s""",
        (model, f"sha256:{content_hash}"),
    ).fetchone()
    return {
        "document_id": str(doc_id), "content_hash": f"sha256:{content_hash}",
        "duplicate": duplicate, "document_type": doc_type, "status": status,
        "pages": pages, "chunks": chunks, "quality_flags": flags,
        "country_code": country, "timings_ms": timings,
    }

# Entry point function
def process(conn, data: bytes, filename=None, content_type=None, country_code=None,run_id=None, model=None) -> dict:
    """Ingest one uploaded file end to end. Raises PipelineError for rejected files."""
    check_media_type(content_type)  # 400: nothing stored, nothing to quarantine
    model = model or embed.MODEL
    content_hash = hashlib.sha256(data).hexdigest()

    # Idempotency check from SQL query if duplicated and done:marked as duplicate. If duplicate but failed embedding: retry
    status = existing_status(conn, content_hash)
    if status == "embedded":
        return _result(conn, content_hash, model, True, {})

    timings = {}
    try:
        run_id = run_id or new_run(conn, embedding_model=model) #create new Lineage record with code, normaliser and embedding model
        if status is None:
            with _timed(timings, "normalise"):
                land_raw(data, content_hash)
                envelope = read_envelope(data, content_hash, filename)
                normalised_path = normalise_to_zone(envelope, content_hash, country_code)
            with _timed(timings, "load"):
                load_document(conn, normalised_path, run_id)
                conn.commit()
        # else: an earlier upload loaded but never embedded; resume from the embed stage

        with _timed(timings, "embed"):
            embed.embed_document(conn, document_id(content_hash), model, run_id)
            conn.commit()
    except psycopg.errors.UniqueViolation:
        # a concurrent upload of the same bytes won the insert: it is the document
        conn.rollback()
        return _result(conn, content_hash, model, True, {})
    except PipelineError as exc:
        conn.rollback()
        quarantine(conn, content_hash, filename, exc.code)
        raise
    except Exception:
        conn.rollback()
        quarantine(conn, content_hash, filename, "internal_error")
        raise
    return _result(conn, content_hash, model, False, timings)
