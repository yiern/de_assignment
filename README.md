# OCR Document Ingestion & Embedding Pipeline

Ingests raw OCR engine exports of medical documents, normalises them into one engine-agnostic
shape, and lands them as structured relational records plus vector embeddings for semantic search.

Stack: FastAPI · PostgreSQL 16 + pgvector · Docker Compose · Python 3.12.

## Setup and run

```bash
docker compose up -d --build          # Postgres + pgvector and the API on :8000
curl localhost:8000/health            # {"status":"ok","postgres":"16.15","pgvector":"0.8.7"}
scripts/seed.sh                       # ingest the three sample documents (idempotent)
```

Useful lifecycle commands:

```bash
docker compose down                   # stop; add -v to also wipe pgdata and the data zones
docker compose exec api python -m app.db                    # apply pending migrations by hand
docker compose exec api python -m pytest tests -q           # run tests
```

## API

| Method | Path | Purpose |
|---|---|---|
| `POST` | `/documents` | ingest one file: raw zone → normalise → load → embed |
| `GET` | `/documents/{id}` | status, quality flags, extracted fields |
| `POST` | `/search` | semantic search over chunks |




### `POST /documents`

`multipart/form-data` with `file` (required) and `country_code` (optional). Accepts the three OCR
export JSON formats plus `application/pdf` and `image/png|jpeg|tiff|bmp|webp` (local OCR).

```bash
# new document -> 201
curl -i -F "file=@path/to/your_export.json" localhost:8000/documents

# country override -> resolved as VN (form field, then the envelope, then SG)
curl -F "file=@path/to/your_export.json" -F "country_code=VN" localhost:8000/documents

# PDF/image -> OCRed locally, then the same pipeline
curl -F "file=@scan.pdf" -F "country_code=SG" localhost:8000/documents

# same bytes again (any filename) -> 200 with "duplicate": true
curl -i -F "file=@path/to/your_export.json" localhost:8000/documents
```

Response (`201` new / `200` duplicate):

```json
{
  "document_id": "4efa1082-aedc-521d-82b2-e313f17bad1e",
  "content_hash": "sha256:00fce3375b92d0d0f581ed923d5288bd0d9b697cfe677b034f9ae08d4bb6272e",
  "duplicate": false,
  "document_type": "receipt",
  "status": "embedded",
  "pages": 1,
  "chunks": 9,
  "quality_flags": [],
  "country_code": "SG",
  "timings_ms": { "normalise": 3, "load": 21, "embed": 140 }
}
```

| Status | Condition |
|---|---|
| `201` | new document (or a resumed one), now `embedded` |
| `200` | duplicate — same bytes already ingested, nothing written |
| `400` | `{"error": "file_missing"}` or `{"error": "invalid_mime_type"}` |
| `413` | `{"error": "file_too_large"}` (over 10 MB, or a PDF over 50 pages) |
| `422` | `unreadable_file`, `unsupported_ocr_format`, or `unsupported_document_type` |
| `500` | `{"error": "internal_server_error"}` (unhandled) |

### `GET /documents/{id}`

```bash
curl -s localhost:8000/documents/4efa1082-aedc-521d-82b2-e313f17bad1e
```

Returns `document_id`, `status`, `status_reason`, `quality_flags`, `document_type`,
`country_code`, stage timestamps, and `fields[]` (`field_name`, `raw_value`, `value`,
`validation_status`). Unknown or malformed id → `404 {"error":"not_found"}`.

### `POST /search`

```bash
curl -s localhost:8000/search -H 'content-type: application/json' -d '{
  "query": "medical leave for acute gastroenteritis",
  "top_k": 5,
  "filters": {"document_type": "medical_certificate", "country_code": "SG"}
}'
```

Body: `query` (required), `top_k` (default 5, 1–50), `filters` (optional, combinable:
`document_type`, `country_code`), `model` (optional, defaults to the current embedding model).
Unknown filter keys → `400`; empty query → `400`. No match → `200` with `"results": []`.
Each result carries `chunk_id`, `document_id`, `document_type`, `page`, `score` (cosine
similarity), `text`, `embedding_model`, and `bbox` — `[x0, y0, x1, y1]` in normalised ratios
0–1 of page width/height, origin top-left.

## CLI

```bash
docker compose cp ./my_exports api:/data/inbox && \
  docker compose exec api python -m app.cli ingest /data/inbox   # full pipeline over a folder
docker compose exec api python -m app.cli load             # rebuild relational tables from the normalised zone
docker compose exec api python -m app.cli reembed          # chunk + embed all loaded/embedded docs
docker compose exec api python -m app.cli reembed --model <name>   # re-embed under another model
docker compose exec api python -m app.cli search "medical leave" \
  -k 5 --type medical_certificate --country SG             # semantic search over embedded chunks
```

All are idempotent: `ingest`/`load` skip documents already ingested/loaded; `reembed` replaces the
same model's chunks and coexists with other models'.

## How to add a new document type


**1. Classify it — `app/pipeline.py`**

```python
_TYPE_MARKERS = {
    "medical_certificate": ("MEDICALCERTIFICATE",),
    "referral_letter":     ("REFERRALLETTER",),
    "receipt":             ("TAXINVOICE", "OFFICIALRECEIPT", "RECEIPT"),
    "discharge_summary":   ("DISCHARGESUMMARY",),      # <-- add
}
```


**2. Allow the value in the schema — new migration `db/migrations/002_discharge_summary.sql`**

```sql
ALTER TABLE documents DROP CONSTRAINT documents_document_type_check;
ALTER TABLE documents ADD CONSTRAINT documents_document_type_check
    CHECK (document_type IN ('medical_certificate', 'referral_letter', 'receipt',
                             'discharge_summary'));
```

Migrations apply automatically at API startup, or by hand:
`docker compose exec api python -m app.db`.

**3. Extract and validate its fields — `app/quality.py`**

```python
FIELD_RULES["discharge_summary"] = [
    Rule("claimant_name",       "Patient Name:",        "text"),
    Rule("discharge_date_time", "Discharge Date/Time:", "datetime"),
    Rule("icd_code",            "ICD-10 Code:",         "text"),
    Rule("provider_name",       "letterhead",           "provider"),
]
```

How a `Rule` is matched:

- `anchor` — a string (matched case-insensitively at the start of a line), a regex, or
  `"letterhead"` (the first line of page 1). The value is the text after `:` on the anchor
  line, or the line below it when `next_line=True`.
- `kind` — the validator: `text`, `date` (DD/MM/YYYY), `datetime`, `amount`
  (minor-unit integer: `S$93.20` → `9320`), `mc_days`, `provider` (must not contain
  "Fullerton Health"), `signature`. A new value grammar is one function plus an entry in
  `_PARSERS`.
- A label that never matches becomes a `missing` row; an unparsable value becomes an `invalid`
  row with a NULL value. Rows are never dropped.

**4. Test the layout — `tests/test_quality.py`**

Add a fixture page carrying your labels and assert the extracted `value`/`validation_status`
per field.

**5. Verify**

```bash
docker compose up -d --build        # rebuild so the container runs your code
docker compose cp ./discharge_samples api:/data/inbox
docker compose exec api python -m app.cli ingest /data/inbox
docker compose exec api python -m app.cli search "discharge" --type discharge_summary
```

`GET /documents/{id}` `fields[]` and the search filter accept the new type with no further
change.


## How to add a new OCR engine format


**1. Write the parser — `app/normalise.py`**

A parser is a function `parse_<engine>(raw)` taking the envelope's `raw_output` and returning
the canonical page list. The envelope around it is engine-independent: `normalise()` already
checks `ocr`/`source`, dispatches on `ocr.engine`, and maps parser exceptions to 422s — the
parser only decides what counts as *its* engine's `raw_output`.

**What the export needs to have (input):**

- a page identifier per page (the page number or equivalent)
- each page's width and height **in the same units as its line coordinates** — pixels,
  inches or already-normalised ratios all work; any unit is fine
- at least one OCR line somewhere in the export (an export with zero lines is rejected)
- per line: the recognised text and a positional box — four corner points or x0/y0/x1/y1;
  a line without geometry is dropped, not failed
- per line, optionally: a confidence value on the engine's own scale (0–100 or 0–1), or none

**What the parser must output :**


- one `page(page_number, lines, native_size=(w, h), unit=…, size_source=…)` per export page.
  
  `native_size` keeps the engine's own units (`page_size_known` is false if the engine
  publishes no dimensions), `size_source` records where the dimensions came from
- one `line(index, text, bbox, confidence, source)` per OCR line, in reading order
  
  `bbox` from `bbox(...)`: `[x0, y0, x1, y1]` in ratios 0–1, origin top-left — divide every
  coordinate by **that page's own** width/height, so pixels or inches cancel and no
  cross-engine conversion ever happens
   
  `confidence` from `confidence(value, scale)`: 0–1, or `None` where the engine gives none;
  
  `source=` records whether the value is engine-reported or derived
- errors: raise `UnsupportedOcrFormat` when `raw_output` isn't this engine's shape at all,
  `MalformedExport` when it is but is broken (missing page size, wrong types). Both become
  422 rejections; neither crashes the batch.

`parse_rapidocr` in the same file is the smallest complete example.

**2. Register it: one line in `PARSERS`**

Add `PARSERS["<engine-string>"] = parse_<engine>`, where the key matches the export
envelope's `ocr.engine` string exactly. Dispatch, the unknown-engine rejection and the
wrong-payload rejection are all handled by `normalise()` — no other stage knows engines exist.

**3. Tests `tests/test_normalise.py`**

Feed a small sample export and assert line counts and text, bbox conversion, confidence handling
and page-size availability, like the existing per-engine tests. Bump `NORMALISER_VERSION` only if
the change would alter the normalised output of documents that already exist.

**4. Try it**

```bash
docker compose exec api python -m pytest tests/test_normalise.py -q
docker compose cp ./new_engine_samples api:/data/inbox
docker compose exec api python -m app.cli ingest /data/inbox
```

Nothing downstream changes: classify, load, chunking and search read only the canonical shape.

## How to switch the embedding model and re-embed

Switching from `BAAI/bge-small-en-v1.5` to `sentence-transformers/all-MiniLM-L6-v2`
(both 384d, so no migration needed):

```bash
# 1. re-embed every loaded/embedded document under the new model
docker compose exec api python -m app.cli reembed --model sentence-transformers/all-MiniLM-L6-v2

# 2. compare scores against the old model; output includes the model per hit
docker compose exec api python -m app.cli search "medical leave" \
  --model sentence-transformers/all-MiniLM-L6-v2

# 3. scores look right -> make it the default (edit MODEL in app/embed.py, rebuild) and
#    retire the old model; the chunks of both models coexist until this DELETE
docker compose exec db psql -U postgres -d ocr_pipeline \
  -c "DELETE FROM chunks WHERE model = 'BAAI/bge-small-en-v1.5'"
```

Notes:

- Or pass `--model` per run as above (any model [fastembed](https://github.com/qdrant/fastembed)
  supports); the model is downloaded on first use unless baked into the image (see the `Dockerfile`).
- **For different vector sizes:** `chunks.embedding` is `vector(384)`. Models with another dimension
  need a migration first (`TRUNCATE chunks`, `ALTER TABLE chunks ALTER COLUMN embedding TYPE
  vector(N)`, rebuild the HNSW index) and `DIMENSION` updated in `app/embed.py`; `reembed`
  refuses a mismatched model rather than writing bad vectors.
