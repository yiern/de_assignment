-- 001_init.sql — initial schema
--
-- Conventions:
--   * identity of a document = SHA-256 of the uploaded bytes; document_id = uuid5 of it, so
--     re-ingest cannot fork a document.
--   * geometry lives in the canonical ratio frame 0-1, origin top-left (see README).
--   * confidences are stored 0-1; confidence_source says whether the engine reported the value
--     or it was derived from words.

CREATE EXTENSION IF NOT EXISTS vector;

-- Lineage root: which run, code version, normaliser version and model produced records.
CREATE TABLE pipeline_runs (
  id                uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  started_at        timestamptz NOT NULL DEFAULT now(),
  finished_at       timestamptz,
  code_version      text,          -- git sha of the running pipeline
  normaliser_version text,         -- app/normalise.py NORMALISER_VERSION that produced artefacts
  embedding_model   text,          -- set once this run embeds chunks
  files_new         int NOT NULL DEFAULT 0,
  files_duplicate   int NOT NULL DEFAULT 0,
  files_failed      int NOT NULL DEFAULT 0
);

-- One row per unique uploaded file.
CREATE TABLE documents (
  id                uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  content_hash      text NOT NULL UNIQUE,   -- 'sha256:<hex>'; THE identity key across all layers
  source_sha256     text,                   -- hash of the original scan. Deliberately NOT unique:
                                            -- a re-OCR of the same scan (new bytes) is a new
                                            -- document version, and this column is how re-exports
                                            -- of one scan are grouped.
  original_filename text,
  mime_type         text,
  size_bytes        bigint CHECK (size_bytes IS NULL OR size_bytes >= 0),
  storage_uri       text NOT NULL,          -- raw zone object for these exact bytes
  document_type     text CHECK (document_type IN ('medical_certificate', 'referral_letter', 'receipt')),
  country_code      text NOT NULL DEFAULT 'SG' CHECK (country_code ~ '^[A-Z]{2}$'),
  status            text NOT NULL DEFAULT 'received' CHECK (status IN
                      ('received', 'normalised', 'loaded', 'embedded', 'flagged', 'rejected')),
  status_reason     text,                   -- quarantine reason when rejected/flagged
  quality_flags     jsonb NOT NULL DEFAULT '[]'::jsonb,  -- quality check results, e.g. ["low_confidence:page_1"]
  run_id            uuid REFERENCES pipeline_runs(id),
  received_at       timestamptz NOT NULL DEFAULT now(),
  normalised_at     timestamptz,            -- per-stage timestamps = the status ladder, queryable
  loaded_at         timestamptz,
  embedded_at       timestamptz,
  normaliser_version text                   -- app/normalise.py NORMALISER_VERSION for this artefact
);

CREATE INDEX documents_status_idx       ON documents (status);
CREATE INDEX documents_type_country_idx ON documents (document_type, country_code);  -- /search filters
CREATE INDEX documents_source_sha_idx   ON documents (source_sha256);                -- re-OCR grouping

-- One row per page: page number, dimensions, engine facts from the envelope, mean confidence.
CREATE TABLE document_pages (
  document_id      uuid NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
  page_number      int  NOT NULL CHECK (page_number >= 1),
  width            real NOT NULL,           -- canonical ratios; the page frame is 1 x 1
  height           real NOT NULL,
  native_width     real,                    -- the engine's own dimensions and unit; NULL when the
  native_height    real,                    -- engine provides none (Textract) — documented reason
  native_unit      text,                    -- 'pixel' | 'inch'
  page_size_known  boolean NOT NULL,
  page_size_source text,
  engine           text NOT NULL,
  engine_version   text,
  processed_at     timestamptz,
  mean_confidence  real CHECK (mean_confidence IS NULL OR mean_confidence BETWEEN 0 AND 1),
  PRIMARY KEY (document_id, page_number)
);

-- Line-level OCR text with bbox + confidence, traceable to page and document.
CREATE TABLE ocr_lines (
  document_id       uuid NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
  page_number       int  NOT NULL,
  line_index        int  NOT NULL,
  text              text NOT NULL,          -- contains PII; see ocr_lines_safe view
  x0 real NOT NULL, y0 real NOT NULL,       -- ratios of page width/height, origin top-left
  x1 real NOT NULL, y1 real NOT NULL,
  confidence        real CHECK (confidence IS NULL OR confidence BETWEEN 0 AND 1),
  confidence_source text CHECK (confidence_source IN ('engine', 'words_mean')),  -- NULL: engine gave none
  PRIMARY KEY (document_id, page_number, line_index),
  FOREIGN KEY (document_id, page_number) REFERENCES document_pages(document_id, page_number) ON DELETE CASCADE
);

-- Key fields per document type. Key-value shape: one consistent, queryable shape for
-- all types — a new document type ships marker rules and field names, not DDL. Missing or
-- unparsable values are stored as NULL value with a validation_status, never dropped.
CREATE TABLE extracted_fields (
  document_id       uuid NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
  field_name        text NOT NULL,
  raw_value         text,                   -- exactly as found in OCR text ('S$93.20')
  value             text,                   -- normalised ('9320'); NULL when missing/unparsable
  validation_status text NOT NULL CHECK (validation_status IN ('valid', 'invalid', 'missing')),
  run_id            uuid REFERENCES pipeline_runs(id),
  created_at        timestamptz NOT NULL DEFAULT now(),
  PRIMARY KEY (document_id, field_name)
);

-- Text chunks + embeddings. Rows are keyed per model so old and new embeddings coexist
-- during a model migration and search keeps working; the switch is a WHERE model = …
-- cutover, then old rows are deleted. bbox is the union of the chunk's OCR lines, so a search
-- result traces back to a region of the original scan.
CREATE TABLE chunks (
  id           uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  document_id  uuid NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
  chunk_index  int  NOT NULL,
  page_number  int  NOT NULL,
  line_start   int  NOT NULL,               -- inclusive range into ocr_lines on that page
  line_end     int  NOT NULL,
  x0 real NOT NULL, y0 real NOT NULL, x1 real NOT NULL, y1 real NOT NULL,
  text         text NOT NULL,
  model        text NOT NULL,
  model_version text,
  dimension    int  NOT NULL CHECK (dimension > 0),
  embedding    vector(384) NOT NULL,        -- typmod must match the model; a dimension change is
                                            -- an ALTER alongside the model switch (documented)
  run_id       uuid REFERENCES pipeline_runs(id),
  created_at   timestamptz NOT NULL DEFAULT now(),
  UNIQUE (document_id, chunk_index, model),
  FOREIGN KEY (document_id, page_number) REFERENCES document_pages(document_id, page_number)
);

CREATE INDEX chunks_model_idx ON chunks (model);
CREATE INDEX chunks_doc_idx   ON chunks (document_id);
-- Distance metric: cosine.
-- For bulk loads, drop and recreate this index after loading — far faster at scale.
CREATE INDEX chunks_embedding_hnsw ON chunks USING hnsw (embedding vector_cosine_ops);

-- Quarantine: a view over status and flags, never a second copy of the truth.
-- Covers rejected/flagged documents AND processed documents carrying quality flags.
CREATE VIEW quarantine AS
SELECT id, status, status_reason, quality_flags, content_hash, original_filename,
       document_type, country_code, received_at
FROM documents
WHERE status IN ('flagged', 'rejected') OR jsonb_array_length(quality_flags) > 0;

-- Concrete control: consumers that do not need raw text read this view instead of the base
-- table. Singapore NRIC/FIN-shaped identifiers (e.g. S1234567D) are masked in place; geometry and
-- confidence stay exact for layout work. In production, REVOKE SELECT ON ocr_lines and GRANT
-- SELECT ON ocr_lines_safe to non-pipeline roles — this compose stack runs one owner role, so the
-- grant wiring is documented rather than executed.
CREATE VIEW ocr_lines_safe AS
SELECT document_id, page_number, line_index,
       regexp_replace(text, '[STFG][0-9]{7}[A-Z]', 'X0000000X', 'g') AS text_masked,
       x0, y0, x1, y1, confidence, confidence_source
FROM ocr_lines;
