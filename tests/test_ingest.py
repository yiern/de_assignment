"""End-to-end ingest tests: full pipeline, idempotency, the API error contract, the CLI."""

import hashlib
import json
import shutil
from pathlib import Path

import psycopg
import pytest
from fastapi.testclient import TestClient
from psycopg_pool import ConnectionPool

from app import cli, embed, main, pipeline
from test_embed import FakeEmbedder

SAMPLES = Path(__file__).resolve().parents[1] / "Assessment_Documents"

SAMPLED_TYPES = {
    "medical_certificate.json": ("aws-textract", "medical_certificate"),
    "receipt.json": ("tesseract", "receipt"),
    "referral_letter.json": ("azure-document-intelligence", "referral_letter"),
}


@pytest.fixture()
def zones(tmp_path, monkeypatch):
    raw, normalised = tmp_path / "raw", tmp_path / "normalised"
    monkeypatch.setattr(pipeline, "RAW_ZONE", raw)
    monkeypatch.setattr(pipeline, "NORMALISED_ZONE", normalised)
    return {"raw": raw, "normalised": normalised}


@pytest.fixture()
def conn_url(db_url, zones, monkeypatch):
    """Fresh tables per test on the shared session database; embedder stubbed."""
    monkeypatch.setattr(embed, "embedder", lambda model=None: FakeEmbedder())
    with psycopg.connect(db_url) as conn:
        conn.execute("TRUNCATE documents, pipeline_runs CASCADE")
    return db_url


@pytest.fixture()
def client(conn_url, monkeypatch):
    # No context manager: the lifespan is skipped, so the pool is injected here.
    with ConnectionPool(conn_url, open=True) as pool:
        monkeypatch.setattr(main, "pool", pool)
        yield TestClient(main.app, raise_server_exceptions=False)


def post(client, data, filename="x.json", content_type="application/json", **form):
    return client.post("/documents", files={"file": (filename, data, content_type)}, data=form)


def upload(client, name, filename=None, **kwargs):
    return post(client, (SAMPLES / name).read_bytes(), filename or name, **kwargs)


def count(conn_url, table):
    with psycopg.connect(conn_url) as conn:
        return conn.execute(f"SELECT count(*) FROM {table}").fetchone()[0]


@pytest.mark.parametrize("name", SAMPLED_TYPES)
def test_each_engine_format_runs_the_full_pipeline(client, zones, name):
    engine, document_type = SAMPLED_TYPES[name]
    response = upload(client, name)

    assert response.status_code == 201
    body = response.json()
    assert body["duplicate"] is False
    assert body["status"] == "embedded"
    assert body["document_type"] == document_type
    assert body["pages"] == 1 and body["chunks"] > 0
    assert body["country_code"] == "SG"
    assert set(body["timings_ms"]) == {"normalise", "load", "embed"}

    raw_file = zones["raw"] / body["content_hash"].removeprefix("sha256:")
    assert raw_file.read_bytes() == (SAMPLES / name).read_bytes()
    normalised = json.loads(next(zones["normalised"].iterdir()).read_text())
    assert normalised["ocr"]["engine"] == engine

    detail = client.get(f"/documents/{body['document_id']}").json()
    assert detail["status"] == "embedded" and detail["fields"]


def test_ingest_then_search(client):
    upload(client, "receipt.json")
    results = client.post("/search", json={"query": "TOTAL", "filters": {"document_type": "receipt"}})

    assert results.status_code == 200
    hit = results.json()["results"][0]
    assert hit["document_type"] == "receipt" and len(hit["bbox"]) == 4
    assert hit["embedding_model"] == embed.MODEL


# The contract fixes the response field names, so pin them: a renamed or leaked key breaks consumers
SEARCH_RESULT_FIELDS = {"chunk_id", "document_id", "document_type", "page", "score",
                       "text", "bbox", "embedding_model"}


def test_search_response_matches_the_contract(client):
    upload(client, "receipt.json")
    upload(client, "referral_letter.json")
    response = client.post("/search", json={"query": "TOTAL S$93.20", "top_k": 3,
                                            "filters": {"document_type": "receipt"}})

    assert response.status_code == 200
    body = response.json()
    assert set(body) == {"results"}
    assert 0 < len(body["results"]) <= 3                                  # top_k is a ceiling
    for hit in body["results"]:
        assert set(hit) == SEARCH_RESULT_FIELDS
        assert hit["document_type"] == "receipt"                          # the filter applies
        assert 0.0 <= hit["score"] <= 1.0                                 # cosine similarity
        assert min(hit["bbox"]) >= 0.0 and max(hit["bbox"]) <= 1.0        # ratio frame
        assert hit["bbox"][0] <= hit["bbox"][2] and hit["bbox"][1] <= hit["bbox"][3]
        assert hit["page"] == 1 and hit["embedding_model"] == embed.MODEL

    # no match is an empty result set, never a 404
    empty = client.post("/search", json={"query": "TOTAL", "filters": {"country_code": "VN"}})
    assert empty.status_code == 200 and empty.json() == {"results": []}


def test_unknown_filter_key_is_rejected_not_ignored(client):
    """A typo'd filter must not answer over the whole corpus."""
    upload(client, "receipt.json")
    bad = client.post("/search", json={"query": "TOTAL", "filters": {"document_typ": "receipt"}})
    assert bad.status_code == 400
    assert bad.json() == {"error": "unknown_filter", "parameter": "document_typ"}

    both = client.post("/search", json={"query": "TOTAL",
                                        "filters": {"document_type": "receipt", "country_code": "SG"}})
    assert both.status_code == 200 and both.json()["results"]


def test_reingest_same_bytes_is_a_duplicate(client, conn_url, zones):
    first, second = upload(client, "receipt.json"), upload(client, "receipt.json")

    assert (first.status_code, second.status_code) == (201, 200)
    assert second.json()["duplicate"] is True
    assert second.json()["document_id"] == first.json()["document_id"]
    assert second.json()["chunks"] == first.json()["chunks"]
    assert [count(conn_url, t) for t in ("documents", "document_pages", "chunks")] == [
        1, 1, first.json()["chunks"]]
    assert len(list(zones["raw"].iterdir())) == 1


def test_same_bytes_under_a_different_filename_is_a_duplicate(client):
    first = upload(client, "receipt.json")
    second = upload(client, "receipt.json", filename="copy-of-receipt.json")

    assert (first.status_code, second.status_code) == (201, 200)
    assert second.json()["document_id"] == first.json()["document_id"]


def test_country_code_precedence(client):
    assert upload(client, "receipt.json", country_code="VN").json()["country_code"] == "VN"

    envelope = json.loads((SAMPLES / "receipt.json").read_text())
    envelope["source"]["country_code"] = "PH"
    assert post(client, json.dumps(envelope).encode()).json()["country_code"] == "PH"

    del envelope["source"]["country_code"]
    envelope["source"]["original_filename"] = "other.pdf"  # new bytes, so not a duplicate
    assert post(client, json.dumps(envelope).encode()).json()["country_code"] == "SG"


def test_invalid_country_codes_fall_through_and_are_flagged(client):
    # lowercase is normalised, not rejected
    assert upload(client, "receipt.json", country_code="sg").json()["country_code"] == "SG"

    envelope = json.loads((SAMPLES / "receipt.json").read_text())
    envelope["source"]["country_code"] = "ph"
    # invalid form value -> falls through to the envelope
    response = post(client, json.dumps(envelope).encode(), country_code="S1")
    assert response.status_code == 201
    assert response.json()["country_code"] == "PH"
    assert response.json()["quality_flags"] == ["country_code_invalid:form"]

    # invalid everywhere -> SG, both skips flagged
    envelope["source"]["country_code"] = "Singapore"
    envelope["source"]["original_filename"] = "again.pdf"
    body = post(client, json.dumps(envelope).encode(), country_code="123").json()
    assert body["country_code"] == "SG"
    assert body["quality_flags"] == ["country_code_invalid:form", "country_code_invalid:envelope"]


def test_reupload_of_a_loaded_but_unembedded_document_resumes_at_embed(client, conn_url):
    first = upload(client, "receipt.json").json()
    with psycopg.connect(conn_url) as conn:  # simulate a crash between load and embed
        conn.execute("DELETE FROM chunks")
        conn.execute("UPDATE documents SET status = 'loaded', embedded_at = NULL")
    response = upload(client, "receipt.json")

    assert response.status_code == 201
    body = response.json()
    assert body["duplicate"] is False and body["status"] == "embedded"
    assert body["document_id"] == first["document_id"] and body["chunks"] == first["chunks"]
    assert set(body["timings_ms"]) == {"embed"}
    assert count(conn_url, "documents") == 1


def test_embedded_duplicate_has_no_timings_and_adds_nothing(client, conn_url):
    upload(client, "receipt.json")
    body = upload(client, "receipt.json").json()
    assert body["duplicate"] is True and body["timings_ms"] == {}


def test_422_body_points_at_the_quarantined_document(client):
    response = post(client, b"{not json")

    assert response.status_code == 422
    body = response.json()
    assert body["error"] == "unreadable_file"
    assert body["content_hash"] == "sha256:" + hashlib.sha256(b"{not json").hexdigest()
    detail = client.get(f"/documents/{body['document_id']}").json()
    assert (detail["status"], detail["status_reason"]) == ("rejected", "unreadable_file")


def test_non_422_errors_carry_no_document_id(client):
    for response in (upload(client, "receipt.json", content_type="text/plain"),
                     client.post("/documents")):
        assert set(response.json()) == {"error"}


def test_malformed_json_is_rejected_stored_and_quarantined(client, conn_url, zones):
    data = b'{"export_format_version": "1.0", "raw_output": {'
    response = post(client, data)

    assert response.status_code == 422
    assert response.json()["error"] == "unreadable_file"
    digest = hashlib.sha256(data).hexdigest()
    assert (zones["raw"] / digest).read_bytes() == data
    with psycopg.connect(conn_url) as conn:
        row = conn.execute("SELECT status, status_reason FROM quarantine").fetchone()
    assert row == ("rejected", "unreadable_file")


def test_rejected_file_can_later_succeed_without_duplicates(client, conn_url):
    # a rejected placeholder row must not block, or duplicate with, a later good ingest
    data = (SAMPLES / "receipt.json").read_bytes()
    with psycopg.connect(conn_url) as conn:
        pipeline.quarantine(conn, hashlib.sha256(data).hexdigest(), "r.json", "x")
    assert post(client, data).status_code == 201
    assert count(conn_url, "documents") == 1


def test_json_that_is_not_an_ocr_export_is_unsupported(client, zones):
    data = json.dumps({"hello": "world"}).encode()
    response = post(client, data)

    assert response.status_code == 422
    assert response.json()["error"] == "unsupported_ocr_format"
    assert (zones["raw"] / hashlib.sha256(data).hexdigest()).exists()


def test_broken_engine_payload_is_422_not_500(client):
    envelope = json.loads((SAMPLES / "medical_certificate.json").read_text())
    for block in envelope["raw_output"]["Blocks"]:
        if block.get("BlockType") == "LINE":
            block["Geometry"]["BoundingBox"] = {"Width": 0.1}  # missing Left/Top/Height
    response = post(client, json.dumps(envelope).encode())

    assert response.status_code == 422
    assert response.json()["error"] == "unreadable_file"


def test_unsupported_document_type_is_422_and_quarantined(client, conn_url, zones):
    envelope = {
        "export_format_version": "1.0",
        "source": {"original_filename": "scan.pdf", "mime_type": "application/pdf"},
        "ocr": {"engine": "tesseract", "engine_version": "5.3.4"},
        "raw_output": {
            "level": [1, 5], "page_num": [1, 1], "block_num": [0, 1], "par_num": [0, 1],
            "line_num": [0, 1], "word_num": [0, 1], "left": [0, 10], "top": [0, 10],
            "width": [100, 40], "height": [100, 10], "conf": [-1, 92.0],
            "text": ["", "SPECIMEN"],
        },
    }
    data = json.dumps(envelope).encode()
    response = post(client, data)

    assert response.status_code == 422
    assert response.json()["error"] == "unsupported_document_type"
    assert (zones["raw"] / hashlib.sha256(data).hexdigest()).exists()
    with psycopg.connect(conn_url) as conn:
        assert conn.execute("SELECT status_reason FROM quarantine").fetchone() == (
            "unsupported_document_type",)


def test_non_json_media_type_is_rejected_before_anything_is_stored(client, zones):
    response = upload(client, "receipt.json", content_type="text/plain")

    assert response.status_code == 400
    assert response.json() == {"error": "invalid_mime_type"}
    assert not zones["raw"].exists()


def test_oversized_upload_is_rejected(client, monkeypatch):
    monkeypatch.setattr(main, "MAX_UPLOAD_BYTES", 100)
    response = upload(client, "receipt.json")

    assert (response.status_code, response.json()) == (413, {"error": "file_too_large"})


def test_losing_a_concurrent_insert_race_reports_a_duplicate(client, conn_url, monkeypatch):
    first = upload(client, "receipt.json")

    def collide(conn, path, run_id):
        raise psycopg.errors.UniqueViolation("documents_content_hash_key")

    monkeypatch.setattr(pipeline, "load_document", collide)
    # skip the up-front duplicate check, as a second worker that read before the commit would
    monkeypatch.setattr(pipeline, "existing_status", lambda conn, content_hash: None)
    with psycopg.connect(conn_url) as conn:
        result = pipeline.process(conn, (SAMPLES / "receipt.json").read_bytes())

    assert result["duplicate"] is True
    assert result["document_id"] == first.json()["document_id"]


def test_missing_file_field(client):
    response = client.post("/documents", data={"country_code": "SG"})

    assert response.status_code == 400
    assert response.json() == {"error": "file_missing"}


def test_unknown_document_is_404(client):
    for bad_id in ("00000000-0000-0000-0000-000000000000", "not-a-uuid"):
        response = client.get(f"/documents/{bad_id}")
        assert (response.status_code, response.json()) == (404, {"error": "not_found"})


def test_unhandled_exception_is_500_with_fixed_body(client, monkeypatch):
    def boom(*args, **kwargs):
        raise RuntimeError("patient name leaked here")

    monkeypatch.setattr(main, "process", boom)
    response = upload(client, "receipt.json")

    assert (response.status_code, response.json()) == (500, {"error": "internal_server_error"})


def test_cli_ingest_summarises_and_isolates_failures(conn_url, tmp_path):
    folder = tmp_path / "batch"
    folder.mkdir()
    shutil.copy(SAMPLES / "receipt.json", folder / "a.json")
    shutil.copy(SAMPLES / "receipt.json", folder / "b_copy.json")  # duplicate bytes
    shutil.copy(SAMPLES / "referral_letter.json", folder / "c.json")
    (folder / "bad.json").write_text("{not json")
    (folder / "notes.txt").write_text("hello")  # unsupported format

    with psycopg.connect(conn_url) as conn:
        processed, skipped, failed = cli.ingest_folder(conn, folder)

    assert (processed, skipped) == (2, 1)
    assert dict(failed) == {"bad.json": "unreadable_file", "notes.txt": "invalid_mime_type"}
    assert count(conn_url, "documents WHERE status = 'embedded'") == 2


def _pdf_bytes(*pages_text):
    import pymupdf

    doc = pymupdf.open()
    for text in pages_text:
        page = doc.new_page(width=595, height=842)
        for i, ln in enumerate(text.splitlines()):
            page.insert_text((72, 120 + 28 * i), ln, fontsize=18)
    return doc.tobytes()


def _png_bytes(text):
    import pymupdf

    doc = pymupdf.open()
    doc.new_page(width=595, height=842).insert_text((72, 120), text, fontsize=18)
    return doc[0].get_pixmap(dpi=200).tobytes("png")


MC = "MEDICAL CERTIFICATE\nDr Tan Wei Ming\nDiagnosis: Acute bronchitis\nMC Days: 3"


def test_ocr_pdf_upload_runs_full_pipeline(client, zones):
    response = post(client, _pdf_bytes(MC, "Follow up notes"), filename="mc.pdf",
                    content_type="application/pdf")

    assert response.status_code == 201
    body = response.json()
    assert body["status"] == "embedded" and body["document_type"] == "medical_certificate"
    assert body["pages"] == 2 and body["chunks"] > 0
    digest = body["content_hash"].removeprefix("sha256:")
    assert (zones["raw"] / f"{digest}.ocr.json").exists()  # re-normalisation stays possible
    normalised = json.loads((zones["normalised"] / f"{digest}.normalised.json").read_text())
    assert normalised["ocr"]["engine"] == "rapidocr"
    assert normalised["pages"][0]["native_size"]["unit"] == "point"
    assert any("MEDICALCERTIFICATE" in ln["text"].replace(" ", "").upper()
               for ln in normalised["pages"][0]["lines"])


def test_ocr_image_upload_then_search(client):
    response = post(client, _png_bytes("OFFICIAL RECEIPT\nTOTAL SGD 93.20"),
                    filename="r.png", content_type="image/png")

    assert response.status_code == 201 and response.json()["document_type"] == "receipt"
    hits = client.post("/search", json={"query": "TOTAL"}).json()["results"]
    assert hits and len(hits[0]["bbox"]) == 4


def test_ocr_reupload_same_bytes_is_idempotent(client, conn_url):
    data = _png_bytes("TAX INVOICE\nTotal: SGD 12.34")
    assert post(client, data, filename="a.png", content_type="image/png").status_code == 201
    second = post(client, data, filename="b.png", content_type="image/png")

    assert second.status_code == 200 and second.json()["duplicate"] is True
    assert count(conn_url, "documents") == 1


def test_corrupt_pdf_is_422_quarantined_and_reusable(client, conn_url, zones):
    data = b"%PDF-1.4\nthis is not really a pdf"
    response = post(client, data, filename="x.pdf", content_type="application/pdf")

    assert response.status_code == 422
    assert response.json()["error"] == "unreadable_file"
    assert (zones["raw"] / hashlib.sha256(data).hexdigest()).read_bytes() == data
    with psycopg.connect(conn_url) as conn:
        assert conn.execute(
            "SELECT status, status_reason FROM quarantine"
        ).fetchone() == ("rejected", "unreadable_file")


def test_reembed_under_new_model_keeps_old_embeddings(client, conn_url):
    upload(client, "receipt.json")
    with psycopg.connect(conn_url) as conn:
        assert embed.embed_loaded(conn, model="other-model") == (1, 0)
        models = conn.execute("SELECT DISTINCT model FROM chunks ORDER BY model").fetchall()
    assert models == [(embed.MODEL,), ("other-model",)]
