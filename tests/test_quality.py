"""Quality checks: value normalisers, field rules on the real sample layouts,
page confidence flags, and quarantine visibility of flagged documents."""

import json
from pathlib import Path

import psycopg

from app import quality as qc
from app import normalise, pipeline

SAMPLES = Path(__file__).resolve().parents[1] / "Assessment_Documents"


def sample_pages(name):
    """Normalise a real sample export and return (by_page, document_type)."""
    envelope = json.loads((SAMPLES / name).read_text(encoding="utf-8"))
    doc = normalise.normalise(envelope, content_sha256="0" * 64)
    by_page = {page["page_number"]: page["lines"] for page in doc["pages"]}
    return by_page, pipeline.classify(doc)


def by_status(rows):
    return {row["field_name"]: row for row in rows}


# --- value normalisers ---------------------------------------------------------

def test_amount_minor_units():
    assert qc.get_amount("S$93.20") == "9320"
    assert qc.get_amount("SGD 1,200.00") == "120000"
    assert qc.get_amount("$45.00") == "4500"
    assert qc.get_amount("85.50") == "8550"
    assert qc.get_amount("S$abc") is None
    assert qc.get_amount("-5") is None


def test_date_validation():
    assert qc.get_date_value("05/09/2026") == "05/09/2026"
    assert qc.get_date_value("02/09/2026 10:20") == "02/09/2026"  # date fields: date only
    assert qc.get_datetime_value("02/09/2026 9:05") == "02/09/2026 09:05"
    assert qc.get_datetime_value("02/09/2026 25:00") is None
    assert qc.get_date_value("32/13/2026") is None                # parses not
    assert qc.get_date_value("2026-09-05") is None                # wrong format
    assert qc.get_date_value("hello") is None


def test_mc_days():
    assert qc.get_mc_days("unfit for a period of 2 day(s) from 02/09/2026") == "2"
    assert qc.get_mc_days("no duration mentioned") is None


def test_provider_tripwire():
    # The planted footer string must mark the provider invalid, not pass silently
    value, status = qc.check("provider", "Fullerton Health Panel Clinic")
    assert (value, status) == (None, "invalid")  # never reaches consumers
    assert qc.check("provider", "Harbourview Family Clinic Pte Ltd")[1] == "valid"


# --- the three real sample layouts ---------------------------------------------

def test_receipt_fields():
    by_page, doc_type = sample_pages("receipt.json")
    rows = by_status(qc.extract_fields(by_page, doc_type))
    assert set(rows) == {"claimant_name", "claimant_address", "claimant_date_of_birth",
                         "provider_name", "tax_amount", "total_amount"}
    assert rows["claimant_name"]["value"] == "SITI NURHALIZA BINTE AHMAD"
    assert rows["claimant_date_of_birth"]["value"] == "22/11/1992"
    assert rows["provider_name"]["value"] == "NOVENA WELLNESS MEDICAL CENTRE"
    assert rows["provider_name"]["validation_status"] == "valid"  # letterhead, not footer
    assert rows["tax_amount"]["value"] == "770"      # S$7.70
    assert rows["total_amount"]["value"] == "9320"   # S$93.20, not the Subtotal line
    assert all(row["validation_status"] == "valid" for row in rows.values())


def test_medical_certificate_fields():
    by_page, doc_type = sample_pages("medical_certificate.json")
    rows = by_status(qc.extract_fields(by_page, doc_type))
    assert rows["diagnosis_name"]["value"] == "Acute gastroenteritis"
    assert rows["icd_code"]["value"] == "A09"
    assert rows["discharge_date_time"]["raw_value"] == "02/09/2026 10:20"
    assert rows["discharge_date_time"]["value"] == "02/09/2026 10:20"
    assert rows["mc_days"]["value"] == "2"
    assert all(row["validation_status"] == "valid" for row in rows.values())
    assert len(rows) == 10  # the full reference field set, every field present


def test_referral_letter_fields():
    by_page, doc_type = sample_pages("referral_letter.json")
    rows = by_status(qc.extract_fields(by_page, doc_type))
    assert rows["claimant_name"]["value"] == "MUHAMMAD RIZAL BIN OSMAN"
    assert rows["total_requested_amount"]["value"] == "120000"  # SGD 1,200.00
    assert rows["total_approved_amount"]["value"] == "100000"
    assert rows["total_amount_paid"]["value"] == "85000"
    assert rows["signature_presence"] == {"field_name": "signature_presence",
                                          "raw_value": rows["signature_presence"]["raw_value"],
                                          "value": "true", "validation_status": "valid"}


def test_label_and_value_on_one_line_without_colon():
    page = [{"text": "TOTAL S$93.20", "line_index": 0}, {"text": "GST 9% S$7.70", "line_index": 1},
            {"text": "Thank you", "line_index": 2}]
    rows = by_status(qc.extract_fields({1: page}, "receipt"))
    assert rows["total_amount"]["value"] == "9320"
    assert rows["tax_amount"]["value"] == "770"


def test_missing_and_unknown():
    # a label the document lacks -> an explicit missing row, never dropped
    page = [{"text": "Patient:", "line_index": 0}, {"text": "TAN WEI MING", "line_index": 1}]
    rows = by_status(qc.extract_fields({1: page}, "receipt"))
    assert rows["claimant_name"]["validation_status"] == "valid"
    assert rows["total_amount"]["validation_status"] == "missing"
    assert rows["total_amount"]["value"] is None
    # unknown document type -> no rules, no rows, no crash
    assert qc.extract_fields({1: page}, None) == []


# --- page confidence flags ------------------------------------------------------

def test_page_flags():
    good = {1: [{"text": "a", "confidence": 0.96}, {"text": "b", "confidence": 0.92}]}
    assert qc.get_page_flags(good) == []
    low = {1: [{"text": "a", "confidence": 0.75}, {"text": "b", "confidence": 0.77}]}
    assert qc.get_page_flags(low) == ["low_confidence:page_1:avg=0.8"]
    # azure reports no confidence at all -> explicit flag, never a fabricated number
    assert qc.get_page_flags({1: [{"text": "a"}]}) == ["confidence_unavailable:page_1"]


# --- quarantine visibility (bullet 4, SQL half) ---------------------------------

def test_quarantine_view_covers_flagged_and_rejected(db_url):
    with psycopg.connect(db_url) as conn:
        doc = conn.execute(
            """INSERT INTO documents (id, content_hash, mime_type, storage_uri, status,
                   status_reason, quality_flags)
               VALUES (gen_random_uuid(), 'sha256:q1', 'application/json', 'raw://q1',
                       'rejected', 'unreadable_file', '[]') RETURNING id"""
        ).fetchone()[0]
        flagged = conn.execute(
            """INSERT INTO documents (id, content_hash, mime_type, storage_uri, status,
                   quality_flags)
               VALUES (gen_random_uuid(), 'sha256:q2', 'application/json', 'raw://q2',
                       'embedded', '["low_confidence:page_1:avg=0.8"]') RETURNING id"""
        ).fetchone()[0]
        clean = conn.execute(
            """INSERT INTO documents (id, content_hash, mime_type, storage_uri, status,
                   quality_flags)
               VALUES (gen_random_uuid(), 'sha256:q3', 'application/json', 'raw://q3',
                       'embedded', '[]') RETURNING id"""
        ).fetchone()[0]
        quarantined = {str(r[0]) for r in conn.execute("SELECT id FROM quarantine")}
        conn.rollback()  # the fixture database is shared per session; leave it as found
    assert {str(doc), str(flagged)} <= quarantined
    assert clean not in quarantined
