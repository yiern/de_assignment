"""One normaliser test per engine format, plus the rejection paths."""

import json
from pathlib import Path

import pytest

from app.normalise import (
    NORMALISER_VERSION,
    MalformedExport,
    UnsupportedOcrFormat,
    normalise,
    parse_rapidocr,
)

SAMPLES = Path(__file__).resolve().parents[1] / "Assessment_Documents"


def load(name):
    with open(SAMPLES / name) as fh:
        return json.load(fh)


def test_textract_normalises_lines_without_page_size():
    doc = normalise(load("medical_certificate.json"))

    assert doc["normaliser_version"] == NORMALISER_VERSION
    assert doc["ocr"]["engine"] == "aws-textract"
    assert doc["ocr"]["engine_version"] == "DetectDocumentText/1.0"
    assert doc["ocr"]["processed_at"] == "2026-09-02T03:12:44Z"

    assert len(doc["pages"]) == 1
    page = doc["pages"][0]
    assert page["page_number"] == 1
    assert len(page["lines"]) == 31
    # Documented reason: Textract returns no absolute page dimensions.
    assert page["page_size_known"] is False
    assert page["native_size"] is None
    assert page["page_size_source"] == "not_provided_by_engine"

    first = page["lines"][0]
    assert first["text"] == "HARBOURVIEW FAMILY CLINIC PTE LTD"
    assert first["bbox"] == [0.080645, 0.062714, 0.5, 0.079818]
    assert first["confidence"] == round(97.189 / 100, 4)
    assert first["confidence_source"] == "engine"


def test_tesseract_rebuilds_lines_from_words():
    doc = normalise(load("receipt.json"))

    assert doc["ocr"]["engine"] == "tesseract"
    assert len(doc["pages"]) == 1
    page = doc["pages"][0]

    # 38 level-4 rows are empty; lines must come from grouping the 92 words.
    assert len(page["lines"]) == 38
    assert all(line["text"].strip() for line in page["lines"])
    assert page["page_size_known"] is True
    assert page["native_size"] == {"width": 1240, "height": 1754, "unit": "pixel"}

    total = next(line for line in page["lines"] if line["text"].startswith("TOTAL"))
    assert total["bbox"][0] == round(700 / 1240, 6)
    assert total["bbox"][1] == round(734 / 1754, 6)
    # -1 sentinel on non-word rows must never leak into a confidence.
    assert all(
        line["confidence"] is None or 0.0 <= line["confidence"] <= 1.0
        for line in page["lines"]
    )
    assert total["confidence_source"] == "words_mean"


def test_azure_normalises_inch_polygons_to_ratios():
    doc = normalise(load("referral_letter.json"))

    assert doc["ocr"]["engine"] == "azure-document-intelligence"
    assert doc["ocr"]["engine_version"] == "prebuilt-read/2024-11-30"
    page = doc["pages"][0]

    assert len(page["lines"]) == 37
    assert page["native_size"] == {"width": 8.2667, "height": 11.6933, "unit": "inch"}
    assert page["page_size_known"] is True

    first = page["lines"][0]
    assert first["text"] == "HARBOURVIEW FAMILY CLINIC PTE LTD"
    # Azure polygons are rounded to 4 dp in the export, so inch->ratio lands within ~1e-4 of
    # the same letterhead coordinates Textract reports natively as ratios (0.080645/0.062714).
    assert first["bbox"] == pytest.approx([0.080645, 0.062714, 0.5, 0.079818], abs=1e-3)
    # Azure scores words only: line confidence is the mean of the words inside the line's span.
    assert 0.0 < first["confidence"] <= 1.0
    assert first["confidence_source"] == "words_mean"
    assert 0.0 < page["mean_confidence"] <= 1.0


def test_azure_line_confidence_is_the_span_window_word_mean():
    def word(offset, length, conf, content):
        return {"content": content, "confidence": conf, "span": {"offset": offset, "length": length}}

    page = {
        "width": 8.27, "height": 11.69, "unit": "inch",
        "words": [word(0, 5, 0.8, "Alpha"), word(6, 5, 0.6, "Bravo"), word(11, 4, 0.2, "Next")],
        "lines": [
            {"content": "Alpha Bravo", "polygon": [0, 0, 2, 0, 2, 1, 0, 1],
             "spans": [{"offset": 0, "length": 11}]},
            {"content": "Next", "polygon": [0, 1, 2, 1, 2, 2, 0, 2],
             "spans": [{"offset": 11, "length": 4}]},
        ],
    }
    doc = normalise({"source": {}, "ocr": {"engine": "azure-document-intelligence"},
                     "raw_output": {"status": "succeeded", "analyzeResult": {"pages": [page]}}})

    first, second = doc["pages"][0]["lines"]
    assert first["confidence"] == pytest.approx(0.7)  # (0.8 + 0.6) / 2; "Next" starts at the boundary
    assert first["confidence_source"] == "words_mean"
    assert second["confidence"] == pytest.approx(0.2)
    assert doc["pages"][0]["mean_confidence"] == pytest.approx(0.45)


@pytest.mark.parametrize(
    "name", ["medical_certificate.json", "receipt.json", "referral_letter.json"]
)
def test_all_engines_normalise_into_the_ratio_frame(name):
    doc = normalise(load(name))
    for page in doc["pages"]:
        for line in page["lines"]:
            x0, y0, x1, y1 = line["bbox"]
            assert 0.0 <= x0 < x1 <= 1.0
            assert 0.0 <= y0 < y1 <= 1.0


def test_envelope_metadata_is_carried_through():
    doc = normalise(load("receipt.json"), content_sha256="abc123")
    assert doc["source"]["content_sha256"] == "abc123"
    assert doc["source"]["country_code"] == "SG"
    assert doc["source"]["export_format_version"] == "1.0"
    assert doc["source"]["original_filename"] == "receipt.pdf"


def test_unknown_engine_is_unsupported():
    with pytest.raises(UnsupportedOcrFormat) as exc:
        normalise({"ocr": {"engine": "abba"}, "source": {}, "raw_output": {}})
    assert exc.value.code == "unsupported_ocr_format"


def test_raw_output_not_matching_declared_engine_is_unsupported():
    envelope = load("medical_certificate.json")
    envelope["ocr"]["engine"] = "tesseract"
    with pytest.raises(UnsupportedOcrFormat):
        normalise(envelope)


def test_missing_envelope_is_unsupported():
    with pytest.raises(UnsupportedOcrFormat) as exc:
        normalise({"raw_output": {}})
    assert exc.value.code == "unsupported_ocr_format"


def test_non_object_json_is_unsupported():
    with pytest.raises(UnsupportedOcrFormat):
        normalise([1, 2, 3])


def test_tesseract_without_page_dimensions_is_malformed():
    envelope = load("receipt.json")
    raw = envelope["raw_output"]
    raw["level"][0] = 2  # remove the only level-1 page row
    with pytest.raises(MalformedExport) as exc:
        normalise(envelope)
    assert exc.value.code == "unreadable_file"


def _rapidocr_raw():
    return [{
        "image_size": [1000, 2000],
        "pdf_page_size": [595.0, 842.0],
        "result": [
            [[[100, 100], [400, 100], [400, 200], [100, 200]], "MEDICAL CERTIFICATE", 0.9876],
            [[[100, 300], [300, 320], [300, 360], [100, 340]], "Dr Tan", 0.5],
        ],
    }]


def test_parse_rapidocr_maps_quads_to_ratio_bboxes():
    (pg,) = parse_rapidocr(_rapidocr_raw())

    assert pg["native_size"] == {"width": 595.0, "height": 842.0, "unit": "point"}
    assert pg["page_size_source"] == "rendered"
    first, second = pg["lines"]
    assert first["bbox"] == [0.1, 0.05, 0.4, 0.1]
    assert first["confidence"] == 0.9876 and first["confidence_source"] == "engine"
    assert second["bbox"] == [0.1, 0.15, 0.3, 0.18]  # rotated quad -> axis-aligned hull
    assert pg["mean_confidence"] == 0.7438


def test_parse_rapidocr_image_page_uses_pixels_and_allows_empty():
    (pg,) = parse_rapidocr([{"image_size": [800, 600], "result": []}])

    assert pg["native_size"]["unit"] == "pixel" and pg["lines"] == []


def test_parse_rapidocr_rejects_non_list():
    with pytest.raises(UnsupportedOcrFormat):
        parse_rapidocr({"pages": []})
