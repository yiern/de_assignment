"""Engine-agnostic OCR normalisation.

Canonical frame: normalised ratios 0-1, origin top-left, bbox as [x0, y0, x1, y1].
Textract already reports ratios and provides no page dimensions, so ratios are the only
frame all three engines share; native dimensions are kept in pages[].native_size so
pixel/inch coordinates can be reconstructed exactly.
"""

from __future__ import annotations

from datetime import datetime, timezone

NORMALISER_VERSION = "1.0"


class NormaliseError(Exception):
    """Base class; `code` is the API error code for this failure."""

    code = "normalise_error"


class MalformedExport(NormaliseError):
    """A recognised engine export whose payload is broken or incomplete."""

    code = "unreadable_file"


class UnsupportedOcrFormat(NormaliseError):
    """JSON that cannot be mapped to any supported engine export."""

    code = "unsupported_ocr_format"


def bbox(x0, y0, x1, y1):
    return [round(v, 6) for v in (min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1))]


def confidence(value, scale):
    """Normalise engine confidence to 0-1. Tesseract uses -1 on non-word rows."""
    if value is None or value < 0:
        return None
    return round(value / scale, 4)


def line(index, text, box, conf, source):
    return {
        "line_index": index,
        "text": text,
        "bbox": box,
        "confidence": conf,
        "confidence_source": source if conf is not None else None,
    }


def page(page_number, lines, native_size=None, unit=None, size_source="engine"):
    known = native_size is not None
    confidences = [ln["confidence"] for ln in lines if ln["confidence"] is not None]
    return {
        "page_number": page_number,
        "size": {"width": 1.0, "height": 1.0},
        "native_size": (
            {"width": native_size[0], "height": native_size[1], "unit": unit} if known else None
        ),
        "page_size_known": known,
        "page_size_source": size_source,
        "mean_confidence": round(sum(confidences) / len(confidences), 4) if confidences else None,
        "lines": lines,
    }


# --------------------------------------------------------------------------- textract


def parse_textract(raw):
    blocks = raw.get("Blocks") if isinstance(raw, dict) else None
    if not isinstance(blocks, list) or not any(
        isinstance(b, dict) and b.get("BlockType") == "PAGE" for b in blocks
    ):
        raise UnsupportedOcrFormat("raw_output does not look like an AWS Textract response")

    by_page = {}
    for block in blocks:
        if not isinstance(block, dict) or block.get("BlockType") != "LINE":
            continue
        box = (block.get("Geometry") or {}).get("BoundingBox")
        if not box:
            continue  # a LINE without geometry cannot be placed; nothing to normalise
        by_page.setdefault(block.get("Page", 1), []).append(
            line(
                index=0,  # renumbered below, in block (reading) order
                text=block.get("Text") or "",
                box=bbox(
                    box["Left"], box["Top"], box["Left"] + box["Width"], box["Top"] + box["Height"]
                ),
                conf=confidence(block.get("Confidence"), 100),
                source="engine",
            )
        )

    pages = []
    for number in sorted(by_page):
        lines = by_page[number]
        for i, ln in enumerate(lines):
            ln["line_index"] = i
        # DetectDocumentText returns no page size: it reports geometry already normalised.
        pages.append(page(number, lines, size_source="not_provided_by_engine"))
    return pages


# -------------------------------------------------------------------------- tesseract

_TESSERACT_COLUMNS = (
    "level",
    "page_num",
    "block_num",
    "par_num",
    "line_num",
    "word_num",
    "left",
    "top",
    "width",
    "height",
    "conf",
    "text",
)


def parse_tesseract(raw):
    if not isinstance(raw, dict) or any(
        not isinstance(raw.get(c), list) for c in _TESSERACT_COLUMNS
    ):
        raise UnsupportedOcrFormat(
            "raw_output does not look like a tesseract image_to_data export"
        )
    n = len(raw["level"])
    if any(len(raw[c]) != n for c in _TESSERACT_COLUMNS):
        raise UnsupportedOcrFormat("tesseract parallel arrays have mismatched lengths")

    by_page = {}
    for i in range(n):
        level = raw["level"][i]
        pg = by_page.setdefault(raw["page_num"][i] or 1, {"size": None, "words": {}, "boxes": {}})
        if level == 1:
            if raw["width"][i] and raw["height"][i]:
                pg["size"] = (raw["width"][i], raw["height"][i])
            continue
        key = (raw["block_num"][i], raw["par_num"][i], raw["line_num"][i])
        if level == 4:
            # Level 4 rows carry the engine's own line box but no text and no confidence.
            pg["boxes"][key] = (
                raw["left"][i],
                raw["top"][i],
                raw["left"][i] + raw["width"][i],
                raw["top"][i] + raw["height"][i],
            )
        elif level == 5 and (raw["text"][i] or "").strip():
            pg["words"].setdefault(key, []).append(
                (
                    raw["text"][i],
                    raw["left"][i],
                    raw["top"][i],
                    raw["left"][i] + raw["width"][i],
                    raw["top"][i] + raw["height"][i],
                    confidence(raw["conf"][i], 100),
                )
            )

    pages = []
    for number in sorted(by_page):
        pg = by_page[number]
        size = pg["size"]
        if not size:
            raise MalformedExport(f"tesseract page {number} has no page dimensions")
        w, h = size
        lines = []
        for i, key in enumerate(sorted(pg["words"])):
            words = pg["words"][key]
            box = pg["boxes"].get(key) or (
                min(w_[1] for w_ in words),
                min(w_[2] for w_ in words),
                max(w_[3] for w_ in words),
                max(w_[4] for w_ in words),
            )
            confidences = [w_[5] for w_ in words if w_[5] is not None]
            lines.append(
                line(
                    index=i,
                    text=" ".join(w_[0] for w_ in words),
                    box=bbox(box[0] / w, box[1] / h, box[2] / w, box[3] / h),
                    conf=(
                        round(sum(confidences) / len(confidences), 4) if confidences else None
                    ),
                    source="words_mean",  # tesseract reports confidence on words only
                )
            )
        pages.append(page(number, lines, native_size=size, unit="pixel", size_source="page_row"))
    return pages


# ----------------------------------------------------------------------------- azure


def parse_azure(raw):
    ar = raw.get("analyzeResult") if isinstance(raw, dict) else None
    if not isinstance(ar, dict) or not isinstance(ar.get("pages"), list):
        raise UnsupportedOcrFormat(
            "raw_output does not look like an Azure Document Intelligence result"
        )
    status = raw.get("status")
    if status is not None and status != "succeeded":
        raise MalformedExport(f"azure ocr status is {status!r}, expected 'succeeded'")

    pages = []
    for i, pg in enumerate(ar["pages"]):
        if not isinstance(pg, dict):
            raise MalformedExport("azure page entry is not an object")
        w, h = pg.get("width"), pg.get("height")
        if not w or not h:
            raise MalformedExport(f"azure page {i + 1} has no width/height")
        words = [wd for wd in pg.get("words") or [] if isinstance(wd, dict)]
        lines = []
        for ln in pg.get("lines") or []:
            polygon = ln.get("polygon") or []
            if len(polygon) < 8:
                continue  # cannot place a line without a usable polygon
            xs, ys = polygon[0::2], polygon[1::2]
            # Azure scores words only: a line's confidence is the mean of the words whose
            # span falls inside the line's own span.
            spans = ln.get("spans") or []
            lo = spans[0]["offset"] if spans else None
            hi = lo + sum(sp["length"] for sp in spans) if spans else None
            confs = [
                confidence(wd["confidence"], 1) for wd in words
                if lo is not None and wd.get("confidence") is not None
                and lo <= wd["span"]["offset"] < hi
            ]
            confs = [c for c in confs if c is not None]
            lines.append(
                line(
                    index=len(lines),
                    text=ln.get("content") or "",
                    box=bbox(min(xs) / w, min(ys) / h, max(xs) / w, max(ys) / h),
                    conf=round(sum(confs) / len(confs), 4) if confs else None,
                    source="words_mean",
                )
            )
        pages.append(
            page(
                pg.get("pageNumber", i + 1),
                lines,
                native_size=(w, h),
                unit=pg.get("unit", "inch"),
                size_source="page",
            )
        )
    return pages


# -------------------------------------------------------------------------- rapidocr
# Locally OCRed PDF/image uploads (app/ocr.py). raw_output is a list of pages, each
# {"image_size": [w, h], "pdf_page_size": [w, h] (pdfs only), "result": [[quad, text, score]]}
# where quad is the engine's 4-corner box in rendered pixels and score is already 0-1.


def parse_rapidocr(raw):
    if not isinstance(raw, list):
        raise UnsupportedOcrFormat("raw_output is not a rapidocr page list")
    pages = []
    for i, p in enumerate(raw):
        if not isinstance(p, dict):
            raise MalformedExport(f"rapidocr page {i + 1} is not an object")
        w, h = p.get("image_size") or (None, None)
        if not w or not h:
            raise MalformedExport(f"rapidocr page {i + 1} has no image_size")
        lines = []
        for quad, text, score in p.get("result") or []:
            xs = [pt[0] for pt in quad]
            ys = [pt[1] for pt in quad]
            lines.append(
                line(
                    index=len(lines),
                    text=text or "",
                    box=bbox(min(xs) / w, min(ys) / h, max(xs) / w, max(ys) / h),
                    conf=confidence(score, 1),  # rapidocr scores are already 0-1
                    source="engine",
                )
            )
        pdf_size = p.get("pdf_page_size")
        pages.append(
            page(
                i + 1,
                lines,
                native_size=(pdf_size[0], pdf_size[1]) if pdf_size else (w, h),
                unit="point" if pdf_size else "pixel",
                size_source="rendered",
            )
        )
    return pages


PARSERS = {
    "aws-textract": parse_textract,
    "tesseract": parse_tesseract,
    "azure-document-intelligence": parse_azure,
    "rapidocr": parse_rapidocr,
}


def normalise(envelope, *, content_sha256=None, normalised_at=None):
    """Normalise one OCR export envelope into the engine-agnostic document shape."""
    if not isinstance(envelope, dict):
        raise UnsupportedOcrFormat("JSON is not an OCR export envelope")

    ocr, source = envelope.get("ocr"), envelope.get("source")

    if not isinstance(ocr, dict) or not isinstance(source, dict):
        raise UnsupportedOcrFormat("JSON has no 'ocr'/'source' export envelope")

    engine = ocr.get("engine")
    parser = PARSERS.get(engine)

    if parser is None:
        raise UnsupportedOcrFormat(f"unsupported ocr.engine: {engine!r}")

    try:
        pages = parser(envelope.get("raw_output"))
    except NormaliseError:
        raise
    except (KeyError, TypeError, ValueError, AttributeError) as exc:
        # right engine, broken payload: a missing key or wrong type is a 422, not a 500
        raise MalformedExport(f"{engine} raw_output is malformed: {type(exc).__name__}") from exc
    if not any(pg["lines"] for pg in pages):
        raise MalformedExport("export contains no OCR lines")

    return {
        "normaliser_version": NORMALISER_VERSION,
        "normalised_at": normalised_at
        or datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
        "source": {
            "original_filename": source.get("original_filename"),
            "mime_type": source.get("mime_type"),
            "source_sha256": source.get("source_sha256"),
            "country_code": source.get("country_code"),
            "export_format_version": envelope.get("export_format_version"),
            "content_sha256": content_sha256,
        },
        "ocr": {
            "engine": engine,
            "engine_version": ocr.get("engine_version"),
            "processed_at": ocr.get("processed_at"),
        },
        "pages": pages,
    }
