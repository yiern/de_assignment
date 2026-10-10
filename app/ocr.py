"""Optional OCR stage: a PDF or image upload is rasterised and OCRed
locally, then wrapped in the same envelope shape as the three engine exports
("rapidocr"). Everything downstream — normalise, load, embed — is unchanged."""

import hashlib
import io
from datetime import datetime, timezone
from importlib.metadata import version

import cv2
import numpy as np
import pymupdf
from PIL import Image, UnidentifiedImageError
from rapidocr_onnxruntime import RapidOCR

MAX_PAGES = 50  # ponytail: hard cap so one upload can't pin the CPU; raise for longer scans
DPI = 200       # A4 renders 1653x2339 px; ratios are DPI-independent

_IMAGE_MAGICS = (
    (b"\x89PNG\r\n\x1a\n", "image/png"),
    (b"\xff\xd8\xff", "image/jpeg"),
    (b"II*\x00", "image/tiff"),
    (b"MM\x00*", "image/tiff"),
    (b"BM", "image/bmp"),
)


class UnreadableDocument(Exception):
    """Magic bytes say PDF/image, but the payload cannot be decoded or OCRed."""


class DocumentTooLarge(Exception):
    """More pages than MAX_PAGES."""


def sniff(data: bytes) -> str | None:
    """MIME of a PDF/image upload by magic bytes, else None (caller falls back to JSON)."""
    if data.startswith(b"%PDF-"):
        return "application/pdf"
    for magic, mime in _IMAGE_MAGICS:
        if data.startswith(magic):
            return mime
    if data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return "image/webp"
    return None


def build_envelope(data: bytes, *, filename, mime_type: str) -> dict:
    """bytes -> synthetic rapidocr engine export (same contract as the three samples)."""
    pages = _render_pdf(data) if mime_type == "application/pdf" else _read_image(data)
    engine = _rapidocr()
    raw_pages = []
    for image, pdf_page_size in pages:
        result, _ = engine(image)
        raw_pages.append(
            {
                "image_size": [int(image.shape[1]), int(image.shape[0])],
                **({"pdf_page_size": pdf_page_size} if pdf_page_size else {}),
                "result": result or [],
            }
        )
    return {
        "export_format_version": "1.0",
        "source": {
            "original_filename": filename,
            "mime_type": mime_type,  # sniffed from the bytes, not what the client claimed
            "source_sha256": hashlib.sha256(data).hexdigest(),
        },
        "ocr": {
            "engine": "rapidocr",
            "engine_version": _VERSION,
            "processed_at": datetime.now(timezone.utc)
            .isoformat(timespec="seconds")
            .replace("+00:00", "Z"),
        },
        "raw_output": raw_pages,
    }


_VERSION = version("rapidocr-onnxruntime")
_engine = None


def _rapidocr():
    """Build the ONNX sessions once per process."""
    global _engine
    if _engine is None:
        _engine = RapidOCR()
    return _engine


def _render_pdf(data: bytes):
    try:
        doc = pymupdf.open(stream=data, filetype="pdf")
        n = doc.page_count
        if n > MAX_PAGES:
            raise DocumentTooLarge(f"{n} pages > cap {MAX_PAGES}")
        pages = []
        with doc:
            for pg in doc:
                pix = pg.get_pixmap(dpi=DPI)
                arr = np.frombuffer(pix.samples, np.uint8).reshape(
                    pix.height, pix.width, pix.n
                )
                if pix.n == 4:
                    arr = arr[:, :, :3]
                pages.append(
                    (
                        cv2.cvtColor(arr, cv2.COLOR_RGB2BGR),
                        [round(pg.rect.width, 4), round(pg.rect.height, 4)],  # points
                    )
                )
    except DocumentTooLarge:
        raise
    except Exception as exc:  # pymupdf raises a zoo of exception types
        raise UnreadableDocument(f"cannot rasterise pdf: {type(exc).__name__}") from exc
    return pages


def _read_image(data: bytes):
    try:
        img = Image.open(io.BytesIO(data))
        img.load()  # force decode now: truncated files fail here, not mid-OCR
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        raise UnreadableDocument(f"cannot decode image: {type(exc).__name__}") from exc
    # ponytail: EXIF orientation ignored — synthetic scans don't carry it
    return [(cv2.cvtColor(np.asarray(img.convert("RGB")), cv2.COLOR_RGB2BGR), None)]
