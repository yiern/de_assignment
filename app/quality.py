"""Automated checks: per-page OCR-confidence flags and reference-field validation.

Runs inside the load stage's transaction. Input is the normalised page/line shape;
output is quality flag strings plus extracted_fields rows (raw, value, status).
Extraction lives here because a field must be located before it can be validated.

Confidence: the normaliser is the single conversion point and canonicalises every engine's
score to 0-1, so this module compares against one scale. Pages with no confidence values at
all get an explicit confidence_unavailable flag instead of an invented number.
"""

import re
from datetime import datetime
from typing import NamedTuple

LOW_CONFIDENCE_THRESHOLD = 0.9  # canonical 0-1 scale, set by the normaliser

# Amount grammar: optional currency/whitespace, digits with , groups, optional decimals.
_AMOUNT_RE = re.compile(r"^[^0-9.,-]*([0-9][0-9,]*(?:\.\d+)?)[^0-9.,-]*$")
# Date grammar: DD/MM/YYYY, optionally followed by HH:MM (the *_date_time fields).
_DATE_RE = re.compile(r"^(\d{2}/\d{2}/\d{4})(?:\s+(\d{1,2}:\d{2}))?$")
_MC_DAYS_RE = re.compile(r"period of (\d+) day")


class Rule(NamedTuple):
    field: str
    anchor: str | re.Pattern  # "letterhead" = first line of the first page
    kind: str                 # key into check(): amount, date, datetime, mc_days, provider, ...
    next_line: bool = False   # value is always the line after the anchor line


# The reference field set as data, so a new document type is table rows, not code. A string
# anchor is matched case-insensitively at line start; the value is the text after ':' on
# that line, else the next line. next_line forces the next-line value (labels whose own text
# ends in the value word, e.g. "Re: Patient").
FIELD_RULES = {
    "receipt": [
        Rule("claimant_name", "Patient:", "text"),
        Rule("claimant_address", "Address:", "text"),
        Rule("claimant_date_of_birth", "DOB:", "date"),
        Rule("provider_name", "letterhead", "provider"),
        Rule("tax_amount", re.compile(r"^gst \d+%", re.IGNORECASE), "amount"),
        Rule("total_amount", "TOTAL", "amount"),
    ],
    "medical_certificate": [
        Rule("claimant_name", "Patient Name:", "text"),
        Rule("claimant_address", "Address:", "text"),
        Rule("claimant_date_of_birth", "Date of Birth:", "date"),
        Rule("diagnosis_name", "Diagnosis:", "text"),
        Rule("discharge_date_time", "Discharge Date/Time:", "datetime"),
        Rule("icd_code", "ICD-10 Code:", "text"),
        Rule("provider_name", "letterhead", "provider"),
        Rule("submission_date_time", "Submission Date/Time:", "datetime"),
        Rule("date_of_mc", "Date of MC:", "date"),
        Rule("mc_days", _MC_DAYS_RE, "mc_days"),
    ],
    "referral_letter": [
        Rule("claimant_name", "Re: Patient", "text", next_line=True),
        Rule("provider_name", "Referring provider:", "provider"),
        Rule("signature_presence", "Signature:", "signature"),
        Rule("total_amount_paid", "Total Amount Paid:", "amount"),
        Rule("total_approved_amount", "Total Approved Amount:", "amount"),
        Rule("total_requested_amount", "Total Requested Amount:", "amount"),
    ],
}


def get_amount(raw: str) -> str | None:
    """'S$93.20' -> '9320', 'SGD 1,200.00' -> '120000': minor units, symbols/separators/decimal point removed."""
    match = _AMOUNT_RE.match(raw.strip()) if raw else None
    return str(int(match.group(1).replace(",", "").replace(".", ""))) if match else None


def get_date_value(raw: str) -> str | None:
    """'05/09/2026' -> itself; '02/09/2026 10:20' -> '02/09/2026'; anything unparseable -> None."""
    match = _DATE_RE.match(raw.strip()) if raw else None
    if not match:
        return None
    try:
        datetime.strptime(match.group(1), "%d/%m/%Y")
        if match.group(2):
            datetime.strptime(match.group(2), "%H:%M")
    except ValueError:
        return None
    return match.group(1)


def get_datetime_value(raw: str) -> str | None:
    """'02/09/2026 10:20' -> itself; a bare date is accepted; anything unparseable -> None."""
    date = get_date_value(raw)
    if date is None:
        return None
    clock = _DATE_RE.match(raw.strip()).group(2)
    return f"{date} {datetime.strptime(clock, '%H:%M'):%H:%M}" if clock else date


def get_mc_days(raw: str) -> str | None:
    """'...period of 2 day(s)...' -> '2'; \\d+ is inherently non-negative."""
    match = _MC_DAYS_RE.search(raw) if raw else None
    return str(int(match.group(1))) if match else None


_PARSERS = {
    "amount": get_amount,
    "date": get_date_value,
    "datetime": get_datetime_value,
    "mc_days": get_mc_days,
}


def check(kind: str, raw: str) -> tuple[str | None, str]:
    """One value verdict: (value, validation_status). raw is non-empty here."""
    if kind in _PARSERS:
        value = _PARSERS[kind](raw)
        return (value, "valid") if value is not None else (None, "invalid")
    if kind == "provider":
        # provider_name must not contain the planted 'Fullerton Health' footer text.
        # A failing value is NULL (it must never reach consumers); raw_value keeps the evidence.
        if "fullerton health" in raw.lower():
            return (None, "invalid")
        return (raw.strip(), "valid")
    if kind == "signature":
        return ("true", "valid") if "[signed]" in raw.lower() else (None, "invalid")
    return (raw.strip(), "valid")  # text: any non-empty value is a valid value


def get_page_flags(lines_by_page: dict[int, list[dict]]) -> list[str]:
    """One flag per page below threshold, one per page with no confidence data."""
    flags = []
    for page_number in sorted(lines_by_page):
        confidences = [line["confidence"] for line in lines_by_page[page_number]
                       if line.get("confidence") is not None]
        if not confidences:
            flags.append(f"confidence_unavailable:page_{page_number}")
        else:
            avg = sum(confidences) / len(confidences)
            if avg < LOW_CONFIDENCE_THRESHOLD:
                flags.append(f"low_confidence:page_{page_number}:avg={avg:.1f}")
    return flags


def get_anchor_value(lines: list[dict], anchor, next_line: bool = False) -> str | None:
    """Value text for a label anchor: after the ':' on the anchor line, else the next line.
    A compiled-regex anchor matches anywhere in the line (e.g. r'^gst \\d+%')."""
    needle = anchor.lower() if isinstance(anchor, str) else None
    for index, line in enumerate(lines):
        text = line["text"].strip()
        if not (text.lower().startswith(needle) if needle else anchor.search(text)):
            continue
        if next_line:
            candidate = ""
        elif ":" in line["text"]:
            candidate = line["text"].split(":", 1)[1].strip()
        else:  # label and value share a line without a colon ("TOTAL S$93.20")
            candidate = text[len(anchor) if needle else anchor.search(text).end():].strip()
        if not candidate and index + 1 < len(lines):
            candidate = lines[index + 1]["text"].strip()
        return candidate or None
    return None


def locate(rule: Rule, lines: list[dict], first_page: list[dict]) -> str | None:
    """The raw OCR string a rule points at, or None when the document lacks it."""
    if rule.anchor == "letterhead":
        return (first_page[0]["text"].strip() or None) if first_page else None
    if rule.kind == "mc_days":  # the value sits inside a sentence: take the matched phrase
        return next((m.group(0) for ln in lines if (m := rule.anchor.search(ln["text"]))), None)
    return get_anchor_value(lines, rule.anchor, rule.next_line)


def extract_fields(lines_by_page: dict[int, list[dict]], document_type: str | None) -> list[dict]:
    """One extracted_fields row per reference field; missing labels are stored as
    missing, malformed values as invalid - rows are never dropped."""
    pages = sorted(lines_by_page)
    first_page = lines_by_page[pages[0]] if pages else []
    all_lines = [line for page in pages for line in lines_by_page[page]]
    rows = []
    for rule in FIELD_RULES.get(document_type or "", []):
        raw = locate(rule, all_lines, first_page)
        value, status = check(rule.kind, raw) if raw is not None else (None, "missing")
        rows.append({"field_name": rule.field, "raw_value": raw,
                     "value": value, "validation_status": status})
    return rows
