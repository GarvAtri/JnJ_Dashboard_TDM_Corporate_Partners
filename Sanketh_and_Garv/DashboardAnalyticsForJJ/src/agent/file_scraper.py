"""Scrape uploaded files to extract column names, headings, and metadata."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import pandas as pd


def scrape_file(file_path: str) -> dict[str, Any]:
    """Dispatch to the appropriate scraper based on file extension.

    Returns a dict with standardised keys:
        name, type, rows, columns, column_names, sample_rows, summary, sheets (Excel only)
    """
    path = Path(file_path)
    if not path.exists():
        return {"name": path.name, "error": f"File not found: {path}"}

    ext = path.suffix.lower()
    try:
        if ext in (".xlsx", ".xls"):
            return _scrape_excel(path)
        if ext == ".json":
            return _scrape_json(path)
        if ext == ".pdf":
            return _scrape_pdf(path)
        return {"name": path.name, "error": f"Unsupported file type: {ext}"}
    except Exception as exc:
        return {"name": path.name, "error": str(exc)}


# ── Excel ────────────────────────────────────────────────────────

def _scrape_excel(path: Path) -> dict[str, Any]:
    """Extract sheet names, column names, dtypes, and sample rows."""
    xls = pd.ExcelFile(path)
    sheets_info: list[dict[str, Any]] = []

    for sheet_name in xls.sheet_names:
        df = pd.read_excel(xls, sheet_name=sheet_name, nrows=100)
        col_details = []
        for col in df.columns:
            dtype = str(df[col].dtype)
            non_null = int(df[col].notna().sum())
            unique = int(df[col].nunique())
            sample_vals = (
                df[col].dropna().head(3).astype(str).tolist()
            )
            col_details.append({
                "name": str(col),
                "dtype": dtype,
                "non_null_count": non_null,
                "unique_count": unique,
                "sample_values": sample_vals,
            })

        sample_rows = (
            df.head(5)
            .fillna("")
            .astype(str)
            .to_dict("records")
        )

        sheets_info.append({
            "sheet_name": sheet_name,
            "row_count": int(len(df)),
            "column_count": int(len(df.columns)),
            "column_names": [str(c) for c in df.columns],
            "columns": col_details,
            "sample_rows": sample_rows,
        })

    total_rows = sum(s["row_count"] for s in sheets_info)
    all_cols = []
    for s in sheets_info:
        all_cols.extend(s["column_names"])

    return {
        "name": path.name,
        "type": "Excel",
        "sheets": [s["sheet_name"] for s in sheets_info],
        "sheet_count": len(sheets_info),
        "total_rows": total_rows,
        "total_columns": len(set(all_cols)),
        "column_names": sorted(set(all_cols)),
        "sheet_details": sheets_info,
        "summary": (
            f"Excel workbook with {len(sheets_info)} sheet(s), "
            f"{total_rows} total rows, and {len(set(all_cols))} unique columns."
        ),
    }


# ── JSON ─────────────────────────────────────────────────────────

def _scrape_json(path: Path) -> dict[str, Any]:
    """Extract keys, structure, and sample entries from a JSON file."""
    raw = path.read_text(encoding="utf-8")
    data = json.loads(raw)

    if isinstance(data, list):
        return _scrape_json_array(path.name, data)
    if isinstance(data, dict):
        return _scrape_json_object(path.name, data)
    return {
        "name": path.name,
        "type": "JSON",
        "structure": type(data).__name__,
        "summary": f"JSON file contains a scalar value of type {type(data).__name__}.",
    }


def _scrape_json_array(name: str, data: list) -> dict[str, Any]:
    """Handle JSON that is an array of records."""
    record_count = len(data)
    # Collect all keys from dict records
    all_keys: set[str] = set()
    sample_records: list[dict] = []
    for item in data[:100]:
        if isinstance(item, dict):
            all_keys.update(item.keys())
            if len(sample_records) < 5:
                sample_records.append(
                    {k: _truncate(v) for k, v in item.items()}
                )

    if all_keys:
        # Infer types from first non-null occurrence
        key_types: dict[str, str] = {}
        for key in sorted(all_keys):
            for item in data[:50]:
                if isinstance(item, dict) and key in item and item[key] is not None:
                    key_types[key] = type(item[key]).__name__
                    break

        return {
            "name": name,
            "type": "JSON",
            "structure": "Array of Objects",
            "record_count": record_count,
            "column_names": sorted(all_keys),
            "column_count": len(all_keys),
            "column_types": key_types,
            "sample_rows": sample_records,
            "summary": (
                f"JSON array with {record_count} records and "
                f"{len(all_keys)} fields: {', '.join(sorted(all_keys)[:10])}"
                f"{'...' if len(all_keys) > 10 else ''}."
            ),
        }

    return {
        "name": name,
        "type": "JSON",
        "structure": "Array",
        "record_count": record_count,
        "element_types": list({type(i).__name__ for i in data[:50]}),
        "summary": f"JSON array with {record_count} elements.",
    }


def _scrape_json_object(name: str, data: dict) -> dict[str, Any]:
    """Handle JSON that is a single object / nested structure."""
    top_keys = list(data.keys())
    key_info: list[dict[str, Any]] = []
    for key in top_keys[:50]:
        val = data[key]
        info: dict[str, Any] = {"name": key, "type": type(val).__name__}
        if isinstance(val, list):
            info["length"] = len(val)
        elif isinstance(val, dict):
            info["sub_keys"] = list(val.keys())[:10]
        else:
            info["sample"] = _truncate(val)
        key_info.append(info)

    return {
        "name": name,
        "type": "JSON",
        "structure": "Object",
        "column_names": top_keys,
        "column_count": len(top_keys),
        "keys": key_info,
        "summary": (
            f"JSON object with {len(top_keys)} top-level keys: "
            f"{', '.join(top_keys[:10])}{'...' if len(top_keys) > 10 else ''}."
        ),
    }


# ── PDF ──────────────────────────────────────────────────────────

def _scrape_pdf(path: Path) -> dict[str, Any]:
    """Extract text headings, page count, and table-like structures from a PDF."""
    try:
        from reportlab.lib.pagesizes import letter  # noqa: F401 — verify reportlab available
    except ImportError:
        pass

    # Use PyPDF2 or fallback to pdfminer-like raw text extraction
    text_pages: list[str] = []
    page_count = 0

    try:
        import PyPDF2  # type: ignore[import-untyped]
        reader = PyPDF2.PdfReader(str(path))
        page_count = len(reader.pages)
        for page in reader.pages[:20]:
            text_pages.append(page.extract_text() or "")
    except ImportError:
        # Fallback: read raw bytes and extract ASCII text
        raw = path.read_bytes()
        text_content = raw.decode("latin-1", errors="ignore")
        # Rough page split
        pages = text_content.split("%%Page")
        page_count = max(len(pages) - 1, 1)
        text_pages = [_extract_readable(p) for p in pages[:20]]

    full_text = "\n".join(text_pages)
    headings = _extract_headings(full_text)
    tables = _detect_table_structures(full_text)
    word_count = len(full_text.split())

    table_note = f", {tables['table_count']} possible table(s)" if tables.get("table_count") else ""
    return {
        "name": path.name,
        "type": "PDF",
        "page_count": page_count,
        "word_count": word_count,
        "headings": headings[:30],
        "column_names": tables.get("column_names", []),
        "detected_tables": tables.get("table_count", 0),
        "sample_text": full_text[:1500].strip(),
        "summary": (
            f"PDF document with {page_count} page(s), ~{word_count} words, "
            f"{len(headings)} heading(s) detected{table_note}."
        ),
    }


# ── helpers ──────────────────────────────────────────────────────

def _truncate(value: Any, max_len: int = 80) -> str:
    s = str(value)
    return s if len(s) <= max_len else s[:max_len] + "…"


def _extract_readable(raw: str) -> str:
    """Keep only printable ASCII-ish content."""
    return re.sub(r"[^\x20-\x7E\n\r\t]", "", raw)


def _extract_headings(text: str) -> list[str]:
    """Heuristic heading extraction from PDF text.

    Looks for lines that are short, title-cased or ALL-CAPS,
    and preceded/followed by blank lines.
    """
    lines = text.split("\n")
    headings: list[str] = []
    for i, line in enumerate(lines):
        stripped = line.strip()
        if not stripped or len(stripped) > 120:
            continue
        prev_blank = i == 0 or not lines[i - 1].strip()
        is_upper = stripped.isupper() and len(stripped) > 2
        is_title = stripped.istitle() and len(stripped.split()) <= 10
        is_numbered = bool(re.match(r"^\d+[\.\)]\s+\S", stripped))
        if (prev_blank and (is_upper or is_title)) or is_numbered:
            headings.append(stripped)
    return headings


def _detect_table_structures(text: str) -> dict[str, Any]:
    """Detect potential table headers from tab/pipe/comma-aligned lines."""
    column_names: list[str] = []
    table_count = 0

    lines = text.split("\n")
    for i, line in enumerate(lines):
        # Pipe-delimited tables
        if "|" in line and line.count("|") >= 2:
            parts = [p.strip() for p in line.split("|") if p.strip()]
            if all(re.match(r"^[A-Za-z_][\w\s]*$", p) for p in parts) and len(parts) >= 2:
                column_names.extend(parts)
                table_count += 1
                continue
        # Tab-delimited header-like rows
        if "\t" in line:
            parts = [p.strip() for p in line.split("\t") if p.strip()]
            if len(parts) >= 3 and all(re.match(r"^[A-Za-z_][\w\s]*$", p) for p in parts):
                column_names.extend(parts)
                table_count += 1
                continue
        # Comma-separated header-like rows
        if "," in line and line.count(",") >= 2:
            parts = [p.strip() for p in line.split(",") if p.strip()]
            if all(re.match(r"^[A-Za-z_][\w\s]*$", p) for p in parts) and len(parts) >= 3:
                column_names.extend(parts)
                table_count += 1

    return {
        "column_names": sorted(set(column_names)),
        "table_count": table_count,
    }


def scrape_all_files(file_paths: list[str]) -> list[dict[str, Any]]:
    """Scrape multiple files and return combined results."""
    return [scrape_file(fp) for fp in file_paths]
