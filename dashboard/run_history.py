"""Scan outputs/reports/ and parse past backtest report markdown files."""
from __future__ import annotations

import re
from pathlib import Path
from typing import Optional


def scan_reports(reports_dir: Path) -> list[dict]:
    """Return sorted list of report metadata dicts (newest first).

    Each dict has keys: path, filename, title, date, size_kb.
    """
    if not reports_dir.exists():
        return []

    rows = []
    for p in sorted(reports_dir.glob("*.md"), reverse=True):
        meta = _parse_metadata(p)
        meta["path"] = str(p)
        meta["filename"] = p.name
        meta["size_kb"] = round(p.stat().st_size / 1024, 1)
        rows.append(meta)
    return rows


def read_report(path: str) -> str:
    """Return full markdown content of a report file."""
    try:
        return Path(path).read_text(encoding="utf-8")
    except Exception as e:
        return f"*Error reading report: {e}*"


def _parse_metadata(p: Path) -> dict:
    """Extract title and date from the first few lines of a markdown report."""
    meta = {"title": p.stem, "date": "", "horizon": None, "phase": None}
    try:
        lines = p.read_text(encoding="utf-8").splitlines()[:20]
        for line in lines:
            # Title from first H1
            if line.startswith("# ") and meta["title"] == p.stem:
                meta["title"] = line[2:].strip()
            # Date line: **Date:** YYYY-MM-DD
            if "**Date:**" in line or "Date:" in line:
                m = re.search(r"\d{4}-\d{2}-\d{2}", line)
                if m:
                    meta["date"] = m.group()
            # Horizon
            if "horizon" in line.lower():
                m = re.search(r"horizon[=\s]*(\d+)", line, re.IGNORECASE)
                if m:
                    meta["horizon"] = int(m.group(1))
            # Phase
            m = re.search(r"[Pp]hase\s*(\d+)", line)
            if m:
                meta["phase"] = int(m.group(1))
    except Exception:
        pass
    return meta
