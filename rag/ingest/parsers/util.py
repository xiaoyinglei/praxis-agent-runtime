from __future__ import annotations

import re
from pathlib import Path

_WHITESPACE_RE = re.compile(r"\s+")
_HEADING_RE = re.compile(r"^#{1,6}\s+(.+?)\s*$")
_SLUG_RE = re.compile(r"[^a-z0-9\u4e00-\u9fff]+")

def normalize_whitespace(text: str) -> str:

    return _WHITESPACE_RE.sub(" ", text).strip()

def slugify(text: str) -> str:

    normalized = normalize_whitespace(text).lower()

    slug = _SLUG_RE.sub("-", normalized).strip("-")

    return slug or "section"

def extract_heading_text(line: str) -> tuple[int, str] | None:

    match = _HEADING_RE.match(line)

    if match is None:

        return None

    return len(line) - len(line.lstrip("#")), normalize_whitespace(match.group(1))

def default_title_from_location(location: str) -> str:

    path = Path(location)

    if path.name:

        stem = path.stem

        if stem:

            return stem

    cleaned = location.rstrip("/")

    if cleaned:

        return cleaned.rsplit("/", 1)[-1] or "document"

    return "document"
