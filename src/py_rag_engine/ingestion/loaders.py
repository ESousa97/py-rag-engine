from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from pypdf import PdfReader


@dataclass(frozen=True, slots=True)
class LoadedPage:
    """Text extracted from a source page or page-like document unit."""

    text: str
    page: int | None


def load_pdf(path: Path) -> list[LoadedPage]:
    """Extract non-empty text pages from a PDF file."""
    reader = PdfReader(str(path))
    pages: list[LoadedPage] = []
    for i, page in enumerate(reader.pages, start=1):
        raw = page.extract_text() or ""
        text = raw.strip()
        if text:
            pages.append(LoadedPage(text=text, page=i))
    return pages


def load_markdown(path: Path) -> list[LoadedPage]:
    """Load a Markdown file as a single page-less document unit."""
    raw = path.read_text(encoding="utf-8")
    text = raw.strip()
    if not text:
        return []
    return [LoadedPage(text=text, page=None)]
