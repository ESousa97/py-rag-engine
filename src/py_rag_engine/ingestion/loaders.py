from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from pypdf import PdfReader


@dataclass(frozen=True, slots=True)
class LoadedPage:
    text: str
    page: int | None


def load_pdf(path: Path) -> list[LoadedPage]:
    reader = PdfReader(str(path))
    pages: list[LoadedPage] = []
    for i, page in enumerate(reader.pages, start=1):
        raw = page.extract_text() or ""
        text = raw.strip()
        if text:
            pages.append(LoadedPage(text=text, page=i))
    return pages


def load_markdown(path: Path) -> list[LoadedPage]:
    raw = path.read_text(encoding="utf-8")
    text = raw.strip()
    if not text:
        return []
    return [LoadedPage(text=text, page=None)]
