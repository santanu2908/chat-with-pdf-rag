"""PDF text extraction with page tracking.

We keep page numbers so cited answers can point to a real page.
"""
import tempfile
import os
from pypdf import PdfReader
from typing import List, Tuple


def load_pdf(file_bytes: bytes, tmp_path: str | None = None) -> List[Tuple[str, int]]:
    """Extract text from a PDF, returning (text, page_number) per page.

    Page numbers are 1-indexed for human readability.
    Empty pages are skipped (scanned PDFs without OCR will return empty).
    """
    # if no tmp_path provided; write the bytes to a temp file
    # this is needed because pypdf works with file paths, not byte streams
    if tmp_path is None:
        tmp_path = os.path.join(tempfile.gettempdir(), "_upload.pdf")

    with open(tmp_path, "wb") as f:
        f.write(file_bytes)

    reader = PdfReader(tmp_path)
    pages: List[Tuple[str, int]] = []

    #reader.pages is a list-like object — each element is a PageObject representing one page of the PDF.
    for i, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        text = text.strip()
        if text:
            pages.append((text, i))

    return pages
