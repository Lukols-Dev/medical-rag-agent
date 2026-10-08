from pathlib import Path

import pypdf
import pytest

from loader import DocumentsLoader

PDF_PATH = Path(__file__).parent.parent / "data" / "bell_palsy.pdf"

def test_loads_one_document_per_page():
    docs = DocumentsLoader(PDF_PATH).load()

    assert len(docs) == 8
    assert all(doc.page_content.strip() for doc in docs)

def test_metadata_has_source_and_page():
    docs = DocumentsLoader(PDF_PATH).load()

    assert docs[0].metadata == {"source": "bell_palsy.pdf", "page": 1}
    assert [doc.metadata["page"] for doc in docs] == list(range(1, 9))

def test_missing_file_raises():
    with pytest.raises(FileNotFoundError):
        DocumentsLoader(Path("nie_istnieje.pdf"))

def test_skips_pages_without_text(tmp_path):
    blank_pdf = tmp_path / "blank.pdf"
    writer = pypdf.PdfWriter()
    writer.add_blank_page(width=200, height=200)
    writer.write(blank_pdf)

    assert DocumentsLoader(blank_pdf).load() == []