from langchain_core.documents import Document
from splitter import split_documents

LONG_PAGE = Document(
    page_content="word " * 1000,
    metadata={"source": "test.pdf", "page": 3},
)

def test_long_page_is_split_into_many_chunks():
    chunks = split_documents([LONG_PAGE])

    assert len(chunks) > 1
    assert all(len(chunk.page_content) <= 1000 for chunk in chunks)


def test_chunks_keep_source_and_page():
    chunks = split_documents([LONG_PAGE])

    for chunk in chunks:
        assert chunk.metadata["source"] == "test.pdf"
        assert chunk.metadata["page"] == 3


def test_short_page_stays_single_chunk():
    short = Document(page_content="Short text.", metadata={"page": 1})

    assert len(split_documents([short])) == 1