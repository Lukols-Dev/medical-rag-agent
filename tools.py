import uuid
from functools import lru_cache

from deepagents.backends import StateBackend
from langchain.tools import tool
from langchain_postgres import PGVectorStore

from vector_store import get_engine, get_vector_store

SEARCH_RESULTS = 4

backend = StateBackend()


@lru_cache(maxsize=1)
def get_store() -> PGVectorStore:
    """
    Connect to the vector store on first use and reuse the connection afterwards.
    """
    return get_vector_store(get_engine())


@tool(parse_docstring=True)
def search_documents(query: str) -> str:
    """Search the medical knowledge base and save matching chunks to the agent filesystem.

    Args:
        query: Natural language search query in English.

    Returns:
        File paths where retrieved chunks were saved under /retrieved/.
    """
    retrieved_docs = get_store().similarity_search(query, k=SEARCH_RESULTS)
    batch_id = uuid.uuid4().hex[:8]
    uploads: list[tuple[str, bytes]] = []

    for index, doc in enumerate(retrieved_docs, start=1):
        path = f"/retrieved/{batch_id}/chunk_{index}.md"
        content = (
            f"# Source: {doc.metadata.get('source', 'unknown')}, "
            f"page {doc.metadata.get('page', '?')}\n\n"
            f"{doc.page_content}"
        )
        uploads.append((path, content.encode("utf-8")))

    backend.upload_files(uploads)
    saved_paths = [path for path, _ in uploads]
    return f"Saved {len(saved_paths)} chunks:\n" + "\n".join(saved_paths)
