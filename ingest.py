from pathlib import Path

from loader import DocumentsLoader
from splitter import split_documents
from vector_store import TABLE_NAME, VECTOR_SIZE, get_engine, get_vector_store

PDF_PATH = Path(__file__).parent / "data" / "bell_palsy.pdf"


def main():
    docs = DocumentsLoader(PDF_PATH).load()
    chunks = split_documents(docs)

    engine = get_engine()
    engine.init_vectorstore_table(
        table_name=TABLE_NAME,
        vector_size=VECTOR_SIZE,
        overwrite_existing=True,  # rebuild the index from scratch on every run
    )
    store = get_vector_store(engine)
    store.add_documents(chunks)
    print(f"Indexed {len(chunks)} chunks from {len(docs)} pages")

    for doc in store.similarity_search("How is Bell's palsy treated?", k=3):
        print(doc.metadata, "->", doc.page_content[:100])


if __name__ == "__main__":
    main()
