from langchain_openai import OpenAIEmbeddings
from langchain_postgres import PGEngine, PGVectorStore

from config import settings

TABLE_NAME = "documents"
EMBEDDING_MODEL = "text-embedding-3-small"
# length of vectors produced by text-embedding-3-small
VECTOR_SIZE = 1536

def get_embeddings()-> OpenAIEmbeddings:
    return OpenAIEmbeddings(
        model=EMBEDDING_MODEL, 
        api_key=settings.openai_api_key
        )

def get_engine() -> PGEngine:
    return PGEngine.from_connection_string(url=settings.database_url)

def get_vector_store(engine: PGEngine) -> PGVectorStore:
    return PGVectorStore.create_sync(
        engine=engine,
        table_name=TABLE_NAME,
        embedding_service=get_embeddings()
        )