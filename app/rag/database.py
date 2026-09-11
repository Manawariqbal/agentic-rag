from app.config import settings
from app.rag.pgvector_store import PGVectorStoreAdapter


class DatabaseManager:

    def __init__(self):
        self.vector_store = PGVectorStoreAdapter(
            database=settings.postgres_database,
            host=settings.postgres_host,
            port=settings.postgres_port,
            user=settings.postgres_user,
            password=settings.postgres_password,
            table_name="rag_documents",
            embed_dim=settings.embedding_dimension
        )

    def get_vector_store(self):
        return self.vector_store