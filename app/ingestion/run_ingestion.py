from app.config import settings
from app.ingestion.parser import DocumentParser
from app.ingestion.chunker import ContextualChunker
from app.ingestion.pipeline import IngestionPipeline
from app.rag.embedding import OllamaEmbeddingProvider
from app.rag.pgvector_store import PGVectorStoreAdapter


def main():
    print("\nStarting Agentic RAG ingestion...\n")

    # Document processing
    parser = DocumentParser()

    # Contextual chunking
    chunker = ContextualChunker(
        chunk_size=800,
        chunk_overlap=100,
    )

    # Ollama embeddings
    embedding_provider = OllamaEmbeddingProvider(
        model=settings.embedding_model,
        base_url=settings.ollama_base_url,
    )

    # PostgreSQL + pgvector
    vector_store = PGVectorStoreAdapter(
        table_name="rag_documents_1024",
        embed_dim=settings.embedding_dimensions,
    )

    # Complete pipeline
    pipeline = IngestionPipeline(
        parser=parser,
        chunker=chunker,
        embedding_provider=embedding_provider,
        vector_store=vector_store,
    )

    # Ingest documents
    results = pipeline.ingest_directory(
        directory="documents"
    )

    print("\n" + "=" * 60)
    print("INGESTION SUMMARY")
    print("=" * 60)

    total = 0

    for filename, count in results.items():
        print(f"{filename}: {count} chunks")
        total += count

    print("-" * 60)
    print(f"Total chunks: {total}")
    print("=" * 60)


if __name__ == "__main__":
    main()