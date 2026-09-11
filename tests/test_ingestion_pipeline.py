from app.ingestion.chunker import ContextualChunker
from app.ingestion.parser import DocumentParser
from app.ingestion.pipeline import IngestionPipeline
from app.rag.embedding import MockEmbeddingProvider
from app.rag.vector_store import InMemoryVectorStore


def main():
    parser = DocumentParser()

    chunker = ContextualChunker(
        chunk_size=800,
        chunk_overlap=100,
    )

    embedding_provider = MockEmbeddingProvider(
        dimensions=384
    )

    vector_store = InMemoryVectorStore(
        embedding_provider=embedding_provider
    )

    pipeline = IngestionPipeline(
        parser=parser,
        chunker=chunker,
        embedding_provider=embedding_provider,
        vector_store=vector_store,
    )

    results = pipeline.ingest_directory("documents")

    print("\nINGESTION RESULTS")
    print("=================")

    for filename, count in results.items():
        print(f"{filename}: {count} chunks")

    print(f"\nTotal chunks: {len(vector_store.documents)}")


if __name__ == "__main__":
    main()