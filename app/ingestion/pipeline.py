from pathlib import Path

from app.ingestion.parser import DocumentParser
from app.ingestion.chunker import ContextualChunker
from app.rag.embedding import EmbeddingProvider
from app.rag.vector_store import VectorStore


class IngestionPipeline:
    def __init__(
        self,
        parser: DocumentParser,
        chunker: ContextualChunker,
        embedding_provider: EmbeddingProvider,
        vector_store: VectorStore,
    ):
        self.parser = parser
        self.chunker = chunker
        self.embedding_provider = embedding_provider
        self.vector_store = vector_store

    def ingest_file(self, file_path: str) -> int:
        # 1. Parse document
        parsed_document = self.parser.parse(file_path)

        filename = parsed_document["filename"]
        content = parsed_document["content"]

        # 2. Create contextual chunks
        chunks = self.chunker.chunk(
            content=content,
            filename=filename,
        )

        if not chunks:
            return 0

        # 3. Generate embeddings
        embeddings = self.embedding_provider.embed_batch(
            [chunk.text for chunk in chunks]
        )

        # 4. Store chunks + embeddings
        self.vector_store.add_chunks(
            chunks=chunks,
            embeddings=embeddings,
        )

        return len(chunks)

    def ingest_directory(self, directory: str) -> dict[str, int]:
        directory_path = Path(directory)

        if not directory_path.exists():
            raise FileNotFoundError(
                f"Directory not found: {directory}"
            )

        if not directory_path.is_dir():
            raise ValueError(
                f"Path is not a directory: {directory}"
            )

        results = {}

        for file_path in sorted(directory_path.glob("*.pdf")):
            count = self.ingest_file(str(file_path))
            results[file_path.name] = count

        return results