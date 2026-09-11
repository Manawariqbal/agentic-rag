from pathlib import Path

from app.ingestion.parser import DocumentParser
from app.ingestion.chunker import ContextualChunker
from app.rag.embedding import EmbeddingProvider
from app.rag.vector_store import VectorStore


class IngestionPipeline:
    """
    End-to-end document ingestion pipeline.

    PDF
      -> Docling
      -> Markdown
      -> Contextual chunks
      -> Embeddings
      -> Vector store
    """

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
        path = Path(file_path)

        if not path.exists():
            raise FileNotFoundError(
                f"Document not found: {file_path}"
            )

        print(f"\n{'=' * 60}")
        print(f"Ingesting: {path.name}")
        print(f"{'=' * 60}")

        # 1. Parse document using Docling
        parsed_document = self.parser.parse(str(path))

        filename = parsed_document["filename"]
        content = parsed_document["content"]

        print(f"Parsed document: {filename}")
        print(f"Content length: {len(content)} characters")

        # 2. Contextual chunking
        chunks = self.chunker.chunk(
            content=content,
            filename=filename,
        )

        print(f"Created {len(chunks)} chunks")

        if not chunks:
            return 0

        # 3. Generate embeddings
        texts = [chunk.text for chunk in chunks]

        print("Generating embeddings...")

        embeddings = self.embedding_provider.embed_batch(texts)

        print(
            f"Generated {len(embeddings)} embeddings "
            f"of dimension {len(embeddings[0])}"
        )

        # 4. Store in PostgreSQL + pgvector
        print("Storing vectors in PostgreSQL...")

        self.vector_store.add_chunks(
            chunks=chunks,
            embeddings=embeddings,
        )

        print(f"Successfully ingested {filename}")

        return len(chunks)

    def ingest_directory(self, directory: str) -> dict[str, int]:
        directory_path = Path(directory)

        if not directory_path.exists():
            raise FileNotFoundError(
                f"Directory not found: {directory}"
            )

        pdf_files = sorted(directory_path.glob("*.pdf"))

        if not pdf_files:
            print(f"No PDF files found in {directory}")
            return {}

        results = {}

        for file_path in pdf_files:
            try:
                count = self.ingest_file(str(file_path))
                results[file_path.name] = count

            except Exception as exc:
                print(
                    f"Failed to ingest {file_path.name}: {exc}"
                )
                results[file_path.name] = 0

        return results