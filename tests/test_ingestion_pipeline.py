from pathlib import Path

import pytest

from app.ingestion.chunker import ContextualChunker, DocumentChunk
from app.ingestion.pipeline import IngestionPipeline
from app.rag.embedding import MockEmbeddingProvider
from app.rag.vector_store import InMemoryVectorStore


class DummyParser:
    def __init__(
        self,
        content="# Annual Leave\n\nEmployees receive 20 days.",
    ):
        self.content = content
        self.last_file_path = None

    def parse(self, file_path: str) -> dict:
        self.last_file_path = file_path

        return {
            "filename": Path(file_path).name,
            "content": self.content,
        }


class DummyChunker:
    def __init__(self, chunks=None):
        self.chunks = (
            chunks
            if chunks is not None
            else [
                DocumentChunk(
                    text="Employees receive 20 days of annual leave.",
                    metadata={
                        "source": "leave_policy.pdf",
                        "section": "Annual Leave",
                    },
                ),
                DocumentChunk(
                    text="Employees receive 10 days of sick leave.",
                    metadata={
                        "source": "leave_policy.pdf",
                        "section": "Sick Leave",
                    },
                ),
            ]
        )

        self.last_content = None
        self.last_filename = None

    def chunk(self, content, filename):
        self.last_content = content
        self.last_filename = filename
        return self.chunks


class DummyEmbeddingProvider:
    def __init__(self, dimensions=4):
        self.dimensions = dimensions
        self.last_texts = None

    def embed_batch(self, texts):
        self.last_texts = texts

        return [
            [float(index + 1)] * self.dimensions
            for index in range(len(texts))
        ]


class DummyVectorStore:
    def __init__(self):
        self.last_chunks = None
        self.last_embeddings = None
        self.documents = []

    def add_chunks(self, chunks, embeddings=None):
        self.last_chunks = chunks
        self.last_embeddings = embeddings

        for chunk, embedding in zip(chunks, embeddings):
            self.documents.append(
                {
                    "text": chunk.text,
                    "metadata": chunk.metadata,
                    "embedding": embedding,
                }
            )


def create_pipeline(
    parser=None,
    chunker=None,
    embedding_provider=None,
    vector_store=None,
):
    parser = parser or DummyParser()
    chunker = chunker or DummyChunker()
    embedding_provider = (
        embedding_provider
        or DummyEmbeddingProvider()
    )
    vector_store = vector_store or DummyVectorStore()

    pipeline = IngestionPipeline(
        parser=parser,
        chunker=chunker,
        embedding_provider=embedding_provider,
        vector_store=vector_store,
    )

    return (
        pipeline,
        parser,
        chunker,
        embedding_provider,
        vector_store,
    )


def create_temp_pdf(tmp_path):
    pdf_path = tmp_path / "leave_policy.pdf"
    pdf_path.write_bytes(b"%PDF-1.4 dummy content")
    return pdf_path


def test_ingest_file_returns_chunk_count(tmp_path):
    pdf_path = create_temp_pdf(tmp_path)

    pipeline, _, _, _, _ = create_pipeline()

    count = pipeline.ingest_file(str(pdf_path))

    assert count == 2


def test_ingest_file_calls_parser(tmp_path):
    pdf_path = create_temp_pdf(tmp_path)

    pipeline, parser, _, _, _ = create_pipeline()

    pipeline.ingest_file(str(pdf_path))

    assert parser.last_file_path == str(pdf_path)


def test_ingest_file_passes_parsed_content_to_chunker(tmp_path):
    pdf_path = create_temp_pdf(tmp_path)

    parser = DummyParser(
        content="# Annual Leave\n\nEmployees receive 20 days."
    )

    pipeline, _, chunker, _, _ = create_pipeline(
        parser=parser
    )

    pipeline.ingest_file(str(pdf_path))

    assert chunker.last_content == parser.content
    assert chunker.last_filename == "leave_policy.pdf"


def test_ingest_file_generates_embeddings_for_chunks(tmp_path):
    pdf_path = create_temp_pdf(tmp_path)

    pipeline, _, chunker, embedding_provider, _ = (
        create_pipeline()
    )

    pipeline.ingest_file(str(pdf_path))

    expected_texts = [
        chunk.text
        for chunk in chunker.chunks
    ]

    assert embedding_provider.last_texts == expected_texts


def test_ingest_file_stores_chunks_and_embeddings(tmp_path):
    pdf_path = create_temp_pdf(tmp_path)

    pipeline, _, chunker, embedding_provider, vector_store = (
        create_pipeline()
    )

    pipeline.ingest_file(str(pdf_path))

    assert vector_store.last_chunks == chunker.chunks

    assert (
        vector_store.last_embeddings
        == embedding_provider.embed_batch(
            [chunk.text for chunk in chunker.chunks]
        )
    )


def test_ingest_file_stores_all_documents(tmp_path):
    pdf_path = create_temp_pdf(tmp_path)

    pipeline, _, _, _, vector_store = create_pipeline()

    count = pipeline.ingest_file(str(pdf_path))

    assert count == len(vector_store.documents)
    assert len(vector_store.documents) == 2


def test_ingest_file_missing_file_raises_error(tmp_path):
    missing_file = tmp_path / "missing.pdf"

    pipeline, _, _, _, _ = create_pipeline()

    with pytest.raises(
        FileNotFoundError,
        match="Document not found",
    ):
        pipeline.ingest_file(str(missing_file))


def test_ingest_file_with_no_chunks_returns_zero(tmp_path):
    pdf_path = create_temp_pdf(tmp_path)

    empty_chunker = DummyChunker(chunks=[])

    pipeline, _, _, embedding_provider, vector_store = (
        create_pipeline(
            chunker=empty_chunker
        )
    )

    count = pipeline.ingest_file(str(pdf_path))

    assert count == 0
    assert embedding_provider.last_texts is None
    assert vector_store.last_chunks is None


def test_ingest_directory_returns_results_for_pdf_files(tmp_path):
    (tmp_path / "b_policy.pdf").write_bytes(
        b"%PDF-1.4 dummy"
    )

    (tmp_path / "a_policy.pdf").write_bytes(
        b"%PDF-1.4 dummy"
    )

    pipeline, _, _, _, _ = create_pipeline()

    results = pipeline.ingest_directory(str(tmp_path))

    assert results == {
        "a_policy.pdf": 2,
        "b_policy.pdf": 2,
    }


def test_ingest_directory_processes_only_pdf_files(tmp_path):
    (tmp_path / "leave.pdf").write_bytes(
        b"%PDF-1.4 dummy"
    )

    (tmp_path / "notes.txt").write_text(
        "This is not a PDF."
    )

    pipeline, _, _, _, _ = create_pipeline()

    results = pipeline.ingest_directory(str(tmp_path))

    assert results == {
        "leave.pdf": 2,
    }


def test_ingest_directory_returns_empty_dict_when_no_pdfs(
    tmp_path,
):
    (tmp_path / "notes.txt").write_text(
        "No PDF files here."
    )

    pipeline, _, _, _, _ = create_pipeline()

    results = pipeline.ingest_directory(str(tmp_path))

    assert results == {}


def test_ingest_directory_missing_directory_raises_error(
    tmp_path,
):
    missing_directory = tmp_path / "does_not_exist"

    pipeline, _, _, _, _ = create_pipeline()

    with pytest.raises(
        FileNotFoundError,
        match="Directory not found",
    ):
        pipeline.ingest_directory(
            str(missing_directory)
        )


def test_ingest_directory_continues_after_file_failure(
    tmp_path,
):
    (tmp_path / "01_broken.pdf").write_bytes(
        b"%PDF-1.4 dummy"
    )

    (tmp_path / "02_valid.pdf").write_bytes(
        b"%PDF-1.4 dummy"
    )

    class FailingParser:
        def parse(self, file_path):
            if Path(file_path).name == "01_broken.pdf":
                raise RuntimeError("Parsing failed")

            return {
                "filename": "02_valid.pdf",
                "content": "# Valid\n\nValid document.",
            }

    pipeline, _, _, _, _ = create_pipeline(
        parser=FailingParser()
    )

    results = pipeline.ingest_directory(str(tmp_path))

    assert results == {
        "01_broken.pdf": 0,
        "02_valid.pdf": 2,
    }


def test_ingest_directory_preserves_sorted_processing_order(
    tmp_path,
):
    (tmp_path / "c.pdf").write_bytes(
        b"%PDF-1.4 dummy"
    )

    (tmp_path / "a.pdf").write_bytes(
        b"%PDF-1.4 dummy"
    )

    (tmp_path / "b.pdf").write_bytes(
        b"%PDF-1.4 dummy"
    )

    class TrackingParser:
        def __init__(self):
            self.processed = []

        def parse(self, file_path):
            self.processed.append(
                Path(file_path).name
            )

            return {
                "filename": Path(file_path).name,
                "content": "# Test\n\nContent.",
            }

    parser = TrackingParser()

    pipeline, _, _, _, _ = create_pipeline(
        parser=parser
    )

    pipeline.ingest_directory(str(tmp_path))

    assert parser.processed == [
        "a.pdf",
        "b.pdf",
        "c.pdf",
    ]


def test_ingestion_pipeline_with_real_in_memory_components(
    tmp_path,
):
    pdf_path = tmp_path / "policy.pdf"
    pdf_path.write_bytes(b"%PDF-1.4 dummy")

    parser = DummyParser(
        content="""
# Annual Leave

Employees receive 20 days of annual leave.

# Sick Leave

Employees receive 10 days of sick leave.
"""
    )

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

    count = pipeline.ingest_file(
        str(pdf_path)
    )

    assert count == 2
    assert len(vector_store.documents) == 2

    assert all(
        len(document["embedding"]) == 384
        for document in vector_store.documents
    )

    assert (
        vector_store.documents[0]["metadata"]["source"]
        == "policy.pdf"
    )