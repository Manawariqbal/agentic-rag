from app.ingestion.chunker import ContextualChunker


def test_chunk_creates_chunks_from_sections():
    content = """
# Annual Leave

Employees are entitled to 20 days of annual leave per year.

Employees should request leave through the HR system.

# Sick Leave

Employees are entitled to 10 days of sick leave per year.
"""

    chunker = ContextualChunker(
        chunk_size=200,
        chunk_overlap=30,
    )

    chunks = chunker.chunk(
        content=content,
        filename="leave_and_attendance_policy.pdf",
    )

    assert len(chunks) == 2

    assert chunks[0].metadata["source"] == (
        "leave_and_attendance_policy.pdf"
    )
    assert chunks[0].metadata["section"] == "Annual Leave"
    assert chunks[0].metadata["chunk_index"] == 0

    assert chunks[1].metadata["source"] == (
        "leave_and_attendance_policy.pdf"
    )
    assert chunks[1].metadata["section"] == "Sick Leave"
    assert chunks[1].metadata["chunk_index"] == 0


def test_chunk_adds_document_context():
    content = """
# Annual Leave

Employees receive 20 days of annual leave.
"""

    chunker = ContextualChunker(
        chunk_size=200,
        chunk_overlap=30,
    )

    chunks = chunker.chunk(
        content=content,
        filename="leave_policy.pdf",
    )

    assert len(chunks) == 1

    text = chunks[0].text

    assert "Document: leave_policy.pdf" in text
    assert "Section: Annual Leave" in text
    assert (
        "Context: This section contains information "
        "from the Annual Leave section of the document."
        in text
    )

    assert "Employees receive 20 days of annual leave." in text


def test_chunk_preserves_metadata():
    content = """
# Entitlement

Employees receive 20 days of annual leave.
"""

    chunker = ContextualChunker()

    chunks = chunker.chunk(
        content=content,
        filename="leave_policy.pdf",
    )

    assert len(chunks) == 1

    metadata = chunks[0].metadata

    assert metadata == {
        "source": "leave_policy.pdf",
        "section": "Entitlement",
        "chunk_index": 0,
    }


def test_chunk_index_resets_for_each_section():
    content = """
# Section One

This is section one content.

# Section Two

This is section two content.
"""

    chunker = ContextualChunker(
        chunk_size=100,
        chunk_overlap=10,
    )

    chunks = chunker.chunk(
        content=content,
        filename="test.pdf",
    )

    section_one_chunks = [
        chunk
        for chunk in chunks
        if chunk.metadata["section"] == "Section One"
    ]

    section_two_chunks = [
        chunk
        for chunk in chunks
        if chunk.metadata["section"] == "Section Two"
    ]

    assert section_one_chunks[0].metadata["chunk_index"] == 0
    assert section_two_chunks[0].metadata["chunk_index"] == 0


def test_chunk_handles_general_content_without_heading():
    content = """
Employees receive company benefits.

Employees must follow company security policies.
"""

    chunker = ContextualChunker()

    chunks = chunker.chunk(
        content=content,
        filename="employee_handbook.pdf",
    )

    assert len(chunks) == 1

    assert chunks[0].metadata["section"] == "General"
    assert chunks[0].metadata["source"] == (
        "employee_handbook.pdf"
    )

    assert "company benefits" in chunks[0].text


def test_chunk_empty_content_returns_empty_list():
    chunker = ContextualChunker()

    chunks = chunker.chunk(
        content="",
        filename="empty.pdf",
    )

    assert chunks == []


def test_chunk_whitespace_content_returns_empty_list():
    chunker = ContextualChunker()

    chunks = chunker.chunk(
        content="   \n\n   ",
        filename="empty.pdf",
    )

    assert chunks == []


def test_split_text_respects_chunk_size():
    chunker = ContextualChunker(
        chunk_size=20,
        chunk_overlap=5,
    )

    text = "abcdefghijklmnopqrstuvwxyz"

    chunks = chunker._split_text(text)

    assert len(chunks) > 1

    for chunk in chunks:
        assert len(chunk) <= 20


def test_split_text_applies_overlap():
    chunker = ContextualChunker(
        chunk_size=20,
        chunk_overlap=5,
    )

    text = "abcdefghijklmnopqrstuvwxyz"

    chunks = chunker._split_text(text)

    assert len(chunks) >= 2

    # With a chunk size of 20 and overlap of 5,
    # the second chunk starts 5 characters before
    # the end of the first chunk.
    assert chunks[0][-5:] == chunks[1][:5]


def test_split_text_empty_input():
    chunker = ContextualChunker(
        chunk_size=100,
        chunk_overlap=10,
    )

    assert chunker._split_text("") == []
    assert chunker._split_text("   ") == []


def test_document_chunk_contains_text_and_metadata():
    content = """
# Benefits

Employees receive benefits.
"""

    chunker = ContextualChunker()

    chunks = chunker.chunk(
        content=content,
        filename="employee_handbook.pdf",
    )

    chunk = chunks[0]

    assert hasattr(chunk, "text")
    assert hasattr(chunk, "metadata")

    assert isinstance(chunk.text, str)
    assert isinstance(chunk.metadata, dict)