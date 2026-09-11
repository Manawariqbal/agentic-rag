from app.ingestion.chunker import ContextualChunker


def main():

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

    print(f"Total chunks: {len(chunks)}")

    for i, chunk in enumerate(chunks):

        print("\n" + "=" * 60)
        print(f"CHUNK {i}")
        print("=" * 60)

        print(chunk.text)
        print("\nMetadata:")
        print(chunk.metadata)


if __name__ == "__main__":
    main()