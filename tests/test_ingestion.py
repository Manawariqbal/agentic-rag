from pathlib import Path

from app.ingestion.parser import DocumentParser


def main():

    parser = DocumentParser()

    documents_directory = Path("documents")
    output_directory = Path("data/processed")

    output_directory.mkdir(
        parents=True,
        exist_ok=True
    )

    pdf_files = list(
        documents_directory.glob("*.pdf")
    )

    print(f"Found {len(pdf_files)} PDF documents.")

    for pdf_file in pdf_files:

        print("\n" + "=" * 70)
        print(f"PROCESSING: {pdf_file.name}")
        print("=" * 70)

        try:

            result = parser.parse(
                str(pdf_file)
            )

            output_file = (
                output_directory /
                f"{pdf_file.stem}.md"
            )

            output_file.write_text(
                result["content"],
                encoding="utf-8"
            )

            print(f"Saved: {output_file}")
            print(
                f"Characters: "
                f"{len(result['content'])}"
            )

        except Exception as e:

            print(f"Failed: {pdf_file.name}")
            print(f"Error: {e}")


if __name__ == "__main__":
    main()