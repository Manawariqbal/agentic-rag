from app.ingestion.parser import DocumentParser


def main():

    parser = DocumentParser()

    file_path = "documents/employee_handbook.pdf"

    result = parser.parse(file_path)

    print("\n")
    print("=" * 70)
    print("DOCUMENT INFORMATION")
    print("=" * 70)

    print(f"Filename: {result['filename']}")

    print("\n")
    print("=" * 70)
    print("EXTRACTED CONTENT")
    print("=" * 70)

    print(result["content"][:5000])


if __name__ == "__main__":
    main()