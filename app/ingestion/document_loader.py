from pathlib import Path

from docling.document_converter import DocumentConverter


class DocumentLoader:
    """
    Loads documents using Docling.
    """

    def __init__(self):
        self.converter = DocumentConverter()

    def load(self, file_path: str):
        """
        Convert a document into a Docling document.

        Args:
            file_path: Path to the input document.

        Returns:
            Docling document object.
        """

        path = Path(file_path)

        if not path.exists():
            raise FileNotFoundError(
                f"Document not found: {file_path}"
            )

        if not path.is_file():
            raise ValueError(
                f"Path is not a file: {file_path}"
            )

        print(f"Processing document: {path.name}")

        result = self.converter.convert(
            str(path)
        )

        print(f"Successfully processed: {path.name}")

        return result.document