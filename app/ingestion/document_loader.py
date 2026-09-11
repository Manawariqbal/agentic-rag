from pathlib import Path

from docling.document_converter import DocumentConverter


class DocumentLoader:
    def __init__(self):
        self.converter = DocumentConverter()

    def load(self, file_path: str):
        path = Path(file_path)

        if not path.exists():
            raise FileNotFoundError(f"Document not found: {file_path}")

        if not path.is_file():
            raise ValueError(f"Path is not a file: {file_path}")

        print(f"Processing: {path.name}")

        result = self.converter.convert(str(path))

        print(f"Processed: {path.name}")

        return result.document