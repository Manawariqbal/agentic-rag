from pathlib import Path

from app.ingestion.document_loader import DocumentLoader


class DocumentParser:

    def __init__(self):
        self.loader = DocumentLoader()

    def parse(self, file_path: str) -> dict:
        document = self.loader.load(file_path)

        markdown_content = document.export_to_markdown()

        return {
            "filename": Path(file_path).name,
            "content": markdown_content,
        }