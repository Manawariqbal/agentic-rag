from pathlib import Path

from docling.document_converter import (
    DocumentConverter,
    PdfFormatOption,
)
from docling.datamodel.pipeline_options import PdfPipelineOptions


class DocumentLoader:

    def __init__(self):
        pipeline_options = PdfPipelineOptions()

        # These PDFs contain selectable text.
        # OCR is therefore not required.
        pipeline_options.do_ocr = False

        self.converter = DocumentConverter(
            format_options={
                "pdf": PdfFormatOption(
                    pipeline_options=pipeline_options
                )
            }
        )

    def load(self, file_path: str):
        path = Path(file_path)

        if not path.exists():
            raise FileNotFoundError(
                f"Document not found: {file_path}"
            )

        if not path.is_file():
            raise ValueError(
                f"Path is not a file: {file_path}"
            )

        print(f"Processing: {path.name}")

        result = self.converter.convert(str(path))

        print(f"Successfully processed: {path.name}")

        return result.document