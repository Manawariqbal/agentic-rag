from pathlib import Path

import pytest

from app.ingestion.parser import DocumentParser


class DummyDocument:
    def __init__(self, markdown):
        self.markdown = markdown

    def export_to_markdown(self):
        return self.markdown


class DummyLoader:
    def __init__(self, document):
        self.document = document
        self.loaded_path = None

    def load(self, file_path):
        self.loaded_path = file_path
        return self.document


def create_parser(markdown="# Employee Handbook\n\nWorking hours: 9 AM to 6 PM."):
    document = DummyDocument(markdown)
    loader = DummyLoader(document)

    parser = DocumentParser()
    parser.loader = loader

    return parser, loader


def test_parser_returns_dictionary():
    parser, _ = create_parser()

    result = parser.parse("documents/employee_handbook.pdf")

    assert isinstance(result, dict)


def test_parser_returns_filename():
    parser, _ = create_parser()

    result = parser.parse(
        "documents/employee_handbook.pdf"
    )

    assert result["filename"] == "employee_handbook.pdf"


def test_parser_returns_markdown_content():
    markdown = (
        "# Employee Handbook\n\n"
        "Working hours are from 9 AM to 6 PM."
    )

    parser, _ = create_parser(markdown)

    result = parser.parse(
        "documents/employee_handbook.pdf"
    )

    assert result["content"] == markdown


def test_parser_calls_document_loader():
    parser, loader = create_parser()

    file_path = "documents/employee_handbook.pdf"

    parser.parse(file_path)

    assert loader.loaded_path == file_path


def test_parser_exports_document_to_markdown():
    markdown = "# Leave Policy\n\nEmployees receive 20 days."

    parser, _ = create_parser(markdown)

    result = parser.parse(
        "documents/leave_and_attendance_policy.pdf"
    )

    assert result["content"] == markdown


def test_parser_handles_nested_file_path():
    parser, _ = create_parser()

    result = parser.parse(
        "/tmp/company/documents/travel_and_expense_policy.pdf"
    )

    assert result["filename"] == "travel_and_expense_policy.pdf"


def test_parser_handles_empty_markdown():
    parser, _ = create_parser("")

    result = parser.parse(
        "documents/empty.pdf"
    )

    assert result["filename"] == "empty.pdf"
    assert result["content"] == ""


def test_parser_propagates_loader_error():
    class FailingLoader:
        def load(self, file_path):
            raise RuntimeError("Unable to load document")

    parser = DocumentParser()
    parser.loader = FailingLoader()

    with pytest.raises(RuntimeError, match="Unable to load document"):
        parser.parse("documents/broken.pdf")


def test_parser_works_with_real_pdf():
    """
    Integration test against one of the actual project PDFs.

    This verifies the complete:
        PDF → DocumentLoader → Docling → Markdown
    path.
    """
    pdf_path = Path("documents/employee_handbook.pdf")

    if not pdf_path.exists():
        pytest.skip("employee_handbook.pdf not found")

    parser = DocumentParser()

    result = parser.parse(str(pdf_path))

    assert isinstance(result, dict)
    assert result["filename"] == "employee_handbook.pdf"
    assert isinstance(result["content"], str)
    assert len(result["content"].strip()) > 0