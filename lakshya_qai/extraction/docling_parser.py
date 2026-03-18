"""Docling parser — high-fidelity table extraction from PDFs.

Docling (IBM Research) achieves 97.9% accuracy on complex tables using
DocLayNet for layout analysis and TableFormer for table structure recognition.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class ExtractedTable:
    """A single table extracted from a PDF."""

    page_number: int
    caption: str
    headers: list[str]
    rows: list[list[str]]
    raw_markdown: str

    def to_dict(self) -> dict:
        """Convert to a dictionary suitable for embedding metadata."""
        return {
            "page_number": self.page_number,
            "caption": self.caption,
            "headers": self.headers,
            "num_rows": len(self.rows),
            "markdown": self.raw_markdown,
        }


@dataclass
class DoclingResult:
    """Complete Docling extraction result."""

    tables: list[ExtractedTable]
    full_markdown: str
    page_count: int


class DoclingParser:
    """Extract tables and structured content from PDFs using Docling.

    Docling uses DocLayNet for layout analysis and TableFormer for table
    structure recognition, achieving best-in-class accuracy on complex tables.
    """

    def __init__(self) -> None:
        # Lazy import — docling is a heavy dependency
        self._converter = None

    def _get_converter(self):
        """Lazily initialize the Docling document converter."""
        if self._converter is None:
            from docling.document_converter import DocumentConverter

            self._converter = DocumentConverter()
        return self._converter

    async def parse_pdf(self, pdf_path: Path) -> DoclingResult:
        """Extract tables and content from a PDF.

        Args:
            pdf_path: Path to the PDF file.

        Returns:
            DoclingResult with extracted tables and full markdown.

        Raises:
            FileNotFoundError: If the PDF does not exist.
        """
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF not found: {pdf_path}")

        # Docling is synchronous — run in thread pool for async compatibility
        import asyncio

        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self._extract_sync, pdf_path)

    def _extract_sync(self, pdf_path: Path) -> DoclingResult:
        """Synchronous extraction using Docling."""
        converter = self._get_converter()
        result = converter.convert(str(pdf_path))
        doc = result.document

        # Export full document as markdown
        full_markdown = doc.export_to_markdown()

        # Extract tables
        tables: list[ExtractedTable] = []
        for table_item in doc.tables:
            table_df = table_item.export_to_dataframe()
            headers = list(table_df.columns)
            rows = table_df.values.tolist()
            markdown = table_item.export_to_markdown()

            # Try to get caption from the table's metadata
            caption = ""
            if hasattr(table_item, "caption") and table_item.caption:
                caption = str(table_item.caption)

            # Get page number
            page_num = 0
            if hasattr(table_item, "prov") and table_item.prov:
                page_num = table_item.prov[0].page_no if table_item.prov else 0

            tables.append(
                ExtractedTable(
                    page_number=page_num,
                    caption=caption,
                    headers=headers,
                    rows=[[str(cell) for cell in row] for row in rows],
                    raw_markdown=markdown,
                )
            )

        return DoclingResult(
            tables=tables,
            full_markdown=full_markdown,
            page_count=getattr(doc, "page_count", 0),
        )
