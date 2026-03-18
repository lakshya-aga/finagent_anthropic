"""Unified extraction pipeline — merges GROBID + Docling + Nougat results.

Each parser handles what it does best:
- GROBID: section-level structure (abstract, methodology, results, etc.)
- Docling: high-fidelity table extraction (97.9% accuracy)
- Nougat: mathematical equation extraction (LaTeX output)

The pipeline merges their outputs into section-aware chunks with metadata,
ready for embedding into the vector store.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from lakshya_qai.extraction.grobid_parser import GrobidParser, GrobidResult
from lakshya_qai.extraction.docling_parser import DoclingParser, DoclingResult
from lakshya_qai.extraction.nougat_parser import NougatParser, NougatResult

logger = logging.getLogger(__name__)


@dataclass
class DocumentChunk:
    """A single chunk ready for embedding into the vector store."""

    text: str
    chunk_type: Literal[
        "abstract", "introduction", "methodology", "results",
        "conclusion", "literature_review", "table", "equation", "other",
    ]
    metadata: dict = field(default_factory=dict)


@dataclass
class ExtractedDocument:
    """Complete extraction result from the unified pipeline."""

    source_path: str
    title: str
    authors: list[str]
    abstract: str
    chunks: list[DocumentChunk]
    tables_count: int
    equations_count: int
    references: list[str]

    # Raw results for debugging / reprocessing
    grobid_result: GrobidResult | None = None
    docling_result: DoclingResult | None = None
    nougat_result: NougatResult | None = None


class ExtractionPipeline:
    """Orchestrates GROBID + Docling + Nougat for comprehensive PDF extraction.

    Usage::

        pipeline = ExtractionPipeline(grobid_url="http://localhost:8070")
        doc = await pipeline.extract_pdf(Path("paper.pdf"))
        for chunk in doc.chunks:
            print(chunk.chunk_type, chunk.text[:100])
    """

    # Max chunk size in characters before splitting
    MAX_CHUNK_SIZE = 1500
    CHUNK_OVERLAP = 200

    def __init__(
        self,
        grobid_url: str = "http://localhost:8070",
        use_nougat: bool = True,
    ) -> None:
        self.grobid = GrobidParser(grobid_url=grobid_url)
        self.docling = DoclingParser()
        self.nougat = NougatParser() if use_nougat else None

    async def extract_pdf(self, pdf_path: Path) -> ExtractedDocument:
        """Run the full extraction pipeline on a PDF.

        Runs GROBID, Docling, and optionally Nougat in parallel, then
        merges their outputs into a unified set of chunks.

        Args:
            pdf_path: Path to the PDF file.

        Returns:
            ExtractedDocument with all chunks and metadata.
        """
        logger.info("Starting extraction pipeline for %s", pdf_path.name)

        # Run parsers in parallel
        tasks = [
            self.grobid.parse_pdf(pdf_path),
            self.docling.parse_pdf(pdf_path),
        ]
        if self.nougat:
            tasks.append(self.nougat.parse_pdf(pdf_path))

        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Unpack results, handling failures gracefully
        grobid_result = results[0] if not isinstance(results[0], Exception) else None
        docling_result = results[1] if not isinstance(results[1], Exception) else None
        nougat_result = None
        if self.nougat and len(results) > 2:
            nougat_result = results[2] if not isinstance(results[2], Exception) else None

        # Log any failures
        for i, r in enumerate(results):
            if isinstance(r, Exception):
                parser_name = ["GROBID", "Docling", "Nougat"][i]
                logger.warning("Parser %s failed: %s", parser_name, r)

        # Build chunks
        chunks = self._merge_results(grobid_result, docling_result, nougat_result)

        # Extract metadata (prefer GROBID for structure)
        title = ""
        authors: list[str] = []
        abstract = ""
        references: list[str] = []

        if grobid_result:
            title = grobid_result.metadata.title
            authors = grobid_result.metadata.authors
            abstract = grobid_result.metadata.abstract
            references = grobid_result.references

        return ExtractedDocument(
            source_path=str(pdf_path),
            title=title,
            authors=authors,
            abstract=abstract,
            chunks=chunks,
            tables_count=len(docling_result.tables) if docling_result else 0,
            equations_count=len(nougat_result.equations) if nougat_result else 0,
            references=references,
            grobid_result=grobid_result,
            docling_result=docling_result,
            nougat_result=nougat_result,
        )

    async def extract_text(self, text_path: Path) -> ExtractedDocument:
        """Extract from a plain text file — simple chunking with overlap.

        Args:
            text_path: Path to the .txt file.

        Returns:
            ExtractedDocument with text chunks.
        """
        content = text_path.read_text(encoding="utf-8")
        chunks = self._chunk_text(content, chunk_type="other", metadata={"source": str(text_path)})

        return ExtractedDocument(
            source_path=str(text_path),
            title=text_path.stem,
            authors=[],
            abstract="",
            chunks=chunks,
            tables_count=0,
            equations_count=0,
            references=[],
        )

    def _merge_results(
        self,
        grobid: GrobidResult | None,
        docling: DoclingResult | None,
        nougat: NougatResult | None,
    ) -> list[DocumentChunk]:
        """Merge outputs from all three parsers into unified chunks."""
        chunks: list[DocumentChunk] = []

        # 1. Section-level chunks from GROBID (primary structure)
        if grobid:
            # Abstract as a single chunk
            if grobid.metadata.abstract:
                chunks.append(
                    DocumentChunk(
                        text=grobid.metadata.abstract,
                        chunk_type="abstract",
                        metadata={
                            "title": grobid.metadata.title,
                            "authors": grobid.metadata.authors,
                        },
                    )
                )

            # Body sections
            for section in grobid.sections:
                section_chunks = self._chunk_section(section)
                chunks.extend(section_chunks)

        # 2. Table chunks from Docling (best table accuracy)
        if docling:
            for table in docling.tables:
                chunks.append(
                    DocumentChunk(
                        text=table.raw_markdown,
                        chunk_type="table",
                        metadata={
                            "caption": table.caption,
                            "headers": table.headers,
                            "page_number": table.page_number,
                            "num_rows": len(table.rows),
                        },
                    )
                )

        # 3. Equation chunks from Nougat (best math extraction)
        if nougat:
            for eq in nougat.equations:
                if eq.is_inline:
                    continue  # skip inline math — it's in the section text
                # Display equations get their own chunk with context
                text = f"Equation: ${eq.latex}$\n\nContext: {eq.context_before} [...] {eq.context_after}"
                chunks.append(
                    DocumentChunk(
                        text=text,
                        chunk_type="equation",
                        metadata={"latex": eq.latex, "is_inline": eq.is_inline},
                    )
                )

        return chunks

    def _chunk_section(self, section) -> list[DocumentChunk]:
        """Convert a GROBID section into one or more chunks."""
        chunks: list[DocumentChunk] = []

        if section.text:
            section_chunks = self._chunk_text(
                section.text,
                chunk_type=section.section_type,
                metadata={"section_title": section.title},
            )
            chunks.extend(section_chunks)

        # Recurse into subsections
        for sub in section.subsections:
            chunks.extend(self._chunk_section(sub))

        return chunks

    def _chunk_text(
        self,
        text: str,
        chunk_type: str,
        metadata: dict | None = None,
    ) -> list[DocumentChunk]:
        """Split text into overlapping chunks of MAX_CHUNK_SIZE."""
        if not text.strip():
            return []

        metadata = metadata or {}
        chunks: list[DocumentChunk] = []

        if len(text) <= self.MAX_CHUNK_SIZE:
            chunks.append(DocumentChunk(text=text, chunk_type=chunk_type, metadata=metadata))
            return chunks

        # Split on paragraph boundaries first, then by size
        paragraphs = text.split("\n\n")
        current_chunk = ""

        for para in paragraphs:
            if len(current_chunk) + len(para) + 2 > self.MAX_CHUNK_SIZE:
                if current_chunk:
                    chunks.append(
                        DocumentChunk(
                            text=current_chunk.strip(),
                            chunk_type=chunk_type,
                            metadata=metadata,
                        )
                    )
                    # Overlap: keep the last portion
                    if len(current_chunk) > self.CHUNK_OVERLAP:
                        current_chunk = current_chunk[-self.CHUNK_OVERLAP :] + "\n\n" + para
                    else:
                        current_chunk = para
                else:
                    # Single paragraph exceeds max size — force split
                    for i in range(0, len(para), self.MAX_CHUNK_SIZE - self.CHUNK_OVERLAP):
                        chunk_text = para[i : i + self.MAX_CHUNK_SIZE]
                        chunks.append(
                            DocumentChunk(
                                text=chunk_text, chunk_type=chunk_type, metadata=metadata
                            )
                        )
                    current_chunk = ""
            else:
                current_chunk = current_chunk + "\n\n" + para if current_chunk else para

        if current_chunk.strip():
            chunks.append(
                DocumentChunk(text=current_chunk.strip(), chunk_type=chunk_type, metadata=metadata)
            )

        return chunks
