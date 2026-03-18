"""GROBID parser — extracts section-level structure from research PDFs.

GROBID outputs TEI-XML with 68+ label types (title, abstract, sections,
references, authors, etc.).  We parse the XML into structured sections
for downstream chunking and embedding.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path

import httpx

# TEI namespace used by GROBID output
TEI_NS = {"tei": "http://www.tei-c.org/ns/1.0"}


@dataclass
class Section:
    """A single section extracted from a research paper."""

    title: str
    text: str
    section_type: str  # abstract, introduction, methodology, results, conclusion, other
    subsections: list[Section] = field(default_factory=list)


@dataclass
class PaperMetadata:
    """Metadata extracted from the paper header."""

    title: str = ""
    authors: list[str] = field(default_factory=list)
    abstract: str = ""
    keywords: list[str] = field(default_factory=list)
    doi: str = ""


@dataclass
class GrobidResult:
    """Complete GROBID extraction result."""

    metadata: PaperMetadata
    sections: list[Section]
    references: list[str]
    raw_tei_xml: str


class GrobidParser:
    """Client for the GROBID REST service that parses TEI-XML into structured sections."""

    # Keywords used to classify sections by type
    SECTION_TYPE_KEYWORDS = {
        "abstract": ["abstract"],
        "introduction": ["introduction", "background", "motivation"],
        "methodology": [
            "method", "methodology", "approach", "model", "framework",
            "data", "dataset", "sample",
        ],
        "results": ["result", "finding", "empirical", "experiment", "analysis"],
        "conclusion": ["conclusion", "summary", "discussion", "future work"],
        "literature_review": ["literature", "related work", "prior work", "review"],
    }

    def __init__(self, grobid_url: str = "http://localhost:8070") -> None:
        self.grobid_url = grobid_url.rstrip("/")

    async def parse_pdf(self, pdf_path: Path) -> GrobidResult:
        """Send a PDF to GROBID and return structured extraction.

        Args:
            pdf_path: Path to the PDF file.

        Returns:
            GrobidResult with metadata, sections, and references.

        Raises:
            httpx.HTTPStatusError: If GROBID returns a non-2xx status.
            FileNotFoundError: If the PDF does not exist.
        """
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF not found: {pdf_path}")

        async with httpx.AsyncClient(timeout=120.0) as client:
            with open(pdf_path, "rb") as f:
                response = await client.post(
                    f"{self.grobid_url}/api/processFulltextDocument",
                    files={"input": (pdf_path.name, f, "application/pdf")},
                    data={
                        "consolidateHeader": "1",
                        "consolidateCitations": "0",
                        "includeRawCitations": "0",
                        "teiCoordinates": "false",
                    },
                )
            response.raise_for_status()

        tei_xml = response.text
        return self._parse_tei(tei_xml)

    def _parse_tei(self, tei_xml: str) -> GrobidResult:
        """Parse GROBID TEI-XML into structured result."""
        root = ET.fromstring(tei_xml)

        metadata = self._extract_metadata(root)
        sections = self._extract_sections(root)
        references = self._extract_references(root)

        return GrobidResult(
            metadata=metadata,
            sections=sections,
            references=references,
            raw_tei_xml=tei_xml,
        )

    def _extract_metadata(self, root: ET.Element) -> PaperMetadata:
        """Extract title, authors, abstract, keywords from TEI header."""
        meta = PaperMetadata()

        # Title
        title_el = root.find(".//tei:titleStmt/tei:title", TEI_NS)
        if title_el is not None and title_el.text:
            meta.title = title_el.text.strip()

        # Authors
        for author in root.findall(".//tei:fileDesc//tei:author", TEI_NS):
            forename = author.find(".//tei:forename", TEI_NS)
            surname = author.find(".//tei:surname", TEI_NS)
            parts = []
            if forename is not None and forename.text:
                parts.append(forename.text.strip())
            if surname is not None and surname.text:
                parts.append(surname.text.strip())
            if parts:
                meta.authors.append(" ".join(parts))

        # Abstract
        abstract_el = root.find(".//tei:profileDesc/tei:abstract", TEI_NS)
        if abstract_el is not None:
            meta.abstract = self._get_all_text(abstract_el).strip()

        # Keywords
        for kw in root.findall(".//tei:keywords/tei:term", TEI_NS):
            if kw.text:
                meta.keywords.append(kw.text.strip())

        # DOI
        doi_el = root.find(".//tei:idno[@type='DOI']", TEI_NS)
        if doi_el is not None and doi_el.text:
            meta.doi = doi_el.text.strip()

        return meta

    def _extract_sections(self, root: ET.Element) -> list[Section]:
        """Extract body sections from TEI."""
        sections: list[Section] = []
        body = root.find(".//tei:body", TEI_NS)
        if body is None:
            return sections

        for div in body.findall("tei:div", TEI_NS):
            section = self._parse_div(div)
            if section:
                sections.append(section)

        return sections

    def _parse_div(self, div: ET.Element) -> Section | None:
        """Parse a single <div> element into a Section."""
        head = div.find("tei:head", TEI_NS)
        title = head.text.strip() if head is not None and head.text else "Untitled"

        # Gather all paragraph text
        paragraphs = []
        for p in div.findall("tei:p", TEI_NS):
            text = self._get_all_text(p).strip()
            if text:
                paragraphs.append(text)

        if not paragraphs and not title:
            return None

        section_type = self._classify_section(title)

        # Recursively parse sub-divs
        subsections = []
        for sub_div in div.findall("tei:div", TEI_NS):
            sub = self._parse_div(sub_div)
            if sub:
                subsections.append(sub)

        return Section(
            title=title,
            text="\n\n".join(paragraphs),
            section_type=section_type,
            subsections=subsections,
        )

    def _classify_section(self, title: str) -> str:
        """Classify a section title into a type based on keywords."""
        title_lower = title.lower()
        for stype, keywords in self.SECTION_TYPE_KEYWORDS.items():
            if any(kw in title_lower for kw in keywords):
                return stype
        return "other"

    def _extract_references(self, root: ET.Element) -> list[str]:
        """Extract reference strings from the bibliography."""
        refs: list[str] = []
        for bibl in root.findall(".//tei:listBibl/tei:biblStruct", TEI_NS):
            ref_text = self._get_all_text(bibl).strip()
            if ref_text:
                refs.append(ref_text)
        return refs

    @staticmethod
    def _get_all_text(element: ET.Element) -> str:
        """Recursively get all text content from an element."""
        return "".join(element.itertext())
