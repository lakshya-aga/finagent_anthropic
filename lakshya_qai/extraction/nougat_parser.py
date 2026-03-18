"""Nougat parser — mathematical equation extraction from research PDFs.

Nougat (Meta) is a Visual Transformer that performs end-to-end OCR on
scientific PDFs, outputting MultiMarkdown with LaTeX equations preserved.
Best-in-class for mathematical content extraction.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class ExtractedEquation:
    """A single equation extracted from a PDF."""

    latex: str
    context_before: str  # surrounding text for embedding
    context_after: str
    is_inline: bool  # inline ($...$) vs display ($$...$$)
    page_estimate: int = 0  # approximate page, based on position in output


@dataclass
class NougatResult:
    """Complete Nougat extraction result."""

    full_mmd: str  # full MultiMarkdown output
    equations: list[ExtractedEquation]


class NougatParser:
    """Extract mathematical content from PDFs using Nougat.

    Nougat is trained on arXiv and PubMed papers and outputs LaTeX
    equations embedded in MultiMarkdown.  Best used specifically for
    equation-heavy content; combine with GROBID/Docling for structure/tables.

    Note:
        Nougat requires GPU for reasonable performance.
        Known to hallucinate on documents outside its training distribution.
    """

    # Regex patterns for LaTeX equations in Nougat output
    DISPLAY_MATH_PATTERN = re.compile(r"\$\$(.*?)\$\$", re.DOTALL)
    INLINE_MATH_PATTERN = re.compile(r"(?<!\$)\$(?!\$)(.*?)(?<!\$)\$(?!\$)")
    # \[ ... \] and \( ... \) patterns
    BRACKET_DISPLAY_PATTERN = re.compile(r"\\\[(.*?)\\\]", re.DOTALL)
    BRACKET_INLINE_PATTERN = re.compile(r"\\\((.*?)\\\)")

    CONTEXT_CHARS = 200  # characters of context around each equation

    def __init__(self, model_tag: str = "0.1.0-small") -> None:
        self.model_tag = model_tag
        self._model = None
        self._processor = None

    def _load_model(self):
        """Lazily load the Nougat model and processor."""
        if self._model is None:
            from nougat import NougatModel
            from nougat.utils.checkpoint import get_checkpoint

            checkpoint = get_checkpoint(model_tag=self.model_tag)
            self._model = NougatModel.from_pretrained(checkpoint)
            self._model.eval()

    async def parse_pdf(self, pdf_path: Path) -> NougatResult:
        """Extract equations and mathematical content from a PDF.

        Args:
            pdf_path: Path to the PDF file.

        Returns:
            NougatResult with full MMD text and extracted equations.

        Raises:
            FileNotFoundError: If the PDF does not exist.
        """
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF not found: {pdf_path}")

        import asyncio

        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self._extract_sync, pdf_path)

    def _extract_sync(self, pdf_path: Path) -> NougatResult:
        """Synchronous extraction using Nougat."""
        from nougat import NougatModel
        from nougat.utils.dataset import LazyDataset
        from nougat.utils.checkpoint import get_checkpoint
        from nougat.postprocessing import markdown_compatible

        self._load_model()

        # Process PDF pages
        dataset = LazyDataset(
            pdf=str(pdf_path),
            prepare=self._model.encoder.prepare_input,
        )

        pages_output: list[str] = []
        for idx, sample in enumerate(dataset):
            if sample is None:
                continue
            # Run inference on each page
            output = self._model.inference(image_tensors=sample.unsqueeze(0))
            # Post-process
            page_text = markdown_compatible(output["predictions"][0])
            pages_output.append(page_text)

        full_mmd = "\n\n".join(pages_output)
        equations = self._extract_equations(full_mmd)

        return NougatResult(full_mmd=full_mmd, equations=equations)

    def _extract_equations(self, text: str) -> list[ExtractedEquation]:
        """Extract all equations from the MultiMarkdown output."""
        equations: list[ExtractedEquation] = []

        # Display math: $$...$$ and \[...\]
        for pattern, is_inline in [
            (self.DISPLAY_MATH_PATTERN, False),
            (self.BRACKET_DISPLAY_PATTERN, False),
            (self.INLINE_MATH_PATTERN, True),
            (self.BRACKET_INLINE_PATTERN, True),
        ]:
            for match in pattern.finditer(text):
                latex = match.group(1).strip()
                if not latex:
                    continue

                start, end = match.span()
                context_before = text[max(0, start - self.CONTEXT_CHARS) : start].strip()
                context_after = text[end : end + self.CONTEXT_CHARS].strip()

                equations.append(
                    ExtractedEquation(
                        latex=latex,
                        context_before=context_before,
                        context_after=context_after,
                        is_inline=is_inline,
                    )
                )

        return equations
