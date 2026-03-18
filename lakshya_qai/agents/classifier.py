"""Artifact Classifier Agent — routes uploads to the correct pipeline.

Confidence-gated: below threshold → asks user to confirm classification.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path

from claude_agent_sdk import query, ClaudeAgentOptions, AssistantMessage, ResultMessage

from lakshya_qai.agents.prompts import CLASSIFIER_PROMPT
from lakshya_qai.config.settings import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()


@dataclass
class ClassificationResult:
    """Result from the artifact classifier."""

    classification: str  # research_paper, research_tool, signal
    confidence: float
    reasoning: str
    suggested_name: str
    needs_human_confirmation: bool


async def classify_artifact(
    user_text: str,
    file_path: Path | None = None,
) -> ClassificationResult:
    """Classify a user upload into research_paper, research_tool, or signal.

    Args:
        user_text: The user's request text.
        file_path: Optional path to the uploaded file.

    Returns:
        ClassificationResult with classification and confidence.
    """
    # Build context for the classifier
    context_parts = [f"User request: {user_text}"]

    if file_path:
        context_parts.append(f"File name: {file_path.name}")
        context_parts.append(f"File extension: {file_path.suffix}")

        # Read first ~2000 chars for context
        try:
            if file_path.suffix in (".txt", ".py", ".md"):
                content = file_path.read_text(encoding="utf-8")[:2000]
                context_parts.append(f"File content preview:\n{content}")
            elif file_path.suffix == ".ipynb":
                import nbformat

                nb = nbformat.read(str(file_path), as_version=4)
                cells_text = []
                for cell in nb.cells[:5]:
                    cells_text.append(f"[{cell.cell_type}] {cell.source[:300]}")
                context_parts.append(f"Notebook cells:\n" + "\n---\n".join(cells_text))
            elif file_path.suffix == ".pdf":
                context_parts.append("[PDF file — content not previewed for classification]")
        except Exception as e:
            logger.warning("Could not read file %s: %s", file_path, e)
            context_parts.append(f"[Could not read file: {e}]")

    prompt = "\n\n".join(context_parts)

    # Run classifier agent
    response_text = ""
    async for message in query(
        prompt=prompt,
        options=ClaudeAgentOptions(
            system_prompt=CLASSIFIER_PROMPT,
            model=settings.classifier_model,
            max_turns=settings.classifier_max_turns,
            max_budget_usd=settings.classifier_budget,
            allowed_tools=[],  # classifier needs no tools
        ),
    ):
        if isinstance(message, AssistantMessage):
            for block in message.content:
                if hasattr(block, "text"):
                    response_text += block.text

    # Parse JSON response
    try:
        # Extract JSON from response (may be wrapped in markdown code block)
        json_text = response_text
        if "```json" in json_text:
            json_text = json_text.split("```json")[1].split("```")[0]
        elif "```" in json_text:
            json_text = json_text.split("```")[1].split("```")[0]

        result = json.loads(json_text.strip())

        needs_confirmation = result["confidence"] < settings.classifier_confidence_threshold

        return ClassificationResult(
            classification=result["classification"],
            confidence=result["confidence"],
            reasoning=result["reasoning"],
            suggested_name=result.get("suggested_name", "unnamed"),
            needs_human_confirmation=needs_confirmation,
        )

    except (json.JSONDecodeError, KeyError) as e:
        logger.error("Classifier returned invalid JSON: %s\nRaw: %s", e, response_text)
        return ClassificationResult(
            classification="signal",  # safe default
            confidence=0.0,
            reasoning=f"Classifier error: {e}. Defaulting to signal for human review.",
            suggested_name="unknown",
            needs_human_confirmation=True,
        )
