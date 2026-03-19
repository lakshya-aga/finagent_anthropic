"""Notebook-to-Module Extractor Agent — converts validated notebooks to .py signal modules."""

from __future__ import annotations

import logging
from pathlib import Path

from claude_agent_sdk import query, ClaudeAgentOptions, AssistantMessage

from lakshya_qai.agents.prompts import EXTRACTOR_PROMPT
from lakshya_qai.config.settings import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()


async def extract_signal_module(
    notebook_path: Path,
    signal_name: str,
) -> Path:
    """Extract signal logic from a notebook into a standalone .py module.

    Args:
        notebook_path: Path to the validated notebook.
        signal_name: Name for the signal (used for filename and class name).

    Returns:
        Path to the generated .py signal module.
    """
    output_path = settings.signals_dir / f"{signal_name}.py"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Read notebook contents for the agent
    import nbformat

    nb = nbformat.read(str(notebook_path), as_version=4)
    notebook_content = []
    for i, cell in enumerate(nb.cells):
        notebook_content.append(f"--- Cell {i} [{cell.cell_type}] ---")
        notebook_content.append(cell.source)
        # Include outputs for code cells
        if cell.cell_type == "code" and cell.outputs:
            for output in cell.outputs:
                if hasattr(output, "text"):
                    notebook_content.append(f"[Output]: {output.text[:500]}")
        notebook_content.append("")

    prompt = f"""
Extract the signal logic from this validated notebook into a standalone Python module.

Notebook: {notebook_path.name}
Signal name: {signal_name}
Output file: {output_path}

## Notebook Contents:
{chr(10).join(notebook_content)}

## Requirements:
1. Create a Signal subclass at {output_path}
2. The class must implement compute() and backtest()
3. Import data functions from lakshya_qai.mcps.data_library.sources
4. Import tool functions from lakshya_qai.mcps.tools_library.tools
5. Ensure NO lookahead bias in compute() — only use data <= as_of_date
6. Convert all hardcoded parameters to config.params with defaults
"""

    response_text = ""
    async for message in query(
        prompt=prompt,
        options=ClaudeAgentOptions(
            system_prompt=EXTRACTOR_PROMPT,
            model=settings.extractor_model,
            max_turns=settings.extractor_max_turns,
            max_budget_usd=settings.extractor_budget,
            allowed_tools=["Read", "Write", "Edit", "Glob"],
            permission_mode="acceptEdits",
            cwd=str(settings.project_root),
        ),
    ):
        if isinstance(message, AssistantMessage):
            for block in message.content:
                if hasattr(block, "text"):
                    response_text += block.text

    if not output_path.exists():
        logger.error("Extractor did not create output file: %s", output_path)
        raise RuntimeError(f"Signal module not created at {output_path}")

    logger.info("Signal module extracted: %s", output_path)
    return output_path
