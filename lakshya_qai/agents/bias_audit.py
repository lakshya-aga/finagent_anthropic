"""Bias Audit Agent — checks for lookahead, survivorship, and snooping biases.

Produces WARNING reports only — does not block the pipeline.
The human gate reviews the report and decides whether to proceed.
"""

from __future__ import annotations

import logging
from pathlib import Path

from claude_agent_sdk import query, ClaudeAgentOptions, AssistantMessage

from lakshya_qai.agents.prompts import BIAS_AUDIT_PROMPT
from lakshya_qai.config.settings import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()


async def audit_for_bias(
    notebook_path: Path,
    signal_module_path: Path,
) -> str:
    """Audit a notebook and its extracted signal module for biases.

    Args:
        notebook_path: Path to the research notebook.
        signal_module_path: Path to the extracted .py signal module.

    Returns:
        Markdown-formatted bias audit report.
    """
    # Read both files
    import nbformat

    nb = nbformat.read(str(notebook_path), as_version=4)
    notebook_content = []
    for i, cell in enumerate(nb.cells):
        notebook_content.append(f"--- Cell {i} [{cell.cell_type}] ---")
        notebook_content.append(cell.source)
        notebook_content.append("")

    module_content = signal_module_path.read_text(encoding="utf-8")

    prompt = f"""
Audit the following research notebook and signal module for statistical biases.

## Research Notebook: {notebook_path.name}
{chr(10).join(notebook_content)}

## Extracted Signal Module: {signal_module_path.name}
```python
{module_content}
```

Produce a detailed bias audit report following your output format.
Check for ALL bias types: lookahead, survivorship, data snooping, selection bias,
and look-ahead in features. Be thorough — false positives are better than misses.
"""

    report = ""
    async for message in query(
        prompt=prompt,
        options=ClaudeAgentOptions(
            system_prompt=BIAS_AUDIT_PROMPT,
            model=settings.bias_audit_model,
            max_turns=settings.bias_audit_max_turns,
            max_budget_usd=settings.bias_audit_budget,
            allowed_tools=["Read", "Glob", "Grep"],  # read-only
            cwd=str(settings.project_root),
        ),
    ):
        if isinstance(message, AssistantMessage):
            for block in message.content:
                if hasattr(block, "text"):
                    report += block.text

    logger.info("Bias audit complete for %s", notebook_path.name)
    return report
