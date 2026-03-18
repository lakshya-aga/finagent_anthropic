"""Planning Agent — creates structured notebook plans from user requests.

Has read access to Knowledge Base, Good Practices, Tools and Data MCPs.
Outputs a structured plan in strict markdown format.
"""

from __future__ import annotations

import logging
from pathlib import Path

from claude_agent_sdk import query, ClaudeAgentOptions, AssistantMessage

from lakshya_qai.agents.prompts import PLANNING_PROMPT
from lakshya_qai.config.settings import get_settings
from lakshya_qai.mcps.tools_library.server import create_tools_library_mcp
from lakshya_qai.mcps.data_library.server import create_data_library_mcp
from lakshya_qai.mcps.knowledge_base.server import create_knowledge_base_mcp

logger = logging.getLogger(__name__)
settings = get_settings()


async def create_plan(
    user_request: str,
    context: str = "",
) -> str:
    """Create a structured notebook plan from a user request.

    Args:
        user_request: The user's research/signal request.
        context: Additional context (e.g., file contents, prior plans).

    Returns:
        Structured markdown plan following the strict format.
    """
    # Read good practices files
    good_practices = _load_good_practices()

    prompt = f"""
User Request: {user_request}

{f"Additional Context: {context}" if context else ""}

## Available Good Practices
{good_practices}

Please search the knowledge base, tools library, and data library for relevant
information, then create a structured notebook plan following your output format.
"""

    tools_mcp = create_tools_library_mcp()
    data_mcp = create_data_library_mcp()
    kb_mcp = create_knowledge_base_mcp()

    plan_text = ""
    async for message in query(
        prompt=prompt,
        options=ClaudeAgentOptions(
            system_prompt=PLANNING_PROMPT,
            model=settings.planning_model,
            max_turns=settings.planning_max_turns,
            max_budget_usd=settings.planning_budget,
            mcp_servers={
                "tools": tools_mcp,
                "data": data_mcp,
                "kb": kb_mcp,
            },
            allowed_tools=[
                "mcp__tools__search_tools",
                "mcp__tools__list_all_tools",
                "mcp__tools__get_tool_details",
                "mcp__data__search_data_sources",
                "mcp__data__list_all_data_sources",
                "mcp__data__get_data_source_details",
                "mcp__kb__search_knowledge_base",
            ],
        ),
    ):
        if isinstance(message, AssistantMessage):
            for block in message.content:
                if hasattr(block, "text"):
                    plan_text += block.text

    return plan_text


def _load_good_practices() -> str:
    """Load all good practices .md files into a single string."""
    practices_dir = settings.good_practices_dir
    if not practices_dir.exists():
        return "(No good practices files found yet)"

    parts = []
    for md_file in sorted(practices_dir.glob("*.md")):
        content = md_file.read_text(encoding="utf-8")
        parts.append(f"### {md_file.stem}\n{content}")

    return "\n\n".join(parts) if parts else "(No good practices files found yet)"
