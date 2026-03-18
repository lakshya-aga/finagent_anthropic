"""Developer Agents — Data Library and Tools Library.

Both agents:
- Receive new tool/source requests with sample code
- Modularize code with enforced docstring standards
- Commit to agent branches (agent/data-lib or agent/tools-lib)
- Human gate approves merge requests
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

from claude_agent_sdk import query, ClaudeAgentOptions, AssistantMessage, ResultMessage

from lakshya_qai.agents.prompts import DEV_DATA_PROMPT, DEV_TOOLS_PROMPT
from lakshya_qai.config.settings import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()


async def run_dev_data_agent(request_file: Path) -> str:
    """Run the Data Library Developer Agent on a new data source request.

    Args:
        request_file: Path to the JSON request file.

    Returns:
        Summary of what was done (branch, files modified, etc.).
    """
    request = json.loads(request_file.read_text())

    prompt = f"""
New data source request:
- Source name: {request['source_name']}
- Description: {request['description']}
- Sample code:
```python
{request['sample_code']}
```

Integrate this into the data library following the standards in your system prompt.
Commit to the `{settings.data_lib_branch}` branch.
"""

    response_text = ""
    async for message in query(
        prompt=prompt,
        options=ClaudeAgentOptions(
            system_prompt=DEV_DATA_PROMPT,
            model=settings.dev_model,
            max_turns=20,
            max_budget_usd=settings.dev_budget,
            allowed_tools=["Read", "Write", "Edit", "Glob", "Grep", "Bash"],
            permission_mode="acceptEdits",
            cwd=str(settings.project_root),
        ),
    ):
        if isinstance(message, AssistantMessage):
            for block in message.content:
                if hasattr(block, "text"):
                    response_text += block.text

    # Update request status
    request["status"] = "completed"
    request_file.write_text(json.dumps(request, indent=2))

    return response_text


async def run_dev_tools_agent(request_file: Path) -> str:
    """Run the Tools Library Developer Agent on a new tool request.

    Args:
        request_file: Path to the JSON request file.

    Returns:
        Summary of what was done.
    """
    request = json.loads(request_file.read_text())

    prompt = f"""
New tool request:
- Tool name: {request['tool_name']}
- Description: {request['description']}
- Category: {request.get('category', 'uncategorized')}
- Sample code:
```python
{request['sample_code']}
```

Integrate this into the tools library following the standards in your system prompt.
Commit to the `{settings.tools_lib_branch}` branch.
"""

    response_text = ""
    async for message in query(
        prompt=prompt,
        options=ClaudeAgentOptions(
            system_prompt=DEV_TOOLS_PROMPT,
            model=settings.dev_model,
            max_turns=20,
            max_budget_usd=settings.dev_budget,
            allowed_tools=["Read", "Write", "Edit", "Glob", "Grep", "Bash"],
            permission_mode="acceptEdits",
            cwd=str(settings.project_root),
        ),
    ):
        if isinstance(message, AssistantMessage):
            for block in message.content:
                if hasattr(block, "text"):
                    response_text += block.text

    request["status"] = "completed"
    request_file.write_text(json.dumps(request, indent=2))

    return response_text
