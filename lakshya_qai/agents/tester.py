"""Test & Edit Agent — runs notebooks until clean execution.

Has read access to MCPs, write access to notebook cells,
and ability to install packages.
"""

from __future__ import annotations

import logging
from pathlib import Path

import nbformat
from nbconvert.preprocessors import ExecutePreprocessor, CellExecutionError

from claude_agent_sdk import query, ClaudeAgentOptions, AssistantMessage, tool, create_sdk_mcp_server

from lakshya_qai.agents.prompts import TESTING_PROMPT
from lakshya_qai.config.settings import get_settings
from lakshya_qai.mcps.tools_library.server import create_tools_library_mcp
from lakshya_qai.mcps.data_library.server import create_data_library_mcp

logger = logging.getLogger(__name__)
settings = get_settings()


# ── Notebook Execution Tool ──────────────────────────────────────────────

_active_notebook_path: Path | None = None


async def run_notebook(args: dict) -> dict:
    """Execute the notebook with 'Run All' and return results.

    Returns execution status, any errors with cell index and traceback.
    """
    global _active_notebook_path
    if not _active_notebook_path:
        return {"content": [{"type": "text", "text": "Error: no active notebook."}]}

    nb = nbformat.read(str(_active_notebook_path), as_version=4)

    ep = ExecutePreprocessor(
        timeout=300,
        kernel_name="python3",
    )

    try:
        ep.preprocess(nb, {"metadata": {"path": str(_active_notebook_path.parent)}})
        nbformat.write(nb, str(_active_notebook_path))
        return {"content": [{"type": "text", "text": f"NOTEBOOK STATUS: PASS\nAll {len(nb.cells)} cells executed successfully."}]}
    except CellExecutionError as e:
        nbformat.write(nb, str(_active_notebook_path))

        # Find which cell failed
        error_info = f"NOTEBOOK STATUS: FAIL\nError in cell execution.\n\nTraceback:\n{e.traceback}\n\nCell source:\n{e.ename}: {e.evalue}"
        return {"content": [{"type": "text", "text": error_info}]}
    except Exception as e:
        return {"content": [{"type": "text", "text": f"NOTEBOOK STATUS: ERROR\nUnexpected error: {e}"}]}


async def install_package(args: dict) -> dict:
    """Install a Python package.

    Args:
        package: Package name (e.g., "scikit-learn").
    """
    import subprocess

    package = args.get("package", "")
    if not package:
        return {"content": [{"type": "text", "text": "Error: 'package' required."}]}

    # Basic validation
    if any(c in package for c in [";", "&", "|", "`"]):
        return {"content": [{"type": "text", "text": "Error: invalid package name."}]}

    try:
        result = subprocess.run(
            ["pip", "install", package],
            capture_output=True,
            text=True,
            timeout=120,
        )
        if result.returncode == 0:
            return {"content": [{"type": "text", "text": f"Installed {package} successfully."}]}
        else:
            return {"content": [{"type": "text", "text": f"Failed to install {package}:\n{result.stderr}"}]}
    except subprocess.TimeoutExpired:
        return {"content": [{"type": "text", "text": f"Timeout installing {package}."}]}


async def write_cell(args: dict) -> dict:
    """Write a new cell (delegated to coder module)."""
    from lakshya_qai.agents.coder import write_cell as _wc, _load_notebook, _save_notebook

    global _active_notebook_path
    from lakshya_qai.agents import coder
    coder._active_notebook_path = _active_notebook_path
    return await _wc(args)


async def edit_cell(args: dict) -> dict:
    """Edit a cell (delegated to coder module)."""
    from lakshya_qai.agents.coder import edit_cell as _ec
    from lakshya_qai.agents import coder
    coder._active_notebook_path = _active_notebook_path
    return await _ec(args)


async def delete_cell(args: dict) -> dict:
    """Delete a cell (delegated to coder module)."""
    from lakshya_qai.agents.coder import delete_cell as _dc
    from lakshya_qai.agents import coder
    coder._active_notebook_path = _active_notebook_path
    return await _dc(args)


async def read_notebook_tool(args: dict) -> dict:
    """Read notebook (delegated to coder module)."""
    from lakshya_qai.agents.coder import read_notebook as _rn
    from lakshya_qai.agents import coder
    coder._active_notebook_path = _active_notebook_path
    return await _rn(args)


def _create_test_mcp():
    """Create the testing MCP server with notebook + execution tools."""
    return create_sdk_mcp_server(
        name="test_tools",
        tools=[
            tool("run_notebook", "Execute the notebook (Run All) and return results", {})(run_notebook),
            tool("install_package", "Install a Python package via pip", {"package": str})(install_package),
            tool("write_cell", "Write a new cell", {"cell_type": str, "source": str, "position": int})(write_cell),
            tool("edit_cell", "Edit an existing cell", {"cell_index": int, "source": str})(edit_cell),
            tool("delete_cell", "Delete a cell", {"cell_index": int})(delete_cell),
            tool("read_notebook", "Read all notebook cells", {})(read_notebook_tool),
        ],
    )


async def test_and_fix_notebook(notebook_path: Path) -> bool:
    """Run the notebook and fix errors until clean execution.

    Args:
        notebook_path: Path to the notebook to test.

    Returns:
        True if notebook passes, False if max attempts exceeded.
    """
    global _active_notebook_path
    _active_notebook_path = notebook_path

    # Load good practices
    good_practices = ""
    gp_dir = settings.good_practices_dir
    if gp_dir.exists():
        for md_file in gp_dir.glob("*.md"):
            good_practices += f"\n### {md_file.stem}\n{md_file.read_text()}\n"

    prompt = f"""
Test this notebook by running it end-to-end: {notebook_path.name}

First, read the notebook to understand its contents, then run it.
If there are errors, fix them and re-run until it passes.

Good practices to be aware of:
{good_practices}

Use run_notebook to execute, then read_notebook + edit_cell/write_cell to fix issues.
You can install_package if imports fail.
"""

    tools_mcp = create_tools_library_mcp()
    data_mcp = create_data_library_mcp()
    test_mcp = _create_test_mcp()

    passed = False
    async for message in query(
        prompt=prompt,
        options=ClaudeAgentOptions(
            system_prompt=TESTING_PROMPT,
            model=settings.testing_model,
            max_turns=settings.testing_max_turns,
            max_budget_usd=settings.testing_budget,
            mcp_servers={
                "tools": tools_mcp,
                "data": data_mcp,
                "test": test_mcp,
            },
            allowed_tools=[
                "mcp__tools__search_tools",
                "mcp__tools__get_unit_source",
                "mcp__tools__get_module_summary",
                "mcp__data__search_data_sources",
                "mcp__data__get_data_source_doc",
                "mcp__test__run_notebook",
                "mcp__test__install_package",
                "mcp__test__write_cell",
                "mcp__test__edit_cell",
                "mcp__test__delete_cell",
                "mcp__test__read_notebook",
            ],
        ),
    ):
        if isinstance(message, AssistantMessage):
            for block in message.content:
                if hasattr(block, "text") and "NOTEBOOK STATUS: PASS" in block.text:
                    passed = True

    _active_notebook_path = None
    return passed
