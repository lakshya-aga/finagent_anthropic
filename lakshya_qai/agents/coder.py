"""Coding Agent — builds notebooks cell-by-cell from structured plans.

Has read-only access to MCPs and good practices.
Uses notebook manipulation tools (write_cell, edit_cell, delete_cell).
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import nbformat

from claude_agent_sdk import query, ClaudeAgentOptions, AssistantMessage, tool, create_sdk_mcp_server

from lakshya_qai.agents.prompts import CODING_PROMPT
from lakshya_qai.config.settings import get_settings
from lakshya_qai.mcps.tools_library.server import create_tools_library_mcp
from lakshya_qai.mcps.data_library.server import create_data_library_mcp

logger = logging.getLogger(__name__)
settings = get_settings()


# ── Notebook Manipulation Tools (exposed as MCP to the agent) ────────────

def _load_notebook(path: Path) -> nbformat.NotebookNode:
    if path.exists():
        return nbformat.read(str(path), as_version=4)
    nb = nbformat.v4.new_notebook()
    nb.metadata["kernelspec"] = {
        "display_name": "Python 3",
        "language": "python",
        "name": "python3",
    }
    return nb


def _save_notebook(nb: nbformat.NotebookNode, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    nbformat.write(nb, str(path))


# Global reference to the notebook being edited
_active_notebook_path: Path | None = None


async def write_cell(args: dict) -> dict:
    """Write a new cell to the notebook.

    Args:
        cell_type: "code" or "markdown"
        source: Cell content
        position: Insert position (0-indexed). -1 or omit for append.
    """
    global _active_notebook_path
    if not _active_notebook_path:
        return {"content": [{"type": "text", "text": "Error: no active notebook set."}]}

    nb = _load_notebook(_active_notebook_path)
    cell_type = args.get("cell_type", "code")
    source = args.get("source", "")
    position = args.get("position", -1)

    if cell_type == "markdown":
        cell = nbformat.v4.new_markdown_cell(source)
    else:
        cell = nbformat.v4.new_code_cell(source)

    if position < 0 or position >= len(nb.cells):
        nb.cells.append(cell)
        pos = len(nb.cells) - 1
    else:
        nb.cells.insert(position, cell)
        pos = position

    _save_notebook(nb, _active_notebook_path)
    return {"content": [{"type": "text", "text": f"Cell written at position {pos}. Total cells: {len(nb.cells)}"}]}


async def edit_cell(args: dict) -> dict:
    """Edit an existing cell in the notebook.

    Args:
        cell_index: 0-based index of the cell to edit.
        source: New cell content.
        cell_type: Optionally change cell type.
    """
    global _active_notebook_path
    if not _active_notebook_path:
        return {"content": [{"type": "text", "text": "Error: no active notebook set."}]}

    nb = _load_notebook(_active_notebook_path)
    idx = args.get("cell_index", 0)

    if idx < 0 or idx >= len(nb.cells):
        return {"content": [{"type": "text", "text": f"Error: cell index {idx} out of range (0-{len(nb.cells)-1})."}]}

    nb.cells[idx].source = args.get("source", nb.cells[idx].source)
    if "cell_type" in args:
        nb.cells[idx].cell_type = args["cell_type"]

    _save_notebook(nb, _active_notebook_path)
    return {"content": [{"type": "text", "text": f"Cell {idx} updated."}]}


async def delete_cell(args: dict) -> dict:
    """Delete a cell from the notebook.

    Args:
        cell_index: 0-based index of the cell to delete.
    """
    global _active_notebook_path
    if not _active_notebook_path:
        return {"content": [{"type": "text", "text": "Error: no active notebook set."}]}

    nb = _load_notebook(_active_notebook_path)
    idx = args.get("cell_index", 0)

    if idx < 0 or idx >= len(nb.cells):
        return {"content": [{"type": "text", "text": f"Error: cell index {idx} out of range."}]}

    removed = nb.cells.pop(idx)
    _save_notebook(nb, _active_notebook_path)
    return {"content": [{"type": "text", "text": f"Deleted cell {idx} ({removed.cell_type}). Remaining: {len(nb.cells)}"}]}


async def read_notebook(args: dict) -> dict:
    """Read the current notebook contents.

    Returns all cells with their index, type, and source.
    """
    global _active_notebook_path
    if not _active_notebook_path:
        return {"content": [{"type": "text", "text": "Error: no active notebook set."}]}

    nb = _load_notebook(_active_notebook_path)
    lines = [f"Notebook: {_active_notebook_path.name} ({len(nb.cells)} cells)\n"]

    for i, cell in enumerate(nb.cells):
        lines.append(f"--- Cell {i} [{cell.cell_type}] ---")
        lines.append(cell.source[:500])
        if len(cell.source) > 500:
            lines.append(f"... ({len(cell.source)} chars total)")
        lines.append("")

    return {"content": [{"type": "text", "text": "\n".join(lines)}]}


def _create_notebook_mcp():
    """Create the notebook manipulation MCP server."""
    return create_sdk_mcp_server(
        name="notebook",
        tools=[
            tool("write_cell", "Write a new cell to the notebook", {"cell_type": str, "source": str, "position": int})(write_cell),
            tool("edit_cell", "Edit an existing notebook cell", {"cell_index": int, "source": str, "cell_type": str})(edit_cell),
            tool("delete_cell", "Delete a cell from the notebook", {"cell_index": int})(delete_cell),
            tool("read_notebook", "Read all cells in the current notebook", {})(read_notebook),
        ],
    )


async def build_notebook(
    plan: str,
    notebook_name: str,
) -> Path:
    """Build a Jupyter notebook from a structured plan.

    Args:
        plan: Structured markdown plan from the Planning Agent.
        notebook_name: Name for the notebook file (without extension).

    Returns:
        Path to the created notebook.
    """
    global _active_notebook_path

    notebook_path = settings.notebooks_dir / f"{notebook_name}.ipynb"
    notebook_path.parent.mkdir(parents=True, exist_ok=True)
    _active_notebook_path = notebook_path

    # Create fresh notebook
    nb = nbformat.v4.new_notebook()
    nb.metadata["kernelspec"] = {
        "display_name": "Python 3",
        "language": "python",
        "name": "python3",
    }
    _save_notebook(nb, notebook_path)

    # Load good practices
    good_practices = ""
    gp_dir = settings.good_practices_dir
    if gp_dir.exists():
        for md_file in gp_dir.glob("*.md"):
            good_practices += f"\n### {md_file.stem}\n{md_file.read_text()}\n"

    prompt = f"""
Build a Jupyter notebook following this plan:

{plan}

## Good Practices to Follow
{good_practices}

Use the notebook tools (write_cell, edit_cell, delete_cell, read_notebook) to
build the notebook cell by cell. Also search the tools and data libraries
to find the right functions to use.

The notebook is at: {notebook_path}
"""

    tools_mcp = create_tools_library_mcp()
    data_mcp = create_data_library_mcp()
    notebook_mcp = _create_notebook_mcp()

    async for message in query(
        prompt=prompt,
        options=ClaudeAgentOptions(
            system_prompt=CODING_PROMPT,
            model=settings.coding_model,
            max_turns=settings.coding_max_turns,
            max_budget_usd=settings.coding_budget,
            mcp_servers={
                "tools": tools_mcp,
                "data": data_mcp,
                "notebook": notebook_mcp,
            },
            allowed_tools=[
                "mcp__tools__search_tools",
                "mcp__tools__get_unit_source",
                "mcp__tools__get_module_summary",
                "mcp__data__search_data_sources",
                "mcp__data__get_data_source_doc",
                "mcp__notebook__write_cell",
                "mcp__notebook__edit_cell",
                "mcp__notebook__delete_cell",
                "mcp__notebook__read_notebook",
            ],
        ),
    ):
        pass  # Agent works via tool calls; we just need to consume the stream

    _active_notebook_path = None
    logger.info("Notebook built: %s", notebook_path)
    return notebook_path
