"""MCP server for the QAI Tools Library (mlfinlab + custom tools).

Built on the fruit-thrower code-RAG pattern:
- AST-based parser extracts all functions/classes/methods
- TF-IDF (dev) or Anthropic embeddings (prod) for semantic search
- MCP exposes search, browse, and tool-request capabilities

The tools library contains:
- mlfinlab: Full quantitative finance toolkit (filters, labeling, bars, etc.)
- Custom tools added via the Dev Agent (Tools)

This server is consumed by the Planning, Coding, and Testing agents (read-only).
"""

from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path
from typing import Optional

from lakshya_qai.config.settings import get_settings

logger = logging.getLogger(__name__)
_settings = get_settings()

TOOLS_DIR = _settings.project_root / "lakshya_qai" / "mcps" / "tools_library" / "tools"
INDEX_DIR = _settings.project_root / ".tools_index"

# ── Lazy-loaded store singleton ──────────────────────────────────────────

_store = None


def _get_store():
    """Get or create the TF-IDF vector store (lazy init)."""
    global _store
    if _store is not None:
        return _store

    from lakshya_qai.mcps.code_rag_vector_store import SimpleTFIDFStore

    _store = SimpleTFIDFStore(persist_dir=str(INDEX_DIR))
    # Auto-index if empty
    if _store.count() == 0:
        _reindex()
    return _store


def _reindex(exclude_dirs: set[str] | None = None) -> int:
    """Parse the tools directory and update the index."""
    global _store
    from lakshya_qai.mcps.code_rag_parser import parse_repository
    from lakshya_qai.mcps.code_rag_vector_store import SimpleTFIDFStore

    if _store is None:
        _store = SimpleTFIDFStore(persist_dir=str(INDEX_DIR))

    default_exclude = {"__pycache__", ".venv", "venv", ".git", "dist", "build", "node_modules"}
    exclude = exclude_dirs or default_exclude

    units = parse_repository(str(TOOLS_DIR), exclude_dirs=exclude)
    count = _store.upsert(units)
    logger.info("Indexed %d units from tools library", len(units))
    return len(units)


# ── MCP Tool Handlers ───────────────────────────────────────────────────

async def search_tools(args: dict) -> dict:
    """Semantic search over the tools library (functions, classes, methods).

    Returns function signatures, docstrings, and source snippets ranked
    by relevance.  Use this to find relevant internal library code before
    writing new code.

    Args (via MCP):
        query: Natural language or code query, e.g. 'CUSUM filter for events'
        n_results: Maximum results (default 5)
        kind: Filter by 'function', 'class', 'method', or 'module'
        module: Filter by module name prefix, e.g. 'filters'
    """
    store = _get_store()
    query = args.get("query", "")
    if not query:
        return {"content": [{"type": "text", "text": "Error: 'query' is required."}]}

    results = store.search(
        query=query,
        n_results=args.get("n_results", 5),
        kind_filter=args.get("kind"),
        module_filter=args.get("module"),
    )

    if not results:
        return {"content": [{"type": "text", "text": f"No results for '{query}'."}]}

    output = []
    for r in results:
        block = (
            f"## {r['kind'].upper()}: {r['module']}.{r['name']}\n"
            f"**Score**: {r['score']:.3f}  |  "
            f"**File**: {r['file_path']}:{r['line_start']}\n\n"
            f"**Signature**:\n```python\n{r['signature']}\n```\n\n"
        )
        if r.get("docstring"):
            block += f"**Docstring**:\n{r['docstring'][:500]}\n\n"
        block += f"**Source preview**:\n```python\n{r['source'][:800]}\n```\n"
        block += f"\n**Unit ID**: `{r['id']}`\n---\n"
        output.append(block)

    return {"content": [{"type": "text", "text": "\n".join(output)}]}


async def get_unit_source(args: dict) -> dict:
    """Fetch full source code of a specific function or class by unit ID.

    Use the 'id' field from search_tools results.

    Args (via MCP):
        unit_id: Unit ID from search results.
    """
    store = _get_store()
    unit_id = args.get("unit_id", "")
    if not unit_id:
        return {"content": [{"type": "text", "text": "Error: 'unit_id' is required."}]}

    match = next((r for r in store._records if r["id"] == unit_id), None)
    if not match:
        return {"content": [{"type": "text", "text": f"Unit '{unit_id}' not found."}]}

    text = (
        f"# {match['kind'].upper()}: {match['module']}.{match['name']}\n"
        f"File: {match['file_path']} (lines {match['line_start']}–{match['line_end']})\n\n"
        f"```python\n{match['source']}\n```"
    )
    return {"content": [{"type": "text", "text": text}]}


async def list_modules(args: dict) -> dict:
    """List all indexed Python modules in the tools library."""
    store = _get_store()
    modules = sorted(set(r["module"] for r in store._records))
    text = "## Indexed Modules\n" + "\n".join(f"- `{m}`" for m in modules)
    return {"content": [{"type": "text", "text": text}]}


async def get_module_summary(args: dict) -> dict:
    """Get all functions and classes defined in a specific module.

    Args (via MCP):
        module: Dotted module name, e.g. 'mlfinlab.filters.filters'
    """
    store = _get_store()
    module = args.get("module", "")
    if not module:
        return {"content": [{"type": "text", "text": "Error: 'module' is required."}]}

    units = [r for r in store._records if r["module"] == module or module in r["module"]]
    if not units:
        return {"content": [{"type": "text", "text": f"No units found in module '{module}'."}]}

    lines = [f"## Module: {module}\n"]
    for u in sorted(units, key=lambda x: x["line_start"]):
        lines.append(f"### {u['kind']}: {u['name']} (line {u['line_start']})")
        lines.append(f"```python\n{u['signature']}\n```")
        if u.get("docstring"):
            lines.append(u["docstring"][:300])
        lines.append("")

    return {"content": [{"type": "text", "text": "\n".join(lines)}]}


async def index_repository(args: dict) -> dict:
    """Re-index the tools library to pick up new or changed files.

    Run this after the Dev Agent adds new tools.

    Args (via MCP):
        exclude_dirs: List of directory names to skip (optional).
    """
    exclude = set(args.get("exclude_dirs", []))
    count = _reindex(exclude_dirs=exclude or None)
    store = _get_store()
    stats = store.stats()
    return {"content": [{"type": "text", "text": f"Indexed {count} units.\n{json.dumps(stats, indent=2)}"}]}


async def get_index_stats(args: dict) -> dict:
    """Return statistics about the current tools library index."""
    store = _get_store()
    stats = store.stats()
    return {"content": [{"type": "text", "text": f"```json\n{json.dumps(stats, indent=2)}\n```"}]}


async def request_new_tool(args: dict) -> dict:
    """Request addition of a new tool to the library.

    Creates a request file for the Dev Agent (Tools) to pick up.
    The sample code will be modularized and integrated.

    Args (via MCP):
        tool_name: Proposed name for the tool.
        description: What the tool does.
        sample_code: Working Python code to be modularized.
        category: Category (e.g., 'risk', 'portfolio', 'signal').
    """
    tool_name = args.get("tool_name", "")
    description = args.get("description", "")
    sample_code = args.get("sample_code", "")

    if not all([tool_name, description, sample_code]):
        return {"content": [{"type": "text", "text": "Error: 'tool_name', 'description', 'sample_code' all required."}]}

    requests_dir = _settings.project_root / "pending_requests" / "tools"
    requests_dir.mkdir(parents=True, exist_ok=True)

    request_id = f"tool_req_{int(time.time())}_{tool_name}"
    request_file = requests_dir / f"{request_id}.json"
    request_file.write_text(json.dumps({
        "request_id": request_id,
        "type": "new_tool",
        "tool_name": tool_name,
        "description": description,
        "sample_code": sample_code,
        "category": args.get("category", "uncategorized"),
        "status": "pending",
    }, indent=2))

    return {"content": [{"type": "text", "text": f"Tool request created: {request_id}\nDev Agent (Tools) will integrate '{tool_name}'."}]}


# ── MCP Server Factory ──────────────────────────────────────────────────

def create_tools_library_mcp():
    """Create the MCP server for the tools library.

    Uses the fruit-thrower code-RAG pattern: AST parsing + TF-IDF search
    over the entire mlfinlab library and any custom tools.
    """
    from claude_agent_sdk import tool, create_sdk_mcp_server

    return create_sdk_mcp_server(
        name="qai_tools_library",
        tools=[
            tool(
                "search_tools",
                "Semantic search over internal Python tools library (mlfinlab + custom). "
                "Returns signatures, docstrings, source snippets ranked by relevance.",
                {"query": str, "n_results": int, "kind": str, "module": str},
            )(search_tools),
            tool(
                "get_unit_source",
                "Fetch full source code of a function/class by its ID from search results",
                {"unit_id": str},
            )(get_unit_source),
            tool(
                "list_modules",
                "List all indexed Python modules in the tools library",
                {},
            )(list_modules),
            tool(
                "get_module_summary",
                "Get all functions and classes defined in a specific module",
                {"module": str},
            )(get_module_summary),
            tool(
                "index_repository",
                "Re-index the tools library after new tools are added",
                {"exclude_dirs": list},
            )(index_repository),
            tool(
                "get_index_stats",
                "Return statistics about the tools library index",
                {},
            )(get_index_stats),
            tool(
                "request_new_tool",
                "Request addition of a new tool (sample code for Dev Agent to integrate)",
                {"tool_name": str, "description": str, "sample_code": str, "category": str},
            )(request_new_tool),
        ],
    )
