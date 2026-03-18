"""MCP server for the QAI Data Library.

Built on the data-mcp pattern:
- Registry of wrapper functions with metadata, tags, examples
- Search scoring with weighted name/tag/summary matching
- Rich markdown documentation rendering
- New source request routing to Dev Agent (Data)

Data sources available:
- get_equity_prices: yfinance OHLCV wrapper
- get_sp500_composition: Point-in-time S&P 500 members
- get_file_data: CSV/Parquet/Excel loader
- get_bloomberg_data: Bloomberg blpapi wrapper (stub)

This server is consumed by the Planning, Coding, and Testing agents (read-only).
"""

from __future__ import annotations

import inspect
import json
import logging
import textwrap
import time
from typing import Any, Dict, List

from lakshya_qai.config.settings import get_settings
from lakshya_qai.mcps.data_library.sources.findata.equity_prices import get_equity_prices
from lakshya_qai.mcps.data_library.sources.findata.sp500_composition import (
    get_sp500_composition,
    refresh_sp500_cache,
)
from lakshya_qai.mcps.data_library.sources.findata.file_reader import get_file_data
from lakshya_qai.mcps.data_library.sources.findata.bloomberg import get_bloomberg_data

logger = logging.getLogger(__name__)
_settings = get_settings()


# ── Registry ─────────────────────────────────────────────────────────────
# One entry per public data wrapper. Add new data sources here.

_REGISTRY: List[Dict[str, Any]] = [
    {
        "name": "get_equity_prices",
        "callable": get_equity_prices,
        "module": "lakshya_qai.mcps.data_library.sources.findata.equity_prices",
        "tags": [
            "equity", "prices", "ohlcv", "historical", "daily", "weekly",
            "monthly", "stocks", "close", "open", "high", "low", "volume",
            "time series", "market data", "yfinance", "bars",
        ],
        "stub": False,
        "install_requires": ["yfinance"],
        "summary": (
            "Fetch historical OHLCV bars for one or more equities. "
            "Wraps yfinance with normalised MultiIndex column output and input validation."
        ),
        "example": textwrap.dedent("""\
            from lakshya_qai.mcps.data_library.sources.findata.equity_prices import get_equity_prices

            # Single ticker — daily close
            df = get_equity_prices(["AAPL"], "2024-01-01", "2024-12-31", fields=["Close"])

            # Multiple tickers — all OHLCV fields
            df = get_equity_prices(["AAPL", "MSFT", "NVDA"], "2023-01-01", "2024-01-01")
            close = df["Close"]           # DataFrame: rows=dates, cols=tickers
            aapl  = df["Close"]["AAPL"]   # Series

            # Weekly bars
            df = get_equity_prices(["SPY"], "2020-01-01", "2024-01-01", fields=["Close"], frequency="1wk")
        """),
    },
    {
        "name": "get_sp500_composition",
        "callable": get_sp500_composition,
        "module": "lakshya_qai.mcps.data_library.sources.findata.sp500_composition",
        "tags": [
            "sp500", "s&p 500", "index", "composition", "constituents",
            "members", "point in time", "pit", "historical", "benchmark",
            "universe", "index membership",
        ],
        "stub": False,
        "install_requires": ["git (system)"],
        "summary": (
            "Return the point-in-time S&P 500 membership for any date. "
            "Clones fja05680/sp500 to ~/.cache/findata/sp500/ on first use "
            "and reads from disk thereafter."
        ),
        "example": textwrap.dedent("""\
            from lakshya_qai.mcps.data_library.sources.findata.sp500_composition import get_sp500_composition

            # List of ~503 tickers as of a given date
            members = get_sp500_composition("2024-12-31")

            # Historical point-in-time
            members_2010 = get_sp500_composition("2010-06-30")

            # Combine: prices for index constituents at a point in time
            from lakshya_qai.mcps.data_library.sources.findata.equity_prices import get_equity_prices
            tickers = get_sp500_composition("2020-01-31")
            prices  = get_equity_prices(tickers, "2020-02-01", "2020-06-30", fields=["Close"])
        """),
    },
    {
        "name": "get_file_data",
        "callable": get_file_data,
        "module": "lakshya_qai.mcps.data_library.sources.findata.file_reader",
        "tags": [
            "file", "csv", "parquet", "excel", "xlsx", "local", "flat file",
            "read", "load", "custom data", "proprietary", "vendor", "disk",
        ],
        "stub": False,
        "install_requires": ["pandas", "openpyxl (for Excel)", "pyarrow (for Parquet)"],
        "summary": (
            "Load financial time-series from a local CSV, Parquet, or Excel file. "
            "Handles date parsing, ticker filtering, and field selection in one call."
        ),
        "example": textwrap.dedent("""\
            from lakshya_qai.mcps.data_library.sources.findata.file_reader import get_file_data

            # Load an entire CSV
            df = get_file_data("data/prices.csv")

            # Parquet — filtered to two tickers and a date range
            df = get_file_data(
                "data/prices.parquet",
                tickers=["AAPL", "MSFT"],
                start_date="2023-01-01",
                end_date="2023-12-31",
                fields=["close", "volume"],
            )
        """),
    },
    {
        "name": "get_bloomberg_data",
        "callable": get_bloomberg_data,
        "module": "lakshya_qai.mcps.data_library.sources.findata.bloomberg",
        "tags": [
            "bloomberg", "blp", "blpapi", "bbg", "terminal", "reference data",
            "historical", "fundamentals", "overrides", "b-pipe", "px_last",
        ],
        "stub": True,
        "install_requires": [
            "blpapi (pip install blpapi --index-url https://bcms.bloomberg.com/pip/simple/)"
        ],
        "summary": (
            "Fetch Bloomberg data via blpapi.  Supports HistoricalDataRequest "
            "and ReferenceDataRequest with optional field overrides.  "
            "STUB — implement the session body before use."
        ),
        "example": textwrap.dedent("""\
            from lakshya_qai.mcps.data_library.sources.findata.bloomberg import get_bloomberg_data

            # Historical daily prices
            df = get_bloomberg_data(
                tickers=["AAPL US Equity", "MSFT US Equity"],
                fields=["PX_LAST", "VOLUME"],
                start_date="2024-01-01",
                end_date="2024-12-31",
            )

            # Reference data — current values
            ref = get_bloomberg_data(
                tickers=["AAPL US Equity"],
                fields=["CUR_MKT_CAP", "GICS_SECTOR_NAME"],
                request_type="ReferenceDataRequest",
            )
        """),
    },
]

_REGISTRY_BY_NAME: Dict[str, Dict[str, Any]] = {e["name"]: e for e in _REGISTRY}


# ── Doc renderer (from data-mcp) ────────────────────────────────────────

def _render_doc(entry: Dict[str, Any]) -> str:
    """Render full markdown documentation for a registry entry."""
    fn = entry["callable"]
    sig = inspect.signature(fn)
    docstring = inspect.getdoc(fn) or "(no docstring)"
    indented_doc = "\n".join("    " + line for line in docstring.splitlines())

    stub_banner = ""
    if entry["stub"]:
        stub_banner = (
            "\n> **STUB** — raises `NotImplementedError` until you "
            f"implement the session body in `{entry['module']}`.\n"
        )

    deps = ", ".join(f"`{d}`" for d in entry.get("install_requires", []))
    install_line = f"\n**Dependencies:** {deps}\n" if deps else ""

    return (
        f"## `{entry['name']}`\n\n"
        f"{entry['summary']}\n"
        f"{install_line}"
        f"{stub_banner}\n"
        f"**Import**\n"
        f"```python\n"
        f"from {entry['module']} import {entry['name']}\n"
        f"```\n\n"
        f"### Signature\n"
        f"```python\n"
        f"{entry['name']}{sig}\n"
        f"```\n\n"
        f"### Parameters & return type\n\n"
        f"{indented_doc}\n\n"
        f"### Example\n"
        f"```python\n"
        f"{entry['example'].rstrip()}\n"
        f"```\n"
    )


# ── Search scoring (from data-mcp) ──────────────────────────────────────

def _score(entry: Dict[str, Any], query: str) -> int:
    """Score a registry entry against a search query."""
    q = query.lower()
    words = set(q.split())
    score = 0
    if q in entry["name"].lower():
        score += 12
    for tag in entry["tags"]:
        if tag in q or q in tag:
            score += 4
        if words & set(tag.split()):
            score += 2
    if words & set(entry["summary"].lower().split()):
        score += 1
    return score


# ── MCP Tool Handlers ───────────────────────────────────────────────────

async def search_data_sources(args: dict) -> dict:
    """Search the data library for the right wrapper function.

    Returns function signature, full parameter docs, return type,
    and a ready-to-paste code example.

    Args (via MCP):
        query: Natural-language description of the data you need.
        top_k: Max results (default 2).
    """
    query = args.get("query", "").strip()
    top_k = int(args.get("top_k", 2))

    if not query:
        return {"content": [{"type": "text", "text": "Provide a non-empty query."}]}

    ranked = sorted(_REGISTRY, key=lambda e: _score(e, query), reverse=True)
    top = ranked[:top_k]

    if _score(top[0], query) == 0:
        names = ", ".join(f"`{e['name']}`" for e in _REGISTRY)
        return {"content": [{"type": "text", "text": f"No tools matched '{query}'. Available: {names}. Try `list_all_data_sources`."}]}

    parts = [f"# findata tools matching: `{query}`\n"]
    for entry in top:
        parts.append(_render_doc(entry))
        parts.append("\n---\n")

    return {"content": [{"type": "text", "text": "\n".join(parts)}]}


async def get_data_source_doc(args: dict) -> dict:
    """Get complete documentation for one specific data source by exact name.

    Args (via MCP):
        source_name: Exact function name (e.g., 'get_equity_prices').
    """
    name = args.get("source_name", "").strip()
    entry = _REGISTRY_BY_NAME.get(name)
    if entry is None:
        available = ", ".join(f"`{n}`" for n in _REGISTRY_BY_NAME)
        return {"content": [{"type": "text", "text": f"Unknown source `{name}`. Available: {available}"}]}
    return {"content": [{"type": "text", "text": _render_doc(entry)}]}


async def list_all_data_sources(args: dict) -> dict:
    """List every data source with a one-line summary and tags."""
    lines = ["# QAI Data Library — available wrapper functions\n"]
    for entry in _REGISTRY:
        stub_flag = "  *(stub — implement before use)*" if entry["stub"] else ""
        lines.append(
            f"### `{entry['name']}`{stub_flag}\n"
            f"- **Module:** `{entry['module']}`\n"
            f"- **Summary:** {entry['summary']}\n"
            f"- **Tags:** {', '.join(entry['tags'])}\n"
            f"- **Install:** {', '.join(entry.get('install_requires', []))}\n"
        )
    return {"content": [{"type": "text", "text": "\n".join(lines)}]}


async def request_new_data_source(args: dict) -> dict:
    """Request addition of a new data source.

    Creates a request file for the Dev Agent (Data) to pick up.

    Args (via MCP):
        source_name: Proposed function name.
        description: What data the source provides.
        sample_code: Working Python code to be modularized.
    """
    source_name = args.get("source_name", "")
    description = args.get("description", "")
    sample_code = args.get("sample_code", "")

    if not all([source_name, description, sample_code]):
        return {"content": [{"type": "text", "text": "Error: 'source_name', 'description', 'sample_code' all required."}]}

    requests_dir = _settings.project_root / "pending_requests" / "data_sources"
    requests_dir.mkdir(parents=True, exist_ok=True)

    request_id = f"data_req_{int(time.time())}_{source_name}"
    request_file = requests_dir / f"{request_id}.json"
    request_file.write_text(json.dumps({
        "request_id": request_id,
        "type": "new_data_source",
        "source_name": source_name,
        "description": description,
        "sample_code": sample_code,
        "status": "pending",
    }, indent=2))

    return {"content": [{"type": "text", "text": f"Data source request created: {request_id}\nDev Agent (Data) will integrate '{source_name}'."}]}


# ── MCP Server Factory ──────────────────────────────────────────────────

def create_data_library_mcp():
    """Create the MCP server for the data library.

    Uses the data-mcp registry pattern: curated entries with metadata,
    tags, and examples for each data wrapper function.
    """
    from claude_agent_sdk import tool, create_sdk_mcp_server

    return create_sdk_mcp_server(
        name="qai_data_library",
        tools=[
            tool(
                "search_data_sources",
                "Search the data library for the right data wrapper. "
                "Returns signature, parameter docs, return type, and copy-paste example.",
                {"query": str, "top_k": int},
            )(search_data_sources),
            tool(
                "get_data_source_doc",
                "Get complete documentation for one data source by exact name",
                {"source_name": str},
            )(get_data_source_doc),
            tool(
                "list_all_data_sources",
                "List every data source with summary and tags",
                {},
            )(list_all_data_sources),
            tool(
                "request_new_data_source",
                "Request addition of a new data source (sample code for Dev Agent to integrate)",
                {"source_name": str, "description": str, "sample_code": str},
            )(request_new_data_source),
        ],
    )
