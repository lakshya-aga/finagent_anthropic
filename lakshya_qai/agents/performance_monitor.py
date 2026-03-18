"""Performance Monitor Agent — continuous signal health monitoring.

Tracks PnL, analyzes against research + market events, recommends
CONTINUE / REVIEW / PAUSE for each live signal.
"""

from __future__ import annotations

import logging
from datetime import date, timedelta

import httpx
import pandas as pd

from claude_agent_sdk import query, ClaudeAgentOptions, AssistantMessage

from lakshya_qai.agents.prompts import PERFORMANCE_MONITOR_PROMPT
from lakshya_qai.config.settings import get_settings
from lakshya_qai.signals.base import SignalHealthReport

logger = logging.getLogger(__name__)
settings = get_settings()


async def monitor_signal(
    signal_id: str,
    signal_api_base: str | None = None,
) -> SignalHealthReport:
    """Run a comprehensive health check on a live signal.

    Args:
        signal_id: ID of the signal to monitor.
        signal_api_base: Base URL for the signal API.

    Returns:
        SignalHealthReport with status and recommendation.
    """
    if signal_api_base is None:
        signal_api_base = f"http://localhost:{settings.signal_api_port}"

    # Fetch current data from Signal API
    async with httpx.AsyncClient(timeout=30.0) as client:
        pnl_resp = await client.get(
            f"{signal_api_base}/signals/{signal_id}/pnl",
            params={"start": str(date.today() - timedelta(days=365))},
        )
        meta_resp = await client.get(f"{signal_api_base}/signals/{signal_id}/current")

    pnl_data = pnl_resp.json() if pnl_resp.status_code == 200 else {}
    meta_data = meta_resp.json() if meta_resp.status_code == 200 else {}

    # Read source notebook if available
    source_notebook_content = ""
    source_nb = meta_data.get("source_notebook", "")
    if source_nb:
        from pathlib import Path
        import nbformat

        nb_path = Path(source_nb)
        if nb_path.exists():
            nb = nbformat.read(str(nb_path), as_version=4)
            for i, cell in enumerate(nb.cells[:15]):  # first 15 cells
                source_notebook_content += f"[Cell {i}] {cell.source[:300]}\n"

    prompt = f"""
Analyze the performance of signal: {signal_id}

## PnL Data (last 12 months)
{_format_pnl_summary(pnl_data)}

## Current Signal Values
{meta_data}

## Source Research Notebook (excerpt)
{source_notebook_content or "(not available)"}

Produce a comprehensive health report and recommend CONTINUE, REVIEW, or PAUSE.
"""

    report_text = ""
    async for message in query(
        prompt=prompt,
        options=ClaudeAgentOptions(
            system_prompt=PERFORMANCE_MONITOR_PROMPT,
            model=settings.monitor_model,
            max_turns=10,
            max_budget_usd=settings.monitor_budget,
            allowed_tools=[],
        ),
    ):
        if isinstance(message, AssistantMessage):
            for block in message.content:
                if hasattr(block, "text"):
                    report_text += block.text

    # Parse recommendation from report
    status = "CONTINUE"
    for keyword in ["PAUSE", "REVIEW", "CONTINUE"]:
        if f"Recommendation: {keyword}" in report_text or f"## Recommendation: {keyword}" in report_text:
            status = keyword
            break

    return SignalHealthReport(
        signal_id=signal_id,
        timestamp=str(date.today()),
        status=status,
        sharpe_ratio=pnl_data.get("sharpe_ratio"),
        max_drawdown=pnl_data.get("max_drawdown"),
        total_return=pnl_data.get("total_return"),
        analysis=report_text,
        recommendation=status,
    )


async def monitor_all_signals(signal_api_base: str | None = None) -> list[SignalHealthReport]:
    """Monitor all live signals and return health reports.

    Returns:
        List of health reports, one per signal.
    """
    if signal_api_base is None:
        signal_api_base = f"http://localhost:{settings.signal_api_port}"

    async with httpx.AsyncClient(timeout=30.0) as client:
        resp = await client.get(f"{signal_api_base}/signals")

    if resp.status_code != 200:
        logger.error("Failed to list signals: %s", resp.status_code)
        return []

    signals = resp.json().get("signals", [])
    reports = []

    for sig in signals:
        try:
            report = await monitor_signal(sig["signal_id"], signal_api_base)
            reports.append(report)
        except Exception as e:
            logger.error("Failed to monitor signal %s: %s", sig["signal_id"], e)

    return reports


def _format_pnl_summary(pnl_data: dict) -> str:
    """Format PnL data for the prompt."""
    if not pnl_data:
        return "(No PnL data available)"

    lines = [
        f"Total Return: {pnl_data.get('total_return', 'N/A')}",
        f"Sharpe Ratio: {pnl_data.get('sharpe_ratio', 'N/A')}",
        f"Max Drawdown: {pnl_data.get('max_drawdown', 'N/A')}",
        f"Period: {pnl_data.get('start', '?')} to {pnl_data.get('end', '?')}",
    ]

    data_points = pnl_data.get("data", [])
    if data_points:
        lines.append(f"Data points: {len(data_points)}")
        # Show last 10 entries
        for dp in data_points[-10:]:
            lines.append(f"  {dp}")

    return "\n".join(lines)
