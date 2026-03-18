"""Trading Agent — advisory-only trade suggestion system.

Reads signal values from the Signal API and produces trade recommendations.
No execution capability — human review required.
"""

from __future__ import annotations

import logging
from datetime import date

import httpx

from claude_agent_sdk import query, ClaudeAgentOptions, AssistantMessage

from lakshya_qai.agents.prompts import TRADING_PROMPT
from lakshya_qai.config.settings import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()


async def generate_trade_suggestions(
    signal_api_base: str | None = None,
) -> str:
    """Generate trade suggestions from all live signals.

    Returns:
        Markdown-formatted trade suggestions report.
    """
    if signal_api_base is None:
        signal_api_base = f"http://localhost:{settings.signal_api_port}"

    # Fetch all signal values
    async with httpx.AsyncClient(timeout=30.0) as client:
        signals_resp = await client.get(f"{signal_api_base}/signals")

        if signals_resp.status_code != 200:
            return "Error: Could not fetch signals from API."

        signals = signals_resp.json().get("signals", [])
        if not signals:
            return "No live signals available for trade suggestions."

        # Fetch current values and health for each signal
        signal_data = []
        for sig in signals:
            sid = sig["signal_id"]
            try:
                current = await client.get(f"{signal_api_base}/signals/{sid}/current")
                health = await client.get(f"{signal_api_base}/signals/{sid}/health")
                signal_data.append({
                    "signal_id": sid,
                    "metadata": sig,
                    "current_values": current.json() if current.status_code == 200 else {},
                    "health": health.json() if health.status_code == 200 else {},
                })
            except Exception as e:
                logger.warning("Failed to fetch data for signal %s: %s", sid, e)

    prompt = f"""
Generate trade suggestions based on these live signals:

Date: {date.today()}

## Signal Data
"""
    for sd in signal_data:
        prompt += f"""
### {sd['signal_id']}
Description: {sd['metadata'].get('description', 'N/A')}
Current Values: {sd['current_values']}
Health Status: {sd['health'].get('status', 'UNKNOWN')}
"""

    prompt += "\nProduce trade suggestions following your output format."

    report = ""
    async for message in query(
        prompt=prompt,
        options=ClaudeAgentOptions(
            system_prompt=TRADING_PROMPT,
            model=settings.trading_model,
            max_turns=10,
            max_budget_usd=settings.trading_budget,
            allowed_tools=[],
        ),
    ):
        if isinstance(message, AssistantMessage):
            for block in message.content:
                if hasattr(block, "text"):
                    report += block.text

    return report
