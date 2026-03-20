"""Tracing module — logs every agent message, tool call, and result.

Wraps claude_agent_sdk.query() to capture and display the full trace
of each agent invocation, similar to OpenAI SDK's request logging.

Usage:
    from lakshya_qai.tracing import traced_query, set_trace_level

    # Enable detailed tracing
    set_trace_level("full")  # "off", "summary", "full"

    # Use traced_query as a drop-in replacement for query
    async for message in traced_query(prompt=..., options=..., agent_name="planner"):
        ...
"""

from __future__ import annotations

import json
import time
from typing import AsyncIterator

from rich.console import Console
from rich.panel import Panel
from rich.syntax import Syntax
from rich.table import Table
from rich.text import Text

from claude_agent_sdk import (
    query,
    ClaudeAgentOptions,
    AssistantMessage,
    UserMessage,
    SystemMessage,
    ResultMessage,
    TextBlock,
    ThinkingBlock,
    ToolUseBlock,
    ToolResultBlock,
)

console = Console()

# ── Trace level ──────────────────────────────────────────────────────────

_trace_level = "summary"  # "off", "summary", "full"


def set_trace_level(level: str):
    """Set tracing verbosity: 'off', 'summary', or 'full'."""
    global _trace_level
    assert level in ("off", "summary", "full"), f"Invalid trace level: {level}"
    _trace_level = level


def get_trace_level() -> str:
    return _trace_level


# ── Colors per agent ────────────────────────────────────────────────────

_AGENT_COLORS = {
    "classifier": "bright_cyan",
    "planner": "bright_green",
    "coder": "bright_yellow",
    "tester": "bright_magenta",
    "extractor": "bright_blue",
    "bias_audit": "bright_red",
    "monitor": "white",
    "trading": "orange3",
    "dev_tools": "dark_orange",
    "dev_data": "dark_orange",
}


def _agent_color(name: str) -> str:
    return _AGENT_COLORS.get(name, "white")


# ── Traced query ─────────────────────────────────────────────────────────

async def traced_query(
    *,
    prompt: str,
    options: ClaudeAgentOptions | None = None,
    agent_name: str = "agent",
) -> AsyncIterator:
    """Drop-in replacement for claude_agent_sdk.query() that logs all messages.

    Yields the same messages as query(), but prints a trace to the console.
    """
    if _trace_level == "off":
        async for msg in query(prompt=prompt, options=options):
            yield msg
        return

    color = _agent_color(agent_name)
    turn = 0
    tool_calls = 0
    text_chars = 0
    start_time = time.time()

    console.print(Panel(
        f"[bold {color}]{agent_name.upper()}[/bold {color}] starting\n"
        f"[dim]Model: {options.model if options else 'default'} | "
        f"Max turns: {options.max_turns if options else '?'} | "
        f"Budget: ${options.max_budget_usd if options else '?'}[/dim]",
        border_style=color,
        expand=False,
    ))

    if _trace_level == "full":
        console.print(f"  [{color}]Prompt:[/{color}]")
        # Show first 500 chars of prompt
        preview = prompt[:500] + ("..." if len(prompt) > 500 else "")
        console.print(f"  [dim]{preview}[/dim]\n")

    async for message in query(prompt=prompt, options=options):
        if isinstance(message, AssistantMessage):
            turn += 1

            for block in message.content:
                if isinstance(block, TextBlock):
                    text_chars += len(block.text)
                    if _trace_level == "full":
                        preview = block.text[:300] + ("..." if len(block.text) > 300 else "")
                        console.print(f"  [{color}][Turn {turn}] Text:[/{color}] {preview}")
                    elif _trace_level == "summary":
                        if turn <= 2 or turn % 5 == 0:
                            preview = block.text[:100] + ("..." if len(block.text) > 100 else "")
                            console.print(f"  [{color}][Turn {turn}] {preview}[/{color}]")

                elif isinstance(block, ThinkingBlock):
                    if _trace_level == "full":
                        preview = block.thinking[:200] + ("..." if len(block.thinking) > 200 else "")
                        console.print(f"  [{color}][Turn {turn}] Thinking:[/{color}] [dim]{preview}[/dim]")

                elif isinstance(block, ToolUseBlock):
                    tool_calls += 1
                    if _trace_level == "full":
                        input_str = json.dumps(block.input, indent=2, default=str)
                        if len(input_str) > 500:
                            input_str = input_str[:500] + "..."
                        console.print(
                            f"  [{color}][Turn {turn}] Tool call #{tool_calls}:[/{color}] "
                            f"[bold]{block.name}[/bold]"
                        )
                        console.print(f"    [dim]{input_str}[/dim]")
                    elif _trace_level == "summary":
                        # Compact: just tool name + key args
                        key_args = {k: str(v)[:50] for k, v in list(block.input.items())[:3]}
                        console.print(
                            f"  [{color}][Turn {turn}] >> {block.name}({key_args})[/{color}]"
                        )

                elif isinstance(block, ToolResultBlock):
                    if _trace_level == "full":
                        content = block.content
                        if isinstance(content, str):
                            preview = content[:300] + ("..." if len(content) > 300 else "")
                        elif isinstance(content, list):
                            preview = json.dumps(content, default=str)[:300]
                        else:
                            preview = str(content)[:300]
                        error_flag = " [red]ERROR[/red]" if block.is_error else ""
                        console.print(
                            f"  [{color}][Turn {turn}] Tool result:{error_flag}[/{color}] "
                            f"[dim]{preview}[/dim]"
                        )

        elif isinstance(message, ResultMessage):
            elapsed = time.time() - start_time
            cost = message.total_cost_usd or 0.0
            usage = message.usage or {}

            table = Table(
                title=f"{agent_name.upper()} completed",
                border_style=color,
                show_header=False,
                expand=False,
            )
            table.add_column("Metric", style="bold")
            table.add_column("Value")
            table.add_row("Duration", f"{elapsed:.1f}s (API: {message.duration_api_ms/1000:.1f}s)")
            table.add_row("Turns", str(message.num_turns))
            table.add_row("Tool calls", str(tool_calls))
            table.add_row("Cost", f"${cost:.4f}")
            table.add_row("Tokens in", str(usage.get("input_tokens", "?")))
            table.add_row("Tokens out", str(usage.get("output_tokens", "?")))
            if usage.get("cache_read_input_tokens"):
                table.add_row("Cache hits", str(usage["cache_read_input_tokens"]))
            table.add_row("Stop reason", message.stop_reason or "?")
            if message.is_error:
                table.add_row("Error", "[red]Yes[/red]")
            console.print(table)
            console.print()

        yield message
