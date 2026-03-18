"""Lakshya QAI — main entry point.

Usage:
    qai process "Build a momentum signal for US large caps" --file paper.pdf
    qai monitor
    qai trade
    qai api
"""

from __future__ import annotations

import asyncio
import logging
import sys
from pathlib import Path

from rich.console import Console
from rich.logging import RichHandler

console = Console()


def setup_logging(verbose: bool = False) -> None:
    """Configure logging with Rich handler."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(message)s",
        datefmt="[%X]",
        handlers=[RichHandler(console=console, rich_tracebacks=True)],
    )


def main() -> None:
    """CLI entry point."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Lakshya QAI — Agentic AI for Quantitative Finance",
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable debug logging")

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # ── process command ──────────────────────────────────────────────
    process_parser = subparsers.add_parser("process", help="Process a user request")
    process_parser.add_argument("request", type=str, help="User request text")
    process_parser.add_argument("--file", type=str, default=None, help="Path to uploaded file")

    # ── monitor command ──────────────────────────────────────────────
    subparsers.add_parser("monitor", help="Run performance monitoring on all live signals")

    # ── trade command ────────────────────────────────────────────────
    subparsers.add_parser("trade", help="Generate trade suggestions from live signals")

    # ── api command ──────────────────────────────────────────────────
    api_parser = subparsers.add_parser("api", help="Start the Signal API server")
    api_parser.add_argument("--host", default="0.0.0.0", help="API host")
    api_parser.add_argument("--port", type=int, default=8000, help="API port")

    args = parser.parse_args()
    setup_logging(verbose=args.verbose)

    if args.command is None:
        parser.print_help()
        sys.exit(0)

    if args.command == "process":
        file_path = Path(args.file) if args.file else None
        if file_path and not file_path.exists():
            console.print(f"[bold red]File not found: {file_path}[/bold red]")
            sys.exit(1)

        from lakshya_qai.orchestrator import process_request

        result = asyncio.run(process_request(args.request, file_path))

        console.print()
        if result.success:
            console.print(f"[bold green]Success:[/bold green] {result.summary}")
        else:
            console.print(f"[bold red]Failed:[/bold red] {result.summary}")

        if result.artifacts:
            console.print("\n[bold]Artifacts:[/bold]")
            for k, v in result.artifacts.items():
                console.print(f"  {k}: {v}")

    elif args.command == "monitor":
        from lakshya_qai.orchestrator import run_monitoring_cycle

        asyncio.run(run_monitoring_cycle())

    elif args.command == "trade":
        from lakshya_qai.orchestrator import run_trading_suggestions

        report = asyncio.run(run_trading_suggestions())
        console.print(report)

    elif args.command == "api":
        from lakshya_qai.signals.api import app

        import uvicorn

        uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
