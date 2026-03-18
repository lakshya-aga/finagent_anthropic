"""Orchestrator — main workflow engine for Lakshya QAI.

Routes user requests through the full agent pipeline:
1. Classify artifact
2. Route to appropriate pipeline
3. Manage the signal lifecycle (plan → code → test → extract → audit → gate → deploy)
4. Handle dev agent requests
5. Run performance monitoring
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from pathlib import Path

from rich.console import Console
from rich.panel import Panel

from lakshya_qai.config.settings import get_settings

logger = logging.getLogger(__name__)
console = Console()
settings = get_settings()


@dataclass
class WorkflowResult:
    """Result from a complete workflow execution."""

    success: bool
    workflow_type: str  # research_paper, research_tool, signal
    summary: str
    artifacts: dict  # paths to produced artifacts


async def process_request(
    user_text: str,
    file_path: Path | None = None,
) -> WorkflowResult:
    """Process a user request through the full QAI pipeline.

    This is the main entry point for the orchestrator.

    Args:
        user_text: The user's request text.
        file_path: Optional path to an uploaded file.

    Returns:
        WorkflowResult with status and produced artifacts.
    """
    console.print(Panel("[bold cyan]Lakshya QAI — Processing Request[/bold cyan]", expand=False))
    console.print(f"[bold]Request:[/bold] {user_text}")
    if file_path:
        console.print(f"[bold]File:[/bold] {file_path}")
    console.print()

    # ── Step 1: Classify ──────────────────────────────────────────────
    console.print("[bold]Step 1:[/bold] Classifying artifact...")
    from lakshya_qai.agents.classifier import classify_artifact

    classification = await classify_artifact(user_text, file_path)

    console.print(
        f"  Classification: [bold]{classification.classification}[/bold] "
        f"(confidence: {classification.confidence:.0%})"
    )
    console.print(f"  Reasoning: {classification.reasoning}")

    # If low confidence, ask user
    if classification.needs_human_confirmation:
        from lakshya_qai.agents.human_gate import classification_confirmation_gate

        confirmed = await classification_confirmation_gate(
            classification=classification.classification,
            confidence=classification.confidence,
            reasoning=classification.reasoning,
            file_name=file_path.name if file_path else "(no file)",
        )
        classification.classification = confirmed
        console.print(f"  [bold green]User confirmed: {confirmed}[/bold green]")

    # ── Step 2: Route ─────────────────────────────────────────────────
    if classification.classification == "research_paper":
        return await _handle_research_paper(file_path, classification.suggested_name)
    elif classification.classification == "research_tool":
        return await _handle_research_tool(user_text, file_path, classification.suggested_name)
    elif classification.classification == "signal":
        return await _handle_signal(user_text, file_path, classification.suggested_name)
    else:
        return WorkflowResult(
            success=False,
            workflow_type="unknown",
            summary=f"Unknown classification: {classification.classification}",
            artifacts={},
        )


async def _handle_research_paper(
    file_path: Path | None,
    name: str,
) -> WorkflowResult:
    """Pipeline for research papers: extract → embed in knowledge base."""
    console.print("\n[bold]Pipeline: Research Paper → Knowledge Base[/bold]")

    if not file_path:
        return WorkflowResult(
            success=False,
            workflow_type="research_paper",
            summary="No file provided for research paper classification.",
            artifacts={},
        )

    # Extract content
    console.print("  Extracting content with GROBID + Docling + Nougat...")
    from lakshya_qai.extraction.pipeline import ExtractionPipeline

    pipeline = ExtractionPipeline(grobid_url=settings.grobid_url)

    if file_path.suffix == ".pdf":
        doc = await pipeline.extract_pdf(file_path)
    else:
        doc = await pipeline.extract_text(file_path)

    console.print(f"  Extracted: {len(doc.chunks)} chunks, {doc.tables_count} tables, {doc.equations_count} equations")

    # Embed in knowledge base
    console.print("  Embedding in knowledge base...")
    from lakshya_qai.mcps.knowledge_base.server import _store

    chunks_data = [
        {"text": c.text, "chunk_type": c.chunk_type, "metadata": c.metadata}
        for c in doc.chunks
    ]
    count = _store.add_chunks(chunks_data, source_id=name)
    console.print(f"  [bold green]Added {count} chunks to knowledge base.[/bold green]")

    return WorkflowResult(
        success=True,
        workflow_type="research_paper",
        summary=f"Paper '{doc.title or name}' extracted and embedded ({count} chunks).",
        artifacts={"source": str(file_path), "chunks_count": count},
    )


async def _handle_research_tool(
    user_text: str,
    file_path: Path | None,
    name: str,
) -> WorkflowResult:
    """Pipeline for research tools: create request → dev agent → human gate."""
    console.print("\n[bold]Pipeline: Research Tool → Dev Agent[/bold]")

    # Read sample code
    sample_code = ""
    if file_path:
        sample_code = file_path.read_text(encoding="utf-8")

    # Create a tool request
    import json
    import time

    requests_dir = settings.project_root / "pending_requests" / "tools"
    requests_dir.mkdir(parents=True, exist_ok=True)

    request_id = f"tool_req_{int(time.time())}_{name}"
    request_file = requests_dir / f"{request_id}.json"
    request_file.write_text(json.dumps({
        "request_id": request_id,
        "type": "new_tool",
        "tool_name": name,
        "description": user_text,
        "sample_code": sample_code,
        "category": "uncategorized",
        "status": "pending",
    }, indent=2))

    console.print(f"  Request created: {request_id}")

    # Run dev agent
    console.print("  Running Dev Agent (Tools)...")
    from lakshya_qai.agents.dev_agents import run_dev_tools_agent

    summary = await run_dev_tools_agent(request_file)
    console.print("  Dev agent completed.")

    # Human gate for merge
    from lakshya_qai.agents.human_gate import merge_approval_gate

    decision = await merge_approval_gate(
        branch_name=settings.tools_lib_branch,
        summary=summary,
    )

    if decision.approved:
        console.print("  [bold green]Merge approved![/bold green]")
        # Merge would happen here in production
    else:
        console.print(f"  [bold red]Merge rejected.[/bold red] Feedback: {decision.feedback}")

    return WorkflowResult(
        success=decision.approved,
        workflow_type="research_tool",
        summary=f"Tool '{name}' processed. Merge: {'approved' if decision.approved else 'rejected'}.",
        artifacts={"request_file": str(request_file), "branch": settings.tools_lib_branch},
    )


async def _handle_signal(
    user_text: str,
    file_path: Path | None,
    name: str,
) -> WorkflowResult:
    """Full signal pipeline: plan → code → test → extract → audit → gate → deploy."""
    console.print("\n[bold]Pipeline: Signal Lifecycle[/bold]")

    # Gather additional context from file
    context = ""
    if file_path:
        try:
            if file_path.suffix in (".py", ".txt"):
                context = file_path.read_text(encoding="utf-8")
            elif file_path.suffix == ".ipynb":
                import nbformat

                nb = nbformat.read(str(file_path), as_version=4)
                context = "\n".join(cell.source for cell in nb.cells)
        except Exception as e:
            logger.warning("Could not read context file: %s", e)

    # ── Step 2a: Plan ─────────────────────────────────────────────────
    console.print("\n  [bold]Step 2a: Planning...[/bold]")
    from lakshya_qai.agents.planner import create_plan

    plan = await create_plan(user_text, context=context)
    console.print("  Plan created.")

    # ── Step 2b: Code ─────────────────────────────────────────────────
    console.print("\n  [bold]Step 2b: Building notebook...[/bold]")
    from lakshya_qai.agents.coder import build_notebook

    notebook_path = await build_notebook(plan, notebook_name=name)
    console.print(f"  Notebook: {notebook_path}")

    # ── Step 2c: Test ─────────────────────────────────────────────────
    console.print("\n  [bold]Step 2c: Testing notebook...[/bold]")
    from lakshya_qai.agents.tester import test_and_fix_notebook

    passed = await test_and_fix_notebook(notebook_path)

    if not passed:
        console.print("  [bold red]Notebook failed testing after max attempts.[/bold red]")
        return WorkflowResult(
            success=False,
            workflow_type="signal",
            summary=f"Signal '{name}' failed at testing stage.",
            artifacts={"notebook": str(notebook_path)},
        )
    console.print("  [bold green]Notebook passed![/bold green]")

    # ── Step 2d: Extract to .py ───────────────────────────────────────
    console.print("\n  [bold]Step 2d: Extracting signal module...[/bold]")
    from lakshya_qai.agents.extractor import extract_signal_module

    signal_module = await extract_signal_module(notebook_path, signal_name=name)
    console.print(f"  Module: {signal_module}")

    # ── Step 2e: Bias Audit ───────────────────────────────────────────
    console.print("\n  [bold]Step 2e: Running bias audit...[/bold]")
    from lakshya_qai.agents.bias_audit import audit_for_bias

    audit_report = await audit_for_bias(notebook_path, signal_module)
    console.print("  Bias audit complete.")

    # ── Step 2f: Human Gate ───────────────────────────────────────────
    console.print("\n  [bold]Step 2f: Human approval required.[/bold]")
    from lakshya_qai.agents.human_gate import signal_approval_gate

    decision = await signal_approval_gate(
        notebook_path=notebook_path,
        signal_module_path=signal_module,
        bias_audit_report=audit_report,
        signal_name=name,
    )

    if not decision.approved:
        console.print(f"  [bold red]Signal rejected.[/bold red] Feedback: {decision.feedback}")
        return WorkflowResult(
            success=False,
            workflow_type="signal",
            summary=f"Signal '{name}' rejected at human gate. Feedback: {decision.feedback}",
            artifacts={
                "notebook": str(notebook_path),
                "module": str(signal_module),
                "audit_report": audit_report,
            },
        )

    # ── Step 2g: Deploy to Dashboard ─────────────────────────────────
    console.print("\n  [bold]Step 2g: Deploying signal...[/bold]")
    from lakshya_qai.signals.api import register_signal
    from lakshya_qai.signals.base import SignalConfig

    config = SignalConfig(
        signal_id=name,
        description=user_text,
        source_notebook=str(notebook_path),
    )

    register_signal(signal_module, config)
    console.print(f"  [bold green]Signal '{name}' is now live![/bold green]")
    console.print(f"  API: http://localhost:{settings.signal_api_port}/signals/{name}/current")

    return WorkflowResult(
        success=True,
        workflow_type="signal",
        summary=f"Signal '{name}' deployed successfully.",
        artifacts={
            "notebook": str(notebook_path),
            "module": str(signal_module),
            "signal_id": name,
            "api_url": f"http://localhost:{settings.signal_api_port}/signals/{name}",
        },
    )


async def run_monitoring_cycle() -> None:
    """Run a single performance monitoring cycle on all live signals."""
    console.print("\n[bold]Running Performance Monitor...[/bold]")

    from lakshya_qai.agents.performance_monitor import monitor_all_signals
    from lakshya_qai.signals.api import update_health_report

    reports = await monitor_all_signals()

    for report in reports:
        update_health_report(report)
        status_color = {
            "CONTINUE": "green",
            "REVIEW": "yellow",
            "PAUSE": "red",
        }.get(report.status, "white")

        console.print(
            f"  {report.signal_id}: [{status_color}]{report.status}[/{status_color}]"
        )

    console.print(f"  Monitored {len(reports)} signals.")


async def run_trading_suggestions() -> str:
    """Generate trade suggestions from all live signals."""
    from lakshya_qai.agents.trading import generate_trade_suggestions

    return await generate_trade_suggestions()
