"""Human Gate — interactive approval checkpoints.

Presents information to the human and waits for approval/rejection.
Used at two points:
1. Signal approval (after bias audit, before dashboard deployment)
2. Merge request approval (for dev agent branches)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.prompt import Confirm, Prompt

logger = logging.getLogger(__name__)
console = Console()


@dataclass
class GateDecision:
    """Result of a human gate review."""

    approved: bool
    feedback: str = ""


async def signal_approval_gate(
    notebook_path: Path,
    signal_module_path: Path,
    bias_audit_report: str,
    signal_name: str,
) -> GateDecision:
    """Present signal for human approval.

    Displays:
    1. Summary of the signal
    2. Bias audit report (warnings)
    3. Ask for approval

    Args:
        notebook_path: Path to the research notebook.
        signal_module_path: Path to the extracted .py module.
        bias_audit_report: Markdown bias audit report.
        signal_name: Name of the signal.

    Returns:
        GateDecision with approval status and optional feedback.
    """
    console.print()
    console.print(Panel(f"[bold cyan]HUMAN GATE: Signal Approval[/bold cyan]\n\nSignal: {signal_name}", expand=False))
    console.print()

    # Show notebook info
    console.print(f"[bold]Research Notebook:[/bold] {notebook_path}")
    console.print(f"[bold]Signal Module:[/bold] {signal_module_path}")
    console.print()

    # Show bias audit report
    console.print(Panel("[bold yellow]Bias Audit Report[/bold yellow]", expand=False))
    console.print(Markdown(bias_audit_report))
    console.print()

    # Show signal module preview
    if signal_module_path.exists():
        code = signal_module_path.read_text(encoding="utf-8")
        console.print(Panel("[bold]Signal Module Preview[/bold]", expand=False))
        # Show first 50 lines
        lines = code.split("\n")[:50]
        console.print("\n".join(lines))
        if len(code.split("\n")) > 50:
            console.print(f"... ({len(code.split(chr(10)))} lines total)")
    console.print()

    # Ask for approval
    approved = Confirm.ask("[bold green]Approve this signal for live deployment?[/bold green]")

    feedback = ""
    if not approved:
        feedback = Prompt.ask("Feedback (what should be changed?)", default="")

    decision = GateDecision(approved=approved, feedback=feedback)
    logger.info(
        "Signal gate decision for %s: %s%s",
        signal_name,
        "APPROVED" if approved else "REJECTED",
        f" — {feedback}" if feedback else "",
    )
    return decision


async def merge_approval_gate(
    branch_name: str,
    summary: str,
) -> GateDecision:
    """Present a dev agent's branch for merge approval.

    Args:
        branch_name: The agent branch to merge (e.g., agent/data-lib).
        summary: Summary of changes made by the dev agent.

    Returns:
        GateDecision with approval status.
    """
    console.print()
    console.print(Panel(f"[bold cyan]HUMAN GATE: Merge Request[/bold cyan]\n\nBranch: {branch_name}", expand=False))
    console.print()

    console.print("[bold]Changes Summary:[/bold]")
    console.print(Markdown(summary))
    console.print()

    # Show git diff summary
    import subprocess

    try:
        diff = subprocess.run(
            ["git", "diff", f"main...{branch_name}", "--stat"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if diff.returncode == 0:
            console.print(Panel("[bold]Git Diff (stat)[/bold]", expand=False))
            console.print(diff.stdout)
    except Exception:
        pass

    approved = Confirm.ask(f"[bold green]Approve merge of {branch_name} into main?[/bold green]")
    feedback = ""
    if not approved:
        feedback = Prompt.ask("Feedback", default="")

    return GateDecision(approved=approved, feedback=feedback)


async def classification_confirmation_gate(
    classification: str,
    confidence: float,
    reasoning: str,
    file_name: str,
) -> str:
    """Ask user to confirm a low-confidence classification.

    Args:
        classification: The classifier's guess.
        confidence: Confidence score.
        reasoning: Why the classifier chose this.
        file_name: Name of the uploaded file.

    Returns:
        The confirmed classification string.
    """
    console.print()
    console.print(Panel("[bold yellow]Classification Confirmation Needed[/bold yellow]", expand=False))
    console.print(f"[bold]File:[/bold] {file_name}")
    console.print(f"[bold]Classifier guess:[/bold] {classification} (confidence: {confidence:.0%})")
    console.print(f"[bold]Reasoning:[/bold] {reasoning}")
    console.print()

    choice = Prompt.ask(
        "Confirm classification",
        choices=["research_paper", "research_tool", "signal"],
        default=classification,
    )
    return choice
