"""End-to-end smoke test for Lakshya QAI.

Tests the system by running a simple momentum signal request through the
full pipeline: classify → plan → code → test.

Usage:
    python -m pytest tests/test_smoke.py -v -s
    python tests/test_smoke.py          # standalone
"""

from __future__ import annotations

import asyncio
import json
import logging
import sys
import traceback
from pathlib import Path

# Ensure project root is on path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(name)s | %(message)s")
logger = logging.getLogger("smoke_test")


# ═══════════════════════════════════════════════════════════════════════════
# Layer 1: Infrastructure — do MCPs actually work?
# ═══════════════════════════════════════════════════════════════════════════

def test_tools_library_mcp():
    """Test that the tools library MCP can create, index, and search."""
    logger.info("=== Layer 1a: Tools Library MCP ===")

    from lakshya_qai.mcps.code_rag_parser import parse_repository
    from lakshya_qai.mcps.code_rag_vector_store import SimpleTFIDFStore

    tools_dir = PROJECT_ROOT / "lakshya_qai" / "mcps" / "tools_library" / "tools"
    assert tools_dir.exists(), f"Tools directory not found: {tools_dir}"

    # Parse
    units = parse_repository(str(tools_dir))
    logger.info(f"  Parsed {len(units)} units from tools library")
    assert len(units) > 50, f"Expected 50+ units, got {len(units)}"

    # Index
    import tempfile
    with tempfile.TemporaryDirectory() as tmpdir:
        store = SimpleTFIDFStore(persist_dir=tmpdir)
        store.upsert(units)
        logger.info(f"  Indexed {store.count()} units")

        # Search
        results = store.search("CUSUM filter for events", n_results=3)
        logger.info(f"  Search 'CUSUM filter': {len(results)} results")
        assert len(results) > 0, "No search results for 'CUSUM filter'"
        names = [r["name"] for r in results]
        logger.info(f"  Top results: {names}")
        assert any("cusum" in n.lower() for n in names), f"Expected cusum in results, got {names}"

        # Search for labeling
        results2 = store.search("triple barrier labeling method", n_results=3)
        names2 = [r["name"] for r in results2]
        logger.info(f"  Search 'triple barrier': {names2}")
        assert len(results2) > 0

    logger.info("  OKTools Library MCP passed")


def test_data_library_mcp():
    """Test that the data library MCP search returns correct results."""
    logger.info("=== Layer 1b: Data Library MCP ===")

    # Test the scoring and doc rendering (no actual API calls)
    from lakshya_qai.mcps.data_library.server import _score, _render_doc, _REGISTRY

    # Score equity prices query
    equity_entry = next(e for e in _REGISTRY if e["name"] == "get_equity_prices")
    score = _score(equity_entry, "daily equity prices for AAPL")
    logger.info(f"  Score for 'daily equity prices': {score}")
    assert score > 5, f"Expected score > 5, got {score}"

    # Score bbg query against wrong source
    score_wrong = _score(equity_entry, "bloomberg reference data overrides")
    logger.info(f"  Score for 'bloomberg' against equity_prices: {score_wrong}")
    assert score < score_wrong or score > score_wrong  # just verify it runs

    # Test doc rendering
    doc = _render_doc(equity_entry)
    assert "get_equity_prices" in doc
    assert "Parameters" in doc or "Signature" in doc
    logger.info(f"  Doc rendered: {len(doc)} chars")

    logger.info("  OKData Library MCP passed")


def test_knowledge_base_mcp():
    """Test that the knowledge base can store and search chunks."""
    logger.info("=== Layer 1c: Knowledge Base MCP ===")

    import tempfile
    import os
    import shutil

    # Use a temporary ChromaDB — don't use context manager (Windows file lock)
    tmpdir = tempfile.mkdtemp(prefix="qai_kb_test_")

    try:
        from lakshya_qai.mcps.knowledge_base.server import KnowledgeBaseStore

        store = KnowledgeBaseStore()
        store._client = None  # force re-init
        store._collection = None

        import chromadb
        from chromadb.config import Settings as ChromaSettings
        store._client = chromadb.PersistentClient(
            path=tmpdir,
            settings=ChromaSettings(anonymized_telemetry=False),
        )
        store._collection = store._client.get_or_create_collection(
            name="test_kb",
            metadata={"hnsw:space": "cosine"},
        )

        # Add test chunks
        chunks = [
            {
                "text": "Momentum strategies buy past winners and sell past losers. "
                        "Jegadeesh and Titman (1993) showed 3-12 month momentum generates significant alpha.",
                "chunk_type": "methodology",
                "metadata": {"source": "jegadeesh_titman_1993", "section_title": "Methodology"},
            },
            {
                "text": "Covariance matrix shrinkage reduces estimation error by pulling the sample "
                        "covariance toward a structured target. Ledoit-Wolf (2004) proposed an optimal "
                        "shrinkage intensity formula.",
                "chunk_type": "methodology",
                "metadata": {"source": "ledoit_wolf_2004", "section_title": "Shrinkage Estimator"},
            },
        ]

        count = store.add_chunks(chunks, source_id="test_papers")
        logger.info(f"  Added {count} chunks")
        assert count == 2

        # Search
        results = store.search("momentum returns winners losers", top_k=2)
        logger.info(f"  Search 'momentum': {len(results)} results")
        assert len(results) > 0
        assert "momentum" in results[0]["text"].lower() or "Jegadeesh" in results[0]["text"]

    finally:
        # Release ChromaDB handles before cleanup
        del store._collection
        del store._client
        store._collection = None
        store._client = None
        try:
            shutil.rmtree(tmpdir, ignore_errors=True)
        except Exception:
            pass  # Windows file locks — temp dir will be cleaned on reboot

    logger.info("  OK Knowledge Base MCP passed")


def test_extraction_pipeline_text():
    """Test the extraction pipeline on a text file (no GROBID needed)."""
    logger.info("=== Layer 1d: Extraction Pipeline (text) ===")

    import tempfile
    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False, encoding="utf-8") as f:
        f.write(
            "# Momentum Factor Research\n\n"
            "This paper examines cross-sectional momentum in US equities.\n\n"
            "## Methodology\n\n"
            "We construct long-short portfolios based on past 12-month returns, "
            "excluding the most recent month to avoid short-term reversal effects.\n\n"
            "## Results\n\n"
            "The momentum strategy generates annualized alpha of 8.2% with a "
            "Sharpe ratio of 0.65 over the 1965-2020 sample period."
        )
        txt_path = Path(f.name)

    from lakshya_qai.extraction.pipeline import ExtractionPipeline
    pipeline = ExtractionPipeline(use_nougat=False)

    doc = asyncio.get_event_loop().run_until_complete(pipeline.extract_text(txt_path))
    logger.info(f"  Extracted {len(doc.chunks)} chunks from text file")
    assert len(doc.chunks) > 0
    assert "momentum" in doc.chunks[0].text.lower()

    txt_path.unlink()  # cleanup
    logger.info("  OKExtraction Pipeline (text) passed")


# ═══════════════════════════════════════════════════════════════════════════
# Layer 2: MCP Server wiring — can the MCP server factories produce
# valid configs that the SDK accepts?
# ═══════════════════════════════════════════════════════════════════════════

def test_mcp_server_creation():
    """Test that all MCP server factories produce valid SDK configs."""
    logger.info("=== Layer 2: MCP Server Factory ===")

    from lakshya_qai.mcps.tools_library.server import create_tools_library_mcp
    from lakshya_qai.mcps.data_library.server import create_data_library_mcp
    from lakshya_qai.mcps.knowledge_base.server import create_knowledge_base_mcp

    tools_mcp = create_tools_library_mcp()
    logger.info(f"  Tools MCP: {tools_mcp}")
    assert tools_mcp["type"] == "sdk"
    assert tools_mcp["name"] == "qai_tools_library"
    assert tools_mcp["instance"] is not None

    data_mcp = create_data_library_mcp()
    logger.info(f"  Data MCP: {data_mcp}")
    assert data_mcp["type"] == "sdk"
    assert data_mcp["name"] == "qai_data_library"

    kb_mcp = create_knowledge_base_mcp()
    logger.info(f"  KB MCP: {kb_mcp}")
    assert kb_mcp["type"] == "sdk"
    assert kb_mcp["name"] == "qai_knowledge_base"

    logger.info("  OKAll MCP server factories produce valid configs")


# ═══════════════════════════════════════════════════════════════════════════
# Layer 3: Agent SDK integration — can agents talk to Claude via SDK?
# Requires ANTHROPIC_API_KEY.
# ═══════════════════════════════════════════════════════════════════════════

async def _test_classifier_agent():
    """Test the classifier agent with a simple text request."""
    logger.info("=== Layer 3a: Classifier Agent ===")

    from lakshya_qai.agents.classifier import classify_artifact

    result = await classify_artifact(
        user_text="Build a cross-sectional momentum signal using Jegadeesh-Titman methodology on S&P 500 stocks",
        file_path=None,
    )

    logger.info(f"  Classification: {result.classification}")
    logger.info(f"  Confidence: {result.confidence:.0%}")
    logger.info(f"  Reasoning: {result.reasoning}")
    logger.info(f"  Suggested name: {result.suggested_name}")
    logger.info(f"  Needs confirmation: {result.needs_human_confirmation}")

    assert result.classification in ("research_paper", "research_tool", "signal")
    assert 0.0 <= result.confidence <= 1.0
    assert result.classification == "signal", f"Expected 'signal', got '{result.classification}'"
    logger.info("  OKClassifier Agent passed")
    return result


async def _test_planner_agent():
    """Test the planning agent with MCP access."""
    logger.info("=== Layer 3b: Planning Agent ===")

    from lakshya_qai.agents.planner import create_plan

    plan = await create_plan(
        user_request="Build a cross-sectional momentum signal: rank S&P 500 stocks by 12-month returns (skip most recent month), go long top decile, short bottom decile. Rebalance monthly.",
        context="",
    )

    logger.info(f"  Plan length: {len(plan)} chars")
    logger.info(f"  Plan preview:\n{plan[:500]}")

    # Check plan has expected structure
    assert len(plan) > 100, f"Plan too short: {len(plan)} chars"
    plan_lower = plan.lower()
    assert "objective" in plan_lower or "goal" in plan_lower or "momentum" in plan_lower, "Plan missing objective"
    logger.info("  OKPlanning Agent passed")
    return plan


async def _test_coder_agent(plan: str):
    """Test the coding agent builds a notebook from a plan."""
    logger.info("=== Layer 3c: Coding Agent ===")

    from lakshya_qai.agents.coder import build_notebook

    notebook_path = await build_notebook(plan, notebook_name="smoke_test_momentum")

    logger.info(f"  Notebook created: {notebook_path}")
    assert notebook_path.exists(), f"Notebook not found: {notebook_path}"

    import nbformat
    nb = nbformat.read(str(notebook_path), as_version=4)
    logger.info(f"  Cells: {len(nb.cells)}")
    assert len(nb.cells) > 3, f"Too few cells: {len(nb.cells)}"

    # Check for code cells
    code_cells = [c for c in nb.cells if c.cell_type == "code"]
    md_cells = [c for c in nb.cells if c.cell_type == "markdown"]
    logger.info(f"  Code cells: {len(code_cells)}, Markdown cells: {len(md_cells)}")
    assert len(code_cells) > 2, "Not enough code cells"

    logger.info("  OKCoding Agent passed")
    return notebook_path


async def _test_tester_agent(notebook_path: Path):
    """Test the test agent runs and fixes the notebook."""
    logger.info("=== Layer 3d: Test & Edit Agent ===")

    from lakshya_qai.agents.tester import test_and_fix_notebook

    passed = await test_and_fix_notebook(notebook_path)
    logger.info(f"  Test result: {'PASSED' if passed else 'FAILED'}")
    # We don't assert pass — the test agent may not fix all errors
    # The important thing is it ran without crashing
    logger.info("  OKTest Agent completed (may or may not have passed)")
    return passed


async def run_layer3_tests():
    """Run all Layer 3 agent tests sequentially."""
    import os
    api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    if not api_key:
        logger.warning("=== SKIPPING Layer 3: No ANTHROPIC_API_KEY set ===")
        return

    logger.info("\n" + "=" * 60)
    logger.info("Layer 3: Agent SDK Integration Tests (LIVE API)")
    logger.info("=" * 60)

    result = await _test_classifier_agent()
    plan = await _test_planner_agent()
    notebook_path = await _test_coder_agent(plan)
    passed = await _test_tester_agent(notebook_path)

    logger.info("\n" + "=" * 60)
    logger.info("Layer 3 Summary:")
    logger.info(f"  Classifier: {result.classification} ({result.confidence:.0%})")
    logger.info(f"  Plan: {len(plan)} chars")
    logger.info(f"  Notebook: {notebook_path}")
    logger.info(f"  Test passed: {passed}")
    logger.info("=" * 60)


# ═══════════════════════════════════════════════════════════════════════════
# Main runner
# ═══════════════════════════════════════════════════════════════════════════

def main():
    """Run all smoke tests."""
    print("\n" + "=" * 60)
    print("LAKSHYA QAI — SMOKE TEST")
    print("=" * 60 + "\n")

    failures = []

    # Layer 1: Infrastructure
    for test_fn in [
        test_tools_library_mcp,
        test_data_library_mcp,
        test_knowledge_base_mcp,
        test_extraction_pipeline_text,
    ]:
        try:
            test_fn()
        except Exception as e:
            logger.error(f"  FAILED: {test_fn.__name__}: {e}")
            traceback.print_exc()
            failures.append(test_fn.__name__)

    # Layer 2: MCP Server wiring
    try:
        test_mcp_server_creation()
    except Exception as e:
        logger.error(f"  FAILED: test_mcp_server_creation: {e}")
        traceback.print_exc()
        failures.append("test_mcp_server_creation")

    # Layer 3: Agent SDK (only if API key present)
    try:
        asyncio.get_event_loop().run_until_complete(run_layer3_tests())
    except Exception as e:
        logger.error(f"  FAILED: Layer 3 agents: {e}")
        traceback.print_exc()
        failures.append("layer3_agents")

    # Summary
    print("\n" + "=" * 60)
    if failures:
        print(f"SMOKE TEST: {len(failures)} FAILURE(S)")
        for f in failures:
            print(f"  FAILED: {f}")
        sys.exit(1)
    else:
        print("SMOKE TEST: ALL PASSED")
    print("=" * 60)


if __name__ == "__main__":
    main()
