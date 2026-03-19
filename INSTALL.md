# Lakshya QAI — Installation Guide

## Prerequisites

- **Python 3.10+** (tested on 3.10–3.12)
- **Git** (for S&P 500 historical composition data)
- **GROBID** (for PDF extraction — optional, only needed for research paper ingestion)

## Quick Start

```bash
# Clone the repository
git clone https://github.com/lakshya-aga/finagent_anthropic.git
cd finagent_anthropic

# Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# .venv\Scripts\activate   # Windows

# Install with all dependencies
pip install -e ".[dev]"

# Or install from requirements.txt (exact versions)
pip install -r requirements.txt
```

## Environment Variables

Create a `.env` file in the project root:

```bash
# Required
ANTHROPIC_API_KEY=sk-ant-...

# Optional — Voyage AI for code embeddings (falls back to TF-IDF without this)
VOYAGE_API_KEY=pa-...

# Optional — override defaults
QAI_SIGNAL_API_HOST=0.0.0.0
QAI_SIGNAL_API_PORT=8000
QAI_GROBID_URL=http://localhost:8070
QAI_CLASSIFIER_CONFIDENCE_THRESHOLD=0.80
```

## External Services

### GROBID (PDF parsing — optional)

Only needed if you plan to ingest research papers as PDFs.

```bash
docker run --rm -p 8070:8070 lfoppiano/grobid:0.8.1
```

Verify: `curl http://localhost:8070/api/isalive` should return `true`.

### Bloomberg (optional)

For Bloomberg data access, install the blpapi package separately:

```bash
pip install "lakshya-qai[bloomberg]"
# or
pip install blpapi --index-url https://bcms.bloomberg.com/pip/simple/
```

Requires a Bloomberg Terminal or B-PIPE connection.

## Verify Installation

```bash
# Check CLI works
python -m lakshya_qai.main --help

# Check all modules import cleanly
python -c "from lakshya_qai.orchestrator import process_request; print('OK')"

# Start the Signal API server
python -m lakshya_qai.main api --port 8000
```

## Usage

### Process a user request (full pipeline)

```bash
# Text-only request
python -m lakshya_qai.main process "Build a momentum signal using Jegadeesh-Titman methodology"

# With a file attachment
python -m lakshya_qai.main process "Extract tools from this paper" --file paper.pdf

# With a notebook
python -m lakshya_qai.main process "Turn this into a production signal" --file research.ipynb
```

### Start the Signal API

```bash
python -m lakshya_qai.main api --host 0.0.0.0 --port 8000
```

Endpoints:
- `GET /signals` — list all live signals
- `GET /signals/{id}/current` — current signal value
- `GET /signals/{id}/timeseries?start=&end=` — historical values
- `GET /signals/{id}/pnl?start=&end=` — PnL timeseries
- `GET /signals/{id}/health` — latest health report

### Run performance monitoring

```bash
python -m lakshya_qai.main monitor
```

### Generate trade suggestions

```bash
python -m lakshya_qai.main trade
```

## Project Structure

```
lakshya_qai/
    agents/              # 11 agents (classifier, planner, coder, tester, etc.)
    config/              # Pydantic settings
    extraction/          # GROBID + Docling + Nougat PDF pipeline
    mcps/                # MCP servers
        tools_library/   # mlfinlab (71 modules) + code-RAG search
        data_library/    # findata (yfinance, S&P 500, Bloomberg, file reader)
        knowledge_base/  # ChromaDB vector store for research papers
        code_rag_parser.py
        code_rag_vector_store.py
        docstring_generator.py
    signals/             # Signal base class + FastAPI server
good_practices/          # Quant best practices (.md files)
ARCHITECTURE.md          # Full system architecture
```

## Optional Dependencies

```bash
# Excel file support
pip install "lakshya-qai[excel]"

# Parquet file support
pip install "lakshya-qai[parquet]"

# Development tools (pytest, ruff)
pip install "lakshya-qai[dev]"
```

## Troubleshooting

**`ModuleNotFoundError: No module named 'claude_agent_sdk'`**
Run `pip install claude-agent-sdk`. This is the Claude Agent SDK package.

**`ConnectionError` when processing PDFs**
GROBID is not running. Start it with `docker run --rm -p 8070:8070 lfoppiano/grobid:0.8.1`.

**`ImportError: blpapi`**
Bloomberg SDK requires a Terminal/B-PIPE. See the Bloomberg section above.

**Signal API not responding**
Check that port 8000 is free: `python -m lakshya_qai.main api --port 8001`
