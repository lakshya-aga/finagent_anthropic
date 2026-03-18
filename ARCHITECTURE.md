# Lakshya QAI — Agentic AI Ecosystem for Quant Finance

## System Overview

An agentic AI ecosystem that takes user requests (text + files: pdf, txt, ipynb, py) and orchestrates a pipeline of specialized agents for quantitative finance research, signal generation, monitoring, and trade advisory.

---

## High-Level Architecture

```
User Request (text + pdf/txt/ipynb/py)
        │
        ▼
┌─────────────────────┐
│   Artifact Classifier│
│   (confidence-gated) │──► confidence < threshold → ask user to confirm
└────────┬────────────┘
         │
    ┌────┴──────────┬──────────────────┐
    ▼               ▼                  ▼
Research Paper   Research Tool      Signal
    │               │                  │
    ▼               ▼                  ▼
Vector Store    Dev Agent         Notebook Pipeline
(Knowledge      (agent branch     (Plan → Code →
 Base)           + human gate)     Test → Extract →
                                   Bias Audit →
                                   Human Gate →
                                   Dashboard)
```

---

## 1. Infrastructure Layer: MCPs

### 1.1 QAI Tools Library MCP (mlfinlab-style)
- Python library with enforced docstring standard (params, return types, examples)
- Docstrings → auto-generated documentation → vectorized and searchable via MCP
- Read-only access for coding/testing agents
- `request_new_tool(sample_code, description)` → routes to Dev Agent (Tools)

### 1.2 Data Library MCP
- Wrapper functions with predictable output signatures over:
  - yfinance, blpapi, Fama-French factors, alternative data sources
- Same docstring → documentation → vector search pattern
- Read-only access for coding/testing agents
- `request_new_data_source(sample_code, description)` → routes to Dev Agent (Data)

### 1.3 Knowledge Base MCP
- Vector store over research papers (PDFs) and text documents
- Ingestion via structured extraction pipeline (see §3)
- Semantic search for planning and coding agents

### 1.4 Good Practices Store
- Growing collection of `.md` files with quant best practices
- Examples: covariance shrinkage, transaction cost modeling, regime detection caveats
- Read access for planning, coding, testing, and bias audit agents

---

## 2. Agent Layer

### 2.1 Artifact Classifier
- Classifies uploaded files into: **research paper**, **research tool**, **signal**
- Outputs a confidence score
- **Below confidence threshold → asks user to confirm classification**
- Routes to appropriate downstream pipeline

### 2.2 Dev Agent — Data Library
- Receives requests to add new data sources
- Modularizes sample code into library-standard wrapper functions
- Enforces docstring standard via prompt instructions
- **Commits to `agent/data-lib` branch (creates if not exists)**
- **Human gate: merge request requires manual approval**

### 2.3 Dev Agent — Tools Library
- Receives requests to add new quant tools
- Modularizes sample code into library-standard functions
- Enforces docstring standard via prompt instructions
- **Commits to `agent/tools-lib` branch (creates if not exists)**
- **Human gate: merge request requires manual approval**

### 2.4 Planning Agent
- **Input:** User request + read access to Knowledge Base MCP + Good Practices
- **Output:** Structured research notebook outline (enforced format)
- Output format:

```
## Notebook Plan
### Objective
<one-liner>
### Data Requirements
- source: <mcp_function>, frequency: <>, date_range: <>
### Methodology
- Step 1: ...
- Step 2: ...
### Expected Outputs
- figures: [...]
- tables: [...]
- signal_definition: <if applicable>
### Known Pitfalls
- <from good practices store>
```

### 2.5 Coding Agent
- **Input:** Structured plan from Planning Agent
- **Access:** QAI Tools MCP (read), Data Library MCP (read), Good Practices (.md)
- **Tools:** `write_cell`, `delete_cell`, `edit_cell`
- Builds the notebook cell by cell following the plan

### 2.6 Test & Edit Agent
- **Input:** Completed notebook from Coding Agent
- **Access:** QAI Tools MCP (read), Data Library MCP (read), Good Practices (.md)
- **Tools:** `write_cell`, `delete_cell`, `edit_cell`, `install_package`, `run_notebook`
- Runs notebook with "Run All"
- Iterates on errors until clean execution
- **On success → triggers Notebook-to-Module Extractor**

### 2.7 Notebook-to-Module Extractor
- **Triggered after:** Test & Edit Agent passes
- Extracts signal logic from validated notebook into a standalone `.py` module
- Module follows a standard interface:

```python
class Signal:
    def __init__(self, config: dict): ...
    def compute(self, as_of_date: date) -> pd.Series: ...
    def backtest(self, start: date, end: date) -> pd.DataFrame: ...
```

- Notebook is preserved as documentation/research artifact
- Module is what gets served via the Signal API

### 2.8 Bias Audit Agent
- **Triggered after:** Test & Edit Agent passes + module extraction
- **Checks for and warns about:**
  - **Lookahead bias:** Future data leaking into past computations (e.g., using full-sample statistics for normalization, forward-filled data used before publication date)
  - **Survivorship bias:** Universe construction ignoring delisted/bankrupt securities
  - **Data snooping / overfitting:** Excessive parameter tuning, no out-of-sample holdout, too many strategy variants tested without multiple-testing correction (Bonferroni, BHY)
  - **Selection bias:** Cherry-picked backtest windows, favorable market regimes only
  - **Look-ahead in features:** Using data that wouldn't have been available at signal generation time (e.g., restated financials, revised economic data)
- **Output:** Warning report with severity levels (CRITICAL / WARNING / INFO)
- **Does NOT block** — only produces warnings for human review
- **After this → Human Gate**

### 2.9 Human Gate (Signal Approval)
- Human reviews:
  - The research notebook
  - The extracted `.py` module
  - The Bias Audit Agent's warning report
- **Approve** → Signal goes live on Dashboard + API
- **Reject** → Feedback loops back to user/planning agent

### 2.10 Performance Monitor Agent
- **Runs continuously** on all live signals
- Tracks per-signal: live PnL, Sharpe ratio, drawdown, turnover
- **Analyzes:**
  - PnL attribution and decomposition
  - The original research notebook(s) that produced the signal
  - Current market events / regime context
  - What is going right and why
  - What is going wrong and why
- **Outputs:**
  - Signal health report
  - Recommendation: **CONTINUE** / **REVIEW** / **PAUSE**
  - If REVIEW or PAUSE → notifies user with justification
- Live PnL displayed on Dashboard per signal

### 2.11 Trading Agent (Advisory Only)
- **Current scope:** Suggestion system only — no execution
- Accesses signal values via Signal APIs
- Produces trade recommendations with rationale
- **Future additions (not in current scope):**
  - Risk management module
  - Position sizing
  - Microstructure analysis suite
  - Stop-loss signal integration
  - Execution integration with broker

---

## 3. Document Extraction Pipeline

Hybrid approach using best-in-class tools for each dimension:

### Pipeline Architecture

```
PDF Input
    │
    ├─► GROBID ──────────► Section-level structure
    │                       (abstract, methodology, results,
    │                        conclusions, references, authors)
    │                       Output: TEI-XML → parsed to sections
    │
    ├─► Docling ─────────► High-fidelity table extraction
    │                       (97.9% accuracy on complex tables)
    │                       Output: structured table data
    │
    └─► Nougat ──────────► Mathematical equation extraction
                            (LaTeX output)
                            Output: equations as LaTeX strings
```

### Merge & Embed

```
Parsed Sections + Tables + Equations
    │
    ▼
Section-aware chunking:
  - Abstract     → single chunk with metadata {type: "abstract"}
  - Methodology  → chunked by subsection {type: "methodology"}
  - Results      → chunked by subsection, tables as separate chunks {type: "results"}
  - Tables       → individual chunks with caption {type: "table"}
  - Equations    → grouped with surrounding context {type: "equation"}
  - References   → structured list {type: "references"}
    │
    ▼
Embedding + Vector Store (with section-type metadata for filtered search)
```

### Tool Selection Rationale

| Tool | Role | Why |
|------|------|-----|
| **GROBID** | Section structure | Purpose-built for academic papers; 68 label types; understands paper anatomy natively; production-proven at scale (ResearchGate, CERN) |
| **Docling** | Table extraction | 97.9% accuracy on complex tables (best-in-class); runs locally; MIT license; fast on CPU |
| **Nougat** | Equation parsing | Best tool for LaTeX equation extraction from PDFs; trained on arXiv/PubMed |

For non-PDF text files (.txt): direct chunking with overlap, no special parsing needed.

---

## 4. Dashboard

### Signal Dashboard
- **Per signal:**
  - Timeseries chart of signal values over time
  - Current signal value (latest)
  - Live PnL timeseries
  - Signal health status from Performance Monitor Agent (CONTINUE / REVIEW / PAUSE)
  - Link to source research notebook
  - Bias Audit warnings summary

### API Layer
- Each approved signal exposed as a REST API:
  - `GET /signals/{signal_id}/current` → current value
  - `GET /signals/{signal_id}/timeseries?start=&end=` → historical values
  - `GET /signals/{signal_id}/pnl?start=&end=` → PnL timeseries
  - `GET /signals/{signal_id}/health` → latest health report
- Trading Agent consumes these APIs

---

## 5. Workflow: End-to-End Signal Lifecycle

```
1. User uploads file + request
       │
2. Artifact Classifier (confidence-gated, user fallback)
       │
       ├── Research Paper → Extraction Pipeline → Vector Store
       ├── Research Tool  → Dev Agent → agent branch → human merge gate
       └── Signal Request ──┐
                            │
3. Planning Agent (KB + good practices) → structured outline
       │
4. Coding Agent (MCPs read-only + good practices) → notebook
       │
5. Test & Edit Agent → run all → fix errors → clean notebook
       │
6. Notebook-to-Module Extractor → Signal class in .py
       │
7. Bias Audit Agent → warning report (lookahead, survivorship, snooping)
       │
8. ═══ HUMAN GATE ═══ (review notebook + module + warnings)
       │
9. Signal API deployed + Dashboard updated
       │
10. Performance Monitor Agent (continuous)
       │    - PnL tracking + attribution
       │    - Cross-reference with research + market events
       │    - Health recommendations (CONTINUE / REVIEW / PAUSE)
       │
11. Trading Agent (advisory) → trade suggestions from signal APIs
```

---

## 6. Guardrails Summary

| Guardrail | Where | Type |
|-----------|-------|------|
| Confidence threshold on classification | Artifact Classifier | Automated + user fallback |
| Agent branch + merge request | Dev Agents | Automated + human gate |
| Docstring quality enforcement | Dev Agents | Prompt-enforced |
| Structured output format | Planning Agent | Prompt-enforced |
| Run-all-without-errors | Test Agent | Automated |
| Bias audit warnings | Bias Audit Agent | Automated (advisory) |
| Human approval for signal go-live | Signal Approval | Human gate |
| Signal health monitoring | Performance Monitor | Automated + human notification |
| Trading agent is advisory only | Trading Agent | Architectural constraint |

---

## 7. Tech Stack (Planned)

| Component | Technology |
|-----------|-----------|
| Agent framework | Claude Agent SDK |
| MCP servers | Python (FastMCP or similar) |
| PDF parsing | GROBID + Docling + Nougat |
| Vector store | TBD (ChromaDB / Qdrant / Weaviate) |
| Notebook execution | nbformat + nbconvert / Jupyter kernel |
| Dashboard | TBD (Streamlit / Dash / custom React) |
| Signal API | FastAPI |
| Version control | Git (agent branches with merge gates) |
| Python environment | conda / venv with agent install permissions |
