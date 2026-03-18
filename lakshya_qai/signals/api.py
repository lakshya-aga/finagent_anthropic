"""Signal API — FastAPI service exposing all live signals.

Each approved signal is served via REST endpoints for timeseries,
current values, PnL, and health reports.  The Trading Agent and
Dashboard consume this API.
"""

from __future__ import annotations

import logging
from datetime import date, timedelta
from pathlib import Path
from typing import Any

import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel

from lakshya_qai.config.settings import get_settings
from lakshya_qai.signals.base import Signal, SignalConfig, SignalHealthReport, load_signal_from_file

logger = logging.getLogger(__name__)
settings = get_settings()

app = FastAPI(
    title="Lakshya QAI Signal API",
    description="REST API for quantitative trading signals",
    version="0.1.0",
)

# ── In-memory signal registry ────────────────────────────────────────────
# In production, back this with a database

_live_signals: dict[str, Signal] = {}
_health_reports: dict[str, SignalHealthReport] = {}


# ── Response Models ──────────────────────────────────────────────────────

class SignalValueResponse(BaseModel):
    signal_id: str
    as_of_date: str
    values: dict[str, float]


class TimeseriesResponse(BaseModel):
    signal_id: str
    start: str
    end: str
    data: list[dict[str, Any]]


class PnLResponse(BaseModel):
    signal_id: str
    start: str
    end: str
    total_return: float
    sharpe_ratio: float | None
    max_drawdown: float | None
    data: list[dict[str, Any]]


class SignalListResponse(BaseModel):
    signals: list[dict[str, Any]]


# ── Endpoints ────────────────────────────────────────────────────────────

@app.get("/signals", response_model=SignalListResponse)
async def list_signals():
    """List all live signals with metadata."""
    return SignalListResponse(
        signals=[sig.metadata() for sig in _live_signals.values()]
    )


@app.get("/signals/{signal_id}/current", response_model=SignalValueResponse)
async def get_current_signal(signal_id: str):
    """Get the current (latest) value of a signal."""
    signal = _live_signals.get(signal_id)
    if not signal:
        raise HTTPException(status_code=404, detail=f"Signal '{signal_id}' not found")

    today = date.today()
    try:
        values = signal.compute(today)
        return SignalValueResponse(
            signal_id=signal_id,
            as_of_date=str(today),
            values=values.to_dict() if isinstance(values, pd.Series) else {"value": float(values)},
        )
    except Exception as e:
        logger.error("Failed to compute signal %s: %s", signal_id, e)
        raise HTTPException(status_code=500, detail=f"Signal computation failed: {e}")


@app.get("/signals/{signal_id}/timeseries", response_model=TimeseriesResponse)
async def get_signal_timeseries(
    signal_id: str,
    start: str = Query(default=None, description="Start date (YYYY-MM-DD)"),
    end: str = Query(default=None, description="End date (YYYY-MM-DD)"),
):
    """Get historical signal values over a date range."""
    signal = _live_signals.get(signal_id)
    if not signal:
        raise HTTPException(status_code=404, detail=f"Signal '{signal_id}' not found")

    start_date = date.fromisoformat(start) if start else date.today() - timedelta(days=252)
    end_date = date.fromisoformat(end) if end else date.today()

    try:
        df = signal.backtest(start_date, end_date)
        data = df.to_dict(orient="records")
        # Convert dates to strings
        for row in data:
            for k, v in row.items():
                if isinstance(v, (date, pd.Timestamp)):
                    row[k] = str(v)
        return TimeseriesResponse(
            signal_id=signal_id,
            start=str(start_date),
            end=str(end_date),
            data=data,
        )
    except Exception as e:
        logger.error("Failed to compute timeseries for %s: %s", signal_id, e)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/signals/{signal_id}/pnl", response_model=PnLResponse)
async def get_signal_pnl(
    signal_id: str,
    start: str = Query(default=None),
    end: str = Query(default=None),
):
    """Get PnL timeseries for a signal."""
    signal = _live_signals.get(signal_id)
    if not signal:
        raise HTTPException(status_code=404, detail=f"Signal '{signal_id}' not found")

    start_date = date.fromisoformat(start) if start else date.today() - timedelta(days=252)
    end_date = date.fromisoformat(end) if end else date.today()

    try:
        df = signal.backtest(start_date, end_date)

        # Calculate summary statistics
        if "pnl" in df.columns:
            pnl = df["pnl"]
            total_return = float(pnl.sum())
            if len(pnl) > 1 and pnl.std() > 0:
                sharpe = float(pnl.mean() / pnl.std() * (252 ** 0.5))
            else:
                sharpe = None
            cum = (1 + pnl).cumprod()
            max_dd = float(((cum / cum.cummax()) - 1).min()) if len(cum) > 0 else None
        else:
            total_return = 0.0
            sharpe = None
            max_dd = None

        data = df.to_dict(orient="records")
        for row in data:
            for k, v in row.items():
                if isinstance(v, (date, pd.Timestamp)):
                    row[k] = str(v)

        return PnLResponse(
            signal_id=signal_id,
            start=str(start_date),
            end=str(end_date),
            total_return=total_return,
            sharpe_ratio=sharpe,
            max_drawdown=max_dd,
            data=data,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/signals/{signal_id}/health")
async def get_signal_health(signal_id: str):
    """Get the latest health report from the Performance Monitor Agent."""
    report = _health_reports.get(signal_id)
    if not report:
        raise HTTPException(
            status_code=404,
            detail=f"No health report for '{signal_id}'. Monitor may not have run yet.",
        )
    return report.model_dump()


# ── Signal Management ────────────────────────────────────────────────────

def register_signal(signal_file: Path, config: SignalConfig) -> str:
    """Load and register a signal from a .py file.

    Called by the orchestrator after human gate approval.
    """
    signal = load_signal_from_file(signal_file, config)
    _live_signals[config.signal_id] = signal
    logger.info("Registered live signal: %s from %s", config.signal_id, signal_file)
    return config.signal_id


def update_health_report(report: SignalHealthReport) -> None:
    """Update the health report for a signal.

    Called by the Performance Monitor Agent.
    """
    _health_reports[report.signal_id] = report
    logger.info("Health report updated for %s: %s", report.signal_id, report.status)


def get_all_live_signal_ids() -> list[str]:
    """Return IDs of all live signals."""
    return list(_live_signals.keys())


def run_api():
    """Run the Signal API server."""
    import uvicorn

    uvicorn.run(
        app,
        host=settings.signal_api_host,
        port=settings.signal_api_port,
    )
