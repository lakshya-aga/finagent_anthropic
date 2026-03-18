"""Base signal interface — all extracted signals must implement this.

The Notebook-to-Module Extractor agent produces .py files that follow
this interface.  The Signal API (FastAPI) discovers and serves them.
"""

from __future__ import annotations

import importlib.util
import sys
from abc import ABC, abstractmethod
from datetime import date
from pathlib import Path
from typing import Any

import pandas as pd
from pydantic import BaseModel


class SignalConfig(BaseModel):
    """Configuration passed to a signal at instantiation."""

    signal_id: str
    params: dict[str, Any] = {}
    description: str = ""
    source_notebook: str = ""  # path to the research notebook that produced this


class SignalHealthReport(BaseModel):
    """Health report from the Performance Monitor Agent."""

    signal_id: str
    timestamp: str
    status: str  # CONTINUE, REVIEW, PAUSE
    sharpe_ratio: float | None = None
    max_drawdown: float | None = None
    total_return: float | None = None
    analysis: str = ""
    recommendation: str = ""


class Signal(ABC):
    """Abstract base class for all trading signals.

    Every signal extracted from a research notebook must implement this
    interface.  The Signal API discovers implementations and serves
    current values + timeseries via REST endpoints.

    Example implementation::

        class MomentumSignal(Signal):
            def __init__(self, config: SignalConfig):
                super().__init__(config)
                self.lookback = config.params.get("lookback", 20)

            def compute(self, as_of_date: date) -> pd.Series:
                prices = self._get_prices(as_of_date)
                return prices.pct_change(self.lookback).iloc[-1]

            def backtest(self, start: date, end: date) -> pd.DataFrame:
                dates = pd.bdate_range(start, end)
                signals = [self.compute(d.date()) for d in dates]
                return pd.DataFrame({"date": dates, "signal": signals})
    """

    def __init__(self, config: SignalConfig) -> None:
        self.config = config
        self.signal_id = config.signal_id

    @abstractmethod
    def compute(self, as_of_date: date) -> pd.Series:
        """Compute the signal value(s) as of a given date.

        Args:
            as_of_date: The date to compute the signal for.
                Must not use any data after this date.

        Returns:
            A pandas Series indexed by asset/ticker with signal values.
        """
        ...

    @abstractmethod
    def backtest(self, start: date, end: date) -> pd.DataFrame:
        """Run a backtest of the signal over a date range.

        Args:
            start: Backtest start date (inclusive).
            end: Backtest end date (inclusive).

        Returns:
            DataFrame with at minimum columns: date, signal_value, pnl.
        """
        ...

    def metadata(self) -> dict[str, Any]:
        """Return signal metadata for the dashboard."""
        return {
            "signal_id": self.signal_id,
            "description": self.config.description,
            "source_notebook": self.config.source_notebook,
            "params": self.config.params,
        }


def load_signal_from_file(filepath: Path, config: SignalConfig) -> Signal:
    """Dynamically load a Signal subclass from a .py file.

    The file must contain exactly one class that inherits from Signal.

    Args:
        filepath: Path to the .py signal module.
        config: Configuration to pass to the signal constructor.

    Returns:
        An instantiated Signal object.

    Raises:
        ValueError: If no Signal subclass is found in the file.
    """
    spec = importlib.util.spec_from_file_location(filepath.stem, filepath)
    if spec is None or spec.loader is None:
        raise ValueError(f"Cannot load module from {filepath}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[filepath.stem] = module
    spec.loader.exec_module(module)

    # Find Signal subclasses
    signal_classes = []
    for attr_name in dir(module):
        attr = getattr(module, attr_name)
        if (
            isinstance(attr, type)
            and issubclass(attr, Signal)
            and attr is not Signal
        ):
            signal_classes.append(attr)

    if not signal_classes:
        raise ValueError(f"No Signal subclass found in {filepath}")

    if len(signal_classes) > 1:
        # Use the first one, but warn
        import logging

        logging.warning(
            "Multiple Signal subclasses in %s, using %s",
            filepath,
            signal_classes[0].__name__,
        )

    return signal_classes[0](config)
