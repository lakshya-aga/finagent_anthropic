"""Central configuration for Lakshya QAI."""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

ROOT_DIR = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    """All environment-driven settings for the QAI ecosystem."""

    model_config = SettingsConfigDict(
        env_file=str(ROOT_DIR / ".env"),
        env_prefix="QAI_",
        extra="ignore",
    )

    # ── Paths ────────────────────────────────────────────────────────────
    project_root: Path = ROOT_DIR
    notebooks_dir: Path = ROOT_DIR / "notebooks"
    signals_dir: Path = ROOT_DIR / "signals_output"
    good_practices_dir: Path = ROOT_DIR / "good_practices"
    knowledge_base_dir: Path = ROOT_DIR / "knowledge_base"

    # ── Models ───────────────────────────────────────────────────────────
    planning_model: str = "sonnet"
    coding_model: str = "sonnet"
    testing_model: str = "sonnet"
    classifier_model: str = "haiku"
    bias_audit_model: str = "sonnet"
    monitor_model: str = "sonnet"
    trading_model: str = "sonnet"
    dev_model: str = "sonnet"
    extractor_model: str = "sonnet"

    # ── Budget Caps (USD per invocation) ─────────────────────────────────
    planning_budget: float = 1.0
    coding_budget: float = 2.0
    testing_budget: float = 2.0
    classifier_budget: float = 0.10
    bias_audit_budget: float = 0.50
    monitor_budget: float = 0.50
    trading_budget: float = 0.50
    dev_budget: float = 2.0
    extractor_budget: float = 1.0

    # ── Turn Limits ──────────────────────────────────────────────────────
    planning_max_turns: int = 15
    coding_max_turns: int = 40
    testing_max_turns: int = 30
    classifier_max_turns: int = 5
    bias_audit_max_turns: int = 15
    extractor_max_turns: int = 20

    # ── Classifier ───────────────────────────────────────────────────────
    classifier_confidence_threshold: float = 0.80

    # ── GROBID ───────────────────────────────────────────────────────────
    grobid_url: str = "http://localhost:8070"

    # ── Vector Store ─────────────────────────────────────────────────────
    chroma_persist_dir: Path = ROOT_DIR / ".chromadb"
    embedding_model: str = "all-MiniLM-L6-v2"

    # ── Signal API ───────────────────────────────────────────────────────
    signal_api_host: str = "0.0.0.0"
    signal_api_port: int = 8000

    # ── Git (for dev agents) ─────────────────────────────────────────────
    git_repo_path: Path = ROOT_DIR
    data_lib_branch: str = "agent/data-lib"
    tools_lib_branch: str = "agent/tools-lib"


@lru_cache
def get_settings() -> Settings:
    """Return a cached singleton of the application settings."""
    return Settings()
