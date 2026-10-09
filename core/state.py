"""Shared app state: the trained pipeline (cached in memory and on disk) and
bank-scoped views of the data."""
from __future__ import annotations

import hashlib

import joblib
import pandas as pd
import streamlit as st

from . import model as M
from .config import BANKS, DATA_DIR, RAW_CSV

CACHE_DIR = DATA_DIR / ".cache"
MODEL_VERSION = "v4"


def _cache_key() -> str:
    h = hashlib.sha256(RAW_CSV.read_bytes()).hexdigest()[:16]
    return f"pipeline_{MODEL_VERSION}_{h}.joblib"


@st.cache_resource(show_spinner="Cleaning data and training risk models…")
def get_pipeline() -> M.Pipeline:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    path = CACHE_DIR / _cache_key()
    if path.exists():
        try:
            return joblib.load(path)
        except Exception:
            path.unlink(missing_ok=True)
    pipe = M.run_pipeline()
    joblib.dump(pipe, path)
    return pipe


def scope_banks(emp: dict) -> list[str]:
    """Banks whose customers the employee may see."""
    if emp["role"] == "admin":
        choice = st.session_state.get("bank_scope", "ALL")
        return list(BANKS) if choice == "ALL" else [choice]
    return [emp["bank_id"]]


def scope_label(emp: dict) -> str:
    banks = scope_banks(emp)
    return "All banks" if len(banks) > 1 else BANKS[banks[0]]


def scoped(df: pd.DataFrame, emp: dict, col: str = "bank_id") -> pd.DataFrame:
    return df[df[col].isin(scope_banks(emp))]
