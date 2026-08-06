"""
IDX REPORT SCHEMA — Pisah Klasik vs SMC
============================================================
"""

from __future__ import annotations
import pandas as pd
from datetime import datetime
import os
from typing import Optional

def get_report_filename(group: str) -> str:
    """group: 'klasik' atau 'smc'"""
    today = datetime.now().strftime("%Y-%m-%d")
    return f"idx_master_report_{group}_{today}.csv"


def save_report(
    df: pd.DataFrame,
    strategy_name: str,
    group: str,          # "klasik" atau "smc"
) -> str:
    """
    Simpan hasil screener ke file sesuai grup.
    group = "klasik" → V2/V3
    group = "smc"    → V4/V5
    """
    if df is None or df.empty:
        return ""

    df = df.copy()
    if "Strategy" not in df.columns:
        df["Strategy"] = strategy_name

    filename = get_report_filename(group)
    file_exists = os.path.exists(filename)

    # Append jika file sudah ada, tulis header hanya sekali
    df.to_csv(
        filename,
        mode="a" if file_exists else "w",
        header=not file_exists,
        index=False,
    )
    return filename