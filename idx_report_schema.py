"""
IDX REPORT SCHEMA
- save_version_report: 1 file per versi (v2/v3/v4/v5) — DISARANKAN
- save_report: legacy grup klasik/smc (opsional)
"""

from __future__ import annotations
import os
from datetime import datetime
from typing import Optional
import pandas as pd


def save_version_report(df: pd.DataFrame, version: str) -> str:
    """
    Simpan hasil screener ke file terpisah per versi.
    version: 'v2' | 'v3' | 'v4' | 'v5'
    → idx_report_v2_YYYY-MM-DD.csv (overwrite hari yang sama)
    """
    if df is None or (isinstance(df, pd.DataFrame) and df.empty):
        return ""

    ver = str(version).lower().strip().replace(" ", "")
    if not ver.startswith("v"):
        ver = f"v{ver}"

    fname = f"idx_report_{ver}_{datetime.now().strftime('%Y-%m-%d')}.csv"
    df = df.copy()
    df.to_csv(fname, index=False)
    print(f"Hasil disimpan ke: {fname}")
    return fname


def get_version_report_path(version: str, day: Optional[str] = None) -> str:
    day = day or datetime.now().strftime("%Y-%m-%d")
    ver = str(version).lower().strip()
    if not ver.startswith("v"):
        ver = f"v{ver}"
    return f"idx_report_{ver}_{day}.csv"


# --- Legacy (boleh tetap ada) ---
def get_report_filename(group: str) -> str:
    today = datetime.now().strftime("%Y-%m-%d")
    return f"idx_master_report_{group}_{today}.csv"


def save_report(df: pd.DataFrame, strategy_name: str, group: str) -> str:
    """Legacy: klasik / smc. Lebih baik pakai save_version_report."""
    if df is None or df.empty:
        return ""
    df = df.copy()
    if "Strategy" not in df.columns:
        df["Strategy"] = strategy_name
    filename = get_report_filename(group)
    file_exists = os.path.exists(filename)
    df.to_csv(filename, mode="a" if file_exists else "w", header=not file_exists, index=False)
    return filename