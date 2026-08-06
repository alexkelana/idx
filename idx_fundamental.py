"""
IDX FUNDAMENTAL ANALYZER
============================================================
Mengambil data fundamental saham IDX dari Yahoo Finance
dan menilai apakah undervalued / fair value / overvalued.

Cara pakai:
    from idx_fundamental import get_fundamental, enrich_with_fundamental

    # Satu ticker
    info = get_fundamental("BBCA")

    # Enrich DataFrame hasil screener (V4/V5)
    df = enrich_with_fundamental(df)
"""

from __future__ import annotations
import yfinance as yf
import pandas as pd
import numpy as np
from typing import Optional


def assess_valuation(
    per: Optional[float],
    pbv: Optional[float],
    roe: Optional[float] = None,
) -> dict:
    """
    Menilai valuasi saham berdasarkan PER, PBV, dan ROE.

    Returns
    -------
    dict
        Valuation        : "Undervalued" | "Fair Value" | "Overvalued" | "Unknown"
        ValuationScore   : 0–100 (semakin tinggi = semakin murah)
        ValuationNote    : keterangan singkat
    """
    result = {
        "Valuation": "Unknown",
        "ValuationScore": 50,
        "ValuationNote": "Data PER/PBV tidak tersedia",
    }

    if per is None and pbv is None:
        return result

    score = 50
    notes = []

    # --- PER ---
    if per is not None:
        if per < 0:
            score -= 15
            notes.append("PER negatif (rugi)")
        elif per <= 8:
            score += 25
            notes.append(f"PER sangat rendah ({per:.1f})")
        elif per <= 12:
            score += 15
            notes.append(f"PER rendah ({per:.1f})")
        elif per <= 20:
            score += 5
            notes.append(f"PER wajar ({per:.1f})")
        elif per <= 30:
            score -= 10
            notes.append(f"PER agak tinggi ({per:.1f})")
        else:
            score -= 25
            notes.append(f"PER tinggi ({per:.1f})")

    # --- PBV ---
    if pbv is not None and pbv > 0:
        if pbv <= 1.0:
            score += 25
            notes.append(f"PBV sangat rendah ({pbv:.2f})")
        elif pbv <= 1.5:
            score += 15
            notes.append(f"PBV rendah ({pbv:.2f})")
        elif pbv <= 3.0:
            score += 5
            notes.append(f"PBV wajar ({pbv:.2f})")
        elif pbv <= 5.0:
            score -= 10
            notes.append(f"PBV agak tinggi ({pbv:.2f})")
        else:
            score -= 25
            notes.append(f"PBV tinggi ({pbv:.2f})")

    # --- ROE (kualitas) ---
    if roe is not None:
        if roe >= 15:
            score += 10
            notes.append(f"ROE kuat ({roe:.1f}%)")
        elif roe >= 8:
            score += 5
            notes.append(f"ROE cukup ({roe:.1f}%)")
        elif roe < 0:
            score -= 10
            notes.append("ROE negatif")

    score = max(0, min(100, int(score)))

    if score >= 70:
        label = "Undervalued"
    elif score >= 45:
        label = "Fair Value"
    else:
        label = "Overvalued"

    result["Valuation"] = label
    result["ValuationScore"] = score
    result["ValuationNote"] = "; ".join(notes) if notes else "Penilaian terbatas"

    return result


def get_fundamental(ticker: str) -> dict:
    """
    Ambil data fundamental + status valuasi untuk satu ticker IDX.

    Parameters
    ----------
    ticker : str
        Kode saham tanpa .JK (contoh: "BBCA", "CPIN")

    Returns
    -------
    dict dengan key:
        MarketCap, PER, PBV, ROE, DivYield, Sector,
        Valuation, ValuationScore, ValuationNote
    """
    result = {
        "MarketCap": None,          # miliar Rp
        "PER": None,
        "PBV": None,
        "ROE": None,                # %
        "DivYield": None,           # %
        "Sector": None,
        "Valuation": "Unknown",
        "ValuationScore": 50,
        "ValuationNote": "",
    }

    if not ticker or not isinstance(ticker, str):
        return result

    ticker = ticker.strip().upper().replace(".JK", "")

    try:
        stock = yf.Ticker(f"{ticker}.JK")
        info = stock.info or {}

        # Market Cap → miliar
        mcap = info.get("marketCap")
        if mcap is not None and mcap > 0:
            result["MarketCap"] = round(mcap / 1_000_000_000, 1)

        # PER
        pe = info.get("trailingPE") or info.get("forwardPE")
        if pe is not None:
            try:
                pe = float(pe)
                if not np.isnan(pe):
                    result["PER"] = round(pe, 1)
            except (TypeError, ValueError):
                pass

        # PBV
        pb = info.get("priceToBook")
        if pb is not None:
            try:
                pb = float(pb)
                if pb > 0 and not np.isnan(pb):
                    result["PBV"] = round(pb, 2)
            except (TypeError, ValueError):
                pass

        # ROE (%)
        roe = info.get("returnOnEquity")
        if roe is not None:
            try:
                roe = float(roe) * 100
                if not np.isnan(roe):
                    result["ROE"] = round(roe, 1)
            except (TypeError, ValueError):
                pass

        # Dividend Yield (%)
        dy = info.get("dividendYield")
        if dy is not None:
            try:
                dy = float(dy) * 100
                if dy >= 0 and not np.isnan(dy):
                    result["DivYield"] = round(dy, 2)
            except (TypeError, ValueError):
                pass

        # Sector
        result["Sector"] = info.get("sector") or info.get("industry") or None

        # Penilaian valuasi
        valuation = assess_valuation(result["PER"], result["PBV"], result["ROE"])
        result.update(valuation)

    except Exception:
        pass

    return result


def enrich_with_fundamental(
    df: pd.DataFrame,
    ticker_col: str = "Ticker",
) -> pd.DataFrame:
    """
    Menambahkan kolom fundamental + valuasi ke DataFrame hasil screener.
    Aman dipanggil dari V4 / V5 / master.
    """
    if df is None or df.empty:
        return df

    df = df.copy()

    if ticker_col not in df.columns:
        return df

    rows = []
    cache: dict[str, dict] = {}

    for _, row in df.iterrows():
        ticker = str(row[ticker_col]).strip().upper()
        if not ticker or ticker in ("NAN", "NONE", ""):
            fund = get_fundamental("")  # empty default
        else:
            if ticker not in cache:
                cache[ticker] = get_fundamental(ticker)
            fund = cache[ticker]
        rows.append(fund)

    fund_df = pd.DataFrame(rows)

    # Gabungkan (hindari duplicate column)
    for col in fund_df.columns:
        df[col] = fund_df[col].values

    return df


# ------------------------------------------------------------------
# Testing mandiri
# ------------------------------------------------------------------
if __name__ == "__main__":
    print("=== TEST IDX FUNDAMENTAL ===\n")

    for t in ["BBCA", "ANTM", "CPIN", "GOTO", "ARTO"]:
        info = get_fundamental(t)
        print(f"{t:6} | PER={str(info['PER']):>6} | PBV={str(info['PBV']):>5} | "
              f"ROE={str(info['ROE']):>6} | {info['Valuation']:12} "
              f"(Score {info['ValuationScore']})")
        if info["ValuationNote"]:
            print(f"         → {info['ValuationNote']}")
        print()