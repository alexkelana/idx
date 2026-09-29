"""
IDX FUNDAMENTAL ANALYZER (hybrid)
============================================================
Prioritas sumber:
  1) fundamentals_override.csv (lokal / Google Drive)
  2) Yahoo Finance (dengan sanitasi outlier)

API tetap kompatibel dengan master / dashboard / V4–V5:
    from idx_fundamental import get_fundamental, enrich_with_fundamental

    info = get_fundamental("BBCA")
    df = enrich_with_fundamental(df)
"""

from __future__ import annotations

import os
from typing import Any, Optional

import numpy as np
import pandas as pd

try:
    import yfinance as yf
except ImportError:
    yf = None

# Rentang wajar — di luar = treat missing (hindari PBV Yahoo 10_000+)
PE_MIN, PE_MAX = -50.0, 150.0
PB_MIN, PB_MAX = 0.05, 25.0
ROE_PCT_MIN, ROE_PCT_MAX = -100.0, 150.0
DIV_PCT_MAX = 30.0



def _safe_float(x) -> Optional[float]:
    try:
        if x is None:
            return None
        v = float(x)
        if np.isnan(v):
            return None
        return v
    except (TypeError, ValueError):
        return None


def _parse_metric_number(val) -> Optional[float]:
    """Parse '4,356,000,000,000' | '6.40%' | '2,379.72' | '-'."""
    if val is None:
        return None
    try:
        if isinstance(val, (int, float)):
            if val != val:
                return None
            return float(val)
    except Exception:
        pass
    s = str(val).strip()
    if not s or s.lower() in ("-", "—", "n/a", "na", "null", "none", "#n/a", "nan"):
        return None
    s = s.replace(",", "").replace(" ", "")
    su = s.upper()
    if su.endswith("%"):
        try:
            return float(s[:-1])
        except Exception:
            return None
    mult = 1.0
    if su.endswith("T"):
        mult, s = 1e12, s[:-1]
    elif su.endswith("B"):
        mult, s = 1e9, s[:-1]
    elif su.endswith("M"):
        mult, s = 1e6, s[:-1]
    elif su.endswith("K"):
        mult, s = 1e3, s[:-1]
    try:
        return float(s) * mult
    except Exception:
        return None


def _sanitize_pe(pe: Optional[float]) -> Optional[float]:
    if pe is None:
        return None
    if pe < PE_MIN or pe > PE_MAX:
        return None
    return round(pe, 1)


def _sanitize_pb(pb: Optional[float]) -> Optional[float]:
    if pb is None or pb <= 0:
        return None
    if pb < PB_MIN or pb > PB_MAX:
        return None
    return round(pb, 2)


def _sanitize_roe_pct(roe: Optional[float]) -> Optional[float]:
    if roe is None:
        return None
    v = float(roe)
    if abs(v) <= 1.5:
        v *= 100.0
    if v < ROE_PCT_MIN or v > ROE_PCT_MAX:
        return None
    return round(v, 1)


def _sanitize_div_pct(dy: Optional[float]) -> Optional[float]:
    if dy is None:
        return None
    v = float(dy)
    if abs(v) <= 1.5:
        v *= 100.0
    if v < 0 or v > DIV_PCT_MAX:
        return None
    return round(v, 2)


def _empty_result() -> dict:
    return {
        "MarketCap": None,
        "PER": None,
        "PBV": None,
        "ROE": None,
        "DivYield": None,
        "Sector": None,
        "Valuation": "Unknown",
        "ValuationScore": 50,
        "ValuationNote": "",
    }


_OVERRIDE_ALIASES = {
    "symbol": "ticker",
    "ticker": "ticker",
    "kode": "ticker",
    "code": "ticker",
    "emiten": "ticker",
    "market cap": "market_cap",
    "marketcap": "market_cap",
    "market_cap": "market_cap",
    "mcap": "market_cap",
    "current pe ratio (ttm)": "pe",
    "current pe ratio": "pe",
    "pe ratio (ttm)": "pe",
    "pe ratio": "pe",
    "trailing pe": "pe",
    "pe": "pe",
    "per": "pe",
    "current price to book value": "pb",
    "price to book value": "pb",
    "price to book": "pb",
    "price_to_book": "pb",
    "pbv": "pb",
    "pb": "pb",
    "current price to sales (ttm)": "ps",
    "current eps (ttm)": "eps",
    "current eps": "eps",
    "eps (ttm)": "eps",
    "eps": "eps",
    "current share outstanding": "shares",
    "enterprise value": "ev",
    "average (roe 5 yr)": "roe",
    "average roe 5 yr": "roe",
    "roe 5 yr": "roe",
    "roe": "roe",
    "return on equity": "roe",
    "return_on_equity": "roe",
    "dividend yield": "div_yield",
    "div yield": "div_yield",
    "div_yield": "div_yield",
    "divyield": "div_yield",
    "debt to equity ratio (quarter)": "debt_equity",
    "debt to equity ratio": "debt_equity",
    "debt to equity": "debt_equity",
    "net profit margin (ttm)(%)": "npm",
    "net profit margin (ttm)": "npm",
    "net profit margin": "npm",
    "sector": "sector",
    "sektor": "sector",
}


def _normalize_override_columns(df: pd.DataFrame) -> dict[str, str]:
    out: dict[str, str] = {}
    for c in df.columns:
        key = str(c).strip().lower()
        logical = _OVERRIDE_ALIASES.get(key)
        if logical and logical not in out:
            out[logical] = c
    return out


def _load_override_row(ticker: str) -> Optional[dict[str, Any]]:
    """Baca satu baris fundamentals_override (header export screener / GDrive)."""
    sym = str(ticker).upper().replace(".JK", "").strip()
    frames: list[pd.DataFrame] = []

    try:
        from idx_gdrive_data import load_fundamentals_override_df

        df, _src = load_fundamentals_override_df()
        if df is not None and not getattr(df, "empty", True):
            frames.append(df)
    except Exception:
        pass

    here = os.path.dirname(os.path.abspath(__file__))
    for path in (
        os.path.join(here, "fundamentals_override.csv"),
        os.path.join(os.getcwd(), "fundamentals_override.csv"),
        "/home/workdir/artifacts/fundamentals_override.csv",
        "/home/workdir/attachments/fundamentals_override.csv",
    ):
        if os.path.isfile(path):
            try:
                frames.append(pd.read_csv(path))
            except Exception:
                pass

    for df in frames:
        if df is None or df.empty:
            continue
        first_col = df.columns[0]
        df = df[df[first_col].astype(str).str.strip().str.len() > 0]
        df = df[
            ~df[first_col]
            .astype(str)
            .str.strip()
            .str.lower()
            .isin(("nan", "none", ""))
        ]
        if df.empty:
            continue

        cmap = _normalize_override_columns(df)
        tcol = cmap.get("ticker")
        if not tcol:
            continue
        ser = (
            df[tcol]
            .astype(str)
            .str.upper()
            .str.replace(".JK", "", regex=False)
            .str.strip()
        )
        hit = df.loc[ser == sym]
        if hit.empty:
            continue
        row = hit.iloc[0]

        def cell(logical: str):
            cname = cmap.get(logical)
            if not cname:
                return None
            return row[cname]

        pe = _sanitize_pe(_parse_metric_number(cell("pe")))
        pb = _sanitize_pb(_parse_metric_number(cell("pb")))
        roe = _sanitize_roe_pct(_parse_metric_number(cell("roe")))
        mcap = _parse_metric_number(cell("market_cap"))
        if mcap is not None and mcap > 1e7:
            mcap = round(mcap / 1e9, 1)
        elif mcap is not None:
            mcap = round(mcap, 1)
        # Dividend Yield di CSV sudah pakai '%'; → parse langsung sebagai persen
        _dy_raw = cell("div_yield")
        if _dy_raw is not None and isinstance(_dy_raw, str) and "%" in str(_dy_raw):
            div_y = _parse_metric_number(_dy_raw)  # sudah %
            if div_y is not None and (div_y < 0 or div_y > DIV_PCT_MAX):
                div_y = None
            elif div_y is not None:
                div_y = round(div_y, 2)
        else:
            div_y = _sanitize_div_pct(_parse_metric_number(_dy_raw))
        sector = cell("sector")

        return {
            "MarketCap": mcap,
            "PER": pe,
            "PBV": pb,
            "ROE": roe,
            "DivYield": div_y,
            "Sector": None
            if sector is None or str(sector).lower() in ("nan", "none", "")
            else str(sector),
            "EPS": _parse_metric_number(cell("eps")),
            "DebtEquity": _parse_metric_number(cell("debt_equity")),
            "NPM": _parse_metric_number(cell("npm")),
            "source": "fundamentals_override",
        }
    return None


def _fetch_yfinance(ticker: str) -> dict[str, Any]:
    out = _empty_result()
    out["source"] = "yfinance"
    if yf is None:
        return out

    sym = str(ticker).upper().replace(".JK", "").strip() + ".JK"
    try:
        info = yf.Ticker(sym).info or {}
    except Exception:
        return out

    mcap = _safe_float(info.get("marketCap"))
    if mcap is not None and mcap > 0:
        out["MarketCap"] = round(mcap / 1_000_000_000, 1)

    pe_raw = _safe_float(info.get("trailingPE") or info.get("forwardPE"))
    out["PER"] = _sanitize_pe(pe_raw)

    pb_raw = _safe_float(info.get("priceToBook") or info.get("price_to_book"))
    out["PBV"] = _sanitize_pb(pb_raw)
    out["_pb_raw"] = pb_raw

    roe_raw = _safe_float(info.get("returnOnEquity"))
    out["ROE"] = _sanitize_roe_pct(roe_raw)

    dy_raw = _safe_float(info.get("dividendYield"))
    out["DivYield"] = _sanitize_div_pct(dy_raw)

    out["Sector"] = info.get("sector") or info.get("industry") or None
    return out



def assess_valuation(
    per: Optional[float],
    pbv: Optional[float],
    roe: Optional[float] = None,
) -> dict:
    """Menilai valuasi dari PER, PBV, ROE → label + score 0–100."""
    result = {
        "Valuation": "Unknown",
        "ValuationScore": 50,
        "ValuationNote": "Data PER/PBV tidak tersedia",
    }
    if per is None and pbv is None:
        return result

    score = 50
    notes: list[str] = []

    if per is not None:
        if per < 0:
            score -= 15
            notes.append(f"PER negatif ({per:.1f})")
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
    elif per is not None:
        notes.append("PBV n/a (sumber tidak andal / disanitasi)")

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
    dict
        MarketCap, PER, PBV, ROE, DivYield, Sector,
        Valuation, ValuationScore, ValuationNote
    """
    result = _empty_result()

    if not ticker or not isinstance(ticker, str):
        return result

    ticker = ticker.strip().upper().replace(".JK", "")
    if not ticker:
        return result

    try:
        ov = _load_override_row(ticker)
        yf_data = _fetch_yfinance(ticker)

        if ov and (ov.get("PER") is not None or ov.get("PBV") is not None):
            for k in ("MarketCap", "PER", "PBV", "ROE", "DivYield", "Sector"):
                if ov.get(k) is not None:
                    result[k] = ov[k]
            for k in ("MarketCap", "PER", "PBV", "ROE", "DivYield", "Sector"):
                if result[k] is None and yf_data.get(k) is not None:
                    result[k] = yf_data[k]
            source_tag = "override"
            if any(
                result[k] is not None
                and ov.get(k) is None
                and yf_data.get(k) is not None
                for k in ("PER", "PBV", "ROE")
            ):
                source_tag = "mixed_override_yfinance"
        else:
            for k in ("MarketCap", "PER", "PBV", "ROE", "DivYield", "Sector"):
                result[k] = yf_data.get(k)
            source_tag = "yfinance"

        valuation = assess_valuation(result["PER"], result["PBV"], result["ROE"])
        note = valuation["ValuationNote"]
        pb_raw = yf_data.get("_pb_raw")
        if result["PBV"] is None and pb_raw is not None and pb_raw > PB_MAX:
            extra = f"PBV yfinance ditolak ({pb_raw:.0f})"
            note = f"{note}; {extra}" if note else extra
        valuation["ValuationNote"] = (
            f"{note} [{source_tag}]" if note else f"[{source_tag}]"
        )
        result.update(valuation)

    except Exception:
        pass

    return result


def get_fundamental_snapshot(ticker: str) -> dict:
    """Alias untuk AI Assistant / pemanggil baru."""
    return get_fundamental(ticker)


def enrich_with_fundamental(
    df: pd.DataFrame,
    ticker_col: str = "Ticker",
) -> pd.DataFrame:
    """
    Menambahkan kolom fundamental + valuasi ke DataFrame hasil screener.
    Aman dipanggil dari V4 / V5 / master / dashboard.
    """
    if df is None or df.empty:
        return df

    df = df.copy()

    if ticker_col not in df.columns:
        for c in ("ticker", "Symbol", "symbol"):
            if c in df.columns:
                ticker_col = c
                break
        else:
            return df

    rows = []
    cache: dict[str, dict] = {}

    for _, row in df.iterrows():
        ticker = str(row[ticker_col]).strip().upper().replace(".JK", "")
        if not ticker or ticker in ("NAN", "NONE", ""):
            fund = _empty_result()
        else:
            if ticker not in cache:
                cache[ticker] = get_fundamental(ticker)
            fund = cache[ticker]
        rows.append(
            {
                "MarketCap": fund.get("MarketCap"),
                "PER": fund.get("PER"),
                "PBV": fund.get("PBV"),
                "ROE": fund.get("ROE"),
                "DivYield": fund.get("DivYield"),
                "Sector": fund.get("Sector"),
                "Valuation": fund.get("Valuation"),
                "ValuationScore": fund.get("ValuationScore"),
                "ValuationNote": fund.get("ValuationNote"),
            }
        )

    fund_df = pd.DataFrame(rows)
    for col in fund_df.columns:
        df[col] = fund_df[col].values

    return df


if __name__ == "__main__":
    print("=== TEST IDX FUNDAMENTAL (hybrid) ===\n")
    for t in ["BBCA", "ANTM", "PGEO", "POWR", "MSJA", "BBRI"]:
        info = get_fundamental(t)
        print(
            f"{t:6} | PER={str(info['PER']):>6} | PBV={str(info['PBV']):>6} | "
            f"ROE={str(info['ROE']):>6} | {info['Valuation']:12} "
            f"(Score {info['ValuationScore']})"
        )
        if info.get("ValuationNote"):
            print(f"         → {info['ValuationNote']}")
        print()
