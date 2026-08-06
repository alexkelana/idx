"""
IDX TRAILING STOP CALCULATOR
============================================================
Modul untuk menghitung saran Trailing Stop berbasis ATR & range.

Cara pakai mandiri:
    python idx_trailing_stop.py

Cara pakai dari modul lain:
    from idx_trailing_stop import calculate_trailing_stop, enrich_with_trailing_stop
"""

from __future__ import annotations
import numpy as np
import pandas as pd
from typing import Optional
import yfinance as yf


def _compute_atr_from_yahoo(ticker: str, period: str = "60d", range_days: int = 20,) -> tuple[Optional[float], Optional[float], Optional[float]]:
    """
    Download data dan hitung Close, ATR(14), Range N-hari.
    Returns: (close, atr, range_pct)
    """
    try:
        symbol = ticker if ticker.endswith(".JK") else f"{ticker}.JK"
        # Pastikan period cukup panjang untuk range_days
        lookback = max(60, range_days + 30)
        df = yf.download(
            symbol,
            period=f"{lookback}d",
            interval="1d",
            progress=False,
            auto_adjust=True,
            multi_level_index=False,
        )

        if df is None or df.empty or len(df) < max(20, range_days):
            return None, None, None

        if isinstance(df.columns, pd.MultiIndex):
            try:
                df.columns = df.columns.get_level_values(0)
            except Exception:
                df = df.droplevel(-1, axis=1)

        if "High" not in df.columns or "Low" not in df.columns or "Close" not in df.columns:
            return None, None, None

        df = df.dropna()
        if len(df) < max(20, range_days):
            return None, None, None

        close = float(df["Close"].iloc[-1])

        high, low, close_s = df["High"], df["Low"], df["Close"]
        prev_close = close_s.shift(1)
        tr = pd.concat([
            high - low,
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ], axis=1).max(axis=1)
        atr = float(tr.rolling(14).mean().iloc[-1])

        # Range N hari (bukan fixed 20)
        recent = df.tail(range_days)
        range_pct = float(((recent["High"].max() - recent["Low"].min()) / close) * 100)

        if np.isnan(atr) or atr <= 0:
            atr = None
        if np.isnan(range_pct):
            range_pct = None

        return close, atr, range_pct

    except Exception:
        return None, None, None


def calculate_trailing_stop(
    close: float,
    atr: Optional[float] = None,
    range_pct_20d: Optional[float] = None,
    min_trail: float = 3.5,
    max_trail: float = 12.0,
) -> dict:
    """
    Menghitung trailing stop yang disarankan.
    """
    if close is None or close <= 0 or (isinstance(close, float) and np.isnan(close)):
        return {
            "TrailPct": 6.0,
            "TrailActivateAfterPct": 8.0,
            "TrailDistance(Rp)": 0.0,
            "TrailMethod": "Fallback (invalid close)",
        }

    if atr is not None and atr > 0 and not (isinstance(atr, float) and np.isnan(atr)):
        atr_pct = (atr / close) * 100
        if atr_pct < 2.0:
            atr_mult = 2.2
        elif atr_pct < 4.0:
            atr_mult = 1.9
        else:
            atr_mult = 1.7
        trail_from_atr = atr_pct * atr_mult
        method = f"ATR ({atr_pct:.1f}% × {atr_mult})"
    else:
        atr_pct = None
        trail_from_atr = 6.5
        method = "Default (no ATR)"

    if range_pct_20d is not None and range_pct_20d > 0 and not (
        isinstance(range_pct_20d, float) and np.isnan(range_pct_20d)
    ):
        trail_from_range = range_pct_20d * 0.55
        method += f" + Range20d ({range_pct_20d:.1f}%)"
    else:
        trail_from_range = trail_from_atr

    trail_pct = (trail_from_atr * 0.65) + (trail_from_range * 0.35)
    trail_pct = max(min_trail, min(max_trail, trail_pct))

    activate_after = max(6.0, trail_pct * 1.15)
    activate_after = min(activate_after, 15.0)

    return {
        "TrailPct": round(trail_pct, 1),
        "TrailActivateAfterPct": round(activate_after, 1),
        "TrailDistance(Rp)": round(close * trail_pct / 100, 0),
        "TrailMethod": method,
        "ATR": round(atr, 2) if atr else None,
        "ATR_Pct": round(atr_pct, 2) if atr_pct else None,
        "Range20d_Pct": round(range_pct_20d, 1) if range_pct_20d else None,
    }


def enrich_with_trailing_stop(
    df: pd.DataFrame,
    close_col: str = "Close",
    atr_col: str = "ATR",
    range_col: str = "RangePct",
    ticker_col: str = "Ticker",
) -> pd.DataFrame:
    """Menambahkan kolom trailing stop ke DataFrame hasil screener."""
    if df is None or df.empty:
        return df

    df = df.copy()
    trail_pcts, activate_pcts, trail_distances, methods = [], [], [], []
    atr_cache: dict[str, tuple] = {}

    for _, row in df.iterrows():
        close = row.get(close_col) or row.get("Entry") or row.get("Close")
        try:
            close = float(close)
        except (TypeError, ValueError):
            close = None

        atr = row.get(atr_col) or row.get("ATR14") or row.get("atr")
        try:
            atr = float(atr) if atr is not None else None
            if atr is not None and (np.isnan(atr) or atr <= 0):
                atr = None
        except (TypeError, ValueError):
            atr = None

        range_pct = row.get(range_col) or row.get("RangePct20") or row.get("range_pct")
        try:
            range_pct = float(range_pct) if range_pct is not None else None
            if range_pct is not None and np.isnan(range_pct):
                range_pct = None
        except (TypeError, ValueError):
            range_pct = None

        if (atr is None or range_pct is None) and ticker_col in row.index:
            ticker = str(row[ticker_col]).strip().upper()
            if ticker and ticker not in ("NAN", "NONE", ""):
                if ticker not in atr_cache:
                    _, auto_atr, auto_range = _compute_atr_from_yahoo(ticker)
                    atr_cache[ticker] = (auto_atr, auto_range)
                auto_atr, auto_range = atr_cache[ticker]
                if atr is None and auto_atr is not None:
                    atr = auto_atr
                if range_pct is None and auto_range is not None:
                    range_pct = auto_range

        result = calculate_trailing_stop(close=close, atr=atr, range_pct_20d=range_pct)
        trail_pcts.append(result["TrailPct"])
        activate_pcts.append(result["TrailActivateAfterPct"])
        trail_distances.append(result["TrailDistance(Rp)"])
        methods.append(result["TrailMethod"])

    df["TrailPct"] = trail_pcts
    df["TrailActivateAfterPct"] = activate_pcts
    df["TrailDistance(Rp)"] = trail_distances
    df["TrailMethod"] = methods
    return df


def run_interactive():
    """Mode interaktif: input kode emiten + range hari → tampilkan trailing stop."""
    print("=" * 60)
    print("IDX TRAILING STOP CALCULATOR")
    print("=" * 60)
    print("Masukkan kode emiten (contoh: BBCA, ARTO, CPIN)")
    print("Ketik 'q' atau 'exit' untuk keluar.\n")

    while True:
        raw = input("Kode emiten: ").strip().upper()
        if not raw:
            continue
        if raw in ("Q", "QUIT", "EXIT"):
            print("Selesai.")
            break

        ticker = raw.replace(".JK", "")

        # --- Input range hari (default 20) ---
        raw_range = input("Range hari [default 20]: ").strip()
        if not raw_range:
            range_days = 20
        else:
            try:
                range_days = int(raw_range)
                if range_days < 5:
                    print("  Minimal 5 hari. Memakai 5.")
                    range_days = 5
                elif range_days > 120:
                    print("  Maksimal 120 hari. Memakai 120.")
                    range_days = 120
            except ValueError:
                print("  Input tidak valid. Memakai default 20 hari.")
                range_days = 20

        print(f"\nMengambil data {ticker}.JK (range {range_days} hari) ...")

        close, atr, range_pct = _compute_atr_from_yahoo(ticker, range_days=range_days)

        if close is None:
            print(f"  Gagal mengambil data untuk {ticker}. Cek kode emiten / koneksi.\n")
            continue

        result = calculate_trailing_stop(close=close, atr=atr, range_pct_20d=range_pct)

        print("-" * 60)
        print(f"  Emiten              : {ticker}")
        print(f"  Harga Close         : Rp {close:,.0f}")
        if result.get("ATR"):
            print(f"  ATR (14)            : Rp {result['ATR']:,.1f} ({result.get('ATR_Pct', 0):.2f}%)")
        if result.get("Range20d_Pct") is not None:
            print(f"  Range {range_days} hari      : {result['Range20d_Pct']:.1f}%")
        print("-" * 60)
        print(f"  TrailPct            : {result['TrailPct']}%")
        print(f"  TrailActivateAfter  : +{result['TrailActivateAfterPct']}%")
        print(f"  TrailDistance       : Rp {result['TrailDistance(Rp)']:,.0f}")
        print(f"  Metode              : {result['TrailMethod']}")
        print("-" * 60)
        print()
        print("  Cara pakai:")
        print(f"  • Tunggu harga naik minimal +{result['TrailActivateAfterPct']}% dari entry")
        print(f"  • Setelah itu, jika harga turun {result['TrailPct']}% dari puncak → exit")
        print()


if __name__ == "__main__":
    run_interactive()