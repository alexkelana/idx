"""
IDX CROSSING MOMENTUM + VOLUME SCREENER
=======================================
Setup mirip screen "Crossing Moment + Volume":

- Stochastic (14,1,3) %K naik tajam vs hari sebelumnya
  (prev di zona rendah → current tinggi / crossing up)
- RSI(14) ikut menguat vs previous
- Volume & Value (Close×Volume) melonjak vs previous
- MACD(12,26) line positif atau membaik

Output kolom selaras screenshot:
  Ticker, Stochastic, PrevStochastic, RSI, PrevRSI,
  Volume, PrevVolume, MACD, Value, (+ Close, score, alasan)
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd

try:
    import yfinance as yf
except ImportError:
    yf = None

try:
    from idx_liquidity_scanner import IdxLiquidityScanner
except ImportError:
    IdxLiquidityScanner = None


PARAMS = {
    "min_avg_value_rp": 5_000_000_000,  # rata value 20h
    "min_avg_volume": 500_000,
    "lookback_days": 90,
    "min_price": 50,
    "max_price": 50_000,
    # Stochastic (14, 1, 3) — %K period 14, smooth 1, %D 3 (filter pakai %K)
    "stoch_k": 14,
    "stoch_smooth": 1,
    "stoch_d": 3,
    "stoch_prev_max": 40.0,  # previous %K di bawah ini (belum overbought)
    "stoch_curr_min": 55.0,  # current %K minimal setelah cross
    "stoch_min_jump": 15.0,  # minimal kenaikan %K (curr - prev)
    # RSI
    "rsi_period": 14,
    "rsi_prev_max": 55.0,
    "rsi_curr_min": 50.0,
    "rsi_min_jump": 5.0,
    # Volume / value
    "min_vol_ratio": 1.8,  # volume hari ini vs kemarin
    "min_value_rp": 2_000_000_000,  # value hari ini (Close*Volume)
    # MACD
    "macd_fast": 12,
    "macd_slow": 26,
    "macd_signal": 9,
    "require_macd_positive": False,  # jika True, MACD line > 0
    "require_macd_rising": True,  # MACD line > prev MACD
    "top_n": 25,
    "account_size": 50_000_000,
    "risk_per_trade_pct": 1.0,
    "lot_size": 100,
}


def _download(ticker: str, days: int) -> pd.DataFrame | None:
    if yf is None:
        return None
    try:
        df = yf.download(
            f"{ticker}.JK",
            period=f"{max(days, 60)}d",
            interval="1d",
            progress=False,
            auto_adjust=True,
            multi_level_index=False,
        )
        if df is None or df.empty or len(df) < 35:
            return None
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        for c in ("Open", "High", "Low", "Close", "Volume"):
            if c not in df.columns:
                return None
            df[c] = pd.to_numeric(df[c], errors="coerce")
        df = df.dropna(subset=["Close", "High", "Low"])
        return df if len(df) >= 35 else None
    except Exception:
        return None


def compute_rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = (-delta).clip(lower=0.0)
    avg_gain = gain.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def compute_stochastic(
    df: pd.DataFrame,
    k_period: int = 14,
    smooth: int = 1,
    d_period: int = 3,
) -> tuple[pd.Series, pd.Series]:
    """%K (14,1,3 style) dan %D."""
    low_min = df["Low"].rolling(k_period).min()
    high_max = df["High"].rolling(k_period).max()
    raw_k = 100 * (df["Close"] - low_min) / (high_max - low_min).replace(0, np.nan)
    if smooth and smooth > 1:
        k = raw_k.rolling(smooth).mean()
    else:
        k = raw_k
    d = k.rolling(d_period).mean()
    return k, d


def compute_macd(
    close: pd.Series,
    fast: int = 12,
    slow: int = 26,
    signal: int = 9,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    ema_fast = close.ewm(span=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    hist = macd_line - signal_line
    return macd_line, signal_line, hist


def get_universe(params: dict) -> list[str]:
    # 1) Google Drive / URL dari secrets
    try:
        from idx_gdrive_data import get_ticker_universe

        remote = get_ticker_universe(fallback=[])
        if remote:
            return remote
    except Exception:
        pass

    # 2) Liquidity scanner
    if IdxLiquidityScanner is not None:
        try:
            scanner = IdxLiquidityScanner(
                min_avg_value_rp=params["min_avg_value_rp"],
                min_avg_volume=params["min_avg_volume"],
                lookback_days=20,
                max_workers=15,
            )
            u = scanner.get_liquid_universe()
            if u:
                return u
        except Exception:
            pass

    # 3) Fallback hardcode
    return [
        "BBCA", "BBRI", "BMRI", "BBNI", "TLKM", "ASII", "UNVR", "ICBP",
        "KLBF", "ANTM", "ADRO", "PTBA", "SMGR", "INTP", "PGAS", "MDKA",
        "GOTO", "AMMN", "BRIS", "ARTO", "INDF", "CPIN", "MYOR", "PWON",
        "HRUM", "ITMG", "MEDC", "EXCL", "ISAT", "TOWR", "ACES", "MAPI",
        "UNTR", "DEFI", "DFAM", "BUMI", "BRPT", "TPIA", "EMTK", "SCMA",
    ]


def analyze_ticker(ticker: str, params: dict) -> dict[str, Any] | None:
    df = _download(ticker, params["lookback_days"])
    if df is None:
        return None

    close = df["Close"]
    last = float(close.iloc[-1])
    if last < params["min_price"] or last > params["max_price"]:
        return None

    vol = df["Volume"].fillna(0.0)
    if float(vol.tail(20).mean()) < params["min_avg_volume"]:
        return None
    avg_value_20 = float((close * vol).tail(20).mean())
    if avg_value_20 < params["min_avg_value_rp"]:
        return None

    k, d = compute_stochastic(
        df,
        params["stoch_k"],
        params["stoch_smooth"],
        params["stoch_d"],
    )
    rsi = compute_rsi(close, params["rsi_period"])
    macd_line, signal_line, hist = compute_macd(
        close, params["macd_fast"], params["macd_slow"], params["macd_signal"]
    )

    if any(pd.isna(x.iloc[-1]) or pd.isna(x.iloc[-2]) for x in (k, rsi, macd_line, vol)):
        return None

    stoch = float(k.iloc[-1])
    prev_stoch = float(k.iloc[-2])
    rsi_now = float(rsi.iloc[-1])
    prev_rsi = float(rsi.iloc[-2])
    vol_now = float(vol.iloc[-1])
    prev_vol = float(vol.iloc[-2]) if float(vol.iloc[-2]) > 0 else float(vol.tail(5).mean())
    macd_now = float(macd_line.iloc[-1])
    macd_prev = float(macd_line.iloc[-2])
    value_now = last * vol_now

    # --- Filter crossing momentum + volume ---
    stoch_cross = (
        prev_stoch <= params["stoch_prev_max"]
        and stoch >= params["stoch_curr_min"]
        and (stoch - prev_stoch) >= params["stoch_min_jump"]
    )
    # alternatif: %K memotong %D ke atas hari ini
    stoch_kd_cross = False
    if not pd.isna(d.iloc[-1]) and not pd.isna(d.iloc[-2]):
        stoch_kd_cross = float(k.iloc[-2]) <= float(d.iloc[-2]) and float(k.iloc[-1]) > float(
            d.iloc[-1]
        )

    if not (stoch_cross or stoch_kd_cross):
        return None

    rsi_ok = (
        prev_rsi <= params["rsi_prev_max"]
        and rsi_now >= params["rsi_curr_min"]
        and (rsi_now - prev_rsi) >= params["rsi_min_jump"]
    )
    if not rsi_ok:
        return None

    vol_ratio = vol_now / prev_vol if prev_vol > 0 else 0.0
    if vol_ratio < params["min_vol_ratio"]:
        return None
    if value_now < params["min_value_rp"]:
        return None

    if params["require_macd_positive"] and macd_now <= 0:
        return None
    if params["require_macd_rising"] and macd_now <= macd_prev:
        return None

    # Score 0-100
    score = 0.0
    score += min(25.0, (stoch - prev_stoch) * 0.5)
    score += min(20.0, (rsi_now - prev_rsi) * 1.2)
    score += min(25.0, (vol_ratio - 1.0) * 8.0)
    if macd_now > 0:
        score += 10.0
    if macd_now > macd_prev:
        score += 10.0
    if stoch_kd_cross:
        score += 10.0
    score = round(min(100.0, score), 1)

    reasons = []
    if stoch_cross:
        reasons.append(f"Stoch cross {prev_stoch:.1f}→{stoch:.1f}")
    if stoch_kd_cross:
        reasons.append("%K cross di atas %D")
    reasons.append(f"RSI {prev_rsi:.1f}→{rsi_now:.1f}")
    reasons.append(f"Vol x{vol_ratio:.1f}")
    reasons.append(f"MACD {macd_now:.2f}")

    return {
        "Ticker": ticker,
        "Close": round(last, 2),
        "Stochastic": round(stoch, 2),
        "PrevStochastic": round(prev_stoch, 2),
        "RSI": round(rsi_now, 2),
        "PrevRSI": round(prev_rsi, 2),
        "Volume": int(vol_now),
        "PrevVolume": int(prev_vol),
        "MACD": round(macd_now, 2),
        "MACD_Signal": round(float(signal_line.iloc[-1]), 2)
        if not pd.isna(signal_line.iloc[-1])
        else None,
        "Value": int(value_now),
        "VolRatio": round(vol_ratio, 2),
        "Score": score,
        "Alasan": "; ".join(reasons),
        # Alias mirip header screenshot
        "Stochastic_14_1_3": round(stoch, 2),
        "Previous_Stochastic_14_1_3": round(prev_stoch, 2),
        "RSI_14": round(rsi_now, 2),
        "Previous_RSI_14": round(prev_rsi, 2),
        "Previous_Volume": int(prev_vol),
        "MACD_12_26": round(macd_now, 2),
    }


def run_crossing_momentum_screener(user_params: dict | None = None) -> pd.DataFrame:
    params = PARAMS.copy()
    if user_params:
        params.update(user_params)

    print("=" * 70)
    print("IDX CROSSING MOMENTUM + VOLUME SCREENER")
    print("Stochastic(14,1,3) cross-up · RSI naik · Volume spike · MACD")
    print("=" * 70)

    if yf is None:
        print("ERROR: yfinance tidak terpasang.")
        return pd.DataFrame()

    universe = get_universe(params)
    print(f"Universe: {len(universe)} emiten\n")

    rows: list[dict] = []
    for i, sym in enumerate(universe, 1):
        print(f"  [{i}/{len(universe)}] {sym}...", end="\r")
        try:
            r = analyze_ticker(sym, params)
            if r:
                rows.append(r)
        except Exception:
            continue
    print(" " * 60, end="\r")

    if not rows:
        print("Tidak ada emiten yang memenuhi Crossing Momentum + Volume hari ini.")
        return pd.DataFrame()

    out = pd.DataFrame(rows).sort_values(
        ["Score", "VolRatio", "Stochastic"], ascending=[False, False, False]
    )
    out = out.head(params["top_n"]).reset_index(drop=True)

    show_cols = [
        "Ticker",
        "Stochastic",
        "PrevStochastic",
        "RSI",
        "PrevRSI",
        "Volume",
        "PrevVolume",
        "MACD",
        "Value",
        "Close",
        "VolRatio",
        "Score",
        "Alasan",
    ]
    show_cols = [c for c in show_cols if c in out.columns]

    print("=" * 110)
    print(f"HASIL — {datetime.now().strftime('%Y-%m-%d %H:%M')} · {len(out)} setup")
    print("=" * 110)
    # format angka mirip screenshot
    disp = out[show_cols].copy()
    if "Stochastic" in disp.columns:
        disp["Stochastic"] = disp["Stochastic"].map(lambda x: f"{x:.2f}%")
        disp["PrevStochastic"] = disp["PrevStochastic"].map(lambda x: f"{x:.2f}%")
        disp["RSI"] = disp["RSI"].map(lambda x: f"{x:.2f}%")
        disp["PrevRSI"] = disp["PrevRSI"].map(lambda x: f"{x:.2f}%")
    print(disp.to_string(index=False))
    print("=" * 110)
    print(
        "Catatan: Stochastic/RSI ditampilkan dalam % seperti screen referensi. "
        "Value = Close × Volume (Rp)."
    )

    try:
        from idx_report_schema import save_version_report

        path = save_version_report(out, "xmom")
    except Exception:
        path = f"idx_report_xmom_{datetime.now().strftime('%Y-%m-%d')}.csv"
        out.to_csv(path, index=False)
    print(f"Disimpan: {path}")
    return out


if __name__ == "__main__":
    run_crossing_momentum_screener()
