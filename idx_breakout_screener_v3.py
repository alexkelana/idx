"""
IDX PULLBACK & RETEST SCREENER — v3 (Buy on Weakness setelah Breakout)
======================================================================
[FIXED]
- ATR dihitung (bug NameError diperbaiki)
- Weekly regime sedikit dilonggarkan
- breakout_lookback 15 hari
- Zona Fibo 23.6% – 61.8%
- Volume koreksi hanya mempengaruhi score (bukan hard reject)
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
from idx_liquidity_scanner import IdxLiquidityScanner

# =========================================================================
# PARAMETER
# =========================================================================
PARAMS = {
    "lookback_days": 350,
    "breakout_lookback": 15,       # [FIX] sebelumnya 10
    "breakout_vol_ratio": 1.4,     # [FIX] sedikit longgar dari 1.5
    "adx_period": 14,
    "roc_period": 10,
    "stop_buffer_pct": 2.0,
    "account_size": 50_000_000,
    "risk_per_trade_pct": 1.0,
    "lot_size": 100,
    "min_rr": 1.2,                 # RR minimum
}

# =========================================================================
# UTILITAS BEI
# =========================================================================
def round_to_idx_tick(price: float) -> int:
    if pd.isna(price) or price <= 0:
        return 0
    price = float(price)
    if price < 50:
        return int(round(price))
    price = int(round(price, 0))
    if price < 200:
        return price
    elif price < 500:
        return int(round(price / 2.0) * 2)
    elif price < 2000:
        return int(round(price / 5.0) * 5)
    elif price < 5000:
        return int(round(price / 10.0) * 10)
    else:
        return int(round(price / 25.0) * 25)


def apply_ara_arb_limits(price: float, prev_close: float, is_target: bool) -> float:
    limit = 0.35 if prev_close < 200 else (0.25 if prev_close <= 5000 else 0.20)
    ara = round_to_idx_tick(prev_close * (1 + limit))
    arb = round_to_idx_tick(prev_close * (1 - limit))
    return min(price, ara) if is_target else max(price, arb)


def compute_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """[FIX] ATR yang sebelumnya tidak dihitung."""
    high, low, close = df["High"], df["Low"], df["Close"]
    prev_close = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)
    return tr.rolling(period).mean()


def compute_adx(df: pd.DataFrame, period: int = 14):
    high, low, close = df["High"], df["Low"], df["Close"]
    up = high - high.shift(1)
    down = low.shift(1) - low

    plus_dm = np.where((up > down) & (up > 0), up, 0.0)
    minus_dm = np.where((down > up) & (down > 0), down, 0.0)

    tr = pd.concat([
        high - low,
        (high - close.shift(1)).abs(),
        (low - close.shift(1)).abs(),
    ], axis=1).max(axis=1)

    atr = tr.ewm(alpha=1 / period, adjust=False).mean()
    plus_di = 100 * pd.Series(plus_dm, index=df.index).ewm(alpha=1 / period, adjust=False).mean() / atr
    minus_di = 100 * pd.Series(minus_dm, index=df.index).ewm(alpha=1 / period, adjust=False).mean() / atr
    dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di).replace(0, np.nan)
    adx = dx.ewm(alpha=1 / period, adjust=False).mean()
    return adx, plus_di, minus_di


def compute_roc(series: pd.Series, period: int = 10) -> pd.Series:
    return series.pct_change(periods=period) * 100


# =========================================================================
# UNIVERSE
# =========================================================================
def get_dynamic_liquidity_universe() -> list:
    print("=" * 60)
    print("TAHAP 1: PRE-SCREENER LIKUIDITAS")
    print("=" * 60)
    scanner = IdxLiquidityScanner(
        min_avg_value_rp=10_000_000_000,
        min_avg_volume=1_000_000,
        lookback_days=20,
        max_workers=15,
    )
    universe = scanner.get_liquid_universe()
    print("=" * 60)
    print("TAHAP 2: ANALISA PULLBACK / RETEST (V3)")
    print("=" * 60)
    return universe


# =========================================================================
# ANALISA SATU TICKER
# =========================================================================
def analyze_ticker(symbol: str, params: dict) -> dict | None:
    ticker = symbol + ".JK"
    try:
        df = yf.download(
            ticker,
            period=f"{int(params['lookback_days'])}d",
            interval="1d",
            progress=False,
            auto_adjust=True,
            multi_level_index=False,
        )
    except Exception:
        return None

    if df is None or df.empty or len(df) < 150:
        return None

    if isinstance(df.columns, pd.MultiIndex):
        try:
            df.columns = df.columns.get_level_values(0)
        except Exception:
            try:
                df = df.droplevel(-1, axis=1)
            except Exception:
                return None

    need = ["Open", "High", "Low", "Close", "Volume"]
    if any(c not in df.columns for c in need):
        return None

    df = df.dropna()
    if len(df) < 150:
        return None

    # ------------------------------------------------------------------
    # 1. REGIME MINGGUAN (dilonggarkan)
    #    Lama : Close > MA10w > MA30w AND MA10 naik
    #    Baru : Close > MA10w AND MA10w >= MA30w * 0.98 (hampir di atas)
    #           OR (Close > MA10w > MA30w)
    # ------------------------------------------------------------------
    weekly = df.resample("W").agg({
        "Open": "first", "High": "max", "Low": "min",
        "Close": "last", "Volume": "sum",
    }).dropna()

    if len(weekly) < 30:
        return None

    ma10w = weekly["Close"].rolling(10).mean()
    ma30w = weekly["Close"].rolling(30).mean()

    w_close = float(weekly["Close"].iloc[-1])
    w_ma10 = float(ma10w.iloc[-1])
    w_ma30 = float(ma30w.iloc[-1])
    w_ma10_prev = float(ma10w.iloc[-2])

    if any(np.isnan(x) for x in [w_ma10, w_ma30, w_ma10_prev]):
        return None

    # [FIX] Regime lebih realistis
    strict_bull = (w_close > w_ma10 > w_ma30) and (w_ma10 >= w_ma10_prev)
    soft_bull = (w_close > w_ma10) and (w_ma10 >= w_ma30 * 0.98)
    is_bullish_regime = strict_bull or soft_bull

    if not is_bullish_regime:
        return None

    # ------------------------------------------------------------------
    # 2. BREAKOUT TERKONFIRMASI
    # ------------------------------------------------------------------
    lookback = int(params["breakout_lookback"])
    breakout_idx = -1
    breakout_price = 0.0
    breakout_vol = 0.0

    for i in range(len(df) - 1, max(len(df) - lookback - 1, 20), -1):
        past_high = float(df["High"].iloc[i - 20:i].max())
        past_vol_avg = float(df["Volume"].iloc[i - 20:i].mean())
        if past_vol_avg <= 0:
            continue
        if (
            float(df["Close"].iloc[i]) > past_high
            and float(df["Volume"].iloc[i]) >= past_vol_avg * params["breakout_vol_ratio"]
        ):
            breakout_idx = i
            breakout_price = past_high
            breakout_vol = float(df["Volume"].iloc[i])
            break

    # Tidak ada breakout, atau breakout hari ini (belum koreksi)
    if breakout_idx < 0 or breakout_idx >= len(df) - 1:
        return None

    # ------------------------------------------------------------------
    # 3. FIBONACCI RETRACE
    # ------------------------------------------------------------------
    swing_low = float(df["Low"].iloc[max(0, breakout_idx - 20):breakout_idx].min())
    peak = float(df["High"].iloc[breakout_idx:].max())
    range_up = peak - swing_low
    if range_up <= 0:
        return None

    # [FIX] Zona lebih lebar: 23.6% – 61.8%
    fib_236 = peak - 0.236 * range_up
    fib_382 = peak - 0.382 * range_up
    fib_618 = peak - 0.618 * range_up

    last_close = float(df["Close"].iloc[-1])
    last_open = float(df["Open"].iloc[-1])
    last_vol = float(df["Volume"].iloc[-1])
    prev_vol = float(df["Volume"].iloc[-2])

    in_fibo_zone = fib_618 <= last_close <= fib_236
    above_breakout_support = last_close >= breakout_price * 0.995  # toleransi tipis

    if not (in_fibo_zone and above_breakout_support):
        return None

    # ------------------------------------------------------------------
    # 4. VOLUME KOREKSI (hanya score, bukan reject)
    # ------------------------------------------------------------------
    peak_loc = df["High"].iloc[breakout_idx:].idxmax()
    peak_idx_int = df.index.get_loc(peak_loc)
    if isinstance(peak_idx_int, slice):
        peak_idx_int = peak_idx_int.start or 0

    if peak_idx_int < len(df) - 1:
        vol_correction_avg = float(df["Volume"].iloc[peak_idx_int + 1:].mean())
    else:
        vol_correction_avg = last_vol

    is_vol_valid = vol_correction_avg < breakout_vol

    # ------------------------------------------------------------------
    # 5. MOMENTUM + ATR [FIX]
    # ------------------------------------------------------------------
    atr_series = compute_atr(df, 14)
    atr = float(atr_series.iloc[-1]) if not pd.isna(atr_series.iloc[-1]) else 0.0

    adx, plus_di, minus_di = compute_adx(df, params["adx_period"])
    curr_adx = float(adx.iloc[-1]) if not pd.isna(adx.iloc[-1]) else 0.0
    curr_pdi = float(plus_di.iloc[-1]) if not pd.isna(plus_di.iloc[-1]) else 0.0
    curr_mdi = float(minus_di.iloc[-1]) if not pd.isna(minus_di.iloc[-1]) else 0.0

    roc_val = compute_roc(df["Close"], params["roc_period"]).iloc[-1]
    roc = float(roc_val) if not pd.isna(roc_val) else 0.0

    is_bullish_candle = last_close > last_open
    is_vol_up = last_vol > prev_vol
    reversal_signal = is_bullish_candle and is_vol_up

    # ------------------------------------------------------------------
    # 6. SCORING
    # ------------------------------------------------------------------
    score = 20
    reasons = ["Weekly Bullish" if strict_bull else "Weekly Soft-Bull"]

    if is_vol_valid:
        score += 20
        reasons.append("Vol Koreksi Rendah")
    if curr_adx > 25 and curr_pdi > curr_mdi:
        score += 20
        reasons.append(f"ADX Kuat ({curr_adx:.1f})")
    elif curr_adx > 18 and curr_pdi > curr_mdi:
        score += 10
        reasons.append(f"ADX Moderat ({curr_adx:.1f})")
    if roc > 0:
        score += 10
        reasons.append("ROC Positif")
    if fib_618 <= last_close <= fib_382:
        score += 15
        reasons.append("Fibo Golden Zone 38-62")
    elif in_fibo_zone:
        score += 8
        reasons.append("Fibo Zone 24-62")
    if reversal_signal:
        score += 15
        reasons.append("Reversal Candle")

    # ------------------------------------------------------------------
    # 7. POSITION PLAN
    # ------------------------------------------------------------------
    entry = round_to_idx_tick(last_close)

    stop_loss_raw = breakout_price * (1 - params["stop_buffer_pct"] / 100)
    stop_loss = round_to_idx_tick(stop_loss_raw)
    stop_loss = apply_ara_arb_limits(stop_loss, last_close, is_target=False)

    risk_per_share = entry - stop_loss
    if risk_per_share <= 0:
        return None

    target_1 = apply_ara_arb_limits(round_to_idx_tick(peak), last_close, is_target=True)
    target_2 = apply_ara_arb_limits(
        round_to_idx_tick(peak + 0.272 * range_up), last_close, is_target=True
    )

    rr_ratio = (target_1 - entry) / risk_per_share if risk_per_share > 0 else 0
    if rr_ratio < params.get("min_rr", 1.2):
        return None

    risk_rp = params["account_size"] * params["risk_per_trade_pct"] / 100
    lots = int((risk_rp / risk_per_share) // params["lot_size"]) if risk_per_share > 0 else 0
    shares = lots * params["lot_size"]
    est_capital_used = shares * entry
    est_loss = shares * risk_per_share
    est_profit = shares * (target_1 - entry)

    return {
        "Ticker": symbol,
        "Close": entry,
        "BreakoutDay": df.index[breakout_idx].strftime("%Y-%m-%d"),
        "Fibo382": round_to_idx_tick(fib_382),
        "Fibo618": round_to_idx_tick(fib_618),
        "VolValid": "Ya" if is_vol_valid else "Tidak",
        "Reversal": "Ya" if reversal_signal else "Tidak",
        "ADX": round(curr_adx, 1),
        "ROC(10)": round(roc, 2),
        "Score": score,
        "Alasan": "; ".join(reasons),
        "Entry": entry,
        "StopLoss": stop_loss,
        "Target1(Peak)": target_1,
        "Target2(Ext)": target_2,
        "ATR": round(atr, 0),
        "RiskPerShare": round(risk_per_share, 0),
        "RR_Ratio": round(rr_ratio, 2),
        "Lots": lots,
        "SuggestedShares": shares,
        "EstCapitalUsed(Rp)": round(est_capital_used, 0),
        "EstLoss(Rp)": round(est_loss, 0),
        "EstProfit1(Rp)": round(est_profit, 0),
        "Strategy": "V3 (Retest Fibo)",
    }


# =========================================================================
# RUNNER
# =========================================================================
def run_screener(universe=None, params=None):
    params = params or PARAMS.copy()
    universe = universe or get_dynamic_liquidity_universe()

    print(f"Menjalankan screener V3 (Pullback & Retest) untuk {len(universe)} saham...\n")
    results = []
    for i, sym in enumerate(universe, 1):
        print(f"  [{i}/{len(universe)}] Cek {sym}...", end="\r")
        try:
            row = analyze_ticker(sym, params)
            if row is not None:
                results.append(row)
        except Exception:
            continue

    print(" " * 60, end="\r")

    if not results:
        print("Tidak ada saham yang lolos filter Pullback/Retest hari ini.")
        return pd.DataFrame()

    df_result = pd.DataFrame(results).sort_values("Score", ascending=False)

    print("=" * 120)
    print(f"IDX PULLBACK & RETEST SCREENER V3 — {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print("=" * 120)
    show_cols = [
        "Ticker", "Close", "Score", "BreakoutDay", "VolValid", "Reversal",
        "ADX", "ROC(10)", "Entry", "StopLoss", "Target1(Peak)", "RR_Ratio", "Lots",
    ]
    show_cols = [c for c in show_cols if c in df_result.columns]
    print(df_result[show_cols].to_string(index=False))
    print("=" * 120)

    df_result["Strategy"] = "V3 (Retest Fibo)"
    try:
        from idx_report_schema import save_version_report
        out_file = save_version_report(df_result, "v3")
    except ImportError:
        out_file = f"idx_report_v3_{datetime.now().strftime('%Y-%m-%d')}.csv"
        df_result.to_csv(out_file, index=False)
    print(f"\nHasil disimpan ke: {out_file}")
    return df_result


if __name__ == "__main__":
    run_screener()