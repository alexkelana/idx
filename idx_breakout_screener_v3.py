"""
IDX PULLBACK & RETEST SCREENER — v3 (upgraded)
================================================
Buy on Weakness setelah Breakout + zona Fibonacci.

Upgrade:
  - SL hybrid: min(breakout-buffer, swing_low, entry-k*ATR) + tick BEI
  - Konfirmasi retest: touch zona Fibo + close hold / rejection wick
  - Regime mingguan bertingkat: strict | normal | relaxed
  - TP/RR realistis: TP1 = max(peak, entry+tp1_r*R) terbatas ARA
  - Kolom: DaysSinceBO, DistToFibo382, Quality, NetPnL (fee)
  - Volume koreksi tetap score (bukan hard reject); penalti jika vol naik
"""

from __future__ import annotations

from datetime import datetime

import numpy as np
import pandas as pd
import yfinance as yf

from idx_liquidity_scanner import IdxLiquidityScanner

# =========================================================================
# PARAMETER
# =========================================================================
PARAMS = {
    "lookback_days": 350,
    "breakout_lookback": 15,
    "breakout_vol_ratio": 1.4,
    "adx_period": 14,
    "roc_period": 10,
    "stop_buffer_pct": 2.0,
    "atr_period": 14,
    "atr_sl_mult": 1.2,
    "account_size": 50_000_000,
    "risk_per_trade_pct": 1.0,
    "lot_size": 100,
    "min_rr": 1.2,
    "tp1_r": 1.5,
    "tp2_ext": 0.272,
    # Fee (fraksi)
    "buy_fee": 0.0015,
    "sell_fee": 0.0025,
    # Regime: strict | normal | relaxed
    "regime_mode": "relaxed",
    # Retest confirmation
    "require_retest_touch": False,
    "retest_wick_min_pct": 0.15,  # lower wick vs range (rejection)
    # Quality thresholds
    "min_score": 35,
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
    if price < 500:
        return int(round(price / 2.0) * 2)
    if price < 2000:
        return int(round(price / 5.0) * 5)
    if price < 5000:
        return int(round(price / 10.0) * 10)
    return int(round(price / 25.0) * 25)


def apply_ara_arb_limits(price: float, prev_close: float, is_target: bool) -> float:
    limit = 0.35 if prev_close < 200 else (0.25 if prev_close <= 5000 else 0.20)
    ara = round_to_idx_tick(prev_close * (1 + limit))
    arb = round_to_idx_tick(prev_close * (1 - limit))
    return min(price, ara) if is_target else max(price, arb)


def trade_net_pnl(entry: float, exit_price: float, shares: int, params: dict) -> float:
    if shares <= 0 or entry <= 0:
        return 0.0
    buy_fee = float(params.get("buy_fee", 0.0015))
    sell_fee = float(params.get("sell_fee", 0.0025))
    cost = entry * shares * (1 + buy_fee)
    proceeds = exit_price * shares * (1 - sell_fee)
    return round(proceeds - cost, 0)


def compute_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high, low, close = df["High"], df["Low"], df["Close"]
    prev_close = close.shift(1)
    tr = pd.concat(
        [
            high - low,
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return tr.rolling(period).mean()


def compute_adx(df: pd.DataFrame, period: int = 14):
    high, low, close = df["High"], df["Low"], df["Close"]
    up = high - high.shift(1)
    down = low.shift(1) - low

    plus_dm = np.where((up > down) & (up > 0), up, 0.0)
    minus_dm = np.where((down > up) & (down > 0), down, 0.0)

    tr = pd.concat(
        [
            high - low,
            (high - close.shift(1)).abs(),
            (low - close.shift(1)).abs(),
        ],
        axis=1,
    ).max(axis=1)

    atr = tr.ewm(alpha=1 / period, adjust=False).mean()
    plus_di = (
        100
        * pd.Series(plus_dm, index=df.index).ewm(alpha=1 / period, adjust=False).mean()
        / atr
    )
    minus_di = (
        100
        * pd.Series(minus_dm, index=df.index).ewm(alpha=1 / period, adjust=False).mean()
        / atr
    )
    dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di).replace(0, np.nan)
    adx = dx.ewm(alpha=1 / period, adjust=False).mean()
    return adx, plus_di, minus_di


def compute_roc(series: pd.Series, period: int = 10) -> pd.Series:
    return series.pct_change(periods=period) * 100


def quality_tier(score: int, rr: float, retest_ok: bool, vol_valid: bool) -> str:
    if score >= 70 and rr >= 1.5 and retest_ok and vol_valid:
        return "A"
    if score >= 50 and rr >= 1.2 and retest_ok:
        return "B"
    if score >= 35:
        return "C"
    return "D"


# =========================================================================
# UNIVERSE
# =========================================================================
def get_dynamic_liquidity_universe() -> list:
    print("=" * 60)
    print("TAHAP 1: UNIVERSE (Google Drive → likuiditas)")
    print("=" * 60)

    def _liq():
        scanner = IdxLiquidityScanner(
            min_avg_value_rp=10_000_000_000,
            min_avg_volume=1_000_000,
            lookback_days=20,
            max_workers=15,
        )
        return scanner.get_liquid_universe()

    try:
        from idx_gdrive_data import resolve_screener_universe

        universe = resolve_screener_universe(fallback_fn=_liq)
    except Exception:
        universe = _liq()
    if not universe:
        universe = _liq()

    print("=" * 60)
    print("TAHAP 2: ANALISA PULLBACK / RETEST (V3 upgraded)")
    print("=" * 60)
    return universe


def check_weekly_regime(df: pd.DataFrame, mode: str) -> tuple[bool, str]:
    """
    strict  : Close > MA10w > MA30w dan MA10 naik
    normal  : strict OR soft (Close > MA10w dan MA10w >= MA30w*0.98)
    relaxed : tolak hanya clear bear (Close < MA10w < MA30w)
    """
    weekly = (
        df.resample("W")
        .agg(
            {
                "Open": "first",
                "High": "max",
                "Low": "min",
                "Close": "last",
                "Volume": "sum",
            }
        )
        .dropna()
    )
    if len(weekly) < 30:
        return False, "Weekly data kurang"

    ma10w = weekly["Close"].rolling(10).mean()
    ma30w = weekly["Close"].rolling(30).mean()
    w_close = float(weekly["Close"].iloc[-1])
    w_ma10 = float(ma10w.iloc[-1])
    w_ma30 = float(ma30w.iloc[-1])
    w_ma10_prev = float(ma10w.iloc[-2])
    if any(np.isnan(x) for x in [w_ma10, w_ma30, w_ma10_prev]):
        return False, "MA weekly NaN"

    strict_bull = (w_close > w_ma10 > w_ma30) and (w_ma10 >= w_ma10_prev)
    soft_bull = (w_close > w_ma10) and (w_ma10 >= w_ma30 * 0.98)
    clear_bear = (w_close < w_ma10 < w_ma30)

    mode = (mode or "normal").lower()
    if mode == "strict":
        ok = strict_bull
        label = "Weekly Strict-Bull" if ok else "Bukan strict bull"
    elif mode == "relaxed":
        ok = not clear_bear
        label = (
            "Weekly Strict-Bull"
            if strict_bull
            else ("Weekly Soft-Bull" if soft_bull else "Weekly Non-Bear")
        )
    else:  # normal
        ok = strict_bull or soft_bull
        label = "Weekly Strict-Bull" if strict_bull else "Weekly Soft-Bull"

    return ok, label


# =========================================================================
# ANALISA SATU TICKER
# =========================================================================
def analyze_ticker(symbol: str, params: dict) -> dict | None:
    sym = str(symbol).upper().replace(".JK", "").strip()
    ticker = sym + ".JK"
    try:
        df = yf.download(
            ticker,
            period=f"{int(params['lookback_days'])}d",
            interval="1d",
            progress=False,
            auto_adjust=True,
            multi_level_index=False,
            threads=False,
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
    # 1. REGIME MINGGUAN (tier)
    # ------------------------------------------------------------------
    regime_ok, regime_label = check_weekly_regime(
        df, params.get("regime_mode", "normal")
    )
    if not regime_ok:
        return None

    # ------------------------------------------------------------------
    # 2. BREAKOUT TERKONFIRMASI (bukan hari ini)
    # ------------------------------------------------------------------
    lookback = int(params["breakout_lookback"])
    breakout_idx = -1
    breakout_price = 0.0
    breakout_vol = 0.0

    for i in range(len(df) - 1, max(len(df) - lookback - 1, 20), -1):
        past_high = float(df["High"].iloc[i - 20 : i].max())
        past_vol_avg = float(df["Volume"].iloc[i - 20 : i].mean())
        if past_vol_avg <= 0:
            continue
        cl = float(df["Close"].iloc[i])
        hi = float(df["High"].iloc[i])
        lo = float(df["Low"].iloc[i])
        op = float(df["Open"].iloc[i])
        # body relatif kuat opsional: close di setengah atas range
        bar_range = max(hi - lo, 1e-9)
        strong_close = cl >= lo + 0.45 * bar_range
        if (
            cl > past_high
            and float(df["Volume"].iloc[i]) >= past_vol_avg * params["breakout_vol_ratio"]
            and strong_close
        ):
            breakout_idx = i
            breakout_price = past_high
            breakout_vol = float(df["Volume"].iloc[i])
            break

    if breakout_idx < 0 or breakout_idx >= len(df) - 1:
        return None

    days_since_bo = len(df) - 1 - breakout_idx

    # ------------------------------------------------------------------
    # 3. FIBONACCI RETRACE
    # ------------------------------------------------------------------
    swing_low = float(df["Low"].iloc[max(0, breakout_idx - 20) : breakout_idx].min())
    peak = float(df["High"].iloc[breakout_idx:].max())
    range_up = peak - swing_low
    if range_up <= 0:
        return None

    fib_236 = peak - 0.236 * range_up
    fib_382 = peak - 0.382 * range_up
    fib_618 = peak - 0.618 * range_up

    last = df.iloc[-1]
    last_close = float(last["Close"])
    last_open = float(last["Open"])
    last_high = float(last["High"])
    last_low = float(last["Low"])
    last_vol = float(last["Volume"])
    prev_vol = float(df["Volume"].iloc[-2])

    in_fibo_zone = fib_618 <= last_close <= fib_236
    above_breakout_support = last_close >= breakout_price * 0.995

    if not (in_fibo_zone and above_breakout_support):
        return None

    dist_to_fibo382_pct = (last_close - fib_382) / last_close * 100 if last_close else 0

    # ------------------------------------------------------------------
    # 4. KONFIRMASI RETEST (touch zona + hold / rejection)
    # ------------------------------------------------------------------
    day_range = max(last_high - last_low, 1e-9)
    lower_wick = min(last_open, last_close) - last_low
    # Touch: low masuk zona Fibo
    touched_zone = last_low <= fib_236 and last_low >= fib_618 * 0.98
    hold_close = last_close >= max(fib_618, breakout_price * 0.995)
    # Rejection wick ≥ fraction of day range (default 15%)
    rejection = (lower_wick / day_range) >= float(
        params.get("retest_wick_min_pct", 0.15)
    )
    retest_ok = touched_zone and hold_close and (rejection or last_close > last_open)

    if params.get("require_retest_touch", True) and not retest_ok:
        return None

    # ------------------------------------------------------------------
    # 5. VOLUME KOREKSI (score; penalti jika naik)
    # ------------------------------------------------------------------
    peak_loc = df["High"].iloc[breakout_idx:].idxmax()
    peak_idx_int = df.index.get_loc(peak_loc)
    if isinstance(peak_idx_int, slice):
        peak_idx_int = peak_idx_int.start or 0

    if peak_idx_int < len(df) - 1:
        vol_correction_avg = float(df["Volume"].iloc[peak_idx_int + 1 :].mean())
    else:
        vol_correction_avg = last_vol

    is_vol_valid = vol_correction_avg < breakout_vol
    vol_rising_correction = vol_correction_avg >= breakout_vol * 0.95

    # ------------------------------------------------------------------
    # 6. MOMENTUM + ATR + MA harian (score)
    # ------------------------------------------------------------------
    atr_series = compute_atr(df, int(params.get("atr_period", 14)))
    atr = float(atr_series.iloc[-1]) if not pd.isna(atr_series.iloc[-1]) else 0.0

    adx, plus_di, minus_di = compute_adx(df, params["adx_period"])
    curr_adx = float(adx.iloc[-1]) if not pd.isna(adx.iloc[-1]) else 0.0
    curr_pdi = float(plus_di.iloc[-1]) if not pd.isna(plus_di.iloc[-1]) else 0.0
    curr_mdi = float(minus_di.iloc[-1]) if not pd.isna(minus_di.iloc[-1]) else 0.0

    roc_val = compute_roc(df["Close"], params["roc_period"]).iloc[-1]
    roc = float(roc_val) if not pd.isna(roc_val) else 0.0

    ma20 = float(df["Close"].rolling(20).mean().iloc[-1])
    above_ma20 = last_close > ma20

    is_bullish_candle = last_close > last_open
    is_vol_up = last_vol > prev_vol
    reversal_signal = is_bullish_candle and is_vol_up

    # ------------------------------------------------------------------
    # 7. SCORING
    # ------------------------------------------------------------------
    score = 15
    reasons = [regime_label]

    if retest_ok:
        score += 15
        reasons.append("Retest hold/reject")
    if is_vol_valid:
        score += 15
        reasons.append("Vol Koreksi Rendah")
    elif vol_rising_correction:
        score -= 8
        reasons.append("Vol koreksi tinggi (risiko)")
    if curr_adx > 25 and curr_pdi > curr_mdi:
        score += 15
        reasons.append(f"ADX Kuat ({curr_adx:.1f})")
    elif curr_adx > 18 and curr_pdi > curr_mdi:
        score += 8
        reasons.append(f"ADX Moderat ({curr_adx:.1f})")
    if roc > 0:
        score += 8
        reasons.append("ROC Positif")
    if fib_618 <= last_close <= fib_382:
        score += 12
        reasons.append("Fibo Golden 38-62")
    elif in_fibo_zone:
        score += 6
        reasons.append("Fibo Zone 24-62")
    if reversal_signal:
        score += 12
        reasons.append("Reversal Candle")
    if above_ma20:
        score += 8
        reasons.append("Close>MA20")
    if days_since_bo <= 5:
        score += 5
        reasons.append(f"BO fresh ({days_since_bo}d)")

    if score < int(params.get("min_score", 35)):
        return None

    # ------------------------------------------------------------------
    # 8. POSITION PLAN — SL hybrid + TP realistis
    # ------------------------------------------------------------------
    entry = round_to_idx_tick(last_close)

    stop_by_bo = breakout_price * (1 - params["stop_buffer_pct"] / 100)
    stop_by_swing = swing_low * 0.995
    stop_by_atr = entry - params.get("atr_sl_mult", 1.2) * atr if atr > 0 else stop_by_bo
    # Ambil SL paling konservatif yang masih di bawah entry (jarak lebih lebar)
    stop_loss_raw = min(stop_by_bo, stop_by_swing, stop_by_atr)
    if stop_loss_raw >= entry:
        stop_loss_raw = entry - max(atr * 0.8, entry * 0.015)
    stop_loss = apply_ara_arb_limits(
        round_to_idx_tick(stop_loss_raw), last_close, is_target=False
    )

    risk_per_share = entry - stop_loss
    if risk_per_share <= 0:
        return None

    tp1_r = float(params.get("tp1_r", 1.5))
    target_r = entry + risk_per_share * tp1_r
    target_1 = apply_ara_arb_limits(
        round_to_idx_tick(max(peak, target_r)), last_close, is_target=True
    )
    # Jika peak di bawah entry (shouldn't), pakai R saja
    if target_1 <= entry:
        target_1 = apply_ara_arb_limits(
            round_to_idx_tick(target_r), last_close, is_target=True
        )

    target_2 = apply_ara_arb_limits(
        round_to_idx_tick(
            max(peak + float(params.get("tp2_ext", 0.272)) * range_up, entry + risk_per_share * 2.5)
        ),
        last_close,
        is_target=True,
    )

    rr_ratio = (target_1 - entry) / risk_per_share if risk_per_share > 0 else 0
    if rr_ratio < params.get("min_rr", 1.2):
        return None

    risk_rp = params["account_size"] * params["risk_per_trade_pct"] / 100
    lots = int((risk_rp / risk_per_share) // params["lot_size"]) if risk_per_share > 0 else 0
    shares = lots * params["lot_size"]

    net_sl = trade_net_pnl(entry, stop_loss, shares, params)
    net_tp1 = trade_net_pnl(entry, target_1, shares, params)
    net_tp2 = trade_net_pnl(entry, target_2, shares, params)

    q = quality_tier(score, rr_ratio, retest_ok, is_vol_valid)

    return {
        "Ticker": sym,
        "Close": entry,
        "BreakoutDay": df.index[breakout_idx].strftime("%Y-%m-%d"),
        "DaysSinceBO": days_since_bo,
        "Fibo382": round_to_idx_tick(fib_382),
        "Fibo618": round_to_idx_tick(fib_618),
        "DistToFibo382Pct": round(dist_to_fibo382_pct, 2),
        "RetestOK": "Ya" if retest_ok else "Tidak",
        "VolValid": "Ya" if is_vol_valid else "Tidak",
        "Reversal": "Ya" if reversal_signal else "Tidak",
        "ADX": round(curr_adx, 1),
        "ROC(10)": round(roc, 2),
        "Score": score,
        "Quality": q,
        "Alasan": "; ".join(reasons),
        "Entry": entry,
        "StopLoss": stop_loss,
        "Target1": target_1,
        "Target1(Peak)": target_1,
        "Target2(Ext)": target_2,
        "ATR": round(atr, 0),
        "RiskPerShare": round(risk_per_share, 0),
        "RR_Ratio": round(rr_ratio, 2),
        "Lots": lots,
        "SuggestedShares": shares,
        "EstCapitalUsed(Rp)": round(shares * entry, 0),
        "EstLoss(Rp)": round(shares * risk_per_share, 0),
        "EstProfit1(Rp)": round(shares * (target_1 - entry), 0),
        "NetPnL_SL": net_sl,
        "NetPnL_TP1": net_tp1,
        "NetPnL_TP2": net_tp2,
        "Strategy": "V3 (Retest Fibo)",
    }


# =========================================================================
# RUNNER
# =========================================================================
def run_screener(universe=None, params=None):
    params = {**PARAMS, **(params or {})}
    universe = universe or get_dynamic_liquidity_universe()
    universe = [str(t).upper().replace(".JK", "").strip() for t in universe]

    print(
        f"Menjalankan screener V3 upgraded (regime={params.get('regime_mode')}) "
        f"untuk {len(universe)} saham...\n"
    )
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

    df_result = pd.DataFrame(results).sort_values(
        ["Quality", "Score", "RR_Ratio"], ascending=[True, False, False]
    )
    # Quality A before B — map for sort
    order = {"A": 0, "B": 1, "C": 2, "D": 3}
    df_result["_q"] = df_result["Quality"].map(lambda x: order.get(x, 9))
    df_result = (
        df_result.sort_values(["_q", "Score", "RR_Ratio"], ascending=[True, False, False])
        .drop(columns=["_q"])
        .reset_index(drop=True)
    )

    print("=" * 120)
    print(
        f"IDX PULLBACK & RETEST SCREENER V3 (upgraded) — "
        f"{datetime.now().strftime('%Y-%m-%d %H:%M')}"
    )
    print("=" * 120)
    show_cols = [
        "Ticker",
        "Quality",
        "Close",
        "Score",
        "DaysSinceBO",
        "RetestOK",
        "VolValid",
        "ADX",
        "Entry",
        "StopLoss",
        "Target1",
        "RR_Ratio",
        "Lots",
        "NetPnL_TP1",
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
