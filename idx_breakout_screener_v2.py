"""
IDX BREAKOUT SCREENER + POSITION PLAN — v2
============================================================
+ Filter false breakout:
  1) Liquidity sweep di resistance sebelumnya
  2) Momentum masih mendukung real breakout
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import requests
import yfinance as yf
from datetime import datetime
from idx_liquidity_scanner import IdxLiquidityScanner

# =========================================================================
# UNIVERSE
# =========================================================================
def get_lq45_universe() -> list:
    url = "https://id.wikipedia.org/wiki/LQ45"
    fallback_universe = ["BBCA", "BBRI", "BMRI", "BBNI", "TLKM", "ASII", "GOTO", "AMMN"]
    try:
        headers = {"User-Agent": "Mozilla/5.0"}
        response = requests.get(url, headers=headers, timeout=10)
        tables = pd.read_html(response.text)
        for df in tables:
            cols = [str(c).lower() for c in df.columns]
            if "kode" in cols or "simbol" in cols or "ticker" in cols:
                target_col = next(
                    (c for c in df.columns if str(c).lower() in ["kode", "simbol", "ticker"]),
                    None,
                )
                if target_col:
                    tickers = df[target_col].dropna().astype(str).tolist()
                    return [
                        t.upper().strip()
                        for t in tickers
                        if len(t.strip()) == 4 and t.isalpha()
                    ]
    except Exception:
        pass
    return fallback_universe


def get_dynamic_liquidity_universe(params: dict) -> list:
    print("=" * 60)
    print("TAHAP 1: PRE-SCREENER (MEMINDAI SELURUH PASAR)")
    print("=" * 60)
    scanner = IdxLiquidityScanner(
        min_avg_value_rp=params.get("min_avg_value_traded", 10_000_000_000),
        min_avg_volume=params.get("min_avg_volume", 1_000_000),
        lookback_days=20,
        max_workers=15,
    )
    liquid_universe = scanner.get_liquid_universe()
    print("=" * 60)
    print("TAHAP 2: DEEP TECHNICAL ANALYSIS (BREAKOUT V2 + SWEEP FILTER)")
    print("=" * 60)
    return liquid_universe


# =========================================================================
# PARAMETER
# =========================================================================
PARAMS = {
    "min_avg_value_traded": 10_000_000_000,
    "min_avg_volume": 1_000_000,
    "lookback_days": 150,
    "consolidation_window": 20,
    "swing_lookback": 60,
    "fractal_bars": 2,
    "atr_period": 14,
    "max_range_pct": 8.0,
    "near_high_pct": 3.0,
    "volume_surge_ratio": 1.2,
    "rsi_min": 50,
    "rsi_max": 70,
    "top_n": 20,
    "min_score": 40,
    "entry_buffer_pct": 0.5,
    "stop_buffer_pct": 0.5,
    "atr_stop_mult": 1.5,
    "min_risk_reward": 1.5,
    "account_size": 5_000_000,
    "risk_per_trade_pct": 1.0,
    "lot_size": 100,
    # --- False breakout filter ---
    "require_liquidity_sweep": False,   # True = wajib ada sweep di res
    "sweep_lookback": 12,              # cari sweep dalam N hari
    "sweep_min_wick_pct": 0.25,        # minimal wick di atas res (%)
    "min_breakout_vol_ratio": 1.25,    # volume untuk momentum BO
    "min_body_pct": 0.35,              # body candle vs harga (%)
    "rsi_max_momentum": 72,            # di atas ini rawan false BO
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
    if prev_close < 200:
        limit = 0.35
    elif prev_close <= 5000:
        limit = 0.25
    else:
        limit = 0.20
    ara = round_to_idx_tick(prev_close * (1 + limit))
    arb = round_to_idx_tick(prev_close * (1 - limit))
    return min(price, ara) if is_target else max(price, arb)


# =========================================================================
# INDIKATOR
# =========================================================================
def compute_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(period).mean()
    avg_loss = loss.rolling(period).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def compute_bb_width(close: pd.Series, period: int = 20, std_mult: float = 2.0) -> pd.Series:
    ma = close.rolling(period).mean()
    std = close.rolling(period).std()
    upper = ma + std_mult * std
    lower = ma - std_mult * std
    return (upper - lower) / ma * 100


def compute_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high, low, close = df["High"], df["Low"], df["Close"]
    prev_close = close.shift(1)
    tr = pd.concat(
        [high - low, (high - prev_close).abs(), (low - prev_close).abs()],
        axis=1,
    ).max(axis=1)
    return tr.rolling(period).mean()


def find_swing_points(df: pd.DataFrame, lookback: int, n_bars: int):
    sub = df.tail(lookback)
    highs, lows = sub["High"], sub["Low"]
    swing_highs, swing_lows = [], []
    for i in range(n_bars, len(sub) - n_bars):
        window_h = highs.iloc[i - n_bars : i + n_bars + 1]
        window_l = lows.iloc[i - n_bars : i + n_bars + 1]
        if highs.iloc[i] == window_h.max():
            swing_highs.append(float(highs.iloc[i]))
        if lows.iloc[i] == window_l.min():
            swing_lows.append(float(lows.iloc[i]))
    return (
        sorted(set(round(x, 0) for x in swing_highs), reverse=True),
        sorted(set(round(x, 0) for x in swing_lows), reverse=True),
    )


# =========================================================================
# FALSE BREAKOUT FILTER: LIQUIDITY SWEEP + MOMENTUM
# =========================================================================
def detect_liquidity_sweep(
    df: pd.DataFrame,
    resistance: float,
    lookback: int = 12,
    min_wick_pct: float = 0.25,
) -> dict:
    """
    Sweep di resistance = High tembus res, Close kembali <= res
    (stop-hunt / liquidity grab) dalam lookback bar, sebelum bar terakhir.
    """
    if resistance <= 0 or len(df) < lookback + 2:
        return {"has_sweep": False, "sweep_high": 0.0, "wick_pct": 0.0}

    # bar -lookback … -2 (hari ini dipakai untuk sinyal BO)
    start = max(0, len(df) - lookback - 1)
    end = len(df) - 1
    for i in range(end - 1, start - 1, -1):
        hi = float(df["High"].iloc[i])
        cl = float(df["Close"].iloc[i])
        if hi > resistance and cl <= resistance * 1.002:
            wick_pct = (hi - max(cl, resistance)) / resistance * 100
            if wick_pct >= min_wick_pct or hi > resistance * 1.001:
                return {
                    "has_sweep": True,
                    "sweep_high": hi,
                    "sweep_close": cl,
                    "wick_pct": round(wick_pct, 2),
                    "bars_ago": end - i,
                }
    return {"has_sweep": False, "sweep_high": 0.0, "wick_pct": 0.0}


def breakout_momentum_ok(
    df: pd.DataFrame,
    resistance: float,
    vol_ratio: float,
    rsi: float,
    params: dict,
) -> tuple[bool, list[str]]:
    """Momentum masih mengarah ke real breakout, bukan spike palsu."""
    notes = []
    last = df.iloc[-1]
    prev = df.iloc[-2]
    close = float(last["Close"])
    open_ = float(last["Open"])
    high = float(last["High"])
    low = float(last["Low"])
    body_pct = abs(close - open_) / close * 100 if close > 0 else 0

    ok = True

    # Dekat resistance
    dist = (resistance - close) / close * 100
    if dist > params.get("near_high_pct", 3.0):
        ok = False
        notes.append(f"Jauh dari res ({dist:.1f}%)")
    else:
        notes.append(f"Dekat res ({dist:.1f}%)")

    # Volume
    min_vol = params.get("min_breakout_vol_ratio", 1.25)
    if vol_ratio >= min_vol:
        notes.append(f"Vol {vol_ratio:.2f}x")
    else:
        ok = False
        notes.append(f"Vol lemah {vol_ratio:.2f}x")

    # RSI
    rsi_max = params.get("rsi_max_momentum", 72)
    if params.get("rsi_min", 50) <= rsi <= rsi_max:
        notes.append(f"RSI {rsi:.0f}")
    elif rsi > rsi_max:
        ok = False
        notes.append(f"RSI overbought {rsi:.0f}")
    else:
        notes.append(f"RSI {rsi:.0f}")

    # Struktur candle
    if body_pct >= params.get("min_body_pct", 0.35) and close >= open_:
        notes.append("Body bullish")
    elif close >= resistance * 0.998:
        notes.append("Close di zona res")
    else:
        notes.append("Pre-BO")

    # Higher / equal low
    if float(last["Low"]) >= float(prev["Low"]) * 0.995:
        notes.append("HL/EL")
    else:
        notes.append("Low melemah")

    # Hindari wick atas ekstrem hari ini tanpa close kuat (fake spike)
    day_range = max(high - low, 1e-9)
    upper_wick = high - max(close, open_)
    if upper_wick / day_range > 0.55 and close < resistance:
        ok = False
        notes.append("Wick atas dominan (risiko fake)")

    return ok, notes


# =========================================================================
# POSITION PLAN
# =========================================================================
def build_position_plan(last_close, swing_highs, swing_lows, atr, recent_high, recent_low, params):
    res_candidates = [h for h in swing_highs if h > last_close] or [recent_high]
    resistance = min(res_candidates) if res_candidates else recent_high
    res_candidates_far = [h for h in res_candidates if h > resistance]
    resistance_2 = min(res_candidates_far) if res_candidates_far else resistance * 1.05

    sup_candidates = [l for l in swing_lows if l < last_close] or [recent_low]
    support = max(sup_candidates) if sup_candidates else recent_low

    confluence_notes = []
    if abs(resistance - recent_high) / last_close * 100 <= 1.5:
        confluence_notes.append("Resistance sejalan dgn high konsolidasi 20D")
    if not np.isnan(atr) and atr > 0:
        confluence_notes.append(f"ATR14 = {atr:.0f}")

    entry = round_to_idx_tick(resistance * (1 + params["entry_buffer_pct"] / 100))

    stop_by_support = support * (1 - params["stop_buffer_pct"] / 100)
    stop_by_atr = (
        entry - (atr * params["atr_stop_mult"])
        if not np.isnan(atr) and atr > 0
        else stop_by_support
    )
    stop_loss = min(stop_by_support, stop_by_atr)
    if stop_loss >= entry:
        stop_loss = entry * 0.97
    stop_loss = apply_ara_arb_limits(round_to_idx_tick(stop_loss), last_close, is_target=False)

    risk_per_share = entry - stop_loss
    if risk_per_share <= 0:
        return None

    target_1 = apply_ara_arb_limits(
        round_to_idx_tick(entry + risk_per_share * 2), last_close, is_target=True
    )
    target_2 = apply_ara_arb_limits(
        round_to_idx_tick(max(resistance_2, entry + risk_per_share * 3)),
        last_close,
        is_target=True,
    )

    reward_1 = target_1 - entry
    rr_ratio = reward_1 / risk_per_share if risk_per_share > 0 else 0
    retest_entry = round_to_idx_tick(resistance)

    risk_rp = params["account_size"] * params["risk_per_trade_pct"] / 100
    lots = int((risk_rp / risk_per_share) // params["lot_size"]) if risk_per_share > 0 else 0
    shares = lots * params["lot_size"]

    return {
        "Support": round_to_idx_tick(support),
        "Resistance": round_to_idx_tick(resistance),
        "Resistance2": round_to_idx_tick(resistance_2),
        "EntryBreakout": entry,
        "EntryRetest": retest_entry,
        "StopLoss": stop_loss,
        "Target1": target_1,
        "Target2": target_2,
        "ATR": round(atr, 0) if not np.isnan(atr) else 0,
        "RiskPerShare": risk_per_share,
        "RR_Ratio": round(rr_ratio, 2),
        "LayakRR": rr_ratio >= params["min_risk_reward"],
        "SuggestedLots": lots,
        "SuggestedShares": shares,
        "EstCapitalUsed(Rp)": round(shares * entry, 0),
        "EstLoss(Rp)": round(shares * risk_per_share, 0),
        "EstProfit1(Rp)": round(shares * reward_1, 0),
        "Confluence": "; ".join(confluence_notes) if confluence_notes else "-",
    }


# =========================================================================
# ANALISA TICKER
# =========================================================================
def analyze_ticker(symbol: str, params: dict) -> dict | None:
    ticker = symbol + ".JK"
    try:
        df = yf.download(
            ticker,
            period=f"{int(params['lookback_days'] * 1.6)}d",
            interval="1d",
            progress=False,
            auto_adjust=True,
            multi_level_index=False,
        )
    except Exception:
        return None

    if df is None or df.empty or len(df) < 50:
        return None

    if isinstance(df.columns, pd.MultiIndex):
        try:
            df.columns = df.columns.get_level_values(0)
        except Exception:
            try:
                df = df.droplevel(-1, axis=1)
            except Exception:
                return None

    if "Close" not in df.columns:
        return None

    df = df.dropna()
    if len(df) < 50:
        return None

    close = df["Close"]
    high = df["High"]
    low = df["Low"]
    volume = df["Volume"]
    last_close = float(close.iloc[-1])

    value_traded = close * volume
    avg_value_20 = float(value_traded.tail(20).mean())
    avg_volume_20 = float(volume.tail(20).mean())

    if avg_value_20 < params["min_avg_value_traded"] * 0.7:
        return None
    if avg_volume_20 < params["min_avg_volume"] * 0.7:
        return None

    win = params["consolidation_window"]
    recent_high = float(high.tail(win).max())
    recent_low = float(low.tail(win).min())
    range_pct = (recent_high - recent_low) / last_close * 100
    dist_to_high_pct = (recent_high - last_close) / last_close * 100

    ma20 = float(close.rolling(20).mean().iloc[-1])
    ma50 = float(close.rolling(50).mean().iloc[-1]) if len(close) >= 50 else np.nan
    above_ma20 = last_close > ma20
    ma20_above_ma50 = (ma20 > ma50) if not np.isnan(ma50) else True

    vol_5d_avg = float(volume.tail(5).mean())
    vol_ratio = vol_5d_avg / avg_volume_20 if avg_volume_20 > 0 else 0

    rsi_val = compute_rsi(close).iloc[-1]
    rsi = float(rsi_val) if not pd.isna(rsi_val) else 50.0

    bb_width = compute_bb_width(close)
    bb_width_now = float(bb_width.iloc[-1]) if not pd.isna(bb_width.iloc[-1]) else 0
    bb_width_avg = float(bb_width.tail(60).mean()) if not pd.isna(bb_width.tail(60).mean()) else 0
    is_squeeze = bb_width_now < bb_width_avg if bb_width_avg > 0 else False

    atr_val = compute_atr(df, params["atr_period"]).iloc[-1]
    atr = float(atr_val) if not pd.isna(atr_val) else 0.0
    swing_highs, swing_lows = find_swing_points(
        df, params["swing_lookback"], params["fractal_bars"]
    )

    # --- Scoring dasar ---
    score = 0
    reasons = []

    if range_pct <= params["max_range_pct"]:
        score += 20
        reasons.append(f"Konsolidasi ketat ({range_pct:.1f}%)")
    if dist_to_high_pct <= params["near_high_pct"]:
        score += 20
        reasons.append(f"Dekat resistance ({dist_to_high_pct:.1f}%)")
    if above_ma20 and ma20_above_ma50:
        score += 20
        reasons.append("Uptrend (Close>MA20>MA50)")
    elif above_ma20:
        score += 10
        reasons.append("Close>MA20")
    if vol_ratio >= params["volume_surge_ratio"]:
        score += 20
        reasons.append(f"Volume naik ({vol_ratio:.2f}x)")
    if params["rsi_min"] <= rsi <= params["rsi_max"]:
        score += 10
        reasons.append(f"RSI sehat ({rsi:.0f})")
    if is_squeeze:
        score += 10
        reasons.append("BB squeeze")

    if score < params.get("min_score", 40):
        return None

    plan = build_position_plan(
        last_close, swing_highs, swing_lows, atr, recent_high, recent_low, params
    )
    if plan is None:
        return None

    resistance = float(plan["Resistance"])

    # --- [BARU] Liquidity sweep ---
    sweep = detect_liquidity_sweep(
        df,
        resistance=resistance,
        lookback=params.get("sweep_lookback", 12),
        min_wick_pct=params.get("sweep_min_wick_pct", 0.25),
    )
    if params.get("require_liquidity_sweep", True) and not sweep["has_sweep"]:
        return None

    # --- [BARU] Momentum real breakout ---
    mom_ok, mom_notes = breakout_momentum_ok(df, resistance, vol_ratio, rsi, params)
    if not mom_ok:
        return None

    if sweep["has_sweep"]:
        score += 15
        reasons.append(
            f"Liquidity sweep (wick {sweep.get('wick_pct', 0)}%, "
            f"{sweep.get('bars_ago', '?')} bar lalu)"
        )
    for n in mom_notes:
        if n not in reasons:
            reasons.append(n)

    result = {
        "Ticker": symbol,
        "Close": round_to_idx_tick(last_close),
        "AvgValue20D(Rp Jt)": round(avg_value_20 / 1_000_000, 0),
        "RangePct": round(range_pct, 1),
        "VolRatio5v20": round(vol_ratio, 2),
        "RSI": round(rsi, 0),
        "Squeeze": is_squeeze,
        "Score": score,
        "Alasan": "; ".join(reasons) if reasons else "-",
        "Sweep": "Ya" if sweep["has_sweep"] else "Tidak",
        "SweepHigh": round_to_idx_tick(sweep.get("sweep_high", 0)),
        "SweepWickPct": sweep.get("wick_pct", 0),
        "MomentumOK": "Ya" if mom_ok else "Tidak",
    }
    result.update(plan)
    return result


# =========================================================================
# RUNNER
# =========================================================================
def run_screener(universe=None, params=None):
    params = params or PARAMS.copy()
    universe = universe or get_dynamic_liquidity_universe(params)

    print(f"Menjalankan screener V2 (Breakout + Sweep Filter) untuk {len(universe)} saham...\n")
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
        print("Tidak ada saham yang lolos filter breakout + liquidity sweep hari ini.")
        return pd.DataFrame()

    df_result = pd.DataFrame(results).sort_values("Score", ascending=False)
    df_result = df_result.head(params["top_n"]).reset_index(drop=True)

    summary_cols = [
        "Ticker", "Close", "Score", "Sweep", "RSI", "VolRatio5v20",
        "Support", "Resistance", "EntryBreakout", "StopLoss",
        "Target1", "RR_Ratio", "SuggestedLots",
    ]
    summary_cols = [c for c in summary_cols if c in df_result.columns]

    print("=" * 120)
    print(f"IDX BREAKOUT SCREENER V2 — {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print("=" * 120)
    print(df_result[summary_cols].to_string(index=False))
    print("=" * 120)

    print("\nRENCANA POSISI — TOP 5\n")
    for _, row in df_result.head(5).iterrows():
        rr_flag = "OK" if row["LayakRR"] else "RR KURANG"
        print(f"--- {row['Ticker']} (Score: {row['Score']}) Sweep={row.get('Sweep')} ---")
        print(f"  Alasan : {row['Alasan']}")
        print(f"  S/R    : {row['Support']:.0f} / {row['Resistance']:.0f}")
        print(f"  Entry  : BO {row['EntryBreakout']:.0f} | Retest {row['EntryRetest']:.0f}")
        print(f"  SL/TP1 : {row['StopLoss']:.0f} / {row['Target1']:.0f} | RR 1:{row['RR_Ratio']:.2f} [{rr_flag}]")
        print(f"  Lots   : {row['SuggestedLots']}")
        print()

    df_result["Strategy"] = "V2 (Breakout)"
    try:
        from idx_report_schema import save_version_report
        out_file = save_version_report(df_result, "v2")
    except ImportError:
        out_file = f"idx_report_v2_{datetime.now().strftime('%Y-%m-%d')}.csv"
        df_result.to_csv(out_file, index=False)
    print(f"Hasil disimpan ke: {out_file}")
    return df_result


if __name__ == "__main__":
    run_screener()