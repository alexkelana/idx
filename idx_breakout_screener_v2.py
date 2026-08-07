"""
IDX BREAKOUT SCREENER + POSITION PLAN — v2 (Swing Trading Optimized)
============================================================
Screener ini mencari saham IDX yang LIKUID dan sedang dalam kondisi
"siap breakout", lalu memberikan RENCANA POSISI (entry, stop loss,
target profit, risk-reward, ukuran posisi) berdasarkan support/resistance
dan confluence beberapa indikator.

Pembaruan v2:
- Penyesuaian Fraksi Harga (Tick Size) BEI.
- Pembatasan Auto Rejection Atas/Bawah (ARA/ARB).
- Dioptimalkan untuk Swing Trading (data harian).

CARA PAKAI:
1. Install dependency (sekali saja):
   pip install yfinance pandas numpy

2. Jalankan setiap sore/malam setelah market close untuk mencari setup esok hari:
   python idx_breakout_screener_v2.py

3. Hasil akan tampil di terminal + tersimpan ke file CSV
   (idx_breakout_screener_v2_output_YYYY-MM-DD.csv)

DISCLAIMER:
- Ini alat bantu screening & perencanaan posisi, BUKAN rekomendasi beli/jual.
- Semua level dihitung dari data historis harian yang bisa delay.
- Selalu cek order book, running trade, dan berita sebelum eksekusi.
- Ukuran posisi hanya estimasi matematis dari risk % yang Anda set.

IDX BREAKOUT SCREENER + POSITION PLAN — v2 (Swing Trading Optimized)
============================================================
[FIXED] MultiIndex, data-length guard, CSV konsisten, min score, 
        entry hanya jika dekat resistance, dll.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import requests
import yfinance as yf
from datetime import datetime, timedelta
from idx_liquidity_scanner import IdxLiquidityScanner

# =========================================================================
# 1. PENGAMBILAN DAFTAR SAHAM
# =========================================================================
def get_lq45_universe() -> list:
    url = "https://id.wikipedia.org/wiki/LQ45"
    fallback_universe = ["BBCA", "BBRI", "BMRI", "BBNI", "TLKM", "ASII", "GOTO", "AMMN"]
    try:
        headers = {'User-Agent': 'Mozilla/5.0'}
        response = requests.get(url, headers=headers, timeout=10)
        tables = pd.read_html(response.text)
        for df in tables:
            cols = [str(c).lower() for c in df.columns]
            if 'kode' in cols or 'simbol' in cols or 'ticker' in cols:
                target_col = next((c for c in df.columns if str(c).lower() in ['kode', 'simbol', 'ticker']), None)
                if target_col:
                    tickers = df[target_col].dropna().astype(str).tolist()
                    return [t.upper().strip() for t in tickers if len(t.strip()) == 4 and t.isalpha()]
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
        max_workers=15
    )
    liquid_universe = scanner.get_liquid_universe()
    print("=" * 60)
    print("TAHAP 2: DEEP TECHNICAL ANALYSIS (BREAKOUT V2)")
    print("=" * 60)
    return liquid_universe

# =========================================================================
# 2. PARAMETER
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
    "min_score": 40,                    # [FIX] Filter skor minimum
    "entry_buffer_pct": 0.5,
    "stop_buffer_pct": 0.5,
    "atr_stop_mult": 1.5,
    "min_risk_reward": 1.5,
    "account_size": 5_000_000,
    "risk_per_trade_pct": 1.0,
    "lot_size": 100,
}

# =========================================================================
# UTILITAS IDX
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
    if is_target:
        return min(price, ara)
    else:
        return max(price, arb)

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
    rsi = 100 - (100 / (1 + rs))
    return rsi

def compute_bb_width(close: pd.Series, period: int = 20, std_mult: float = 2.0) -> pd.Series:
    ma = close.rolling(period).mean()
    std = close.rolling(period).std()
    upper = ma + std_mult * std
    lower = ma - std_mult * std
    width = (upper - lower) / ma * 100
    return width

def compute_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high, low, close = df["High"], df["Low"], df["Close"]
    prev_close = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)
    return tr.rolling(period).mean()

def find_swing_points(df: pd.DataFrame, lookback: int, n_bars: int):
    sub = df.tail(lookback)
    highs, lows = sub["High"], sub["Low"]
    swing_highs, swing_lows = [], []
    idx = list(sub.index)
    for i in range(n_bars, len(idx) - n_bars):
        window_h = highs.iloc[i - n_bars: i + n_bars + 1]
        window_l = lows.iloc[i - n_bars: i + n_bars + 1]
        if highs.iloc[i] == window_h.max():
            swing_highs.append(highs.iloc[i])
        if lows.iloc[i] == window_l.min():
            swing_lows.append(lows.iloc[i])
    return sorted(set(round(x, 0) for x in swing_highs), reverse=True), \
           sorted(set(round(x, 0) for x in swing_lows), reverse=True)

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

    # Entry
    entry = resistance * (1 + params["entry_buffer_pct"] / 100)
    entry = round_to_idx_tick(entry)

    # Stop loss
    stop_by_support = support * (1 - params["stop_buffer_pct"] / 100)
    stop_by_atr = entry - (atr * params["atr_stop_mult"]) if not np.isnan(atr) and atr > 0 else stop_by_support
    stop_loss = min(stop_by_support, stop_by_atr)
    if stop_loss >= entry:
        stop_loss = entry * 0.97

    stop_loss = round_to_idx_tick(stop_loss)
    stop_loss = apply_ara_arb_limits(stop_loss, last_close, is_target=False)

    risk_per_share = entry - stop_loss
    if risk_per_share <= 0:
        return None

    # Target
    target_1 = entry + risk_per_share * 2
    target_2 = max(resistance_2, entry + risk_per_share * 3)

    target_1 = round_to_idx_tick(target_1)
    target_1 = apply_ara_arb_limits(target_1, last_close, is_target=True)
    target_2 = round_to_idx_tick(target_2)
    target_2 = apply_ara_arb_limits(target_2, last_close, is_target=True)

    reward_1 = target_1 - entry
    rr_ratio = reward_1 / risk_per_share if risk_per_share > 0 else 0

    retest_entry = round_to_idx_tick(resistance)

    # Position sizing
    risk_rp = params["account_size"] * params["risk_per_trade_pct"] / 100
    raw_shares = risk_rp / risk_per_share if risk_per_share > 0 else 0
    lots = int(raw_shares // params["lot_size"])
    shares = lots * params["lot_size"]
    est_capital_used = shares * entry
    est_loss = shares * risk_per_share
    est_profit = shares * reward_1

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
        "EstCapitalUsed(Rp)": round(est_capital_used, 0),
        "EstLoss(Rp)": round(est_loss, 0),
        "EstProfit1(Rp)": round(est_profit, 0),
        "Confluence": "; ".join(confluence_notes) if confluence_notes else "-",
    }

def analyze_ticker(symbol: str, params: dict) -> dict | None:
    ticker = symbol + ".JK"
    try:
        # [FIX] Gunakan period + auto_adjust + multi_level_index
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

    if df is None or df.empty or len(df) < 50:          # [FIX] Minimal 50 baris
        return None

    # [FIX] MultiIndex fallback
    if isinstance(df.columns, pd.MultiIndex):
        try:
            df.columns = df.columns.get_level_values(0)
        except Exception:
            df = df.droplevel(-1, axis=1)

    if "Close" not in df.columns:
        return None

    df = df.dropna()
    if len(df) < 50:
        return None

    close = df["Close"]
    high = df["High"]
    low = df["Low"]
    volume = df["Volume"]

    last_close = close.iloc[-1]

    # Likuiditas (soft check — universe sudah di-scan)
    value_traded = close * volume
    avg_value_20 = value_traded.tail(20).mean()
    avg_volume_20 = volume.tail(20).mean()

    if avg_value_20 < params["min_avg_value_traded"] * 0.7:   # [FIX] Soft threshold
        return None
    if avg_volume_20 < params["min_avg_volume"] * 0.7:
        return None

    # Konsolidasi
    win = params["consolidation_window"]
    recent_high = high.tail(win).max()
    recent_low = low.tail(win).min()
    range_pct = (recent_high - recent_low) / last_close * 100
    dist_to_high_pct = (recent_high - last_close) / last_close * 100

    ma20 = close.rolling(20).mean().iloc[-1]
    # [FIX] Guard MA50
    ma50 = close.rolling(50).mean().iloc[-1] if len(close) >= 50 else np.nan
    above_ma20 = last_close > ma20
    ma20_above_ma50 = (ma20 > ma50) if not np.isnan(ma50) else True

    vol_5d_avg = volume.tail(5).mean()
    vol_ratio = vol_5d_avg / avg_volume_20 if avg_volume_20 > 0 else 0

    rsi = compute_rsi(close).iloc[-1]
    bb_width = compute_bb_width(close)
    bb_width_now = bb_width.iloc[-1]
    bb_width_avg = bb_width.tail(60).mean()
    is_squeeze = bb_width_now < bb_width_avg if not np.isnan(bb_width_avg) else False

    atr = compute_atr(df, params["atr_period"]).iloc[-1]
    swing_highs, swing_lows = find_swing_points(df, params["swing_lookback"], params["fractal_bars"])

    # SCORING
    score = 0
    reasons = []

    if range_pct <= params["max_range_pct"]:
        score += 20
        reasons.append(f"Konsolidasi ketat ({range_pct:.1f}%)")
    if dist_to_high_pct <= params["near_high_pct"]:
        score += 20
        reasons.append(f"Dekat resistance ({dist_to_high_pct:.1f}% dari high)")
    if above_ma20 and ma20_above_ma50:
        score += 20
        reasons.append("Uptrend (Close>MA20>MA50)")
    elif above_ma20:
        score += 10
        reasons.append("Close>MA20")
    if vol_ratio >= params["volume_surge_ratio"]:
        score += 20
        reasons.append(f"Volume naik ({vol_ratio:.2f}x avg)")
    if params["rsi_min"] <= rsi <= params["rsi_max"]:
        score += 10
        reasons.append(f"RSI sehat ({rsi:.0f})")
    if is_squeeze:
        score += 10
        reasons.append("BB squeeze (volatilitas mengetat)")

    # [FIX] Filter skor minimum
    if score < params.get("min_score", 40):
        return None

    plan = build_position_plan(last_close, swing_highs, swing_lows, atr,
                               recent_high, recent_low, params)
    if plan is None:
        return None

    result = {
        "Ticker": symbol,
        "Close": round_to_idx_tick(last_close),
        "AvgValue20D(Rp Jt)": round(avg_value_20 / 1_000_000, 0),
        "RangePct": round(range_pct, 1),
        "VolRatio5v20": round(vol_ratio, 2),
        "RSI": round(rsi, 0) if not np.isnan(rsi) else None,
        "Squeeze": is_squeeze,
        "Score": score,
        "Alasan": "; ".join(reasons) if reasons else "-",
    }
    result.update(plan)
    return result

def run_screener(universe=None, params=None):
    params = params or PARAMS
    universe = universe or get_dynamic_liquidity_universe(params)

    print(f"Menjalankan screener v2 (Swing Trading) untuk {len(universe)} saham...\n")
    results = []
    for i, sym in enumerate(universe, 1):
        print(f"  [{i}/{len(universe)}] Cek {sym}...", end="\r")
        row = analyze_ticker(sym, params)
        if row is not None:
            results.append(row)

    print(" " * 60, end="\r")

    if not results:
        print("Tidak ada saham yang lolos filter likuiditas & breakout hari ini.")
        return pd.DataFrame()

    df_result = pd.DataFrame(results).sort_values("Score", ascending=False)
    df_result = df_result.head(params["top_n"]).reset_index(drop=True)

    summary_cols = ["Ticker", "Close", "Score", "RSI", "VolRatio5v20",
                    "Support", "Resistance", "EntryBreakout", "StopLoss",
                    "Target1", "RR_Ratio", "SuggestedLots"]
    print("=" * 120)
    print(f"IDX BREAKOUT SCREENER V2 — {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print("=" * 120)
    print(df_result[summary_cols].to_string(index=False))
    print("=" * 120)

    print("\nRENCANA POSISI — TOP 5 KANDIDAT\n")
    for _, row in df_result.head(5).iterrows():
        rr_flag = "OK" if row["LayakRR"] else "RR KURANG, pertimbangkan skip/tunggu setup lebih baik"
        print(f"--- {row['Ticker']} (Score: {row['Score']}/100) ---")
        print(f"  Alasan breakout   : {row['Alasan']}")
        print(f"  Confluence         : {row['Confluence']}")
        print(f"  Support / Resist   : {row['Support']:.0f} / {row['Resistance']:.0f} (next: {row['Resistance2']:.0f})")
        print(f"  Entry (breakout)   : {row['EntryBreakout']:.0f}  |  Entry (retest)  : {row['EntryRetest']:.0f}")
        print(f"  Stop Loss          : {row['StopLoss']:.0f}  (risiko/saham: Rp{row['RiskPerShare']:.0f})")
        print(f"  Target 1 / 2       : {row['Target1']:.0f} / {row['Target2']:.0f}")
        print(f"  Risk:Reward        : 1:{row['RR_Ratio']:.2f}  [{rr_flag}]")
        if row["SuggestedLots"] == 0:
            print(f"  Saran ukuran posisi: 0 lot — risiko per saham terlalu besar dibanding budget.")
        else:
            print(f"  Saran ukuran posisi: {row['SuggestedLots']} lot ({row['SuggestedShares']} lembar, ~Rp {row['EstCapitalUsed(Rp)']:,.0f})")
        print()

    df_result["Strategy"] = "V2 (Breakout)"
    from idx_report_schema import save_version_report
    out_file = save_version_report(df_result, "v2")
    print(f"Hasil disimpan ke: {out_file}")
    return df_result

if __name__ == "__main__":
    run_screener()