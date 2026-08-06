"""
IDX PULLBACK & RETEST SCREENER — v3 (Buy on Weakness setelah Breakout)
[FIXED] atr NameError, params ke scanner, MultiIndex, min RR, data guard.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import requests
import yfinance as yf
from datetime import datetime, timedelta
from idx_liquidity_scanner import IdxLiquidityScanner

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

def get_dynamic_liquidity_universe(params: dict = None) -> list:          # [FIX] Terima params
    print("=" * 60)
    print("TAHAP 1: PRE-SCREENER (MEMINDAI SELURUH PASAR)")
    print("=" * 60)
    params = params or {}
    scanner = IdxLiquidityScanner(
        min_avg_value_rp=params.get("min_avg_value_traded", 10_000_000_000),
        min_avg_volume=params.get("min_avg_volume", 1_000_000),
        lookback_days=20,
        max_workers=15
    )
    liquid_universe = scanner.get_liquid_universe()
    print("=" * 60)
    print("TAHAP 2: DEEP TECHNICAL ANALYSIS (RETEST V3)")
    print("=" * 60)
    return liquid_universe

PARAMS = {
    "lookback_days": 350,
    "breakout_lookback": 10,
    "breakout_vol_ratio": 1.5,
    "adx_period": 14,
    "roc_period": 10,
    "stop_buffer_pct": 2.0,
    "min_rr": 1.2,                          # [FIX] Minimum RR
    "account_size": 5_000_000,
    "risk_per_trade_pct": 1.0,
    "lot_size": 100,
    "min_avg_value_traded": 10_000_000_000,
    "min_avg_volume": 1_000_000,
}

def round_to_idx_tick(price: float) -> int:
    if pd.isna(price) or price <= 0:
        return 0
    price = float(price)
    if price < 50:
        return int(round(price))
    price = int(round(price, 0))
    if price < 200: return price
    elif price < 500: return int(round(price / 2.0) * 2)
    elif price < 2000: return int(round(price / 5.0) * 5)
    elif price < 5000: return int(round(price / 10.0) * 10)
    else: return int(round(price / 25.0) * 25)

def apply_ara_arb_limits(price: float, prev_close: float, is_target: bool) -> float:
    limit = 0.35 if prev_close < 200 else (0.25 if prev_close <= 5000 else 0.20)
    ara = round_to_idx_tick(prev_close * (1 + limit))
    arb = round_to_idx_tick(prev_close * (1 - limit))
    return min(price, ara) if is_target else max(price, arb)

def compute_adx(df: pd.DataFrame, period: int = 14):
    high, low, close = df['High'], df['Low'], df['Close']
    up = high - high.shift(1)
    down = low.shift(1) - low
    plus_dm = np.where((up > down) & (up > 0), up, 0.0)
    minus_dm = np.where((down > up) & (down > 0), down, 0.0)
    tr1 = high - low
    tr2 = abs(high - close.shift(1))
    tr3 = abs(low - close.shift(1))
    tr = pd.DataFrame({'tr1': tr1, 'tr2': tr2, 'tr3': tr3}).max(axis=1)
    atr = tr.ewm(alpha=1/period, adjust=False).mean()
    plus_di = 100 * pd.Series(plus_dm, index=df.index).ewm(alpha=1/period, adjust=False).mean() / atr
    minus_di = 100 * pd.Series(minus_dm, index=df.index).ewm(alpha=1/period, adjust=False).mean() / atr
    dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = dx.ewm(alpha=1/period, adjust=False).mean()
    return adx, plus_di, minus_di

def compute_roc(series: pd.Series, period: int = 10) -> pd.Series:
    return series.pct_change(periods=period) * 100

def compute_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:     # [FIX] Fungsi ATR ditambahkan
    high, low, close = df["High"], df["Low"], df["Close"]
    prev_close = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)
    return tr.rolling(period).mean()

def analyze_ticker(symbol: str, params: dict) -> dict | None:
    ticker = symbol + ".JK"
    try:
        df = yf.download(
            ticker,
            period=f"{params['lookback_days']}d",
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
            df = df.droplevel(-1, axis=1)

    if "Close" not in df.columns:
        return None

    df = df.dropna()
    if len(df) < 150:
        return None

    # 1. REGIME FILTER (Mingguan)
    weekly = df.resample('W').agg({
        'Open': 'first', 'High': 'max', 'Low': 'min',
        'Close': 'last', 'Volume': 'sum'
    }).dropna()
    if len(weekly) < 30:
        return None

    ma10w = weekly['Close'].rolling(10).mean()
    ma30w = weekly['Close'].rolling(30).mean()

    w_close = weekly['Close'].iloc[-1]
    w_ma10 = ma10w.iloc[-1]
    w_ma30 = ma30w.iloc[-1]
    w_ma10_prev = ma10w.iloc[-2]

    if pd.isna(w_ma10) or pd.isna(w_ma30):
        return None

    is_bullish_regime = (w_close > w_ma10 > w_ma30) and (w_ma10 > w_ma10_prev)
    if not is_bullish_regime:
        return None

    # 2. CARI BREAKOUT TERKONFIRMASI
    breakout_idx = -1
    breakout_price = 0
    breakout_vol = 0

    for i in range(len(df)-1, max(len(df) - params["breakout_lookback"] - 1, 20), -1):
        past_high = df['High'].iloc[i-20:i].max()
        past_vol_avg = df['Volume'].iloc[i-20:i].mean()
        if df['Close'].iloc[i] > past_high and df['Volume'].iloc[i] >= (past_vol_avg * params["breakout_vol_ratio"]):
            breakout_idx = i
            breakout_price = past_high
            breakout_vol = df['Volume'].iloc[i]
            break

    if breakout_idx == -1 or breakout_idx >= len(df) - 2:   # [FIX] Minimal 2 candle setelah breakout
        return None

    # 3. DETEKSI KOREKSI & FIBONACCI
    swing_low = df['Low'].iloc[breakout_idx-20:breakout_idx].min()
    peak = df['High'].iloc[breakout_idx:].max()
    range_up = peak - swing_low
    if range_up <= 0:
        return None

    fib_382 = peak - 0.382 * range_up
    fib_618 = peak - 0.618 * range_up

    last_close = df['Close'].iloc[-1]
    last_open = df['Open'].iloc[-1]
    last_vol = df['Volume'].iloc[-1]
    prev_vol = df['Volume'].iloc[-2]

    in_fibo_zone = fib_618 <= last_close <= fib_382
    above_breakout_support = last_close >= breakout_price

    if not (in_fibo_zone and above_breakout_support):
        return None

    # 4. VALIDASI VOLUME KOREKSI
    peak_loc = df['High'].iloc[breakout_idx:].idxmax()
    peak_idx_int = df.index.get_loc(peak_loc)
    if peak_idx_int < len(df) - 1:
        vol_correction_avg = df['Volume'].iloc[peak_idx_int+1:].mean()
    else:
        vol_correction_avg = last_vol
    is_vol_valid = vol_correction_avg < breakout_vol

    # 5. MOMENTUM
    adx, plus_di, minus_di = compute_adx(df, params["adx_period"])
    curr_adx = adx.iloc[-1]
    curr_pdi = plus_di.iloc[-1]
    curr_mdi = minus_di.iloc[-1]
    roc = compute_roc(df['Close'], params["roc_period"]).iloc[-1]

    # [FIX] Hitung ATR
    atr = compute_atr(df, 14).iloc[-1]

    # 6. SINYAL REVERSAL
    is_bullish_candle = last_close > last_open
    is_vol_up = last_vol > prev_vol
    reversal_signal = is_bullish_candle and is_vol_up

    # SCORING
    score = 20
    reasons = ["Weekly Bullish"]
    if is_vol_valid:
        score += 20
        reasons.append("Vol Koreksi Rendah")
    if curr_adx > 25 and curr_pdi > curr_mdi:
        score += 20
        reasons.append(f"ADX Kuat ({curr_adx:.1f})")
    if roc > 0:
        score += 10
        reasons.append("ROC Positif")
    if in_fibo_zone and above_breakout_support:
        score += 15
        reasons.append("Fibo Golden Zone")
    if reversal_signal:
        score += 15
        reasons.append("Reversal Candle")

    # POSITION PLAN
    entry = round_to_idx_tick(last_close)
    stop_loss_raw = breakout_price * (1 - params["stop_buffer_pct"] / 100)
    stop_loss = round_to_idx_tick(stop_loss_raw)
    stop_loss = apply_ara_arb_limits(stop_loss, last_close, is_target=False)

    risk_per_share = entry - stop_loss
    if risk_per_share <= 0:
        return None

    target_1 = round_to_idx_tick(peak)
    target_1 = apply_ara_arb_limits(target_1, last_close, is_target=True)
    target_2_raw = peak + 0.272 * range_up
    target_2 = round_to_idx_tick(target_2_raw)
    target_2 = apply_ara_arb_limits(target_2, last_close, is_target=True)

    rr_ratio = (target_1 - entry) / risk_per_share if risk_per_share > 0 else 0
    if rr_ratio < params.get("min_rr", 1.2):               # [FIX] Filter RR
        return None

    risk_rp = params["account_size"] * params["risk_per_trade_pct"] / 100
    lots = int((risk_rp / risk_per_share) // params["lot_size"]) if risk_per_share > 0 else 0
    shares = lots * params["lot_size"]
    est_capital_used = shares * entry
    est_loss = shares * risk_per_share
    reward_1 = target_1 - entry
    est_profit = shares * reward_1

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
        "ATR": round(atr, 0) if not np.isnan(atr) else 0,   # [FIX] Sekarang aman
        "RiskPerShare": risk_per_share,
        "RR_Ratio": round(rr_ratio, 2),
        "Lots": lots,
        "SuggestedShares": shares,
        "EstCapitalUsed(Rp)": round(est_capital_used, 0),
        "EstLoss(Rp)": round(est_loss, 0),
        "EstProfit1(Rp)": round(est_profit, 0)
    }

def run_screener(universe=None, params=None):
    params = params or PARAMS
    universe = universe or get_dynamic_liquidity_universe(params)

    print(f"Menjalankan screener v3 (Pullback & Retest) untuk {len(universe)} saham...\n")
    results = []
    for i, sym in enumerate(universe, 1):
        print(f"  [{i}/{len(universe)}] Cek {sym}...", end="\r")
        row = analyze_ticker(sym, params)
        if row is not None:
            results.append(row)

    print(" " * 60, end="\r")

    if not results:
        print("Tidak ada saham yang lolos filter Pullback/Retest hari ini.")
        return pd.DataFrame()

    df_result = pd.DataFrame(results).sort_values("Score", ascending=False)

    summary_cols = ["Ticker", "Close", "Score", "BreakoutDay", "VolValid", "Reversal",
                    "ADX", "ROC(10)", "Entry", "StopLoss", "Target1(Peak)", "RR_Ratio", "Lots"]
    print("=" * 140)
    print(f"IDX PULLBACK & RETEST SCREENER V3 — {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print("=" * 140)
    print(df_result[summary_cols].to_string(index=False))
    print("=" * 140)

    print("\nRENCANA POSISI — TOP SETUP\n")
    for _, row in df_result.head(5).iterrows():
        print(f"--- {row['Ticker']} (Score: {row['Score']}/100) ---")
        print(f"  Katalis Setup      : {row['Alasan']}")
        print(f"  Breakout Tanggal   : {row['BreakoutDay']}")
        print(f"  Momentum           : ADX = {row['ADX']} | ROC(10) = {row['ROC(10)']}%")
        print(f"  Zona Fibo Ideal    : {row['Fibo618']} (61.8%) s/d {row['Fibo382']} (38.2%)")
        print(f"  Entry (Buy)        : {row['Entry']}")
        print(f"  Stop Loss          : {row['StopLoss']}")
        print(f"  Target 1 (Puncak)  : {row['Target1(Peak)']}")
        print(f"  Target 2 (Ext 127%): {row['Target2(Ext)']}")
        print(f"  Risk:Reward        : 1:{row['RR_Ratio']}")
        print(f"  Saran Posisi       : {row['Lots']} lot\n")

    # [FIX] Nama file konsisten
    from idx_report_schema import save_report

    out_file = save_report(df_result, strategy_name="V3 (Retest Fibo)", group="klasik")
    print(f"Hasil ditambahkan ke: {out_file}")
    return df_result

if __name__ == "__main__":
    run_screener()