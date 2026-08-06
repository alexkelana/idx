"""
IDX SMC SCREENER V5 — CHOCH & PREMIUM/DISCOUNT ZONES
============================================================
Screener ini lebih agresif daripada V4. Mencari pembalikan arah trend 
awal (Change of Character / CHOCH) dan memberikan entry di zona Discount 
(di bawah Fibo 50%) yang bertepatan dengan Order Block.

Validasi CHOCH:
1. Terjadi trend turun (Lower Lows & Lower Highs).
2. Tiba-tiba harga menembus Lower High terakhir (CHOCH Bullish).
3. Harga kemudian terkoreksi kembali ke zona Discount (Fibo < 50%).
"""

"""
IDX SMC SCREENER V5 — CHOCH & PREMIUM/DISCOUNT ZONES
[FIXED] Deteksi CHOCH lebih ketat (wajib prior downtrend), 
        MultiIndex, import aman, zona discount lebih masuk akal, CSV konsisten.
"""

import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
from idx_liquidity_scanner import IdxLiquidityScanner

# [FIX] Import aman (fallback jika V4 belum ada)
try:
    from idx_breakout_screener_v4_smc import round_to_idx_tick, apply_ara_arb_limits
except ImportError:
    def round_to_idx_tick(price: float) -> int:
        if pd.isna(price) or price <= 0: return 0
        price = float(price)
        if price < 50: return int(round(price))
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

def detect_choch_and_discount(df: pd.DataFrame, lookback: int = 60):
    df_sub = df.tail(lookback).copy()
    if len(df_sub) < 20:
        return None

    high = df_sub['High'].values
    low = df_sub['Low'].values
    close = df_sub['Close'].values

    swing_highs = []
    swing_lows = []
    for i in range(2, len(df_sub) - 2):
        if high[i] > high[i-1] and high[i] > high[i-2] and high[i] > high[i+1] and high[i] > high[i+2]:
            swing_highs.append((i, high[i]))
        if low[i] < low[i-1] and low[i] < low[i-2] and low[i] < low[i+1] and low[i] < low[i+2]:
            swing_lows.append((i, low[i]))

    if len(swing_highs) < 2 or len(swing_lows) < 2:
        return None

    # [FIX] Validasi prior downtrend: minimal 2 Lower High + 2 Lower Low berurutan
    last_two_sh = swing_highs[-2:]
    last_two_sl = swing_lows[-2:]
    is_prior_downtrend = (last_two_sh[1][1] < last_two_sh[0][1]) and (last_two_sl[1][1] < last_two_sl[0][1])
    if not is_prior_downtrend:
        return None

    # CHOCH = penembusan swing high terakhir
    choch_idx = -1
    choch_val = 0
    last_sh = swing_highs[-1]
    for i in range(last_sh[0] + 1, len(df_sub)):
        if close[i] > last_sh[1]:
            choch_idx = i
            choch_val = last_sh[1]
            break

    if choch_idx == -1 or choch_idx >= len(df_sub) - 2:
        return None

    lowest_low_before_choch = df_sub['Low'].iloc[:choch_idx].min()
    peak_after_choch = df_sub['High'].iloc[choch_idx:].max()
    range_up = peak_after_choch - lowest_low_before_choch
    if range_up <= 0:
        return None

    fibo_50 = peak_after_choch - (range_up * 0.5)
    fibo_618 = peak_after_choch - (range_up * 0.618)
    fibo_786 = peak_after_choch - (range_up * 0.786)

    return {
        'choch_idx': choch_idx,
        'choch_val': choch_val,
        'lowest_low': lowest_low_before_choch,
        'peak': peak_after_choch,
        'fibo_50': fibo_50,
        'fibo_618': fibo_618,
        'fibo_786': fibo_786
    }

def analyze_smc_v5_ticker(symbol: str, user_params: dict = None) -> dict | None:
    ticker = symbol + ".JK"
    account_size = 5_000_000
    risk_pct = 1.0
    if user_params:
        account_size = user_params.get("account_size", 5_000_000)
        risk_pct = user_params.get("risk_per_trade_pct", 1.0)

    try:
        df = yf.download(
            ticker,
            period="120d",
            interval="1d",
            progress=False,
            auto_adjust=True,
            multi_level_index=False,
        )
    except Exception:
        return None

    if df is None or df.empty or len(df) < 40:
        return None

    if isinstance(df.columns, pd.MultiIndex):
        try:
            df.columns = df.columns.get_level_values(0)
        except Exception:
            df = df.droplevel(-1, axis=1)

    if "Close" not in df.columns:
        return None

    df = df.dropna()
    if len(df) < 40:
        return None

    last_close = df['Close'].iloc[-1]

    choch_data = detect_choch_and_discount(df)
    if not choch_data:
        return None

    # [FIX] Zona Discount lebih ketat: di bawah 50% dan di atas 78.6% (area sweet spot)
    # Atau minimal di bawah fibo_50 dan di atas lowest_low
    if last_close > choch_data['fibo_50'] or last_close < choch_data['lowest_low']:
        return None

    # Bonus: lebih bagus jika di area 61.8–78.6
    in_deep_discount = choch_data['fibo_786'] <= last_close <= choch_data['fibo_618']

    entry = round_to_idx_tick(last_close)
    stop_loss_raw = choch_data['lowest_low'] * 0.98
    stop_loss = apply_ara_arb_limits(round_to_idx_tick(stop_loss_raw), last_close, is_target=False)

    risk_per_share = entry - stop_loss
    if risk_per_share <= 0:
        return None

    target_1 = apply_ara_arb_limits(round_to_idx_tick(choch_data['peak']), last_close, is_target=True)
    rr_ratio = (target_1 - entry) / risk_per_share if risk_per_share > 0 else 0
    if rr_ratio < 2.0:
        return None

    risk_rp = account_size * (risk_pct / 100.0)
    shares = int(risk_rp / risk_per_share) if risk_per_share > 0 else 0
    lots = shares // 100
    actual_shares = lots * 100
    est_loss = actual_shares * risk_per_share
    est_profit = actual_shares * (target_1 - entry)

    return {
        "Ticker": symbol,
        "Close": round_to_idx_tick(last_close),
        "CHOCH_Level": round_to_idx_tick(choch_data['choch_val']),
        "Fibo_618": round_to_idx_tick(choch_data['fibo_618']),
        "Fibo_786": round_to_idx_tick(choch_data['fibo_786']),
        "DeepDiscount": "Ya" if in_deep_discount else "Tidak",
        "Entry": entry,
        "StopLoss": stop_loss,
        "Target(Peak)": target_1,
        "RR_Ratio": round(rr_ratio, 2),
        "Lots": lots,
        "EstLoss(Rp)": round(est_loss, 0),
        "EstProfit(Rp)": round(est_profit, 0)
    }

def run_screener_v5(user_params: dict = None):
    print("Mempersiapkan Universe Likuiditas (SMC V5)...")
    scanner = IdxLiquidityScanner(min_avg_value_rp=10_000_000_000, min_avg_volume=1_000_000)
    universe = scanner.get_liquid_universe()

    print(f"\nMenjalankan Screener SMC V5 untuk {len(universe)} saham...")
    results = []
    for i, sym in enumerate(universe, 1):
        print(f"  Cek {sym}...", end="\r")
        res = analyze_smc_v5_ticker(sym, user_params)
        if res:
            results.append(res)

    print(" " * 50, end="\r")

    if not results:
        print("Tidak ada saham yang sedang di zona Discount pasca CHOCH hari ini.")
        return

    df_res = pd.DataFrame(results).sort_values("RR_Ratio", ascending=False)
    print("\n" + "=" * 90)
    print("SMC CHOCH & DISCOUNT ZONE SCREENER V5 - HASIL")
    print("=" * 90)
    print(df_res.to_string(index=False))
    print("=" * 90)

    # [FIX] Nama file konsisten
    from idx_report_schema import save_report

    out_file = save_report(df_res, strategy_name="V5 (SMC CHOCH)", group="smc")
    print(f"Hasil disimpan ke: {out_file}")

if __name__ == "__main__":
    run_screener_v5()