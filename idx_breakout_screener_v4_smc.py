"""
IDX SMC SCREENER V4 — ORDER BLOCK & FVG DETECTION
============================================================
Versi awal (aturan original) + bug diperbaiki.

Aturan (tetap sama dengan original):
1. Ambil Order Block paling baru
2. Jarak maksimal dari OB Top = 3%
3. Mitigation: Low masuk ke OB dan Close masih di atas OB Bottom
4. RR minimum 1.5

Bug yang diperbaiki:
- bos_idx sekarang absolut (Target Liquidity tidak loncat jauh)
- MultiIndex & data guard
- Download lebih stabil
"""

import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime
from idx_liquidity_scanner import IdxLiquidityScanner

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

# =========================================================================
# SMC ALGORITHMS (bug indexing diperbaiki)
# =========================================================================
def detect_smc_zones(df: pd.DataFrame, lookback: int = 60):
    """Mendeteksi Order Block dan FVG. Index disimpan secara absolut."""
    if len(df) < 15:
        return []

    # [FIX] Simpan offset agar index absolut
    start_pos = max(0, len(df) - lookback)
    df_sub = df.iloc[start_pos:].copy()

    if len(df_sub) < 10:
        return []

    high = df_sub["High"].values
    low = df_sub["Low"].values
    close = df_sub["Close"].values
    open_p = df_sub["Open"].values

    # 1. Cari Swing High
    swing_highs = []
    for i in range(2, len(df_sub) - 2):
        if (high[i] > high[i-1] and high[i] > high[i-2] and
            high[i] > high[i+1] and high[i] > high[i+2]):
            swing_highs.append((i, high[i]))

    ob_zones = []

    # 2. Cari BOS & FVG
    for sh_idx, sh_val in swing_highs:
        for i in range(sh_idx + 1, len(df_sub) - 2):
            if close[i] > sh_val:  # BOS
                fvg_gap = low[i + 1] - high[i - 1]
                has_fvg = fvg_gap > 0

                if has_fvg:
                    # Cari candle bearish terakhir (Order Block)
                    ob_candle_idx = -1
                    for j in range(i - 1, sh_idx - 1, -1):
                        if close[j] < open_p[j]:
                            ob_candle_idx = j
                            break

                    if ob_candle_idx != -1:
                        # [FIX] Konversi ke index absolut
                        absolute_bos_idx = start_pos + i
                        absolute_ob_idx = start_pos + ob_candle_idx

                        ob_zones.append({
                            "bos_idx": absolute_bos_idx,
                            "ob_idx": absolute_ob_idx,
                            "ob_high": float(high[ob_candle_idx]),
                            "ob_low": float(low[ob_candle_idx]),
                            "fvg_gap": float(fvg_gap),
                        })
                        break

    return ob_zones

# =========================================================================
# ANALISIS (aturan original dipertahankan)
# =========================================================================
def analyze_smc_ticker(symbol: str, user_params: dict = None) -> dict | None:
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

    # MultiIndex fallback
    if isinstance(df.columns, pd.MultiIndex):
        try:
            df.columns = df.columns.get_level_values(0)
        except Exception:
            try:
                df = df.droplevel(-1, axis=1)
            except Exception:
                return None

    if "Close" not in df.columns or "High" not in df.columns or "Low" not in df.columns:
        return None

    df = df.dropna()
    if len(df) < 40:
        return None

    last_close = float(df["Close"].iloc[-1])
    last_low = float(df["Low"].iloc[-1])

    # Deteksi Zona SMC
    ob_zones = detect_smc_zones(df, lookback=60)
    if not ob_zones:
        return None

    # ===== ATURAN ORIGINAL DIPERTAHANKAN =====
    # Ambil OB yang paling baru
    latest_ob = ob_zones[-1]
    ob_top = latest_ob["ob_high"]
    ob_bottom = latest_ob["ob_low"]

    # Maksimal jarak 3%
    dist_to_ob_pct = (last_close - ob_top) / ob_top * 100

    # Harus menyentuh OB (mitigation)
    has_mitigated = last_low <= ob_top and last_close >= ob_bottom

    if dist_to_ob_pct > 3.0 or not has_mitigated:
        return None

    # Trading Plan
    entry = round_to_idx_tick(last_close)

    stop_loss_raw = ob_bottom * 0.985
    stop_loss = round_to_idx_tick(stop_loss_raw)
    stop_loss = apply_ara_arb_limits(stop_loss, last_close, is_target=False)

    risk_per_share = entry - stop_loss
    if risk_per_share <= 0:
        return None

    # [FIX] Target sekarang akurat karena bos_idx absolut
    peak_after_bos = df["High"].iloc[latest_ob["bos_idx"]:].max()
    target_1 = apply_ara_arb_limits(
        round_to_idx_tick(peak_after_bos), last_close, is_target=True
    )

    rr_ratio = (target_1 - entry) / risk_per_share if risk_per_share > 0 else 0
    if rr_ratio < 1.5:
        return None

    # Position Sizing
    risk_rp = account_size * (risk_pct / 100.0)
    shares = int(risk_rp / risk_per_share) if risk_per_share > 0 else 0
    lots = shares // 100
    actual_shares = lots * 100
    est_loss = actual_shares * risk_per_share
    est_profit = actual_shares * (target_1 - entry)

    return {
        "Ticker": symbol,
        "Close": round_to_idx_tick(last_close),
        "OB_Top": round_to_idx_tick(ob_top),
        "OB_Bottom": round_to_idx_tick(ob_bottom),
        "Entry": entry,
        "StopLoss": stop_loss,
        "Target(Liquidity)": target_1,
        "RR_Ratio": round(rr_ratio, 2),
        "Lots": lots,
        "EstLoss(Rp)": round(est_loss, 0),
        "EstProfit(Rp)": round(est_profit, 0),
    }

def run_screener_v4(user_params: dict = None):
    print("Mempersiapkan Universe Likuiditas (SMC V4 - Original Rules + Bug Fixed)...")
    scanner = IdxLiquidityScanner(
        min_avg_value_rp=10_000_000_000,
        min_avg_volume=1_000_000,
    )
    universe = scanner.get_liquid_universe()

    print(f"\nMenjalankan Screener SMC V4 untuk {len(universe)} saham...")
    results = []
    for i, sym in enumerate(universe, 1):
        print(f"  [{i}/{len(universe)}] Cek {sym}...", end="\r")
        res = analyze_smc_ticker(sym, user_params)
        if res:
            results.append(res)

    print(" " * 60, end="\r")

    if not results:
        print("Tidak ada saham yang sedang Mitigasi Order Block hari ini.")
        return

    df_res = pd.DataFrame(results).sort_values("RR_Ratio", ascending=False)

    print("\n" + "=" * 90)
    print("SMC ORDER BLOCK SCREENER V4 - HASIL (Original Rules + Bug Fixed)")
    print("=" * 90)
    print(df_res.to_string(index=False))
    print("=" * 90)

    # CSV konsisten
    from idx_report_schema import save_report

    out_file = save_report(df_res, strategy_name="V4 (SMC Order Block)", group="smc")
    print(f"Hasil disimpan ke: {out_file}")
    return df_res

if __name__ == "__main__":
    run_screener_v4()