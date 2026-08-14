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

Aturan original + bug index fixed + SL hybrid kondisional:
- Jika jarak invalidasi (OB bottom) ≤ max_sl_pct → ATR boleh memperketat
- Jika di luar range → SL = struktur murni (OB_bottom × buffer)
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime
from idx_liquidity_scanner import IdxLiquidityScanner

# =========================================================================
# PARAMETER SL
# =========================================================================
SL_PARAMS = {
    "atr_period": 14,
    "sl_struct_buf": 0.015,   # 1.5% di bawah OB bottom
    "sl_atr_mult": 1.2,
    "max_sl_pct": 3.5,        # ambang "masuk range"
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


def compute_atr(df: pd.DataFrame, period: int = 14) -> float:
    h, l, c = df["High"], df["Low"], df["Close"]
    prev = c.shift(1)
    tr = pd.concat([h - l, (h - prev).abs(), (l - prev).abs()], axis=1).max(axis=1)
    val = float(tr.rolling(period).mean().iloc[-1])
    return val if val == val and val > 0 else 0.0


def stop_loss_struct_with_optional_atr(
    entry: float,
    struct_level: float,
    atr: float,
    prev_close: float,
    params: dict,
) -> tuple[int, str]:
    """
    Masuk range (jarak struct ≤ max_sl_pct) → ATR boleh memperketat.
    Luar range → prinsip awal (struktur × buffer).
    ATR tidak pernah menaruh SL di bawah struktur.
    """
    buf = params.get("sl_struct_buf", 0.015)
    k = params.get("sl_atr_mult", 1.2)
    max_sl_pct = params.get("max_sl_pct", 3.5)

    sl_struct = float(struct_level) * (1.0 - buf)
    if sl_struct >= entry:
        sl_struct = entry * 0.99

    dist_pct = (entry - sl_struct) / entry * 100.0

    if dist_pct <= max_sl_pct and atr and atr > 0:
        sl_atr = entry - k * atr
        stop_raw = max(sl_struct, sl_atr)  # tidak di bawah struktur
        source = "struct+atr"
    else:
        stop_raw = sl_struct
        source = "struct"

    if stop_raw >= entry:
        stop_raw = entry * 0.99

    stop = round_to_idx_tick(stop_raw)
    stop = apply_ara_arb_limits(stop, prev_close, is_target=False)
    if stop >= entry:
        stop = apply_ara_arb_limits(
            round_to_idx_tick(entry * 0.99), prev_close, is_target=False
        )
    return int(stop), source


# =========================================================================
# SMC — OB / FVG
# =========================================================================
def detect_smc_zones(df: pd.DataFrame, lookback: int = 60):
    if len(df) < 15:
        return []

    start_pos = max(0, len(df) - lookback)
    df_sub = df.iloc[start_pos:].copy()
    if len(df_sub) < 10:
        return []

    high = df_sub["High"].values
    low = df_sub["Low"].values
    close = df_sub["Close"].values
    open_p = df_sub["Open"].values

    swing_highs = []
    for i in range(2, len(df_sub) - 2):
        if (
            high[i] > high[i - 1]
            and high[i] > high[i - 2]
            and high[i] > high[i + 1]
            and high[i] > high[i + 2]
        ):
            swing_highs.append((i, high[i]))

    ob_zones = []
    for sh_idx, sh_val in swing_highs:
        for i in range(sh_idx + 1, len(df_sub) - 2):
            if close[i] > sh_val:
                fvg_gap = low[i + 1] - high[i - 1]
                if fvg_gap > 0:
                    ob_candle_idx = -1
                    for j in range(i - 1, sh_idx - 1, -1):
                        if close[j] < open_p[j]:
                            ob_candle_idx = j
                            break
                    if ob_candle_idx != -1:
                        ob_zones.append(
                            {
                                "bos_idx": start_pos + i,
                                "ob_idx": start_pos + ob_candle_idx,
                                "ob_high": float(high[ob_candle_idx]),
                                "ob_low": float(low[ob_candle_idx]),
                                "fvg_gap": float(fvg_gap),
                            }
                        )
                        break
    return ob_zones


# =========================================================================
# ANALISA
# =========================================================================
def analyze_smc_ticker(symbol: str, user_params: dict = None) -> dict | None:
    ticker = symbol + ".JK"

    account_size = 5_000_000
    risk_pct = 1.0
    sl_cfg = SL_PARAMS.copy()
    if user_params:
        account_size = user_params.get("account_size", account_size)
        risk_pct = user_params.get("risk_per_trade_pct", risk_pct)
        for k in SL_PARAMS:
            if k in user_params:
                sl_cfg[k] = user_params[k]

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
            try:
                df = df.droplevel(-1, axis=1)
            except Exception:
                return None

    if any(c not in df.columns for c in ["Open", "High", "Low", "Close"]):
        return None

    df = df.dropna()
    if len(df) < 40:
        return None

    last_close = float(df["Close"].iloc[-1])
    last_low = float(df["Low"].iloc[-1])

    ob_zones = detect_smc_zones(df, lookback=60)
    if not ob_zones:
        return None

    latest_ob = ob_zones[-1]
    ob_top = latest_ob["ob_high"]
    ob_bottom = latest_ob["ob_low"]

    dist_to_ob_pct = (last_close - ob_top) / ob_top * 100
    has_mitigated = last_low <= ob_top and last_close >= ob_bottom
    if dist_to_ob_pct > 3.0 or not has_mitigated:
        return None

    entry = round_to_idx_tick(last_close)
    atr = compute_atr(df, sl_cfg["atr_period"])

    stop_loss, sl_src = stop_loss_struct_with_optional_atr(
        entry=entry,
        struct_level=float(ob_bottom),
        atr=atr,
        prev_close=last_close,
        params=sl_cfg,
    )

    risk_per_share = entry - stop_loss
    if risk_per_share <= 0:
        return None

    peak_after_bos = float(df["High"].iloc[latest_ob["bos_idx"] :].max())
    target_1 = apply_ara_arb_limits(
        round_to_idx_tick(peak_after_bos), last_close, is_target=True
    )

    rr_ratio = (target_1 - entry) / risk_per_share if risk_per_share > 0 else 0
    if rr_ratio < 1.5:
        return None

    risk_rp = account_size * (risk_pct / 100.0)
    shares = int(risk_rp / risk_per_share) if risk_per_share > 0 else 0
    lots = shares // 100
    actual_shares = lots * 100

    return {
        "Ticker": symbol,
        "Close": round_to_idx_tick(last_close),
        "OB_Top": round_to_idx_tick(ob_top),
        "OB_Bottom": round_to_idx_tick(ob_bottom),
        "Entry": entry,
        "StopLoss": stop_loss,
        "SL_Source": sl_src,
        "ATR": round(atr, 0),
        "Target(Liquidity)": target_1,
        "RR_Ratio": round(rr_ratio, 2),
        "Lots": lots,
        "EstLoss(Rp)": round(actual_shares * risk_per_share, 0),
        "EstProfit(Rp)": round(actual_shares * (target_1 - entry), 0),
        "Strategy": "V4 (SMC Order Block)",
    }


def run_screener_v4(user_params: dict = None):
    print("Mempersiapkan Universe Likuiditas (SMC V4)...")
    scanner = IdxLiquidityScanner(
        min_avg_value_rp=10_000_000_000,
        min_avg_volume=1_000_000,
    )
    universe = scanner.get_liquid_universe()

    print(f"\nMenjalankan Screener SMC V4 untuk {len(universe)} saham...")
    results = []
    for i, sym in enumerate(universe, 1):
        print(f"  [{i}/{len(universe)}] Cek {sym}...", end="\r")
        try:
            res = analyze_smc_ticker(sym, user_params)
            if res:
                results.append(res)
        except Exception:
            continue

    print(" " * 60, end="\r")
    if not results:
        print("Tidak ada saham yang sedang Mitigasi Order Block hari ini.")
        return None

    df_res = pd.DataFrame(results).sort_values("RR_Ratio", ascending=False)
    print("\n" + "=" * 90)
    print("SMC ORDER BLOCK SCREENER V4")
    print("=" * 90)
    print(df_res.to_string(index=False))
    print("=" * 90)

    df_res["Strategy"] = "V4 (SMC Order Block)"
    try:
        from idx_report_schema import save_version_report
        out_file = save_version_report(df_res, "v4")
    except ImportError:
        out_file = f"idx_report_v4_{datetime.now().strftime('%Y-%m-%d')}.csv"
        df_res.to_csv(out_file, index=False)
    print(f"Hasil disimpan ke: {out_file}")
    return df_res


if __name__ == "__main__":
    run_screener_v4()