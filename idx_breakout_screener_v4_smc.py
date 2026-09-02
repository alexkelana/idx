"""
IDX SMC SCREENER V4 — ORDER BLOCK & FVG
============================================================
+ SL hybrid kondisional (struct / struct+atr)
+ Estimasi sesi ke Target Liquidity (ATR × momentum k)
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime
from idx_liquidity_scanner import IdxLiquidityScanner

SL_PARAMS = {
    "atr_period": 14,
    "sl_struct_buf": 0.015,
    "sl_atr_mult": 1.2,
    "max_sl_pct": 3.5,
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


def compute_adx(df: pd.DataFrame, period: int = 14):
    high, low, close = df["High"], df["Low"], df["Close"]
    up = high.diff()
    down = -low.diff()
    plus_dm = np.where((up > down) & (up > 0), up, 0.0)
    minus_dm = np.where((down > up) & (down > 0), down, 0.0)
    prev_c = close.shift(1)
    tr = pd.concat(
        [high - low, (high - prev_c).abs(), (low - prev_c).abs()], axis=1
    ).max(axis=1)
    atr = tr.ewm(alpha=1 / period, adjust=False).mean()
    plus_di = 100 * pd.Series(plus_dm, index=df.index).ewm(
        alpha=1 / period, adjust=False
    ).mean() / atr.replace(0, np.nan)
    minus_di = 100 * pd.Series(minus_dm, index=df.index).ewm(
        alpha=1 / period, adjust=False
    ).mean() / atr.replace(0, np.nan)
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    adx = dx.ewm(alpha=1 / period, adjust=False).mean()
    return (
        float(adx.iloc[-1]) if not pd.isna(adx.iloc[-1]) else 0.0,
        float(plus_di.iloc[-1]) if not pd.isna(plus_di.iloc[-1]) else 0.0,
        float(minus_di.iloc[-1]) if not pd.isna(minus_di.iloc[-1]) else 0.0,
    )


def compute_roc(series: pd.Series, period: int = 10) -> float:
    val = series.pct_change(periods=period).iloc[-1]
    return float(val * 100) if not pd.isna(val) else 0.0


def stop_loss_struct_with_optional_atr(
    entry: float,
    struct_level: float,
    atr: float,
    prev_close: float,
    params: dict,
) -> tuple[int, str]:
    buf = params.get("sl_struct_buf", 0.015)
    k = params.get("sl_atr_mult", 1.2)
    max_sl_pct = params.get("max_sl_pct", 3.5)

    sl_struct = float(struct_level) * (1.0 - buf)
    if sl_struct >= entry:
        sl_struct = entry * 0.99

    dist_pct = (entry - sl_struct) / entry * 100.0

    if dist_pct <= max_sl_pct and atr and atr > 0:
        sl_atr = entry - k * atr
        stop_raw = max(sl_struct, sl_atr)
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


def estimate_days_to_target(
    entry: float,
    target: float,
    atr: float,
    adx: float = 0.0,
    plus_di: float = 0.0,
    minus_di: float = 0.0,
    vol_ratio: float = 1.0,
    roc10: float = 0.0,
):
    """
    Estimasi sesi bursa ke target liquidity.
    Returns: (eta, eta_fast, eta_slow, k_used) atau (None,...)
    """
    dist = float(target) - float(entry)
    if dist <= 0 or not atr or atr <= 0:
        return None, None, None, None

    k = 0.35
    if adx > 25 and plus_di > minus_di:
        k += 0.15
    elif adx > 20 and plus_di > minus_di:
        k += 0.08
    if vol_ratio >= 1.3:
        k += 0.10
    elif vol_ratio >= 1.1:
        k += 0.05
    if roc10 > 5:
        k += 0.12
    elif roc10 > 2:
        k += 0.06
    k = max(0.25, min(0.90, k))

    eta = dist / (k * atr)
    eta_fast = dist / (min(0.90, k * 1.25) * atr)
    eta_slow = dist / (max(0.25, k * 0.75) * atr)

    return round(eta, 1), round(eta_fast, 1), round(eta_slow, 1), round(k, 2)


# =========================================================================
# SMC ZONES
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

    if any(c not in df.columns for c in ["Open", "High", "Low", "Close", "Volume"]):
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

    # --- Momentum untuk ETA ---
    vol = df["Volume"]
    avg_vol_20 = float(vol.tail(20).mean()) if len(vol) >= 20 else float(vol.mean())
    vol_ratio = (
        float(vol.tail(5).mean()) / avg_vol_20 if avg_vol_20 and avg_vol_20 > 0 else 1.0
    )
    adx, plus_di, minus_di = compute_adx(df, 14)
    roc10 = compute_roc(df["Close"], 10)

    eta, eta_fast, eta_slow, k_used = estimate_days_to_target(
        entry=entry,
        target=target_1,
        atr=atr,
        adx=adx,
        plus_di=plus_di,
        minus_di=minus_di,
        vol_ratio=vol_ratio,
        roc10=roc10,
    )
    dist_to_target = max(0.0, float(target_1) - float(entry))

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
        "DistToTarget": round(dist_to_target, 0),
        "RR_Ratio": round(rr_ratio, 2),
        "ETA_Sesi": eta,
        "ETA_Cepat": eta_fast,
        "ETA_Lambat": eta_slow,
        "MomentumK": k_used,
        "ADX": round(adx, 1),
        "VolRatio": round(vol_ratio, 2),
        "ROC10": round(roc10, 2),
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

    show_cols = [
        "Ticker", "Close", "Entry", "StopLoss", "Target(Liquidity)",
        "DistToTarget", "RR_Ratio", "ETA_Sesi", "ETA_Cepat", "ETA_Lambat",
        "MomentumK", "ATR", "ADX", "VolRatio", "SL_Source", "Lots",
    ]
    show_cols = [c for c in show_cols if c in df_res.columns]

    print("\n" + "=" * 110)
    print(f"SMC ORDER BLOCK SCREENER V4 — {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print("ETA_Sesi ≈ jarak target / (MomentumK × ATR) | satuan: sesi bursa (perkiraan)")
    print("=" * 110)
    print(df_res[show_cols].to_string(index=False))
    print("=" * 110)

    print("\nRINGKASAN ETA (TOP 5 RR)")
    for _, row in df_res.head(5).iterrows():
        print(
            f"  {row['Ticker']}: target {row['Target(Liquidity)']} | "
            f"ETA ~{row['ETA_Sesi']} sesi "
            f"(cepat {row['ETA_Cepat']} / lambat {row['ETA_Lambat']}) | "
            f"k={row['MomentumK']} ATR={row['ATR']}"
        )

    df_res["Strategy"] = "V4 (SMC Order Block)"
    try:
        from idx_report_schema import save_version_report
        out_file = save_version_report(df_res, "v4")
    except ImportError:
        out_file = f"idx_report_v4_{datetime.now().strftime('%Y-%m-%d')}.csv"
        df_res.to_csv(out_file, index=False)
    print(f"\nHasil disimpan ke: {out_file}")
    return df_res


if __name__ == "__main__":
    run_screener_v4()