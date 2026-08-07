"""
IDX SMC SCREENER V5 — CHOCH & DISCOUNT ZONE
============================================================
Mencari Change of Character (CHOCH) bullish setelah tekanan jual,
lalu entry di zona discount yang masih masuk akal untuk market IDX.

Karakter pasar Indonesia yang dipertahankan:
- Prioritas setup relatif fresh (CHOCH tidak terlalu tua)
- Validasi struktur turun sederhana (lower high / lower low)
- Zona discount praktis (bukan hanya ekstrem)
- RR minimum realistis (1.5), bukan 2.0 kaku
- Tick size + ARA/ARB BEI
- SL di bawah invalidation (swing low)
"""

from __future__ import annotations

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
# DETEKSI CHOCH + DISCOUNT
# =========================================================================
def detect_choch_and_discount(df: pd.DataFrame, lookback: int = 60) -> dict | None:
    """
    Deteksi CHOCH bullish setelah struktur turun sederhana.
    Index disimpan absolut agar aman dipakai di full df.
    """
    if len(df) < 20:
        return None

    start_pos = max(0, len(df) - lookback)
    df_sub = df.iloc[start_pos:].copy()
    if len(df_sub) < 15:
        return None

    high = df_sub["High"].values
    low = df_sub["Low"].values
    close = df_sub["Close"].values

    # 1. Swing points (fractal 5 bar)
    swing_highs = []
    swing_lows = []
    for i in range(2, len(df_sub) - 2):
        if high[i] > high[i - 1] and high[i] > high[i - 2] and high[i] > high[i + 1] and high[i] > high[i + 2]:
            swing_highs.append((i, float(high[i])))
        if low[i] < low[i - 1] and low[i] < low[i - 2] and low[i] < low[i + 1] and low[i] < low[i + 2]:
            swing_lows.append((i, float(low[i])))

    if len(swing_highs) < 2 or len(swing_lows) < 2:
        return None

    # 2. Struktur turun sederhana (karakter bearish sebelum CHOCH)
    #    Minimal: swing high terakhir < swing high sebelumnya
    #    ATAU swing low terakhir <= swing low sebelumnya (LL / non-HH)
    sh1, sh2 = swing_highs[-2], swing_highs[-1]
    sl1, sl2 = swing_lows[-2], swing_lows[-1]
    has_lower_high = sh2[1] < sh1[1]
    has_lower_low = sl2[1] <= sl1[1]
    if not (has_lower_high or has_lower_low):
        return None  # bukan tekanan jual yang jelas

    # 3. CHOCH: close menembus swing high terakhir (setelah struktur turun)
    last_sh_idx, last_sh_val = sh2
    choch_rel = -1
    for i in range(last_sh_idx + 1, len(df_sub)):
        if close[i] > last_sh_val:
            choch_rel = i
            break

    if choch_rel == -1:
        return None

    # Freshness: CHOCH tidak lebih tua dari 20 bar (karakter IDX — setup cepat basi)
    age_bars = len(df_sub) - 1 - choch_rel
    if age_bars > 20:
        return None

    # 4. Range & Fibo
    lowest_low = float(df_sub["Low"].iloc[:choch_rel].min())
    peak_after = float(df_sub["High"].iloc[choch_rel:].max())
    range_up = peak_after - lowest_low
    if range_up <= 0:
        return None

    # Hindari range terlalu kecil (noise)
    last_close_sub = float(close[-1])
    if range_up / last_close_sub < 0.04:  # range < 4%
        return None

    fibo_382 = peak_after - range_up * 0.382
    fibo_50 = peak_after - range_up * 0.50
    fibo_618 = peak_after - range_up * 0.618
    fibo_786 = peak_after - range_up * 0.786

    return {
        "choch_idx": start_pos + choch_rel,  # absolut
        "choch_val": last_sh_val,
        "choch_age": age_bars,
        "lowest_low": lowest_low,
        "peak": peak_after,
        "fibo_382": fibo_382,
        "fibo_50": fibo_50,
        "fibo_618": fibo_618,
        "fibo_786": fibo_786,
        "has_lower_high": has_lower_high,
        "has_lower_low": has_lower_low,
    }


# =========================================================================
# ANALISA TICKER
# =========================================================================
def analyze_smc_v5_ticker(symbol: str, user_params: dict = None) -> dict | None:
    ticker = symbol + ".JK"

    account_size = 50_000_000
    risk_pct = 1.0
    if user_params:
        account_size = user_params.get("account_size", 50_000_000)
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

    choch = detect_choch_and_discount(df, lookback=60)
    if not choch:
        return None

    # ------------------------------------------------------------------
    # ZONA ENTRY — karakter IDX
    # Discount praktis: antara Fibo 38.2% dan 78.6%, di atas lowest low
    # Prioritas lebih tinggi jika di bawah Fibo 50% (discount lebih dalam)
    # ------------------------------------------------------------------
    in_discount_zone = (
        choch["fibo_786"] <= last_close <= choch["fibo_382"]
        and last_close >= choch["lowest_low"] * 0.995
    )
    if not in_discount_zone:
        return None

    deep_discount = last_close <= choch["fibo_50"]

    # Optional: harga sudah menyentuh zona (mitigation-ish)
    # tidak wajib hard reject jika close di zona

    entry = round_to_idx_tick(last_close)

    # SL di bawah swing low (invalidation) + buffer tipis
    stop_loss_raw = choch["lowest_low"] * 0.985
    stop_loss = apply_ara_arb_limits(
        round_to_idx_tick(stop_loss_raw), last_close, is_target=False
    )

    risk_per_share = entry - stop_loss
    if risk_per_share <= 0:
        return None

    # Target: peak pasca CHOCH (liquidity / premium)
    target_1 = apply_ara_arb_limits(
        round_to_idx_tick(choch["peak"]), last_close, is_target=True
    )

    rr_ratio = (target_1 - entry) / risk_per_share if risk_per_share > 0 else 0
    # [FIX] RR min 1.5 — realistis untuk IDX (bukan 2.0 kaku)
    if rr_ratio < 1.5:
        return None

    # Score sederhana (fresh + deep discount + struktur)
    score = 40
    reasons = ["CHOCH Bullish"]
    if choch["choch_age"] <= 8:
        score += 20
        reasons.append(f"Fresh CHOCH ({choch['choch_age']} bar)")
    elif choch["choch_age"] <= 15:
        score += 10
        reasons.append(f"CHOCH {choch['choch_age']} bar")
    if deep_discount:
        score += 20
        reasons.append("Deep Discount (≤50%)")
    else:
        score += 8
        reasons.append("Discount zone (38-50%)")
    if choch["has_lower_high"] and choch["has_lower_low"]:
        score += 15
        reasons.append("LH+LL sebelum CHOCH")
    elif choch["has_lower_high"] or choch["has_lower_low"]:
        score += 8
        reasons.append("Struktur turun")

    # Position sizing
    risk_rp = account_size * (risk_pct / 100.0)
    shares = int(risk_rp / risk_per_share) if risk_per_share > 0 else 0
    lots = shares // 100
    actual_shares = lots * 100
    est_loss = actual_shares * risk_per_share
    est_profit = actual_shares * (target_1 - entry)

    return {
        "Ticker": symbol,
        "Close": round_to_idx_tick(last_close),
        "CHOCH_Level": round_to_idx_tick(choch["choch_val"]),
        "CHOCH_Age": choch["choch_age"],
        "Fibo_50": round_to_idx_tick(choch["fibo_50"]),
        "Fibo_618": round_to_idx_tick(choch["fibo_618"]),
        "Fibo_786": round_to_idx_tick(choch["fibo_786"]),
        "Entry": entry,
        "StopLoss": stop_loss,
        "Target(Peak)": target_1,
        "RR_Ratio": round(rr_ratio, 2),
        "Score": score,
        "Alasan": "; ".join(reasons),
        "Lots": lots,
        "EstLoss(Rp)": round(est_loss, 0),
        "EstProfit(Rp)": round(est_profit, 0),
        "Strategy": "V5 (SMC CHOCH)",
    }


# =========================================================================
# RUNNER
# =========================================================================
def run_screener_v5(user_params: dict = None):
    print("Mempersiapkan Universe Likuiditas (SMC V5)...")
    scanner = IdxLiquidityScanner(
        min_avg_value_rp=10_000_000_000,
        min_avg_volume=1_000_000,
    )
    universe = scanner.get_liquid_universe()

    print(f"\nMenjalankan Screener SMC V5 (CHOCH) untuk {len(universe)} saham...")
    results = []
    for i, sym in enumerate(universe, 1):
        print(f"  [{i}/{len(universe)}] Cek {sym}...", end="\r")
        try:
            res = analyze_smc_v5_ticker(sym, user_params)
            if res:
                results.append(res)
        except Exception:
            continue

    print(" " * 60, end="\r")

    if not results:
        print("Tidak ada saham di zona Discount pasca CHOCH yang valid hari ini.")
        return None

    df_res = pd.DataFrame(results).sort_values(
        ["Score", "RR_Ratio"], ascending=[False, False]
    )

    print("\n" + "=" * 100)
    print(f"SMC CHOCH & DISCOUNT V5 — {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print("=" * 100)
    cols = [
        "Ticker", "Close", "CHOCH_Level", "CHOCH_Age", "Entry", "StopLoss",
        "Target(Peak)", "RR_Ratio", "Score", "Lots", "Alasan",
    ]
    cols = [c for c in cols if c in df_res.columns]
    print(df_res[cols].to_string(index=False))
    print("=" * 100)

    df_res["Strategy"] = "V4 (SMC Order Block)"
    from idx_report_schema import save_version_report
    out_file = save_version_report(df_res, "v4")
    print(f"Hasil disimpan ke: {out_file}")
    return df_res


if __name__ == "__main__":
    run_screener_v5()