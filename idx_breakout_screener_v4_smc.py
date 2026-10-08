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

    # ATR dulu (untuk hybrid entry + SL)
    high_s = df["High"]
    low_s = df["Low"]
    close_s = df["Close"]
    prev_close = close_s.shift(1)
    tr = pd.concat(
        [
            (high_s - low_s),
            (high_s - prev_close).abs(),
            (low_s - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    atr = float(tr.tail(14).mean()) if len(tr) >= 14 else float(tr.mean())
    if not atr or atr != atr or atr <= 0:
        atr = float(last_close) * 0.02

    # --- Entry struktural hybrid (V4) ---
    # Di SMC V4, Order Block = candle bear pra-BOS → EntryStruct = OB_Top
    entry_close = float(round_to_idx_tick(last_close))
    entry_struct = float(round_to_idx_tick(ob_top))
    market_max_dist_pct = float((user_params or {}).get("market_max_dist_pct", 1.2))
    market_max_dist_atr = float((user_params or {}).get("market_max_dist_atr", 0.75))
    limit_expiry_bars = int((user_params or {}).get("limit_expiry_bars", 5))
    use_struct = (user_params or {}).get("use_structural_entry", True)

    dist_struct_pct = abs(entry_close - entry_struct) / entry_close * 100 if entry_close else 0.0
    dist_struct_atr = abs(entry_close - entry_struct) / atr if atr > 0 else 999.0
    inside_ob = float(ob_bottom) <= last_close <= float(ob_top) * 1.002

    if not use_struct:
        entry = entry_close
        entry_mode = "FALLBACK"
        entry_note = "Structural entry off — pakai close"
    elif inside_ob or dist_struct_pct <= market_max_dist_pct or dist_struct_atr <= market_max_dist_atr:
        entry = entry_close
        entry_mode = "MARKET"
        entry_note = (
            f"MARKET @ close (zona OB / dekat OB_Top {entry_struct}, Δ{dist_struct_pct:.1f}%)"
        )
    else:
        # Close di atas OB → LIMIT tunggu retest OB_Top (mitigasi ulang)
        entry = entry_struct
        entry_mode = "LIMIT"
        entry_note = (
            f"LIMIT @ OB_Top {entry_struct} (bear pra-BOS); "
            f"expiry {limit_expiry_bars} bar; close {entry_close:.0f} Δ{dist_struct_pct:.1f}%"
        )

    entry = float(round_to_idx_tick(entry))
    if entry <= 0:
        return None

    # SL struktural di bawah OB bottom
    stop_struct = float(ob_bottom) * 0.985
    # Lantai napas: min(1.2% entry, 0.85×ATR) — mana yang lebih dalam
    min_risk = max(entry * 0.012, 0.85 * atr)
    stop_floor = entry - min_risk
    # Plafon risk 8% (selaras exchange / V3 spirit)
    max_risk = entry * 0.08
    stop_cap = entry - max_risk

    stop_loss_raw = min(stop_struct, stop_floor)  # lebih dalam = lebih kecil angka
    sl_source = "structure"
    if stop_struct > stop_floor:
        # struktur terlalu dangkal → pakai lantai ATR/%
        stop_loss_raw = stop_floor
        sl_source = "ATR_floor"
    if stop_loss_raw < stop_cap:
        stop_loss_raw = stop_cap
        sl_source = sl_source + "+cap"

    stop_loss = round_to_idx_tick(stop_loss_raw)
    stop_loss = apply_ara_arb_limits(stop_loss, last_close, is_target=False)

    risk_per_share = entry - stop_loss
    if risk_per_share <= 0:
        return None
    # Tolak jika risk masih < 0.7×ATR setelah semua clamp (noise)
    if atr > 0 and risk_per_share < 0.7 * atr:
        return None

    # [FIX] Target sekarang akurat karena bos_idx absolut
    peak_after_bos = df["High"].iloc[latest_ob["bos_idx"]:].max()
    target_1 = apply_ara_arb_limits(
        round_to_idx_tick(peak_after_bos), last_close, is_target=True
    )
    if target_1 <= entry:
        return None

    rr_ratio = (target_1 - entry) / risk_per_share if risk_per_share > 0 else 0
    if rr_ratio < 1.5:
        return None
    # RR ekstrem biasanya artefak target jauh + SL residual — cap informasi
    if rr_ratio > 6.0:
        # masih lolos, tapi tandai di output; jangan hard-reject
        pass

    # Position Sizing
    risk_rp = account_size * (risk_pct / 100.0)
    shares = int(risk_rp / risk_per_share) if risk_per_share > 0 else 0
    lots = shares // 100
    actual_shares = lots * 100
    est_loss = actual_shares * risk_per_share
    est_profit = actual_shares * (target_1 - entry)

    risk_atr = round(risk_per_share / atr, 2) if atr > 0 else 0.0
    risk_pct_out = round(risk_per_share / entry * 100, 2) if entry > 0 else 0.0

    # Base score V4 + MA cross trend profile
    score = 50
    reasons = ["OB mitigasi + BOS", entry_note]
    if entry_mode == "LIMIT":
        score += 3  # disiplin tidak chase
        reasons.append("Entry LIMIT struktural")
    elif entry_mode == "MARKET" and inside_ob:
        score += 5
        reasons.append("Harga di dalam OB")
    if rr_ratio >= 2.5:
        score += 15
        reasons.append(f"RR kuat ({rr_ratio:.1f})")
    elif rr_ratio >= 1.8:
        score += 8
        reasons.append(f"RR OK ({rr_ratio:.1f})")
    if dist_to_ob_pct <= 1.0:
        score += 10
        reasons.append("Sangat dekat OB")
    elif dist_to_ob_pct <= 2.0:
        score += 5
        reasons.append("Dekat OB")
    mx = {
        "delta": 0, "signal": "NONE", "age": None, "note": "",
        "ma50": None, "ma200": None, "spread_pct": None, "structural": "FLAT",
    }
    try:
        from idx_ma_cross_screener import apply_ma_cross_score
        score, mx = apply_ma_cross_score(score, df, profile="trend", reasons=reasons)
    except Exception:
        pass
    score = int(max(0, min(100, score)))

    return {
        "Ticker": symbol,
        "Close": round_to_idx_tick(last_close),
        "OB_Top": round_to_idx_tick(ob_top),
        "OB_Bottom": round_to_idx_tick(ob_bottom),
        "Entry": entry,
        "EntryClose": entry_close,
        "EntryStruct": entry_struct,
        "EntryMode": entry_mode,
        "EntryStructNote": entry_note,
        "EntryExpiryBars": limit_expiry_bars,
        "DistEntryStructPct": round(dist_struct_pct, 2),
        "StopLoss": stop_loss,
        "SL_Source": sl_source,
        "Target(Liquidity)": target_1,
        "RR_Ratio": round(rr_ratio, 2),
        "Score": score,
        "Alasan": "; ".join(reasons),
        "MA_Cross": mx.get("signal"),
        "MA_CrossAge": mx.get("age"),
        "MA_CrossDelta": mx.get("delta"),
        "MA200": mx.get("ma200"),
        "MA_SpreadPct": mx.get("spread_pct"),
        "ATR": round(atr, 1),
        "RiskATR": risk_atr,
        "RiskPct": risk_pct_out,
        "Lots": lots,
        "EstLoss(Rp)": round(est_loss, 0),
        "EstProfit(Rp)": round(est_profit, 0),
        "Strategy": "V4 (SMC OB)",
    }

def run_screener_v4(user_params: dict = None, universe=None):
    """
    universe: opsional — daftar ticker dari master/confluence (hindari scan ulang).
    """
    if universe:
        print(
            f"Memakai shared universe ({len(universe)} ticker) — skip scan likuiditas V4."
        )
    else:
        print("Mempersiapkan Universe Likuiditas (SMC V4)...")

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

    df_res["Strategy"] = "V4 (SMC Order Block)"
    from idx_report_schema import save_version_report
    out_file = save_version_report(df_res, "v4")
    print(f"Hasil disimpan ke: {out_file}")
    return df_res

# --- Exchange rules (tick + ARA/ARB BEI Sep 2026) ---
try:
    from idx_exchange_rules import (
        round_to_idx_tick,
        apply_ara_arb_limits,
        get_ara_arb_limits,
        DEFAULT_SCREENER_MIN_PRICE,
        describe_rules,
    )
except ImportError:
    # Fallback minimal jika modul belum ter-deploy
    def round_to_idx_tick(price: float) -> int:
        if price is None or price <= 0:
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
        pc = float(prev_close or 0)
        if pc <= 0:
            return float(round_to_idx_tick(price))
        if pc <= 10:
            ara, arb = pc + 1, max(1.0, pc - 1)
        elif pc <= 200:
            ara, arb = pc * 1.35, pc * 0.85
        elif pc <= 5000:
            ara, arb = pc * 1.25, pc * 0.85
        else:
            ara, arb = pc * 1.20, pc * 0.85
        ara = float(round_to_idx_tick(ara))
        arb = float(round_to_idx_tick(arb))
        p = float(round_to_idx_tick(price))
        return min(p, ara) if is_target else max(p, arb)

    def get_ara_arb_limits(prev_close, asof=None):
        pc = float(prev_close or 0)
        if pc <= 10:
            return pc + 1, max(1.0, pc - 1), "fallback_1_10"
        if pc <= 200:
            return pc * 1.35, pc * 0.85, "fallback"
        if pc <= 5000:
            return pc * 1.25, pc * 0.85, "fallback"
        return pc * 1.20, pc * 0.85, "fallback"

    DEFAULT_SCREENER_MIN_PRICE = 50.0

    def describe_rules(asof=None):
        return "fallback local ARA/ARB"

if __name__ == "__main__":
    run_screener_v4()