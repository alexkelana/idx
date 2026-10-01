"""
IDX SMC SCREENER V5 — CHOCH & DISCOUNT ZONE
============================================================
+ Filter sweep low (spring) di dekat lowest_low / zona support
  sebelum entry di discount

Karakter IDX:
- CHOCH relatif fresh
- Discount praktis (38.2–78.6)
- RR min 1.5
- Tick + ARA/ARB
- SL di bawah invalidation (swing low)
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime
from idx_liquidity_scanner import IdxLiquidityScanner

# Parameter sweep low (bisa dioverride lewat user_params)
SWEEP_PARAMS = {
    "require_sweep_low": False,   # True = wajib ada sweep low
    "sweep_lookback": 15,
    "sweep_min_wick_pct": 0.30,   # minimal penetrasi di bawah level (%)
    "sweep_max_close_above_pct": 0.5,  # close harus reclaim di atas level
}


def detect_liquidity_sweep_low(
    df: pd.DataFrame,
    support: float,
    lookback: int = 15,
    min_wick_pct: float = 0.30,
) -> dict:
    """
    Sweep low = Low menusuk di bawah support, Close reclaim di atas support.
    (stop-hunt / spring) dalam lookback bar, sebelum bar terakhir.
    """
    if support <= 0 or len(df) < lookback + 2:
        return {"has_sweep": False, "sweep_low": 0.0, "wick_pct": 0.0}

    start = max(0, len(df) - lookback - 1)
    end = len(df) - 1
    for i in range(end - 1, start - 1, -1):
        lo = float(df["Low"].iloc[i])
        cl = float(df["Close"].iloc[i])
        # menusuk bawah + close kembali di atas / sangat dekat support
        if lo < support * 0.999 and cl >= support * 0.998:
            wick_pct = (min(cl, support) - lo) / support * 100
            if wick_pct >= min_wick_pct or lo < support * 0.997:
                return {
                    "has_sweep": True,
                    "sweep_low": lo,
                    "sweep_close": cl,
                    "wick_pct": round(wick_pct, 2),
                    "bars_ago": end - i,
                }
    return {"has_sweep": False, "sweep_low": 0.0, "wick_pct": 0.0}


def detect_choch_and_discount(df: pd.DataFrame, lookback: int = 60) -> dict | None:
    if len(df) < 20:
        return None

    start_pos = max(0, len(df) - lookback)
    df_sub = df.iloc[start_pos:].copy()
    if len(df_sub) < 15:
        return None

    high = df_sub["High"].values
    low = df_sub["Low"].values
    close = df_sub["Close"].values

    swing_highs = []
    swing_lows = []
    for i in range(2, len(df_sub) - 2):
        if (
            high[i] > high[i - 1]
            and high[i] > high[i - 2]
            and high[i] > high[i + 1]
            and high[i] > high[i + 2]
        ):
            swing_highs.append((i, float(high[i])))
        if (
            low[i] < low[i - 1]
            and low[i] < low[i - 2]
            and low[i] < low[i + 1]
            and low[i] < low[i + 2]
        ):
            swing_lows.append((i, float(low[i])))

    if len(swing_highs) < 2 or len(swing_lows) < 2:
        return None

    sh1, sh2 = swing_highs[-2], swing_highs[-1]
    sl1, sl2 = swing_lows[-2], swing_lows[-1]
    has_lower_high = sh2[1] < sh1[1]
    has_lower_low = sl2[1] <= sl1[1]
    if not (has_lower_high or has_lower_low):
        return None

    last_sh_idx, last_sh_val = sh2
    choch_rel = -1
    for i in range(last_sh_idx + 1, len(df_sub)):
        if close[i] > last_sh_val:
            choch_rel = i
            break

    if choch_rel == -1:
        return None

    age_bars = len(df_sub) - 1 - choch_rel
    if age_bars > 20:
        return None

    lowest_low = float(df_sub["Low"].iloc[:choch_rel].min())
    peak_after = float(df_sub["High"].iloc[choch_rel:].max())
    range_up = peak_after - lowest_low
    if range_up <= 0:
        return None

    last_close_sub = float(close[-1])
    if range_up / last_close_sub < 0.04:
        return None

    fibo_382 = peak_after - range_up * 0.382
    fibo_50 = peak_after - range_up * 0.50
    fibo_618 = peak_after - range_up * 0.618
    fibo_786 = peak_after - range_up * 0.786

    return {
        "choch_idx": start_pos + choch_rel,
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


def analyze_smc_v5_ticker(symbol: str, user_params: dict = None) -> dict | None:
    ticker = symbol + ".JK"

    account_size = 50_000_000
    risk_pct = 1.0
    sw = SWEEP_PARAMS.copy()
    if user_params:
        account_size = user_params.get("account_size", account_size)
        risk_pct = user_params.get("risk_per_trade_pct", risk_pct)
        for k in SWEEP_PARAMS:
            if k in user_params:
                sw[k] = user_params[k]

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

    choch = detect_choch_and_discount(df, lookback=60)
    if not choch:
        return None

    in_discount_zone = (
        choch["fibo_786"] <= last_close <= choch["fibo_382"]
        and last_close >= choch["lowest_low"] * 0.995
    )
    if not in_discount_zone:
        return None

    deep_discount = last_close <= choch["fibo_50"]

    # --- Sweep low di support (lowest_low / fibo dalam) ---
    support_level = float(choch["lowest_low"])
    # juga coba level Fibo 78.6 sebagai support sekunder
    sweep = detect_liquidity_sweep_low(
        df,
        support=support_level,
        lookback=int(sw["sweep_lookback"]),
        min_wick_pct=float(sw["sweep_min_wick_pct"]),
    )
    if not sweep["has_sweep"]:
        # fallback: sweep di fibo_786
        sweep_f = detect_liquidity_sweep_low(
            df,
            support=float(choch["fibo_786"]),
            lookback=int(sw["sweep_lookback"]),
            min_wick_pct=float(sw["sweep_min_wick_pct"]),
        )
        if sweep_f["has_sweep"]:
            sweep = sweep_f
            sweep["level_note"] = "Fibo786"
        else:
            sweep["level_note"] = "LowestLow"
    else:
        sweep["level_note"] = "LowestLow"

    if sw.get("require_sweep_low", False) and not sweep["has_sweep"]:
        return None

    entry = round_to_idx_tick(last_close)

    stop_loss_raw = choch["lowest_low"] * 0.985
    # jika ada sweep, SL sedikit di bawah sweep low
    if sweep["has_sweep"] and sweep.get("sweep_low", 0) > 0:
        stop_loss_raw = min(stop_loss_raw, float(sweep["sweep_low"]) * 0.99)

    stop_loss = apply_ara_arb_limits(
        round_to_idx_tick(stop_loss_raw), last_close, is_target=False
    )

    risk_per_share = entry - stop_loss
    if risk_per_share <= 0:
        return None

    target_1 = apply_ara_arb_limits(
        round_to_idx_tick(choch["peak"]), last_close, is_target=True
    )

    rr_ratio = (target_1 - entry) / risk_per_share if risk_per_share > 0 else 0
    if rr_ratio < 1.5:
        return None

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

    if sweep["has_sweep"]:
        score += 15
        reasons.append(
            f"Sweep low ({sweep.get('level_note', '')} wick "
            f"{sweep.get('wick_pct', 0)}%, {sweep.get('bars_ago', '?')} bar lalu)"
        )

    # MA50/200 modifier — profil soft (mean-reversion CHOCH)
    mx = {
        "delta": 0, "signal": "NONE", "age": None, "note": "",
        "ma50": None, "ma200": None, "spread_pct": None, "structural": "FLAT",
    }
    try:
        from idx_ma_cross_screener import apply_ma_cross_score
        score, mx = apply_ma_cross_score(score, df, profile="soft", reasons=reasons)
    except Exception:
        pass

    risk_rp = account_size * (risk_pct / 100.0)
    shares = int(risk_rp / risk_per_share) if risk_per_share > 0 else 0
    lots = shares // 100
    actual_shares = lots * 100

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
        "MA_Cross": mx.get("signal"),
        "MA_CrossAge": mx.get("age"),
        "MA_CrossDelta": mx.get("delta"),
        "MA200": mx.get("ma200"),
        "MA_SpreadPct": mx.get("spread_pct"),
        "SweepLow": "Ya" if sweep["has_sweep"] else "Tidak",
        "SweepLowPrice": round_to_idx_tick(sweep.get("sweep_low", 0)),
        "SweepWickPct": sweep.get("wick_pct", 0),
        "Alasan": "; ".join(reasons),
        "Lots": lots,
        "EstLoss(Rp)": round(actual_shares * risk_per_share, 0),
        "EstProfit(Rp)": round(actual_shares * (target_1 - entry), 0),
        "Strategy": "V5 (SMC CHOCH)",
    }


def run_screener_v5(user_params: dict = None, universe=None):
    """
    universe: opsional — daftar ticker dari master/confluence (hindari scan ulang).
    """
    if universe:
        print(
            f"Memakai shared universe ({len(universe)} ticker) — skip scan likuiditas V5."
        )
    else:
        print("Mempersiapkan Universe Likuiditas (SMC V5)...")

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

    print(
        f"\nMenjalankan Screener SMC V5 (CHOCH + Sweep Low) "
        f"untuk {len(universe)} saham..."
    )
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
        "Ticker", "Close", "CHOCH_Level", "CHOCH_Age", "SweepLow", "Entry",
        "StopLoss", "Target(Peak)", "RR_Ratio", "Score", "Lots", "Alasan",
    ]
    cols = [c for c in cols if c in df_res.columns]
    print(df_res[cols].to_string(index=False))
    print("=" * 100)

    df_res["Strategy"] = "V5 (SMC CHOCH)"
    try:
        from idx_report_schema import save_version_report
        out_file = save_version_report(df_res, "v5")
    except ImportError:
        out_file = f"idx_report_v5_{datetime.now().strftime('%Y-%m-%d')}.csv"
        df_res.to_csv(out_file, index=False)
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
    run_screener_v5()
