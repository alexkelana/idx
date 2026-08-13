"""
IDX INTRADAY SCREENER — Confluence Ketat (Teknikal + Fundamental Ringan)
========================================================================
Tujuan: kandidat day trade (bukan swing).
- Universe likuid
- Confluence teknikal (trend mikro, volume, jarak ke level, volatilitas)
- Filter fundamental ringan (hindari PER ekstrem / rugi berat bila data ada)
- Target & SL dalam jangkauan 1 sesi (konservatif)
- Tick size + ARA/ARB BEI
- RR minimum realistis untuk intraday (default 1.5)

Disclaimer: alat bantu screening, bukan saran investasi.
Data harian yfinance = kandidat untuk eksekusi intraday (konfirmasi order book wajib).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime

try:
    from idx_liquidity_scanner import IdxLiquidityScanner
except ImportError:
    IdxLiquidityScanner = None

# =========================================================================
# PARAMETER INTRADAY
# =========================================================================
PARAMS = {
    "lookback_days": 60,
    "min_avg_value_rp": 15_000_000_000,  # lebih ketat dari swing
    "min_avg_volume": 2_000_000,
    "min_price": 100,                   # hindari tick terlalu kasar / noise
    "max_price": 10000,
    # Teknikal
    "max_dist_to_high_pct": 2.5,        # dekat resistance / breakout zone
    "max_range_20_pct": 12.0,           # jangan terlalu wild
    "min_vol_ratio": 1.3,               # volume 5d vs 20d
    "rsi_min": 45,
    "rsi_max": 68,                      # hindari chase overbought
    "min_score": 55,
    # Risk intraday (ketat)
    "sl_atr_mult": 0.6,                 # SL ~ 0.6 x ATR14
    "sl_max_pct": 1.8,                  # SL maks % dari entry
    "tp_atr_mult": 1.1,                 # TP ~ 1.1 x ATR
    "tp_max_pct": 3.0,                  # TP maks % (realistis 1 hari)
    "min_rr": 1.5,
    "account_size": 50_000_000,
    "risk_per_trade_pct": 0.5,          # lebih kecil dari swing
    "lot_size": 100,
    "top_n": 15,
    # Fundamental ringan (opsional; skip jika data kosong)
    "max_pe": 40,
    "min_pe": 0,
    "max_pb": 8,
    "require_fundamental": False,       # True = lebih ketat, hasil lebih sedikit
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
    if price < 500:
        return int(round(price / 2.0) * 2)
    if price < 2000:
        return int(round(price / 5.0) * 5)
    if price < 5000:
        return int(round(price / 10.0) * 10)
    return int(round(price / 25.0) * 25)


def apply_ara_arb_limits(price: float, prev_close: float, is_target: bool) -> float:
    """ARA/ARB BEI (pendekatan retail umum)."""
    if prev_close < 200:
        limit = 0.35
    elif prev_close <= 5000:
        limit = 0.25
    else:
        limit = 0.20
    ara = round_to_idx_tick(prev_close * (1 + limit))
    arb = round_to_idx_tick(prev_close * (1 - limit))
    return min(price, ara) if is_target else max(price, arb)


def compute_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(period).mean()
    avg_loss = loss.rolling(period).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def compute_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    h, l, c = df["High"], df["Low"], df["Close"]
    prev = c.shift(1)
    tr = pd.concat([h - l, (h - prev).abs(), (l - prev).abs()], axis=1).max(axis=1)
    return tr.rolling(period).mean()


def get_fundamental_light(symbol: str) -> dict:
    """PER/PBV/ROE ringan dari yfinance; kosongkan jika gagal."""
    out = {"PE": None, "PB": None, "ROE": None, "FundOK": True, "FundNote": "-"}
    try:
        info = yf.Ticker(symbol + ".JK").info or {}
        pe = info.get("trailingPE") or info.get("forwardPE")
        pb = info.get("priceToBook")
        roe = info.get("returnOnEquity")
        if pe is not None:
            out["PE"] = round(float(pe), 2)
        if pb is not None:
            out["PB"] = round(float(pb), 2)
        if roe is not None:
            out["ROE"] = round(float(roe) * 100, 2) if abs(float(roe)) <= 5 else round(float(roe), 2)

        notes = []
        ok = True
        if out["PE"] is not None:
            if out["PE"] < 0:
                ok = False
                notes.append("PE negatif")
            elif out["PE"] > PARAMS["max_pe"]:
                ok = False
                notes.append(f"PE tinggi ({out['PE']})")
            else:
                notes.append(f"PE {out['PE']}")
        if out["PB"] is not None and out["PB"] > PARAMS["max_pb"]:
            ok = False
            notes.append(f"PB tinggi ({out['PB']})")
        out["FundOK"] = ok
        out["FundNote"] = "; ".join(notes) if notes else "Data fund terbatas"
    except Exception:
        out["FundNote"] = "Fund N/A"
        out["FundOK"] = not PARAMS.get("require_fundamental", False)
    return out


# =========================================================================
# ANALISA SATU TICKER
# =========================================================================
def analyze_intraday(symbol: str, params: dict) -> dict | None:
    ticker = symbol + ".JK"
    try:
        df = yf.download(
            ticker,
            period=f"{int(params['lookback_days'])}d",
            interval="1d",
            progress=False,
            auto_adjust=True,
            multi_level_index=False,
        )
    except Exception:
        return None

    if df is None or df.empty or len(df) < 30:
        return None
    if isinstance(df.columns, pd.MultiIndex):
        try:
            df.columns = df.columns.get_level_values(0)
        except Exception:
            return None

    need = ["Open", "High", "Low", "Close", "Volume"]
    if any(c not in df.columns for c in need):
        return None
    df = df.dropna()
    if len(df) < 30:
        return None

    close = df["Close"]
    high = df["High"]
    low = df["Low"]
    vol = df["Volume"]
    last_close = float(close.iloc[-1])
    prev_close = float(close.iloc[-2])

    if last_close < params["min_price"] or last_close > params["max_price"]:
        return None

    # Likuiditas
    avg_val = float((close * vol).tail(20).mean())
    avg_vol = float(vol.tail(20).mean())
    if avg_val < params["min_avg_value_rp"] * 0.8:
        return None
    if avg_vol < params["min_avg_volume"] * 0.8:
        return None

    # Level & struktur
    high_20 = float(high.tail(20).max())
    low_20 = float(low.tail(20).min())
    range_20_pct = (high_20 - low_20) / last_close * 100
    dist_high = (high_20 - last_close) / last_close * 100
    dist_low = (last_close - low_20) / low_20 * 100

    if range_20_pct > params["max_range_20_pct"]:
        return None

    ma10 = float(close.rolling(10).mean().iloc[-1])
    ma20 = float(close.rolling(20).mean().iloc[-1])
    atr = float(compute_atr(df, 14).iloc[-1])
    if np.isnan(atr) or atr <= 0:
        return None
    atr_pct = atr / last_close * 100

    # Volatilitas harian harus cukup untuk day trade, tapi tidak ekstrem
    if atr_pct < 0.8 or atr_pct > 6.0:
        return None

    vol_ratio = float(vol.tail(5).mean()) / avg_vol if avg_vol > 0 else 0
    rsi = float(compute_rsi(close).iloc[-1])
    if np.isnan(rsi):
        rsi = 50.0

    last_open = float(df["Open"].iloc[-1])
    last_high = float(high.iloc[-1])
    last_low = float(low.iloc[-1])
    bullish_day = last_close > last_open

    # --- Confluence score ---
    score = 0
    reasons = []

    # Trend mikro
    if last_close > ma10 > ma20:
        score += 20
        reasons.append("Trend mikro naik (C>MA10>MA20)")
    elif last_close > ma10:
        score += 10
        reasons.append("Di atas MA10")

    # Dekat high / breakout zone (momentum day trade)
    if dist_high <= params["max_dist_to_high_pct"]:
        score += 20
        reasons.append(f"Dekat high20 ({dist_high:.1f}%)")
    elif dist_low <= 2.0 and last_close > ma20:
        score += 12
        reasons.append("Bounce zona low + di atas MA20")

    # Volume
    if vol_ratio >= params["min_vol_ratio"]:
        score += 20
        reasons.append(f"Volume {vol_ratio:.2f}x")
    elif vol_ratio >= 1.1:
        score += 8
        reasons.append(f"Volume agak naik {vol_ratio:.2f}x")

    # RSI
    if params["rsi_min"] <= rsi <= params["rsi_max"]:
        score += 15
        reasons.append(f"RSI {rsi:.0f}")
    elif rsi > 72:
        score -= 10
        reasons.append(f"RSI overbought {rsi:.0f}")

    # Candle / struktur hari terakhir
    if bullish_day and last_close >= (last_low + 0.6 * (last_high - last_low)):
        score += 10
        reasons.append("Close kuat di half atas range")

    # ATR sehat untuk intraday
    if 1.2 <= atr_pct <= 4.0:
        score += 10
        reasons.append(f"ATR% {atr_pct:.1f} cocok intraday")

    if score < params["min_score"]:
        return None

    # Fundamental ringan
    fund = get_fundamental_light(symbol)
    if params.get("require_fundamental") and not fund["FundOK"]:
        return None
    if fund["FundOK"] and fund["PE"] is not None:
        score += 5
        reasons.append(fund["FundNote"])
    elif not fund["FundOK"]:
        reasons.append("Fund lemah: " + fund["FundNote"])

    # --- Plan intraday (ketat) ---
    entry = round_to_idx_tick(last_close)

    sl_atr = entry - params["sl_atr_mult"] * atr
    sl_pct = entry * (1 - params["sl_max_pct"] / 100)
    stop_raw = max(sl_atr, sl_pct)  # lebih ketat (lebih dekat ke entry)
    # pastikan di bawah entry
    if stop_raw >= entry:
        stop_raw = entry * (1 - 0.01)
    stop_loss = apply_ara_arb_limits(round_to_idx_tick(stop_raw), prev_close, is_target=False)

    risk = entry - stop_loss
    if risk <= 0:
        return None
    risk_pct = risk / entry * 100
    if risk_pct > params["sl_max_pct"] * 1.15:
        return None

    tp_atr = entry + params["tp_atr_mult"] * atr
    tp_pct = entry * (1 + params["tp_max_pct"] / 100)
    # target jangan tembus high20 terlalu jauh dalam 1 hari tanpa alasan
    tp_cap = min(tp_atr, tp_pct, high_20 * 1.01 if high_20 > entry else tp_pct)
    target = apply_ara_arb_limits(round_to_idx_tick(tp_cap), prev_close, is_target=True)

    reward = target - entry
    if reward <= 0:
        return None
    rr = reward / risk
    if rr < params["min_rr"]:
        return None

    # Pastikan target masih "intraday-able": reward % tidak gila
    reward_pct = reward / entry * 100
    if reward_pct > params["tp_max_pct"] * 1.2:
        return None

    risk_rp = params["account_size"] * params["risk_per_trade_pct"] / 100
    lots = int((risk_rp / risk) // params["lot_size"]) if risk > 0 else 0
    shares = lots * params["lot_size"]

    return {
        "Ticker": symbol,
        "Close": entry,
        "Score": score,
        "Alasan": "; ".join(reasons),
        "RSI": round(rsi, 1),
        "VolRatio": round(vol_ratio, 2),
        "ATR": round(atr, 0),
        "ATR_pct": round(atr_pct, 2),
        "DistHigh20%": round(dist_high, 2),
        "Entry": entry,
        "StopLoss": stop_loss,
        "Target": target,
        "RiskPct": round(risk_pct, 2),
        "RewardPct": round(reward_pct, 2),
        "RR_Ratio": round(rr, 2),
        "Lots": lots,
        "EstLoss(Rp)": round(shares * risk, 0),
        "EstProfit(Rp)": round(shares * reward, 0),
        "PE": fund.get("PE"),
        "PB": fund.get("PB"),
        "FundNote": fund.get("FundNote"),
        "Strategy": "Intraday Confluence",
        "Horizon": "1 sesi (intraday)",
    }


# =========================================================================
# RUNNER
# =========================================================================
def get_universe(params: dict) -> list:
    if IdxLiquidityScanner is None:
        return ["BBCA", "BBRI", "BMRI", "BBNI", "TLKM", "ASII", "UNVR", "ICBP", "GOTO", "AMMN"]
    print("=" * 60)
    print("PRE-SCREENER LIKUIDITAS (INTRADAY)")
    print("=" * 60)
    scanner = IdxLiquidityScanner(
        min_avg_value_rp=params["min_avg_value_rp"],
        min_avg_volume=params["min_avg_volume"],
        lookback_days=20,
        max_workers=15,
    )
    return scanner.get_liquid_universe()


def run_intraday_screener(user_params: dict | None = None):
    params = PARAMS.copy()
    if user_params:
        params.update(user_params)

    universe = get_universe(params)
    print(f"\nAnalisa intraday confluence untuk {len(universe)} saham...\n")

    rows = []
    for i, sym in enumerate(universe, 1):
        print(f"  [{i}/{len(universe)}] {sym}...", end="\r")
        try:
            r = analyze_intraday(sym, params)
            if r:
                rows.append(r)
        except Exception:
            continue
    print(" " * 50, end="\r")

    if not rows:
        print("Tidak ada kandidat intraday yang lolos confluence hari ini.")
        return pd.DataFrame()

    df = pd.DataFrame(rows).sort_values(["Score", "RR_Ratio"], ascending=[False, False])
    df = df.head(params["top_n"]).reset_index(drop=True)

    cols = [
        "Ticker", "Close", "Score", "RSI", "VolRatio", "ATR_pct",
        "Entry", "StopLoss", "Target", "RiskPct", "RewardPct", "RR_Ratio", "Lots", "Alasan",
    ]
    cols = [c for c in cols if c in df.columns]

    print("=" * 110)
    print(f"IDX INTRADAY SCREENER — {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print(f"SL maks ~{params['sl_max_pct']}% | TP maks ~{params['tp_max_pct']}% | Min RR {params['min_rr']}")
    print("=" * 110)
    print(df[cols].to_string(index=False))
    print("=" * 110)
    print("Catatan: Konfirmasi di order book / pre-open. Target dirancang untuk 1 sesi, bukan swing.")

    try:
        from idx_report_schema import save_version_report
        path = save_version_report(df, "intraday")
    except Exception:
        path = f"idx_report_intraday_{datetime.now().strftime('%Y-%m-%d')}.csv"
        df.to_csv(path, index=False)
    print(f"Disimpan: {path}")
    return df


if __name__ == "__main__":
    run_intraday_screener()