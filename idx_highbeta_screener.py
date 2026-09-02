"""
IDX HIGH-BETA / SPECULATIVE LIQUID SCREENER
============================================================
Untuk saham berkarakter mirip BUMI:
- Nilai transaksi sangat besar (bukan sekadar volume lembar)
- Volatilitas & range lebar (lawan dari konsolidasi ketat)
- Partisipasi tinggi (volume ratio)
- Momentum mikro OK, tanpa menuntut base rapi / OB / CHOCH

Cocok: diskresi intraday ~ swing sangat pendek
Tidak cocok: investor yang ingin setup tenang & RR struktural murni

Disclaimer: alat bantu screening, bukan rekomendasi.
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

PARAMS = {
    # Likuiditas — lebih ketat dari V2 (karakter "board aktif")
    "min_avg_value_rp": 30_000_000_000,   # 30M+/hari
    "min_avg_volume": 5_000_000,
    "lookback_days": 60,
    # Karakter high-beta
    "min_range_20_pct": 10.0,             # range 20D minimal (bukan max)
    "max_range_20_pct": 45.0,             # buang yang terlalu gila / data aneh
    "min_atr_pct": 2.5,                   # ATR/Close minimal
    "max_atr_pct": 12.0,
    "min_vol_ratio": 1.15,                # 5d vs 20d
    "min_price": 100,
    "max_price": 5000,
    # Momentum mikro (longgar)
    "rsi_min": 35,
    "rsi_max": 75,
    "require_above_ma10": False,          # True = hanya yang masih panas short-term
    # Risk — size kecil, SL ATR
    "sl_atr_mult": 1.2,
    "tp_atr_mult": 1.8,
    "min_rr": 1.3,
    "account_size": 4_000_000,
    "risk_per_trade_pct": 0.5,            # lebih kecil: saham liar
    "lot_size": 100,
    "top_n": 20,
    "min_score": 50,
}


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
    limit = 0.35 if prev_close < 200 else (0.25 if prev_close <= 5000 else 0.20)
    ara = round_to_idx_tick(prev_close * (1 + limit))
    arb = round_to_idx_tick(prev_close * (1 - limit))
    return min(price, ara) if is_target else max(price, arb)


def compute_rsi(close: pd.Series, period: int = 14) -> float:
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    rs = gain.rolling(period).mean() / loss.rolling(period).mean().replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    val = float(rsi.iloc[-1])
    return val if val == val else 50.0


def compute_atr(df: pd.DataFrame, period: int = 14) -> float:
    h, l, c = df["High"], df["Low"], df["Close"]
    prev = c.shift(1)
    tr = pd.concat([h - l, (h - prev).abs(), (l - prev).abs()], axis=1).max(axis=1)
    val = float(tr.rolling(period).mean().iloc[-1])
    return val if val == val and val > 0 else 0.0


def analyze_ticker(symbol: str, params: dict) -> dict | None:
    try:
        df = yf.download(
            symbol + ".JK",
            period=f"{int(params['lookback_days'] * 1.6)}d",
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
    if any(c not in df.columns for c in ["Open", "High", "Low", "Close", "Volume"]):
        return None

    df = df.dropna()
    if len(df) < 30:
        return None

    c, h, l, v = df["Close"], df["High"], df["Low"], df["Volume"]
    last = float(c.iloc[-1])
    prev = float(c.iloc[-2])
    if last < params["min_price"] or last > params["max_price"]:
        return None

    avg_vol = float(v.tail(20).mean())
    avg_val = float((c * v).tail(20).mean())
    if avg_vol < params["min_avg_volume"] or avg_val < params["min_avg_value_rp"]:
        return None

    high20 = float(h.tail(20).max())
    low20 = float(l.tail(20).min())
    range20 = (high20 - low20) / last * 100
    if range20 < params["min_range_20_pct"] or range20 > params["max_range_20_pct"]:
        return None

    atr = compute_atr(df, 14)
    atr_pct = atr / last * 100 if last else 0
    if atr_pct < params["min_atr_pct"] or atr_pct > params["max_atr_pct"]:
        return None

    vol_ratio = float(v.tail(5).mean()) / avg_vol if avg_vol > 0 else 0
    if vol_ratio < params["min_vol_ratio"]:
        return None

    rsi = compute_rsi(c)
    if not (params["rsi_min"] <= rsi <= params["rsi_max"]):
        return None

    ma10 = float(c.rolling(10).mean().iloc[-1])
    ma20 = float(c.rolling(20).mean().iloc[-1])
    if params.get("require_above_ma10") and last < ma10:
        return None

    # Skor karakter spekulatif likuid
    score = 0
    reasons = []

    # Value tier
    if avg_val >= 100e9:
        score += 25
        reasons.append(f"Value sangat besar ({avg_val/1e9:.0f}M)")
    elif avg_val >= 50e9:
        score += 18
        reasons.append(f"Value besar ({avg_val/1e9:.0f}M)")
    else:
        score += 10
        reasons.append(f"Value OK ({avg_val/1e9:.0f}M)")

    # Volatilitas di zona "aktif tapi masih tradeable"
    if 12 <= range20 <= 28:
        score += 20
        reasons.append(f"Range20 spekulatif ({range20:.0f}%)")
    else:
        score += 10
        reasons.append(f"Range20 {range20:.0f}%")

    if 3.0 <= atr_pct <= 7.0:
        score += 15
        reasons.append(f"ATR% {atr_pct:.1f}")
    else:
        score += 8
        reasons.append(f"ATR% {atr_pct:.1f}")

    if vol_ratio >= 1.5:
        score += 15
        reasons.append(f"Vol spike {vol_ratio:.2f}x")
    else:
        score += 8
        reasons.append(f"Vol {vol_ratio:.2f}x")

    if last > ma10:
        score += 10
        reasons.append("Di atas MA10")
    if last > ma20:
        score += 8
        reasons.append("Di atas MA20")

    # Kedekatan high20 — momentum pendek (boleh tidak di puncak)
    dist_high = (high20 - last) / last * 100
    if dist_high <= 5:
        score += 10
        reasons.append(f"Dekat high20 ({dist_high:.1f}%)")
    elif dist_high <= 12:
        score += 5
        reasons.append(f"Area high20 ({dist_high:.1f}%)")

    chg1 = (last / prev - 1) * 100
    if abs(chg1) >= 3:
        score += 5
        reasons.append(f"Hari aktif {chg1:+.1f}%")

    if score < params["min_score"]:
        return None

    # Plan agresif pendek — SL/TP berbasis ATR (bukan struktur SMC)
    entry = round_to_idx_tick(last)
    stop_raw = entry - params["sl_atr_mult"] * atr
    stop = apply_ara_arb_limits(round_to_idx_tick(stop_raw), last, is_target=False)
    if stop >= entry:
        stop = apply_ara_arb_limits(round_to_idx_tick(entry * 0.97), last, is_target=False)

    risk = entry - stop
    if risk <= 0:
        return None

    target_raw = entry + params["tp_atr_mult"] * atr
    # jangan jauh di atas high20 tanpa alasan
    target_raw = min(target_raw, high20 * 1.02 if high20 > entry else target_raw)
    target = apply_ara_arb_limits(round_to_idx_tick(target_raw), last, is_target=True)

    reward = target - entry
    rr = reward / risk if risk > 0 else 0
    if rr < params["min_rr"]:
        return None

    risk_rp = params["account_size"] * params["risk_per_trade_pct"] / 100
    lots = int((risk_rp / risk) // params["lot_size"]) if risk > 0 else 0
    shares = lots * params["lot_size"]

    return {
        "Ticker": symbol,
        "Close": entry,
        "Score": score,
        "AvgValue20D(M)": round(avg_val / 1e9, 1),
        "Range20%": round(range20, 1),
        "ATR%": round(atr_pct, 2),
        "VolRatio": round(vol_ratio, 2),
        "RSI": round(rsi, 1),
        "DistHigh20%": round(dist_high, 1),
        "Chg1D%": round(chg1, 2),
        "Entry": entry,
        "StopLoss": stop,
        "Target": target,
        "RR_Ratio": round(rr, 2),
        "Lots": lots,
        "EstLoss(Rp)": round(shares * risk, 0),
        "EstProfit(Rp)": round(shares * reward, 0),
        "Alasan": "; ".join(reasons),
        "Strategy": "HighBeta Liquid",
        "Horizon": "Intraday / swing sangat pendek",
    }


def get_universe(params: dict) -> list:
    # Prefilter longgar; filter ketat value di analyse
    if IdxLiquidityScanner is None:
        return [
            "BUMI", "BREN", "GOTO", "BBCA", "BBRI", "BMRI", "TLKM", "ASII",
            "AMMN", "MDKA", "ANTM", "ADRO", "PTBA", "ITMG", "BRMS", "CUAN",
            "DSSA", "BYAN", "EMTK", "BUKA", "ARTO", "PANI", "TPIA", "BRPT",
        ]
    scanner = IdxLiquidityScanner(
        min_avg_value_rp=max(15_000_000_000, params["min_avg_value_rp"] * 0.5),
        min_avg_volume=max(2_000_000, int(params["min_avg_volume"] * 0.4)),
        lookback_days=20,
        max_workers=15,
    )
    return scanner.get_liquid_universe()


def run_highbeta_screener(user_params: dict | None = None):
    params = PARAMS.copy()
    if user_params:
        params.update(user_params)

    print("=" * 60)
    print("HIGH-BETA / SPECULATIVE LIQUID SCREENER")
    print("=" * 60)
    universe = get_universe(params)
    print(f"Analisa {len(universe)} kandidat...\n")

    rows = []
    for i, sym in enumerate(universe, 1):
        print(f"  [{i}/{len(universe)}] {sym}...", end="\r")
        try:
            r = analyze_ticker(sym, params)
            if r:
                rows.append(r)
        except Exception:
            continue
    print(" " * 50, end="\r")

    if not rows:
        print("Tidak ada saham high-beta likuid yang lolos filter hari ini.")
        return pd.DataFrame()

    df = pd.DataFrame(rows).sort_values("Score", ascending=False)
    df = df.head(params["top_n"]).reset_index(drop=True)

    cols = [
        "Ticker", "Close", "Score", "AvgValue20D(M)", "Range20%", "ATR%",
        "VolRatio", "RSI", "DistHigh20%", "Entry", "StopLoss", "Target",
        "RR_Ratio", "Lots", "Alasan",
    ]
    cols = [c for c in cols if c in df.columns]

    print("=" * 110)
    print(f"HASIL — {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print("Karakter: likuid besar + range/ATR tinggi + partisipasi aktif")
    print("=" * 110)
    print(df[cols].to_string(index=False))
    print("=" * 110)
    print("Catatan: SL/TP berbasis ATR. Size default 0.5% risk. Wajib cek order book.")

    try:
        from idx_report_schema import save_version_report
        path = save_version_report(df, "highbeta")
    except Exception:
        path = f"idx_report_highbeta_{datetime.now().strftime('%Y-%m-%d')}.csv"
        df.to_csv(path, index=False)
    print(f"Disimpan: {path}")
    return df


if __name__ == "__main__":
    run_highbeta_screener()