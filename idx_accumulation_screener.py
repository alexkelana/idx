"""
IDX ACCUMULATION SCREENER — Late Accumulation / Early Markup
============================================================
Mencari saham yang mendekati akhir fase akumulasi (Wyckoff-ish):
- Ada base/range sideways dengan durasi memadai
- Volume sempat kontraksi di dalam range
- Higher low dan/atau OBV naik (proxy akumulasi)
- Volatilitas sempat menyempit
- Harga di bagian atas range ATAU baru early break

Bukan: breakout ketat V2, OB V4, atau high-beta liar.
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
# PARAMETER
# =========================================================================
PARAMS = {
    # Universe
    "min_avg_value_rp": 10_000_000_000,
    "min_avg_volume": 1_000_000,
    "lookback_days": 120,
    "min_price": 50,
    "max_price": 20000,
    # Deteksi base
    "base_window": 50,            # window utama pencarian range
    "min_days_in_base": 20,
    "max_days_in_base": 80,
    "min_range_pct": 6.0,         # terlalu sempit = noise
    "max_range_pct": 28.0,        # terlalu lebar = bukan akumulasi tenang
    # Late accumulation
    "min_pos_in_range": 0.62,     # di kuartil atas range
    "max_breakout_pct": 6.0,      # boleh sedikit di atas range high (early markup)
    # Volume & volatilitas
    "max_vol_contraction": 0.90,  # vol tengah range <= 90% vol awal range
    "atr_contract_ratio": 0.95,   # ATR akhir < ATR awal window (sempat kontraksi)
    # Scoring
    "min_score": 55,
    "top_n": 20,
    # Risk plan (konservatif — struktur range)
    "sl_buffer_pct": 0.015,       # di bawah range low
    "min_rr": 1.4,
    "account_size": 50_000_000,
    "risk_per_trade_pct": 0.75,
    "lot_size": 100,
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
    limit = 0.35 if prev_close < 200 else (0.25 if prev_close <= 5000 else 0.20)
    ara = round_to_idx_tick(prev_close * (1 + limit))
    arb = round_to_idx_tick(prev_close * (1 - limit))
    return min(price, ara) if is_target else max(price, arb)


def compute_atr_series(df: pd.DataFrame, period: int = 14) -> pd.Series:
    h, l, c = df["High"], df["Low"], df["Close"]
    prev = c.shift(1)
    tr = pd.concat([h - l, (h - prev).abs(), (l - prev).abs()], axis=1).max(axis=1)
    return tr.rolling(period).mean()


def compute_obv(df: pd.DataFrame) -> pd.Series:
    direction = np.sign(df["Close"].diff()).fillna(0.0)
    return (direction * df["Volume"]).cumsum()


def swing_lows(low: np.ndarray, left: int = 2, right: int = 2):
    out = []
    for i in range(left, len(low) - right):
        window = low[i - left : i + right + 1]
        if low[i] == window.min() and list(window).count(low[i]) == 1:
            out.append((i, float(low[i])))
    return out


# =========================================================================
# DETEKSI BASE / AKUMULASI
# =========================================================================
def detect_accumulation(df: pd.DataFrame, params: dict) -> dict | None:
    """
    Deteksi range akumulasi pada window terakhir dan status late-stage.
    """
    n = params["base_window"]
    if len(df) < n + 20:
        return None

    base = df.iloc[-n:].copy()
    high = base["High"].values
    low = base["Low"].values
    close = base["Close"].values
    vol = base["Volume"].values

    range_high = float(np.max(high))
    range_low = float(np.min(low))
    mid = (range_high + range_low) / 2.0
    if mid <= 0:
        return None

    range_pct = (range_high - range_low) / mid * 100.0
    if range_pct < params["min_range_pct"] or range_pct > params["max_range_pct"]:
        return None

    last_close = float(close[-1])
    width = range_high - range_low
    if width <= 0:
        return None

    # Posisi di dalam range (boleh sedikit break di atas)
    pos = (last_close - range_low) / width
    breakout_pct = (last_close - range_high) / range_high * 100.0

    # Harus di bagian atas range ATAU early break terbatas
    late = pos >= params["min_pos_in_range"]
    early_break = 0 < breakout_pct <= params["max_breakout_pct"]
    if not (late or early_break):
        return None
    # Jangan terlalu jauh di atas range (late chase)
    if breakout_pct > params["max_breakout_pct"]:
        return None

    # Volume: awal vs tengah window
    third = max(5, n // 3)
    vol_early = float(np.mean(vol[:third]))
    vol_mid = float(np.mean(vol[third : 2 * third]))
    vol_late = float(np.mean(vol[2 * third :]))
    if vol_early <= 0:
        return None
    vol_contraction = vol_mid / vol_early

    # ATR kontraksi (awal window vs akhir)
    atr = compute_atr_series(df, 14)
    atr_base = atr.iloc[-n:]
    if atr_base.isna().all():
        return None
    atr_early = float(atr_base.iloc[:third].mean())
    atr_late = float(atr_base.iloc[-third:].mean())
    atr_ratio = atr_late / atr_early if atr_early > 0 else 1.0

    # Higher low
    lows = swing_lows(low, 2, 2)
    has_hl = False
    if len(lows) >= 2:
        has_hl = lows[-1][1] > lows[-2][1]

    # OBV slope (regresi linear sederhana pada base)
    obv = compute_obv(df).iloc[-n:].values.astype(float)
    x = np.arange(len(obv), dtype=float)
    if len(obv) >= 10 and np.std(obv) > 0:
        slope = float(np.polyfit(x, obv, 1)[0])
        # normalisasi kasar vs volume rata-rata
        obv_slope_norm = slope / (float(np.mean(vol)) + 1e-9)
    else:
        obv_slope_norm = 0.0
    obv_rising = obv_slope_norm > 0

    # Estimasi "hari di base": dari swing low terdalam signifikan
    # pendekatan: gunakan full window jika syarat volume/range terpenuhi
    days_in_base = n
    # perketat: hitung dari bar saat harga pertama kali masuk 90% dalam range akhir
    # (sederhana: pakai n; durasi divalidasi min/max)
    if days_in_base < params["min_days_in_base"] or days_in_base > params["max_days_in_base"]:
        # window tetap 50; jika Anda ingin adaptif, bisa scan multi-window
        pass

    return {
        "range_high": range_high,
        "range_low": range_low,
        "range_pct": range_pct,
        "pos_in_range": pos,
        "breakout_pct": breakout_pct,
        "vol_contraction": vol_contraction,
        "vol_early": vol_early,
        "vol_mid": vol_mid,
        "vol_late": vol_late,
        "atr_ratio": atr_ratio,
        "has_higher_low": has_hl,
        "obv_rising": obv_rising,
        "obv_slope_norm": obv_slope_norm,
        "days_in_base": days_in_base,
        "early_break": early_break,
        "atr_last": float(atr.iloc[-1]) if not pd.isna(atr.iloc[-1]) else 0.0,
    }


# =========================================================================
# ANALISA TICKER
# =========================================================================
def analyze_ticker(symbol: str, params: dict) -> dict | None:
    try:
        df = yf.download(
            symbol + ".JK",
            period=f"{int(params['lookback_days'] * 1.5)}d",
            interval="1d",
            progress=False,
            auto_adjust=True,
            multi_level_index=False,
        )
    except Exception:
        return None

    if df is None or df.empty or len(df) < params["base_window"] + 15:
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
    if len(df) < params["base_window"] + 15:
        return None

    last = float(df["Close"].iloc[-1])
    if last < params["min_price"] or last > params["max_price"]:
        return None

    avg_vol = float(df["Volume"].tail(20).mean())
    avg_val = float((df["Close"] * df["Volume"]).tail(20).mean())
    if avg_vol < params["min_avg_volume"] or avg_val < params["min_avg_value_rp"]:
        return None

    acc = detect_accumulation(df, params)
    if not acc:
        return None

    # Soft checks volume & ATR (tidak hard-fail semua agar tidak terlalu sepi)
    score = 40
    reasons = []

    # Posisi late accumulation
    if acc["early_break"]:
        score += 18
        reasons.append(f"Early break +{acc['breakout_pct']:.1f}%")
        phase = "Early Markup"
    else:
        score += 12
        reasons.append(f"Pos range {acc['pos_in_range']*100:.0f}%")
        phase = "Late Accumulation"

    # Volume contraction
    if acc["vol_contraction"] <= params["max_vol_contraction"]:
        score += 15
        reasons.append(f"Vol kontrak {acc['vol_contraction']:.2f}")
    elif acc["vol_contraction"] <= 1.05:
        score += 6
        reasons.append(f"Vol stabil {acc['vol_contraction']:.2f}")
    else:
        score -= 5
        reasons.append(f"Vol belum kontrak {acc['vol_contraction']:.2f}")

    # Volume late mulai hidup (partisipasi balik)
    if acc["vol_early"] > 0 and acc["vol_late"] >= acc["vol_mid"] * 1.05:
        score += 8
        reasons.append("Vol akhir naik")

    # ATR contraction
    if acc["atr_ratio"] <= params["atr_contract_ratio"]:
        score += 12
        reasons.append(f"ATR kontrak {acc['atr_ratio']:.2f}")
    elif acc["atr_ratio"] <= 1.05:
        score += 5
        reasons.append(f"ATR stabil {acc['atr_ratio']:.2f}")

    # Higher low
    if acc["has_higher_low"]:
        score += 12
        reasons.append("Higher low")
    else:
        reasons.append("HL belum jelas")

    # OBV
    if acc["obv_rising"]:
        score += 12
        reasons.append("OBV naik")
    else:
        score -= 3
        reasons.append("OBV lemah")

    # Range quality
    if 8 <= acc["range_pct"] <= 20:
        score += 8
        reasons.append(f"Range {acc['range_pct']:.1f}%")
    else:
        score += 3
        reasons.append(f"Range {acc['range_pct']:.1f}%")

    if score < params["min_score"]:
        return None

    # --- Plan: SL di bawah range low, target = range high atau +1R di atas break ---
    entry = round_to_idx_tick(last)
    sl_raw = acc["range_low"] * (1 - params["sl_buffer_pct"])
    stop = apply_ara_arb_limits(round_to_idx_tick(sl_raw), last, is_target=False)
    if stop >= entry:
        stop = apply_ara_arb_limits(round_to_idx_tick(entry * 0.97), last, is_target=False)

    risk = entry - stop
    if risk <= 0:
        return None

    # Target 1: range high (jika masih di dalam), atau ekstensi kecil jika early break
    if acc["early_break"]:
        t1_raw = max(acc["range_high"] * 1.03, entry + risk * 1.6)
    else:
        t1_raw = max(acc["range_high"], entry + risk * 1.4)

    target = apply_ara_arb_limits(round_to_idx_tick(t1_raw), last, is_target=True)
    reward = target - entry
    rr = reward / risk if risk > 0 else 0
    if rr < params["min_rr"]:
        # longgarkan target sedikit ke 1.5R jika struktur mengizinkan
        target = apply_ara_arb_limits(
            round_to_idx_tick(entry + risk * params["min_rr"]), last, is_target=True
        )
        reward = target - entry
        rr = reward / risk if risk > 0 else 0
        if rr < params["min_rr"] * 0.95:
            return None

    risk_rp = params["account_size"] * params["risk_per_trade_pct"] / 100
    lots = int((risk_rp / risk) // params["lot_size"]) if risk > 0 else 0
    shares = lots * params["lot_size"]

    dist_break = (acc["range_high"] - last) / last * 100.0

    return {
        "Ticker": symbol,
        "Close": entry,
        "Phase": phase,
        "Score": score,
        "RangeLow": round_to_idx_tick(acc["range_low"]),
        "RangeHigh": round_to_idx_tick(acc["range_high"]),
        "Range%": round(acc["range_pct"], 1),
        "PosInRange": round(acc["pos_in_range"], 2),
        "DaysBase": acc["days_in_base"],
        "VolContract": round(acc["vol_contraction"], 2),
        "ATRratio": round(acc["atr_ratio"], 2),
        "HL": "Y" if acc["has_higher_low"] else "N",
        "OBV": "Y" if acc["obv_rising"] else "N",
        "DistBreak%": round(dist_break, 2),
        "Entry": entry,
        "StopLoss": stop,
        "Target": target,
        "RR_Ratio": round(rr, 2),
        "Lots": lots,
        "EstLoss(Rp)": round(shares * risk, 0),
        "EstProfit(Rp)": round(shares * reward, 0),
        "Alasan": "; ".join(reasons),
        "Strategy": "Accumulation Late/Early",
    }


# =========================================================================
# RUNNER
# =========================================================================
def get_universe(params: dict) -> list:
    if IdxLiquidityScanner is None:
        return [
            "BBCA", "BBRI", "BMRI", "BBNI", "TLKM", "ASII", "UNVR", "ICBP",
            "KLBF", "ANTM", "ADRO", "PTBA", "SMGR", "INTP", "PGAS", "MDKA",
            "GOTO", "AMMN", "BRIS", "ARTO", "INDF", "CPIN", "MYOR", "PWON",
        ]
    scanner = IdxLiquidityScanner(
        min_avg_value_rp=params["min_avg_value_rp"],
        min_avg_volume=params["min_avg_volume"],
        lookback_days=20,
        max_workers=15,
    )
    return scanner.get_liquid_universe()


def run_accumulation_screener(user_params: dict | None = None):
    params = PARAMS.copy()
    if user_params:
        params.update(user_params)

    print("=" * 60)
    print("IDX ACCUMULATION SCREENER (Late Accumulation / Early Markup)")
    print("=" * 60)
    universe = get_universe(params)
    print(f"Analisa {len(universe)} saham...\n")

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
        print("Tidak ada saham di fase late accumulation / early markup hari ini.")
        return pd.DataFrame()

    df = pd.DataFrame(rows).sort_values(
        ["Score", "PosInRange"], ascending=[False, False]
    )
    df = df.head(params["top_n"]).reset_index(drop=True)

    cols = [
        "Ticker", "Close", "Phase", "Score", "RangeLow", "RangeHigh", "Range%",
        "PosInRange", "VolContract", "ATRratio", "HL", "OBV", "DistBreak%",
        "Entry", "StopLoss", "Target", "RR_Ratio", "Lots", "Alasan",
    ]
    cols = [c for c in cols if c in df.columns]

    print("=" * 110)
    print(f"HASIL — {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print("Phase: Late Accumulation = di atas range | Early Markup = baru pecah tipis")
    print("=" * 110)
    print(df[cols].to_string(index=False))
    print("=" * 110)

    try:
        from idx_report_schema import save_version_report
        path = save_version_report(df, "accumulation")
    except Exception:
        path = f"idx_report_accumulation_{datetime.now().strftime('%Y-%m-%d')}.csv"
        df.to_csv(path, index=False)
    print(f"Disimpan: {path}")
    return df


if __name__ == "__main__":
    run_accumulation_screener()