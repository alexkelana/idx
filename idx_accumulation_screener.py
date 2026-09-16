"""
IDX ACCUMULATION SCREENER — Late Accumulation / Early Markup
============================================================
+ Filter sweep low (spring) di RangeLow:
  Low menusuk di bawah range, Close reclaim ke dalam range

Mencari saham mendekati akhir akumulasi:
- Base/range sideways
- Volume kontraksi
- Higher low / OBV naik
- Volatilitas sempat menyempit
- Harga di bagian atas range ATAU early break
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
    "min_avg_value_rp": 10_000_000_000,
    "min_avg_volume": 1_000_000,
    "lookback_days": 120,
    "min_price": 50,
    "max_price": 20000,
    "base_window": 50,
    "min_days_in_base": 20,
    "max_days_in_base": 80,
    "min_range_pct": 6.0,
    "max_range_pct": 28.0,
    "min_pos_in_range": 0.62,
    "max_breakout_pct": 6.0,
    "max_vol_contraction": 0.90,
    "atr_contract_ratio": 0.95,
    "min_score": 55,
    "top_n": 20,
    "sl_buffer_pct": 0.015,
    "min_rr": 1.4,
    "account_size": 50_000_000,
    "risk_per_trade_pct": 0.75,
    "lot_size": 100,
    # Sweep low (spring)
    "require_sweep_low": False,
    "sweep_lookback": 20,
    "sweep_min_wick_pct": 0.30,
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


def detect_liquidity_sweep_low(
    df: pd.DataFrame,
    support: float,
    lookback: int = 20,
    min_wick_pct: float = 0.30,
) -> dict:
    """
    Spring / sweep low di dasar range:
    Low < support, Close reclaim >= support (hampir).
    """
    if support <= 0 or len(df) < lookback + 2:
        return {"has_sweep": False, "sweep_low": 0.0, "wick_pct": 0.0}

    start = max(0, len(df) - lookback - 1)
    end = len(df) - 1
    for i in range(end - 1, start - 1, -1):
        lo = float(df["Low"].iloc[i])
        cl = float(df["Close"].iloc[i])
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


def detect_accumulation(df: pd.DataFrame, params: dict) -> dict | None:
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

    pos = (last_close - range_low) / width
    breakout_pct = (last_close - range_high) / range_high * 100.0

    late = pos >= params["min_pos_in_range"]
    early_break = 0 < breakout_pct <= params["max_breakout_pct"]
    if not (late or early_break):
        return None
    if breakout_pct > params["max_breakout_pct"]:
        return None

    third = max(5, n // 3)
    vol_early = float(np.mean(vol[:third]))
    vol_mid = float(np.mean(vol[third : 2 * third]))
    vol_late = float(np.mean(vol[2 * third :]))
    if vol_early <= 0:
        return None
    vol_contraction = vol_mid / vol_early

    atr = compute_atr_series(df, 14)
    atr_base = atr.iloc[-n:]
    if atr_base.isna().all():
        return None
    atr_early = float(atr_base.iloc[:third].mean())
    atr_late = float(atr_base.iloc[-third:].mean())
    atr_ratio = atr_late / atr_early if atr_early > 0 else 1.0

    lows = swing_lows(low, 2, 2)
    has_hl = False
    if len(lows) >= 2:
        has_hl = lows[-1][1] > lows[-2][1]

    obv = compute_obv(df).iloc[-n:].values.astype(float)
    x = np.arange(len(obv), dtype=float)
    if len(obv) >= 10 and np.std(obv) > 0:
        slope = float(np.polyfit(x, obv, 1)[0])
        obv_slope_norm = slope / (float(np.mean(vol)) + 1e-9)
    else:
        obv_slope_norm = 0.0
    obv_rising = obv_slope_norm > 0

    days_in_base = n

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

    # Sweep low (spring) di RangeLow
    sweep = detect_liquidity_sweep_low(
        df,
        support=float(acc["range_low"]),
        lookback=int(params.get("sweep_lookback", 20)),
        min_wick_pct=float(params.get("sweep_min_wick_pct", 0.30)),
    )
    if params.get("require_sweep_low", False) and not sweep["has_sweep"]:
        return None

    score = 40
    reasons = []

    if acc["early_break"]:
        score += 18
        reasons.append(f"Early break +{acc['breakout_pct']:.1f}%")
        phase = "Early Markup"
    else:
        score += 12
        reasons.append(f"Pos range {acc['pos_in_range']*100:.0f}%")
        phase = "Late Accumulation"

    if acc["vol_contraction"] <= params["max_vol_contraction"]:
        score += 15
        reasons.append(f"Vol kontrak {acc['vol_contraction']:.2f}")
    elif acc["vol_contraction"] <= 1.05:
        score += 6
        reasons.append(f"Vol stabil {acc['vol_contraction']:.2f}")
    else:
        score -= 5
        reasons.append(f"Vol belum kontrak {acc['vol_contraction']:.2f}")

    if acc["vol_early"] > 0 and acc["vol_late"] >= acc["vol_mid"] * 1.05:
        score += 8
        reasons.append("Vol akhir naik")

    if acc["atr_ratio"] <= params["atr_contract_ratio"]:
        score += 12
        reasons.append(f"ATR kontrak {acc['atr_ratio']:.2f}")
    elif acc["atr_ratio"] <= 1.05:
        score += 5
        reasons.append(f"ATR stabil {acc['atr_ratio']:.2f}")

    if acc["has_higher_low"]:
        score += 12
        reasons.append("Higher low")
    else:
        reasons.append("HL belum jelas")

    if acc["obv_rising"]:
        score += 12
        reasons.append("OBV naik")
    else:
        score -= 3
        reasons.append("OBV lemah")

    if 8 <= acc["range_pct"] <= 20:
        score += 8
        reasons.append(f"Range {acc['range_pct']:.1f}%")
    else:
        score += 3
        reasons.append(f"Range {acc['range_pct']:.1f}%")

    # Bonus spring / sweep low
    if sweep["has_sweep"]:
        score += 15
        reasons.append(
            f"Sweep low/spring (wick {sweep.get('wick_pct', 0)}%, "
            f"{sweep.get('bars_ago', '?')} bar lalu)"
        )

    if score < params["min_score"]:
        return None

    entry = round_to_idx_tick(last)
    sl_raw = acc["range_low"] * (1 - params["sl_buffer_pct"])
    if sweep["has_sweep"] and sweep.get("sweep_low", 0) > 0:
        sl_raw = min(sl_raw, float(sweep["sweep_low"]) * 0.99)

    stop = apply_ara_arb_limits(round_to_idx_tick(sl_raw), last, is_target=False)
    if stop >= entry:
        stop = apply_ara_arb_limits(round_to_idx_tick(entry * 0.97), last, is_target=False)

    risk = entry - stop
    if risk <= 0:
        return None

    if acc["early_break"]:
        t1_raw = max(acc["range_high"] * 1.03, entry + risk * 1.6)
    else:
        t1_raw = max(acc["range_high"], entry + risk * 1.4)

    target = apply_ara_arb_limits(round_to_idx_tick(t1_raw), last, is_target=True)
    reward = target - entry
    rr = reward / risk if risk > 0 else 0
    if rr < params["min_rr"]:
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
        "SweepLow": "Ya" if sweep["has_sweep"] else "Tidak",
        "SweepLowPrice": round_to_idx_tick(sweep.get("sweep_low", 0)),
        "SweepWickPct": sweep.get("wick_pct", 0),
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


def get_universe(params: dict) -> list:
    try:
        from idx_gdrive_data import get_ticker_universe

        remote = get_ticker_universe(fallback=[])
        if remote:
            return remote
    except Exception:
        pass

    if IdxLiquidityScanner is not None:
        try:
            scanner = IdxLiquidityScanner(
                min_avg_value_rp=params["min_avg_value_rp"],
                min_avg_volume=params["min_avg_volume"],
                lookback_days=20,
                max_workers=15,
            )
            u = scanner.get_liquid_universe()
            if u:
                return u
        except Exception:
            pass

    return [
        "BBCA", "BBRI", "BMRI", "BBNI", "TLKM", "ASII", "UNVR", "ICBP",
        "KLBF", "ANTM", "ADRO", "PTBA", "SMGR", "INTP", "PGAS", "MDKA",
        "GOTO", "AMMN", "BRIS", "ARTO", "INDF", "CPIN", "MYOR", "PWON",
    ]


def run_accumulation_screener(user_params: dict | None = None):
    params = PARAMS.copy()
    if user_params:
        params.update(user_params)

    print("=" * 60)
    print("IDX ACCUMULATION SCREENER (Late Base + Sweep Low/Spring)")
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
        "Ticker", "Close", "Phase", "Score", "SweepLow", "RangeLow", "RangeHigh",
        "Range%", "PosInRange", "VolContract", "HL", "OBV", "DistBreak%",
        "Entry", "StopLoss", "Target", "RR_Ratio", "Lots", "Alasan",
    ]
    cols = [c for c in cols if c in df.columns]

    print("=" * 110)
    print(f"HASIL — {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print("SweepLow = spring di RangeLow (Low < support, Close reclaim)")
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
