"""
IDX MA CROSS SCREENER — Golden / Death Cross (MA50 × MA200)
============================================================
Menangkap persilangan MA50 dan MA200:

  GOLDEN CROSS : MA50 memotong naik melewati MA200
  DEATH CROSS  : MA50 memotong turun melewati MA200

Filter:
- Universe likuid (shared scanner / Google Drive)
- Cross relatif fresh (default ≤ 15 bar)
- Opsional: harga konfirmasi (di atas kedua MA untuk golden, di bawah untuk death)
- Score berdasarkan freshness + jarak MA + volume

Output: idx_report_macross_YYYY-MM-DD.csv
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Optional

import numpy as np
import pandas as pd

try:
    import yfinance as yf
except ImportError:
    yf = None

try:
    from idx_liquidity_scanner import IdxLiquidityScanner
except ImportError:
    IdxLiquidityScanner = None

try:
    from idx_exchange_rules import (
        round_to_idx_tick,
        apply_ara_arb_limits,
        DEFAULT_SCREENER_MIN_PRICE,
    )
except ImportError:

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
        if pc <= 200:
            ara, arb = pc * 1.35, pc * 0.85
        elif pc <= 5000:
            ara, arb = pc * 1.25, pc * 0.85
        else:
            ara, arb = pc * 1.20, pc * 0.85
        p = float(round_to_idx_tick(price))
        return min(p, float(round_to_idx_tick(ara))) if is_target else max(
            p, float(round_to_idx_tick(arb))
        )

    DEFAULT_SCREENER_MIN_PRICE = 50.0


PARAMS = {
    "lookback_bars": 15,          # max umur cross (bar) agar dianggap fresh
    "history_days": 320,          # butuh ≥200+ untuk MA200 stabil
    "min_price": 50.0,
    "max_price": 100_000.0,
    "min_avg_value_rp": 5_000_000_000,
    "min_avg_volume": 500_000,
    "require_price_confirm": True,  # golden: close > MA50 & MA200; death: sebaliknya
    "min_sep_pct_at_cross": 0.0,    # |MA50-MA200|/MA200 saat cross (0 = longgar)
    "min_volume_ratio": 0.8,       # vol hari cross vs avg20 (opsional longgar)
    "include_death": True,
    "include_golden": True,
    "account_size": 50_000_000,
    "risk_per_trade_pct": 1.0,
    "lot_size": 100,
    # SL/TP heuristik: golden SL di bawah MA200, TP = risk * RR
    "sl_buffer_pct": 1.5,
    "rr_target": 2.0,
    "top_n": 40,
}


def _download(ticker: str, days: int) -> Optional[pd.DataFrame]:
    if yf is None:
        return None
    try:
        df = yf.download(
            f"{ticker}.JK",
            period=f"{max(days, 250)}d",
            interval="1d",
            progress=False,
            auto_adjust=True,
            multi_level_index=False,
        )
    except Exception:
        return None
    if df is None or df.empty:
        return None
    if isinstance(df.columns, pd.MultiIndex):
        try:
            df.columns = df.columns.get_level_values(0)
        except Exception:
            return None
    need = {"Open", "High", "Low", "Close", "Volume"}
    if not need.issubset(set(df.columns)):
        return None
    df = df.dropna(subset=["Close"]).copy()
    return df if len(df) >= 220 else None


def _find_ma_crosses(
    ma50: pd.Series,
    ma200: pd.Series,
    lookback: int,
) -> list[dict[str, Any]]:
    """
    Cari golden/death cross dalam `lookback` bar terakhir (tidak termasuk bar berjalan
    jika ingin confirmed close — kita scan sampai bar terakhir yang fully closed).
    Cross di bar i: tanda (MA50-MA200) berubah vs bar i-1.
    """
    delta = ma50 - ma200
    crosses: list[dict[str, Any]] = []
    n = len(delta)
    if n < 3:
        return crosses

    start = max(1, n - lookback)
    for i in range(start, n):
        prev = float(delta.iloc[i - 1])
        cur = float(delta.iloc[i])
        if np.isnan(prev) or np.isnan(cur):
            continue
        # golden: prev <= 0 and cur > 0
        if prev <= 0 and cur > 0:
            crosses.append(
                {
                    "kind": "GOLDEN",
                    "bar_index": i,
                    "bars_ago": n - 1 - i,
                    "ma50": float(ma50.iloc[i]),
                    "ma200": float(ma200.iloc[i]),
                }
            )
        # death: prev >= 0 and cur < 0
        elif prev >= 0 and cur < 0:
            crosses.append(
                {
                    "kind": "DEATH",
                    "bar_index": i,
                    "bars_ago": n - 1 - i,
                    "ma50": float(ma50.iloc[i]),
                    "ma200": float(ma200.iloc[i]),
                }
            )
    return crosses



# Bobot modifier per profil strategy
MA_CROSS_PROFILES = {
    # Trend-follow: V2, V3, V4
    "trend": {
        "golden_fresh": 14,   # ≤5 bar
        "golden_ok": 8,       # 6–15
        "golden_weak": 2,     # cross ada tapi harga tidak confirm
        "struct_bull": 3,     # MA50>MA200 tanpa cross fresh
        "struct_bear": -3,
        "death_fresh": -14,
        "death_ok": -8,
        "death_weak": -2,
        "chop_penalty": -5,
    },
    # Mean-reversion / CHOCH: V5 — death tidak menghukum keras
    "soft": {
        "golden_fresh": 8,
        "golden_ok": 5,
        "golden_weak": 1,
        "struct_bull": 2,
        "struct_bear": -1,
        "death_fresh": -5,
        "death_ok": -3,
        "death_weak": 0,
        "chop_penalty": -3,
    },
    # Intraday / short horizon
    "light": {
        "golden_fresh": 5,
        "golden_ok": 3,
        "golden_weak": 0,
        "struct_bull": 1,
        "struct_bear": -1,
        "death_fresh": -5,
        "death_ok": -3,
        "death_weak": 0,
        "chop_penalty": -2,
    },
}


def compute_ma_cross_modifier(
    df: pd.DataFrame,
    *,
    lookback: int = 15,
    profile: str = "trend",
    require_price_confirm: bool = True,
) -> dict:
    """
    Hitung penambah/pengurang skor dari MA50×MA200.

    Returns
    -------
    dict:
      delta, signal (GOLDEN|DEATH|NONE), age, note,
      ma50, ma200, spread_pct, structural (BULL|BEAR|FLAT)
    """
    out = {
        "delta": 0,
        "signal": "NONE",
        "age": None,
        "note": "",
        "ma50": None,
        "ma200": None,
        "spread_pct": None,
        "structural": "FLAT",
        "price_confirm": False,
    }
    if df is None or len(df) < 220:
        out["note"] = "MA cross: data kurang"
        return out

    weights = MA_CROSS_PROFILES.get(profile) or MA_CROSS_PROFILES["trend"]
    close = pd.to_numeric(df["Close"], errors="coerce")
    last = float(close.iloc[-1])
    ma50 = close.rolling(50).mean()
    ma200 = close.rolling(200).mean()
    if pd.isna(ma50.iloc[-1]) or pd.isna(ma200.iloc[-1]):
        out["note"] = "MA cross: MA belum stabil"
        return out

    m50 = float(ma50.iloc[-1])
    m200 = float(ma200.iloc[-1])
    out["ma50"] = round(m50, 2)
    out["ma200"] = round(m200, 2)
    spread = (m50 - m200) / m200 * 100 if m200 else 0.0
    out["spread_pct"] = round(spread, 2)
    if m50 > m200 * 1.001:
        out["structural"] = "BULL"
    elif m50 < m200 * 0.999:
        out["structural"] = "BEAR"

    crosses = _find_ma_crosses(ma50, ma200, lookback)
    chosen = None
    if crosses:
        chosen = sorted(crosses, key=lambda x: x["bars_ago"])[0]

    # Chop: banyak cross dalam 60 bar
    chop = False
    if len(df) >= 260:
        many = _find_ma_crosses(ma50, ma200, 60)
        if len(many) >= 3:
            chop = True

    delta = 0
    notes: list[str] = []

    if chosen:
        kind = chosen["kind"]
        age = int(chosen["bars_ago"])
        out["signal"] = kind
        out["age"] = age
        if kind == "GOLDEN":
            confirm = (not require_price_confirm) or (last > m50 and last > m200)
            out["price_confirm"] = confirm
            if confirm and age <= 5:
                delta += int(weights["golden_fresh"])
                notes.append(f"Golden MA50/200 ({age}b) +{weights['golden_fresh']}")
            elif confirm and age <= lookback:
                delta += int(weights["golden_ok"])
                notes.append(f"Golden MA50/200 ({age}b) +{weights['golden_ok']}")
            else:
                delta += int(weights["golden_weak"])
                notes.append(f"Golden lemah ({age}b) +{weights['golden_weak']}")
        else:  # DEATH
            confirm = (not require_price_confirm) or (last < m50 and last < m200)
            out["price_confirm"] = confirm
            if confirm and age <= 5:
                delta += int(weights["death_fresh"])
                notes.append(f"Death MA50/200 ({age}b) {weights['death_fresh']}")
            elif confirm and age <= lookback:
                delta += int(weights["death_ok"])
                notes.append(f"Death MA50/200 ({age}b) {weights['death_ok']}")
            else:
                delta += int(weights["death_weak"])
                notes.append(f"Death lemah ({age}b) {weights['death_weak']}")
    else:
        # struktural tanpa cross fresh
        if out["structural"] == "BULL":
            delta += int(weights["struct_bull"])
            notes.append(f"Struktur MA50>MA200 +{weights['struct_bull']}")
        elif out["structural"] == "BEAR":
            delta += int(weights["struct_bear"])
            notes.append(f"Struktur MA50<MA200 {weights['struct_bear']}")

    if chop:
        delta += int(weights["chop_penalty"])
        notes.append(f"Chop MA (multi-cross 60b) {weights['chop_penalty']}")

    out["delta"] = int(delta)
    out["note"] = "; ".join(notes) if notes else "MA cross netral"
    return out


def apply_ma_cross_score(
    score: float | int,
    df: pd.DataFrame,
    *,
    profile: str = "trend",
    lookback: int = 15,
    reasons: list | None = None,
) -> tuple[int, dict]:
    """
    Terapkan modifier ke skor strategy. Kembalikan (score_baru, detail_modifier).
    """
    mx = compute_ma_cross_modifier(df, lookback=lookback, profile=profile)
    new_score = int(max(0, min(100, int(score) + int(mx["delta"]))))
    if reasons is not None and mx.get("note"):
        reasons.append(mx["note"])
    return new_score, mx


def analyze_ma_cross_ticker(
    symbol: str,
    params: dict | None = None,
) -> Optional[dict[str, Any]]:
    p = {**PARAMS, **(params or {})}
    df = _download(symbol, int(p["history_days"]))
    if df is None:
        return None

    close = pd.to_numeric(df["Close"], errors="coerce")
    vol = pd.to_numeric(df["Volume"], errors="coerce").fillna(0)
    last = float(close.iloc[-1])
    if last < float(p["min_price"]) or last > float(p["max_price"]):
        return None

    ma50 = close.rolling(50).mean()
    ma200 = close.rolling(200).mean()
    if pd.isna(ma50.iloc[-1]) or pd.isna(ma200.iloc[-1]):
        return None

    crosses = _find_ma_crosses(ma50, ma200, int(p["lookback_bars"]))
    if not crosses:
        return None

    # Ambil cross paling baru yang diizinkan jenisnya
    crosses = sorted(crosses, key=lambda x: x["bars_ago"])
    chosen = None
    for c in crosses:
        if c["kind"] == "GOLDEN" and not p.get("include_golden", True):
            continue
        if c["kind"] == "DEATH" and not p.get("include_death", True):
            continue
        chosen = c
        break
    if chosen is None:
        return None

    kind = chosen["kind"]
    bars_ago = int(chosen["bars_ago"])
    i = int(chosen["bar_index"])
    m50_x = float(chosen["ma50"])
    m200_x = float(chosen["ma200"])
    m50_now = float(ma50.iloc[-1])
    m200_now = float(ma200.iloc[-1])

    sep_pct = abs(m50_x - m200_x) / m200_x * 100 if m200_x else 0.0
    if sep_pct < float(p.get("min_sep_pct_at_cross") or 0):
        # hampir selalu 0 di titik cross — skip filter ketat default
        pass

    # Konfirmasi harga
    price_ok = True
    if p.get("require_price_confirm", True):
        if kind == "GOLDEN":
            price_ok = last > m50_now and last > m200_now
        else:
            price_ok = last < m50_now and last < m200_now
    if not price_ok:
        return None

    # Volume di hari cross vs avg 20
    avg_vol20 = float(vol.iloc[max(0, i - 20) : i].mean()) if i > 5 else float(vol.tail(20).mean())
    vol_cross = float(vol.iloc[i])
    vol_ratio = (vol_cross / avg_vol20) if avg_vol20 > 0 else 0.0
    if vol_ratio < float(p.get("min_volume_ratio") or 0):
        return None

    # ATR14 untuk sizing
    prev_c = close.shift(1)
    tr = pd.concat(
        [
            df["High"] - df["Low"],
            (df["High"] - prev_c).abs(),
            (df["Low"] - prev_c).abs(),
        ],
        axis=1,
    ).max(axis=1)
    atr = float(tr.tail(14).mean())

    entry = float(round_to_idx_tick(last))
    if kind == "GOLDEN":
        # SL di bawah MA200 (invalidasi struktur bullish jangka menengah)
        sl_raw = min(m200_now, m50_now) * (1 - float(p["sl_buffer_pct"]) / 100.0)
        stop = apply_ara_arb_limits(round_to_idx_tick(sl_raw), last, is_target=False)
        risk = entry - stop
        if risk <= 0:
            return None
        tp_raw = entry + risk * float(p["rr_target"])
        target = apply_ara_arb_limits(round_to_idx_tick(tp_raw), last, is_target=True)
        bias = "LONG"
    else:
        # DEATH — setup short-bias / exit long; SL di atas MA200
        sl_raw = max(m200_now, m50_now) * (1 + float(p["sl_buffer_pct"]) / 100.0)
        stop = apply_ara_arb_limits(round_to_idx_tick(sl_raw), last, is_target=True)
        # Untuk short: risk = stop - entry; target di bawah
        risk = stop - entry
        if risk <= 0:
            # fallback long-exit only: tidak hitung short TP ketat
            risk = max(atr, entry * 0.03)
            stop = apply_ara_arb_limits(
                round_to_idx_tick(entry + risk), last, is_target=True
            )
            risk = stop - entry
        tp_raw = entry - risk * float(p["rr_target"])
        target = apply_ara_arb_limits(round_to_idx_tick(tp_raw), last, is_target=False)
        bias = "SHORT_OR_EXIT_LONG"

    rr = abs((target - entry) / risk) if risk else 0.0

    # Score
    score = 40
    reasons = [f"{kind} CROSS MA50/MA200"]
    if bars_ago <= 3:
        score += 25
        reasons.append(f"Sangat fresh ({bars_ago} bar)")
    elif bars_ago <= 7:
        score += 15
        reasons.append(f"Fresh ({bars_ago} bar)")
    else:
        score += 5
        reasons.append(f"Usia cross {bars_ago} bar")

    if price_ok:
        score += 15
        reasons.append("Harga konfirmasi vs MA")

    if vol_ratio >= 1.5:
        score += 10
        reasons.append(f"Vol cross kuat ({vol_ratio:.1f}x)")
    elif vol_ratio >= 1.0:
        score += 5
        reasons.append(f"Vol cross OK ({vol_ratio:.1f}x)")

    # Spread MA sekarang (momentum setelah cross)
    spread_now = (m50_now - m200_now) / m200_now * 100 if m200_now else 0.0
    if kind == "GOLDEN" and spread_now > 0.5:
        score += 5
        reasons.append(f"MA50 > MA200 (+{spread_now:.1f}%)")
    if kind == "DEATH" and spread_now < -0.5:
        score += 5
        reasons.append(f"MA50 < MA200 ({spread_now:.1f}%)")

    score = int(max(0, min(100, score)))

    acct = float(p["account_size"])
    risk_pct = float(p["risk_per_trade_pct"])
    risk_rp = acct * (risk_pct / 100.0)
    risk_ps = abs(entry - stop)
    shares = int(risk_rp / risk_ps) if risk_ps > 0 else 0
    lots = shares // int(p["lot_size"])
    actual = lots * int(p["lot_size"])

    # Tanggal cross jika index datetime
    cross_date = ""
    try:
        cross_date = str(df.index[i].date())
    except Exception:
        cross_date = str(i)

    return {
        "Ticker": symbol,
        "Signal": kind,
        "Bias": bias,
        "CrossDate": cross_date,
        "DaysSinceCross": bars_ago,
        "Close": entry,
        "MA50": round_to_idx_tick(m50_now),
        "MA200": round_to_idx_tick(m200_now),
        "MA50_at_cross": round_to_idx_tick(m50_x),
        "MA200_at_cross": round_to_idx_tick(m200_x),
        "SpreadPct": round(spread_now, 2),
        "VolRatio_at_cross": round(vol_ratio, 2),
        "ATR14": round(atr, 2),
        "Entry": entry,
        "StopLoss": float(stop),
        "Target": float(target),
        "RR_Ratio": round(rr, 2),
        "Score": score,
        "Alasan": "; ".join(reasons),
        "Lots": lots,
        "EstLoss(Rp)": round(actual * risk_ps, 0) if actual else 0,
        "EstProfit(Rp)": round(actual * abs(target - entry), 0) if actual else 0,
        "Strategy": "MA Cross MA50/200",
    }


def _default_universe(params: dict) -> list[str]:
    def _liq():
        if IdxLiquidityScanner is None:
            return []
        scanner = IdxLiquidityScanner(
            min_avg_value_rp=float(params["min_avg_value_rp"]),
            min_avg_volume=float(params["min_avg_volume"]),
            lookback_days=20,
            max_workers=15,
        )
        if hasattr(scanner, "get_liquid_universe"):
            return scanner.get_liquid_universe()
        return []

    try:
        from idx_gdrive_data import resolve_screener_universe

        u = resolve_screener_universe(fallback_fn=_liq)
        if u:
            return list(u)
    except Exception:
        pass
    u = _liq()
    if u:
        return list(u)
    # fallback kecil
    return [
        "BBCA", "BBRI", "BMRI", "BBNI", "TLKM", "ASII", "UNVR", "ICBP",
        "ADRO", "PTBA", "ANTM", "INCO", "MDKA", "BRPT", "GOTO", "BUKA",
    ]


def run_screener_ma_cross(
    user_params: dict | None = None,
    universe: list[str] | None = None,
) -> Optional[pd.DataFrame]:
    """
    universe: opsional — dari master/confluence (skip scan likuiditas).
    """
    p = {**PARAMS, **(user_params or {})}
    if universe:
        print(f"[macross] Shared universe: {len(universe)} ticker")
        uni = list(universe)
    else:
        print("[macross] Menyiapkan universe likuid...")
        uni = _default_universe(p)
        print(f"[macross] Universe: {len(uni)} ticker")

    results: list[dict] = []
    for i, sym in enumerate(uni, 1):
        print(f"  [{i}/{len(uni)}] {sym}...", end="\r")
        try:
            r = analyze_ma_cross_ticker(sym, p)
            if r:
                results.append(r)
        except Exception:
            continue
    print(" " * 50, end="\r")

    if not results:
        print("Tidak ada Golden/Death cross MA50/200 yang memenuhi filter hari ini.")
        return None

    df = pd.DataFrame(results)
    # Golden dulu, lalu score, lalu freshness
    df["_ord"] = df["Signal"].map({"GOLDEN": 0, "DEATH": 1}).fillna(2)
    df = df.sort_values(
        ["_ord", "Score", "DaysSinceCross"],
        ascending=[True, False, True],
    ).drop(columns=["_ord"])

    top_n = int(p.get("top_n") or 40)
    if len(df) > top_n:
        df = df.head(top_n)

    print("\n" + "=" * 100)
    print(f"MA CROSS MA50/MA200 — {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print("=" * 100)
    cols = [
        "Ticker", "Signal", "DaysSinceCross", "Close", "MA50", "MA200",
        "SpreadPct", "Entry", "StopLoss", "Target", "RR_Ratio", "Score", "Alasan",
    ]
    cols = [c for c in cols if c in df.columns]
    print(df[cols].to_string(index=False))
    print("=" * 100)
    n_g = int((df["Signal"] == "GOLDEN").sum())
    n_d = int((df["Signal"] == "DEATH").sum())
    print(f"Golden: {n_g} | Death: {n_d} | Total: {len(df)}")

    try:
        from idx_report_schema import save_version_report

        out = save_version_report(df, "macross")
    except Exception:
        out = f"idx_report_macross_{datetime.now().strftime('%Y-%m-%d')}.csv"
        df.to_csv(out, index=False)
    print(f"Hasil disimpan: {out}")
    return df


# Alias
run_screener = run_screener_ma_cross


if __name__ == "__main__":
    run_screener_ma_cross()
