"""
IDX Structural Entry Helper
===========================
Entry struktural hybrid untuk setup breakout/BOS:

1. Level High yang dibreak (resistance / BOS)
2. Candle swing-high yang memuat level itu (bukan bar breakout)
3. Candle bear terakhir SEBELUM candle high itu
4. EntryStruct = High candle bear itu (hanya jika < last_close → retest)

Mode:
- LIMIT        : tunggu retest ke EntryStruct (struct < close, jarak cukup)
- MARKET       : close dekat struct
- TOO_EXTENDED : struct terlalu jauh di bawah close — jangan chase
- FALLBACK     : struktur tidak valid → entry default

Tidak dipaksa ke V5 / accumulation / MA-cross.
"""

from __future__ import annotations

from typing import Any, Optional

import math
import pandas as pd

try:
    from idx_exchange_rules import round_to_idx_tick
except ImportError:

    def round_to_idx_tick(price: float) -> int:
        """Fallback tick IDX; half-up (bukan banker's rounding)."""
        if price is None:
            return 0
        try:
            p = float(price)
        except (TypeError, ValueError):
            return 0
        if not math.isfinite(p) or p <= 0:
            return 0

        def _half_up(x: float) -> int:
            return int(math.floor(x + 0.5))

        if p < 200:
            return _half_up(p)
        if p < 500:
            return _half_up(p / 2.0) * 2
        if p < 2000:
            return _half_up(p / 5.0) * 5
        if p < 5000:
            return _half_up(p / 10.0) * 10
        return _half_up(p / 25.0) * 25


DEFAULT_STRUCT_PARAMS = {
    "struct_lookback": 40,
    "bear_lookback_extra": 15,
    "market_max_dist_atr": 0.75,
    "market_max_dist_pct": 1.2,
    "limit_expiry_bars": 5,
    "min_bear_body_pct": 0.15,
    "min_bear_body_abs_pct": 0.15,
    "res_tol_pct": 0.002,
    "res_floor_pct": 0.97,
    "res_ceil_pct": 1.01,
    "max_limit_dist_pct": 8.0,
}


def _safe_float(x, default: Optional[float] = None) -> Optional[float]:
    try:
        if x is None:
            return default
        v = float(x)
        if not math.isfinite(v):
            return default
        return v
    except (TypeError, ValueError):
        return default


def _tick_step(price: float) -> int:
    p = float(price)
    if p < 200:
        return 1
    if p < 500:
        return 2
    if p < 2000:
        return 5
    if p < 5000:
        return 10
    return 25


def floor_to_idx_tick(price: float) -> int:
    """Pembulatan ke bawah ke tick IDX (aman untuk stop loss)."""
    p = _safe_float(price)
    if p is None or p <= 0:
        return 0
    t = int(round_to_idx_tick(p))
    if t <= p:
        return t
    step = _tick_step(p)
    return max(0, t - step)


def _has_ohlc(df: pd.DataFrame) -> bool:
    if df is None or not isinstance(df, pd.DataFrame) or df.empty:
        return False
    need = {"Open", "High", "Low", "Close"}
    return need.issubset(set(df.columns))


def _is_bear(
    row: pd.Series,
    min_body_pct: float = 0.15,
    min_body_abs_pct: float = 0.15,
) -> bool:
    """Bear sejati: close < open AND body cukup vs range AND vs open."""
    o = _safe_float(row.get("Open") if hasattr(row, "get") else row["Open"])
    c = _safe_float(row.get("Close") if hasattr(row, "get") else row["Close"])
    h = _safe_float(row.get("High") if hasattr(row, "get") else row["High"])
    l = _safe_float(row.get("Low") if hasattr(row, "get") else row["Low"])
    if o is None or c is None or h is None or l is None:
        return False
    if c >= o:
        return False
    body = o - c
    if o > 0 and (body / o) * 100 < min_body_abs_pct:
        return False
    rng = h - l
    if rng <= 0:
        return body > 0
    return (body / rng) >= min_body_pct


def find_broken_high_candle_idx(
    df: pd.DataFrame,
    resistance: float,
    *,
    end_idx: Optional[int] = None,
    lookback: int = 40,
    res_tol_pct: float = 0.002,
    res_floor_pct: float = 0.97,
    res_ceil_pct: float = 1.01,
) -> Optional[int]:
    """
    Index candle swing yang memuat High ≈ resistance, **sebelum** bar
    pertama yang close menembus resistance + toleransi (bukan candle breakout).
    """
    if not _has_ohlc(df):
        return None
    res = _safe_float(resistance)
    if res is None or res <= 0:
        return None

    n = len(df)
    end = n - 1 if end_idx is None else min(int(end_idx), n - 1)
    start = max(0, end - lookback)
    tol = max(res * res_tol_pct, 1.0)

    # Breakout: close menembus resistance + tol (sama tol High — hindari false break)
    break_i: Optional[int] = None
    for i in range(start, end + 1):
        cl = _safe_float(df["Close"].iloc[i])
        if cl is not None and cl > res + tol:
            break_i = i
            break

    # search_end = bar sebelum breakout; tanpa syarat break_i > start
    search_end = (break_i - 1) if break_i is not None else end
    if search_end < start:
        return None

    # Harga sudah di atas res sepanjang window (tidak ada pre-break) → tidak ada struktur
    if break_i is None:
        closes = [_safe_float(df["Close"].iloc[i]) for i in range(start, end + 1)]
        closes = [c for c in closes if c is not None]
        if closes and all(c > res for c in closes):
            return None

    # Di antara High dalam toleransi: utamakan High tertinggi (swing resistance),
    # lalu bar lebih dulu — hindari retest belakangan yang "lebih dekat" ke angka res.
    candidates_tol: list[tuple[int, float, float]] = []  # i, hi, dist
    for i in range(start, search_end + 1):
        hi = _safe_float(df["High"].iloc[i])
        if hi is None:
            continue
        dist = abs(hi - res)
        if dist <= tol:
            candidates_tol.append((i, hi, dist))
    if candidates_tol:
        candidates_tol.sort(key=lambda x: (-x[1], x[0]))
        return candidates_tol[0][0]

    floor = res * res_floor_pct
    ceil = res * res_ceil_pct
    candidates: list[tuple[int, float]] = []
    for i in range(start, search_end + 1):
        hi = _safe_float(df["High"].iloc[i])
        if hi is None:
            continue
        if floor <= hi <= ceil:
            candidates.append((i, hi))
    if not candidates:
        return None
    candidates.sort(key=lambda x: (-x[1], x[0]))
    return candidates[0][0]


def find_last_bear_high_before(
    df: pd.DataFrame,
    high_candle_idx: int,
    *,
    extra_lookback: int = 15,
    min_body_pct: float = 0.15,
    min_body_abs_pct: float = 0.15,
) -> Optional[dict[str, Any]]:
    if not _has_ohlc(df) or high_candle_idx is None or high_candle_idx <= 0:
        return None
    start = max(0, int(high_candle_idx) - int(extra_lookback))
    for i in range(int(high_candle_idx) - 1, start - 1, -1):
        row = df.iloc[i]
        if _is_bear(row, min_body_pct=min_body_pct, min_body_abs_pct=min_body_abs_pct):
            hi = _safe_float(row["High"])
            if hi is None or hi <= 0:
                continue
            try:
                d = str(df.index[i].date())
            except Exception:
                d = str(df.index[i])
            return {
                "idx": i,
                "high": hi,
                "low": _safe_float(row["Low"]),
                "open": _safe_float(row["Open"]),
                "close": _safe_float(row["Close"]),
                "date": d,
            }
    return None


def compute_structural_entry(
    df: pd.DataFrame,
    resistance: float,
    last_close: float,
    *,
    atr: float = 0.0,
    default_entry: Optional[float] = None,
    params: Optional[dict] = None,
) -> dict[str, Any]:
    """
    Hitung EntryStruct + mode hybrid.

    Distansi diukur ke last_close.
    MARKET memakai max(close, default_entry) jika default = buy-stop di atas close.
    Struct terlalu jauh → TOO_EXTENDED (jangan chase).
    """
    p = {**DEFAULT_STRUCT_PARAMS, **(params or {})}

    close_px = _safe_float(last_close)
    def_entry = _safe_float(default_entry)
    res = _safe_float(resistance)
    atr_v = _safe_float(atr, 0.0) or 0.0

    if close_px is None or close_px <= 0:
        close_px = def_entry if def_entry and def_entry > 0 else None
    if close_px is None or close_px <= 0:
        return {
            "entry": 0.0,
            "entry_struct": None,
            "entry_close": 0.0,
            "entry_default": 0.0,
            "entry_mode": "FALLBACK",
            "dist_pct": None,
            "dist_atr": None,
            "broken_high_idx": None,
            "bear_idx": None,
            "bear_date": None,
            "expiry_bars": int(p["limit_expiry_bars"]),
            "note": "last_close invalid — FALLBACK",
            "struct_found": False,
            "valid": False,
        }

    fallback_entry = float(
        round_to_idx_tick(def_entry if def_entry and def_entry > 0 else close_px)
    )
    # MARKET: jika plan buy-stop di atas close, hormati level itu
    if def_entry is not None and def_entry > close_px:
        market_entry = float(round_to_idx_tick(def_entry))
    else:
        market_entry = float(round_to_idx_tick(close_px))

    out: dict[str, Any] = {
        "entry": fallback_entry,
        "entry_struct": None,
        "entry_close": float(round_to_idx_tick(close_px)),
        "entry_default": fallback_entry,
        "entry_mode": "FALLBACK",
        "dist_pct": None,
        "dist_atr": None,
        "broken_high_idx": None,
        "bear_idx": None,
        "bear_date": None,
        "expiry_bars": int(p["limit_expiry_bars"]),
        "note": "Struktur tidak ditemukan — pakai entry default",
        "struct_found": False,
        "valid": True,  # FALLBACK ke plan default tetap bisa dieksekusi
    }

    if not _has_ohlc(df):
        out["note"] = "OHLC tidak lengkap — FALLBACK"
        out["valid"] = True
        return out

    if res is None or res <= 0:
        out["note"] = "Resistance invalid — FALLBACK"
        out["valid"] = True  # tetap boleh pakai default plan
        out["struct_found"] = False
        return out

    hi_idx = find_broken_high_candle_idx(
        df,
        res,
        lookback=int(p["struct_lookback"]),
        res_tol_pct=float(p["res_tol_pct"]),
        res_floor_pct=float(p["res_floor_pct"]),
        res_ceil_pct=float(p["res_ceil_pct"]),
    )
    if hi_idx is None:
        return out

    bear = find_last_bear_high_before(
        df,
        hi_idx,
        extra_lookback=int(p["bear_lookback_extra"]),
        min_body_pct=float(p["min_bear_body_pct"]),
        min_body_abs_pct=float(p["min_bear_body_abs_pct"]),
    )
    if bear is None:
        out["broken_high_idx"] = hi_idx
        out["note"] = "Tidak ada candle bear pra-high — FALLBACK"
        return out

    struct = float(round_to_idx_tick(bear["high"]))
    out["entry_struct"] = struct
    out["broken_high_idx"] = hi_idx
    out["bear_idx"] = bear["idx"]
    out["bear_date"] = bear.get("date")
    out["struct_found"] = True

    # Struct di atas / sama market → tidak ada retest
    if struct >= close_px:
        out["entry"] = market_entry
        out["entry_mode"] = "MARKET"
        out["dist_pct"] = None  # jangan 0.0 menyesatkan
        out["dist_atr"] = None
        out["note"] = (
            f"EntryStruct {struct} >= close {out['entry_close']} — tidak ada retest; MARKET"
        )
        return out

    dist_pct = (close_px - struct) / close_px * 100.0
    dist_atr = (close_px - struct) / atr_v if atr_v > 0 else None
    out["dist_pct"] = round(dist_pct, 2)
    out["dist_atr"] = round(dist_atr, 2) if dist_atr is not None else None

    max_lim = float(p["max_limit_dist_pct"])
    if dist_pct > max_lim:
        # Jangan chase — struktur sudah basi relatif ke harga
        out["entry"] = fallback_entry
        out["entry_mode"] = "TOO_EXTENDED"
        out["note"] = (
            f"EntryStruct {struct} terlalu jauh (Δ{dist_pct:.1f}% > {max_lim}%) — "
            f"TOO_EXTENDED; jangan chase (pakai default / skip limit)"
        )
        out["valid"] = True  # setup default masih boleh; flag mode jelas
        return out

    near_pct = dist_pct <= float(p["market_max_dist_pct"])
    near_atr = dist_atr is not None and dist_atr <= float(p["market_max_dist_atr"])

    if near_pct or near_atr:
        out["entry"] = market_entry
        out["entry_mode"] = "MARKET"
        out["note"] = (
            f"Dekat EntryStruct {struct} (Δ{dist_pct:.1f}%) — MARKET"
        )
    else:
        out["entry"] = struct
        out["entry_mode"] = "LIMIT"
        out["note"] = (
            f"LIMIT @ EntryStruct {struct} (bear {bear.get('date')}); "
            f"expiry {p['limit_expiry_bars']} bar; close {out['entry_close']:.0f} Δ{dist_pct:.1f}%"
        )

    return out


def apply_structural_entry_to_plan(
    plan: dict,
    df: pd.DataFrame,
    *,
    resistance: float,
    last_close: float,
    atr: float = 0.0,
    params: Optional[dict] = None,
) -> dict:
    """
    Patch plan V2-style.

    EntryValid = boleh dieksekusi (entry > SL, RR masuk akal path).
    StructFound = struktur berhasil diidentifikasi (terpisah dari validitas eksekusi).
    """
    params = params or {}
    orig_breakout = plan.get("EntryBreakout") or plan.get("Entry")
    default_entry = orig_breakout or last_close

    se = compute_structural_entry(
        df,
        resistance=resistance,
        last_close=last_close,
        atr=atr or 0.0,
        default_entry=_safe_float(default_entry),
        params=params,
    )

    stop_raw = _safe_float(plan.get("StopLoss"), 0.0) or 0.0
    stop_tick = float(floor_to_idx_tick(stop_raw)) if stop_raw > 0 else 0.0
    t1 = _safe_float(plan.get("Target1"), 0.0) or 0.0
    t2 = _safe_float(plan.get("Target2"), 0.0) or 0.0

    entry = _safe_float(se.get("entry"), 0.0) or 0.0
    mode = str(se.get("entry_mode") or "FALLBACK")
    note = str(se.get("note") or "")
    struct_found = bool(se.get("struct_found"))

    # LIMIT/entry di bawah atau = SL → fallback ke market/default
    if stop_tick > 0 and entry <= stop_tick:
        entry = (
            _safe_float(se.get("entry_close"))
            or _safe_float(default_entry)
            or entry
        )
        entry = float(round_to_idx_tick(entry))
        if entry <= stop_tick:
            # Masih invalid
            mode = "FALLBACK"
            note = f"Entry <= SL setelah fallback — invalid; " + note
            se = {
                **se,
                "entry": entry,
                "entry_mode": mode,
                "note": note,
                "valid": False,
            }
        else:
            mode = "FALLBACK"
            note = f"EntryStruct/limit <= SL → FALLBACK @ {entry:.0f}; " + note
            se = {
                **se,
                "entry": entry,
                "entry_mode": mode,
                "note": note,
                "valid": True,
            }

    if t1 > 0 and t1 <= entry:
        note = note + "; TP1 <= entry"

    plan = dict(plan)
    plan["EntryClose"] = se.get("entry_close")
    plan["EntryStruct"] = se.get("entry_struct")
    plan["EntryMode"] = mode
    plan["Entry"] = float(round_to_idx_tick(entry)) if entry > 0 else entry
    if orig_breakout is not None:
        plan["EntryBreakout"] = orig_breakout
    plan["EntryDefault"] = se.get("entry_default")
    plan["EntryStructNote"] = note
    plan["EntryExpiryBars"] = se.get("expiry_bars")
    plan["DistEntryStructPct"] = se.get("dist_pct")
    plan["StructFound"] = struct_found

    entry_f = float(plan["Entry"])
    executable = stop_tick <= 0 or entry_f > stop_tick
    plan["EntryValid"] = executable and bool(se.get("valid", True))

    do_sizing = ("SuggestedLots" in plan) or ("account_size" in params)

    if executable and entry_f > stop_tick and stop_tick > 0:
        risk = entry_f - stop_tick
        plan["StopLoss"] = stop_tick
        plan["RiskPerShare"] = round(risk, 2)

        min_rr = float(
            params.get(
                "min_risk_reward",
                params.get("min_rr", plan.get("_min_rr_hint", 1.5)),
            )
        )
        if t1 > entry_f:
            plan["RR_Ratio"] = round((t1 - entry_f) / risk, 2)
            if "LayakRR" in plan or "min_risk_reward" in params or "min_rr" in params:
                plan["LayakRR"] = plan["RR_Ratio"] >= min_rr
        else:
            plan["RR_Ratio"] = 0.0
            if "LayakRR" in plan:
                plan["LayakRR"] = False

        if t2 > entry_f:
            plan["RR_TP2"] = round((t2 - entry_f) / risk, 2)
        elif "RR_TP2" in plan:
            plan["RR_TP2"] = 0.0

        if do_sizing:
            account = float(params.get("account_size", 5_000_000))
            risk_pct = float(params.get("risk_per_trade_pct", 1.0))
            lot_size = int(params.get("lot_size", 100))
            risk_rp = account * (risk_pct / 100.0)
            lots = int((risk_rp / risk) // lot_size) if risk > 0 else 0
            shares = lots * lot_size
            if shares > 0 and entry_f * shares > account:
                shares = int(account // entry_f)
                shares = (shares // lot_size) * lot_size
                lots = shares // lot_size
            plan["SuggestedLots"] = lots
            plan["SuggestedShares"] = shares
            plan["EstCapitalUsed(Rp)"] = round(shares * entry_f, 0)
            plan["EstLoss(Rp)"] = round(shares * risk, 0)
            plan["EstProfit1(Rp)"] = (
                round(shares * (t1 - entry_f), 0) if t1 > entry_f else 0
            )
            plan["EstProfit2(Rp)"] = (
                round(shares * (t2 - entry_f), 0) if t2 > entry_f else 0
            )
            if lots <= 0:
                plan["EntryStructNote"] = (note + "; lots=0 (risk/modal)").strip("; ")
    else:
        # Tidak executable: nolkan sizing hanya jika field sudah ada / sizing aktif
        plan["RiskPerShare"] = 0.0
        plan["RR_Ratio"] = 0.0
        if "RR_TP2" in plan:
            plan["RR_TP2"] = 0.0
        if "LayakRR" in plan:
            plan["LayakRR"] = False
        if do_sizing or "SuggestedLots" in plan:
            plan["SuggestedLots"] = 0
            plan["SuggestedShares"] = 0
            plan["EstCapitalUsed(Rp)"] = 0
            plan["EstLoss(Rp)"] = 0
            plan["EstProfit1(Rp)"] = 0
            plan["EstProfit2(Rp)"] = 0
        if stop_tick > 0:
            plan["StopLoss"] = stop_tick
        plan["EntryValid"] = False

    return plan
