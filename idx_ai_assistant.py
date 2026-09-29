"""
IDX AI TRADER ASSISTANT
============================================================
Multi-role agent yang menganalisa ticker hasil screener.

Roles:
  1. technical   — struktur, entry/SL/TP, RR, sinyal palsu
  2. fundamental — valuasi, sektor, kesehatan laporan (dari data yg ada)
  3. risk        — size, fee/pajak, jarak SL, likuiditas
  4. critic      — alasan GAGAL / invalidation
  5. chief       — gabungan: skor 1-10, bias, checklist

Provider: OpenAI-compatible API (urllib; bypass httpx bila bermasalah)
  Env:
    OPENAI_API_KEY atau XAI_API_KEY
    OPENAI_BASE_URL  (default https://api.openai.com/v1)
    OPENAI_MODEL     (default gpt-4o-mini)
    OPENAI_TIMEOUT   (default 180)
    OPENAI_MAX_RETRIES (default 5)
    OPENAI_ROLE_GAP_SEC (default 2.5, jeda antar role)
"""

from __future__ import annotations

import json
import os
import glob
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd


def _load_dotenv_files() -> list[str]:
    loaded: list[str] = []
    candidates: list[Path] = []
    try:
        here = Path(__file__).resolve().parent
        candidates.append(here / ".env")
        candidates.append(here.parent / ".env")
    except Exception:
        pass
    candidates.append(Path.cwd() / ".env")

    seen: set[str] = set()
    paths: list[Path] = []
    for p in candidates:
        try:
            key = str(p.resolve())
        except Exception:
            key = str(p)
        if key in seen:
            continue
        seen.add(key)
        paths.append(p)

    try:
        from dotenv import load_dotenv

        for p in paths:
            if p.is_file():
                load_dotenv(p, override=False)
                loaded.append(str(p))
        if loaded:
            return loaded
    except ImportError:
        pass

    for p in paths:
        if not p.is_file():
            continue
        try:
            for raw in p.read_text(encoding="utf-8").splitlines():
                line = raw.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                if line.startswith("export "):
                    line = line[7:].strip()
                key, _, val = line.partition("=")
                key = key.strip()
                val = val.strip().strip("'").strip('"')
                if key and key not in os.environ:
                    os.environ[key] = val
            loaded.append(str(p))
        except Exception:
            continue
    return loaded


_DOTENV_LOADED = _load_dotenv_files()


def _default_model() -> str:
    return (
        os.environ.get("OPENAI_MODEL")
        or os.environ.get("XAI_MODEL")
        or "gpt-4o-mini"
    )


def _default_base_url() -> str:
    return (
        os.environ.get("OPENAI_BASE_URL")
        or os.environ.get("XAI_BASE_URL")
        or "https://api.openai.com/v1"
    )


DEFAULT_MODEL = _default_model()
DEFAULT_BASE_URL = _default_base_url()

ROLE_ORDER = ["technical", "fundamental", "risk", "critic", "chief"]

ROLE_MAX_TOKENS = {
    "technical": 700,
    "fundamental": 650,
    "risk": 550,
    "critic": 750,
    "chief": 1000,
}

# Jeda antar role (detik) — bantu TPM limit tier rendah (30k TPM)
ROLE_GAP_SEC = float(os.environ.get("OPENAI_ROLE_GAP_SEC", "2.5"))


# =========================================================================
# DETERMINISTIC ANALYSIS ENGINE (FACT → SIGNAL → SCORE → STANCE)
# Berlaku untuk semua strategy screener. AI menjelaskan; engine menghitung.
# =========================================================================
ENGINE_PARAMS = {
    "fib_at_level_atr": 0.10,
    "fib_near_level_atr": 0.30,
    "fib_away_atr": 0.75,
    "pe_low": 10.0,
    "pe_high": 25.0,
    "pb_low": 1.0,
    "pb_high": 3.0,
    "roe_weak": 8.0,
    "roe_strong": 20.0,
    "rr_tp1_weight": 0.5,
    "rr_tp2_weight": 0.5,
    "score_tech_w": 0.45,
    "score_fund_w": 0.20,
    "score_risk_w": 0.18,
    "score_market_w": 0.07,
    "score_catalyst_w": 0.10,
}


def _fnum(x, default=None):
    try:
        if x is None or (isinstance(x, float) and x != x):
            return default
        return float(x)
    except Exception:
        return default


def _row_get(row: dict, *keys, default=None):
    for k in keys:
        if k in row and row[k] is not None and str(row[k]).strip() != "":
            return row[k]
    return default


def build_facts(context: dict) -> dict[str, Any]:
    """Layer FACT — angka mentah, tanpa interpretasi."""
    row = context.get("screener_row") or {}
    ma = context.get("ma_structure") or {}
    ma_vals = ma.get("ma") or {}
    fund = context.get("canonical_valuation") or context.get("fundamental") or {}
    regime = context.get("market_regime") or context.get("regime") or {}
    acc = context.get("account") or {}

    close = _fnum(_row_get(row, "Close", "Entry", default=ma.get("last_close")))
    entry = _fnum(_row_get(row, "Entry", "Close"), close)
    sl = _fnum(_row_get(row, "StopLoss"))
    tp1 = _fnum(
        _row_get(
            row,
            "Target1",
            "Target1(Peak)",
            "Target(Liquidity)",
            "Target",
            "Target(Peak)",
        )
    )
    tp2 = _fnum(_row_get(row, "Target2(Ext)", "Target2", "Target(Ext)"))
    atr = _fnum(_row_get(row, "ATR", "ATR14"))
    fib382 = _fnum(_row_get(row, "Fibo382", "Fib382"))
    fib618 = _fnum(_row_get(row, "Fibo618", "Fib618"))
    adx = _fnum(_row_get(row, "ADX"))
    roc = _fnum(_row_get(row, "ROC(10)", "ROC10", "ROC"))
    score_scr = _fnum(_row_get(row, "Score"))
    rr_scr = _fnum(_row_get(row, "RR_Ratio", "RR"))
    lots = _fnum(_row_get(row, "Lots"), 0) or 0
    retest_raw = _row_get(row, "RetestOK", "Retest", "Reversal")
    alasan = str(_row_get(row, "Alasan", default="") or "")

    risk_ps = None
    if entry and sl and entry > sl:
        risk_ps = entry - sl
    rr_tp1 = (tp1 - entry) / risk_ps if risk_ps and tp1 and entry else rr_scr
    rr_tp2 = (tp2 - entry) / risk_ps if risk_ps and tp2 and entry else None

    acct = _fnum(acc.get("size_rp"), 50_000_000) or 50_000_000
    risk_pct_cfg = _fnum(acc.get("risk_per_trade_pct"), 1.0) or 1.0
    account_risk_pct = None
    if lots and risk_ps and acct > 0:
        account_risk_pct = (lots * 100 * risk_ps) / acct * 100

    # Regime: normalize UNKNOWN if empty / missing key regime
    reg_label = regime.get("regime") if isinstance(regime, dict) else None
    if not reg_label or str(reg_label).upper() in ("", "NONE", "NULL"):
        reg_label = "UNKNOWN"
    else:
        reg_label = str(reg_label).upper()

    facts_base = {
        "ticker": context.get("ticker"),
        "screener_version": context.get("screener_version"),
        "close": close,
        "entry": entry,
        "sl": sl,
        "tp1": tp1,
        "tp2": tp2,
        "atr": atr,
        "fib382": fib382,
        "fib618": fib618,
        "adx": adx,
        "roc": roc,
        "screener_score": score_scr,
        "rr_screener": rr_scr,
        "rr_tp1": round(rr_tp1, 2) if rr_tp1 is not None else None,
        "rr_tp2": round(rr_tp2, 2) if rr_tp2 is not None else None,
        "lots": lots,
        "account_risk_pct": round(account_risk_pct, 3) if account_risk_pct is not None else None,
        "risk_pct_cfg": risk_pct_cfg,
        "retest_raw": retest_raw,
        "alasan": alasan,
        "ma20": _fnum(ma_vals.get("MA20")),
        "ma50": _fnum(ma_vals.get("MA50")),
        "ma100": _fnum(ma_vals.get("MA100")),
        "ma200": _fnum(ma_vals.get("MA200")),
        "ma_available": bool(ma.get("available")),
        "ma_stack_hint": (ma.get("stack") or ma.get("interpretation_hints") or None),
        "pe": _fnum(fund.get("pe_ratio")),
        "pb": _fnum(fund.get("pb_ratio")),
        "roe": _fnum(fund.get("roe")),
        "div_yield": _fnum(fund.get("dividend_yield_pct")),
        "fund_source": fund.get("source"),
        "fund_confidence": fund.get("confidence"),
        "ihsg_regime_raw": reg_label,
        "ihsg_last_close": _fnum(regime.get("last_close")) if isinstance(regime, dict) else None,
    }
    # Rezim emiten (terpisah dari IHSG)
    tr = derive_ticker_regime(facts_base, ma)
    facts_base.update(tr)
    # pre-computed object for prompts / context
    facts_base["ticker_regime_detail"] = {
        "regime": tr.get("ticker_regime"),
        "strength": tr.get("ticker_regime_strength"),
        "summary": tr.get("ticker_regime_summary"),
        "notes": tr.get("ticker_regime_notes"),
        "above_ma50": tr.get("above_ma50"),
        "above_ma200": tr.get("above_ma200"),
        "ma20_50_cross": tr.get("ma20_50_cross"),
        "ma20_50_cross_days": tr.get("ma20_50_cross_days"),
        "_note": (
            "Rezim EMITEN dari MA/ADX/ROC. "
            "Jangan samakan dengan regime IHSG (market_regime)."
        ),
    }
    return facts_base


def _classify_fib_distance(close, level, atr, p=ENGINE_PARAMS) -> tuple[str | None, float | None]:
    if close is None or level is None or not atr or atr <= 0:
        return None, None
    dist = abs(close - level)
    dist_atr = dist / atr
    if dist_atr <= p["fib_at_level_atr"]:
        state = "AT_LEVEL"
    elif dist_atr <= p["fib_near_level_atr"]:
        state = "NEAR_LEVEL"
    elif dist_atr <= p["fib_away_atr"]:
        state = "AWAY"
    else:
        state = "FAR"
    return state, round(dist_atr, 3)


def _retest_status(facts: dict) -> str:
    """
    NO_RETEST | VALID_RETEST | FAILED_RETEST | UNKNOWN
    NO_RETEST ≠ gagal; tidak ada penalti otomatis.
    """
    raw = facts.get("retest_raw")
    alasan = (facts.get("alasan") or "").lower()
    if raw is not None:
        s = str(raw).strip().lower()
        if s in ("ya", "yes", "true", "1", "valid", "ok"):
            return "VALID_RETEST"
        if s in ("tidak", "no", "false", "0", "failed", "gagal"):
            # V3 "RetestOK=Tidak" bisa berarti belum/ gagal — bedakan via alasan
            if any(x in alasan for x in ("gagal", "fail", "breakdown", "invalid")):
                return "FAILED_RETEST"
            if any(x in alasan for x in ("retest hold", "retest", "fibo golden", "fibo zone")):
                return "VALID_RETEST"
            return "NO_RETEST"
    if any(x in alasan for x in ("retest hold", "retest ok", "fibo golden", "fibo zone")):
        return "VALID_RETEST"
    if "retest gagal" in alasan or "failed retest" in alasan:
        return "FAILED_RETEST"
    # V2/V4/V5 sering tanpa field retest → UNKNOWN (bukan FAILED)
    ver = str(facts.get("screener_version") or "").lower()
    if "v3" in ver:
        return "NO_RETEST"
    return "UNKNOWN"


def _trend_from_ma(facts: dict) -> str:
    c, m20, m50 = facts.get("close"), facts.get("ma20"), facts.get("ma50")
    m100, m200 = facts.get("ma100"), facts.get("ma200")
    if not facts.get("ma_available") or c is None or m20 is None:
        return "UNKNOWN"
    above = sum(
        1
        for m in (m20, m50, m100, m200)
        if m is not None and c > m
    )
    below = sum(
        1
        for m in (m20, m50, m100, m200)
        if m is not None and c < m
    )
    if above >= 3 and (m50 is None or m20 >= m50 * 0.995):
        return "BULLISH"
    if below >= 3:
        return "BEARISH"
    return "NEUTRAL"



def derive_ticker_regime(facts: dict, ma_structure: dict | None = None) -> dict:
    """
    Rezim **emiten** (bukan IHSG): dari MA stack, posisi harga, cross, ADX/ROC screener.
    Output dipakai engine + prompt — dipisah tegas dari market_regime IHSG.
    """
    ma = ma_structure or {}
    c = facts.get("close")
    m20, m50 = facts.get("ma20"), facts.get("ma50")
    m100, m200 = facts.get("ma100"), facts.get("ma200")
    adx = facts.get("adx")
    roc = facts.get("roc")

    label = "UNKNOWN"
    strength = "UNKNOWN"
    notes: list[str] = []

    trend = _trend_from_ma(facts)
    if trend == "BULLISH":
        label = "BULLISH"
    elif trend == "BEARISH":
        label = "BEARISH"
    elif trend == "NEUTRAL":
        label = "NEUTRAL"

    # Perkuat dari stack MA di ma_structure
    stack = ma.get("stack") or []
    stack_txt = " ".join(str(x) for x in stack).lower()
    if "bullish" in stack_txt and label != "BEARISH":
        label = "BULLISH"
        notes.append("Susunan MA bullish")
    elif "bearish" in stack_txt and label != "BULLISH":
        label = "BEARISH"
        notes.append("Susunan MA bearish")
    elif "campur" in stack_txt or "rotasi" in stack_txt:
        if label == "UNKNOWN":
            label = "NEUTRAL"
        notes.append("Susunan MA campur/rotasi")

    # Cross freshness
    crosses = ma.get("crosses") or {}
    cx2050 = crosses.get("MA20_vs_MA50") or {}
    direction = (cx2050.get("direction_last_cross") or "").lower()
    days = cx2050.get("days_since_cross")
    if direction == "golden":
        notes.append(
            f"Golden MA20/50"
            + (f" ({days}d lalu)" if days is not None else "")
        )
        if label == "NEUTRAL":
            label = "BULLISH"
    elif direction == "death":
        notes.append(
            f"Death MA20/50"
            + (f" ({days}d lalu)" if days is not None else "")
        )
        if label == "NEUTRAL":
            label = "BEARISH"

    # Strength dari ADX + posisi vs MA50/200
    above_50 = m50 is not None and c is not None and c > m50
    above_200 = m200 is not None and c is not None and c > m200
    if adx is not None and adx >= 25 and label in ("BULLISH", "BEARISH"):
        strength = "STRONG"
        notes.append(f"ADX kuat ({adx})")
    elif adx is not None and adx >= 18:
        strength = "MODERATE"
        notes.append(f"ADX moderat ({adx})")
    elif adx is not None:
        strength = "WEAK"
        notes.append(f"ADX lemah ({adx})")
    else:
        strength = "UNKNOWN"

    if roc is not None:
        if roc > 2:
            notes.append(f"ROC+ ({roc})")
        elif roc < -2:
            notes.append(f"ROC- ({roc})")

    if above_50 and above_200 and label == "BULLISH":
        notes.append("Harga di atas MA50 & MA200")
    elif (not above_50) and m50 is not None and label == "BEARISH":
        notes.append("Harga di bawah MA50")

    # Align with IHSG later in signals
    summary = f"{label}" + (f"/{strength}" if strength != "UNKNOWN" else "")
    if notes:
        summary += " — " + "; ".join(notes[:4])

    return {
        "ticker_regime": label,
        "ticker_regime_strength": strength,
        "ticker_regime_notes": notes[:6],
        "ticker_regime_summary": summary,
        "above_ma50": above_50 if m50 is not None else None,
        "above_ma200": above_200 if m200 is not None else None,
        "ma20_50_cross": direction or None,
        "ma20_50_cross_days": days,
    }


def detect_emiten_regime(
    ticker: str,
    *,
    screener_row: dict | None = None,
    ma_structure: dict | None = None,
    ihsg_regime: dict | str | None = None,
    fetch_ma: bool = True,
) -> dict:
    """
    Helper publik: deteksi rezim **emiten** SEBELUM agent AI.

    Dipakai di build_context agar model tidak perlu menghitung MA/cross sendiri —
    cukup baca field hasil helper ini.

    Parameters
    ----------
    ticker : str
    screener_row : optional dict dari report (ADX, ROC, Close, …)
    ma_structure : optional hasil fetch_ma_structure; jika None & fetch_ma=True → diunduh
    ihsg_regime : dict regime master (key regime) atau string BULLISH/BEARISH/…
    fetch_ma : bool

    Returns
    -------
    dict siap tempel ke context["ticker_regime"]:
        regime, strength, summary, notes, alignment_with_ihsg, ma (ringkas), …
    """
    t = str(ticker or "").upper().strip().replace(".JK", "")
    row = screener_row or {}
    ma = ma_structure
    if ma is None and fetch_ma and t:
        try:
            ma = fetch_ma_structure(t)
        except Exception as e:
            ma = {"available": False, "error": str(e)}
    ma = ma or {}

    ma_vals = ma.get("ma") or {}
    close = _fnum(
        _row_get(row, "Close", "Entry", default=ma.get("last_close"))
    )
    facts = {
        "close": close,
        "ma20": _fnum(ma_vals.get("MA20")),
        "ma50": _fnum(ma_vals.get("MA50")),
        "ma100": _fnum(ma_vals.get("MA100")),
        "ma200": _fnum(ma_vals.get("MA200")),
        "ma_available": bool(ma.get("available")),
        "adx": _fnum(_row_get(row, "ADX")),
        "roc": _fnum(_row_get(row, "ROC(10)", "ROC10", "ROC")),
    }
    tr = derive_ticker_regime(facts, ma)

    # IHSG label
    if isinstance(ihsg_regime, dict):
        ihsg_label = _normalize_market_regime(
            str(ihsg_regime.get("regime") or ihsg_regime.get("label") or "UNKNOWN")
        )
    else:
        ihsg_label = _normalize_market_regime(str(ihsg_regime or "UNKNOWN"))

    emiten = tr.get("ticker_regime") or "UNKNOWN"
    if emiten in ("BULLISH", "BEARISH") and ihsg_label == emiten:
        align = "ALIGNED"
        align_note = "Rezim emiten searah IHSG — tailwind konteks."
    elif emiten in ("BULLISH", "BEARISH") and ihsg_label in ("BULLISH", "BEARISH"):
        align = "CONFLICT"
        align_note = "Rezim emiten berlawanan IHSG — size lebih hati-hati."
    else:
        align = "NEUTRAL"
        align_note = "Alignment netral (salah satu UNKNOWN/NEUTRAL)."

    return {
        "ticker": t,
        "regime": emiten,
        "strength": tr.get("ticker_regime_strength") or "UNKNOWN",
        "summary": tr.get("ticker_regime_summary") or emiten,
        "notes": tr.get("ticker_regime_notes") or [],
        "above_ma50": tr.get("above_ma50"),
        "above_ma200": tr.get("above_ma200"),
        "ma20_50_cross": tr.get("ma20_50_cross"),
        "ma20_50_cross_days": tr.get("ma20_50_cross_days"),
        "ihsg_regime": ihsg_label,
        "alignment_with_ihsg": align,
        "alignment_note": align_note,
        "ma_available": bool(ma.get("available")),
        "ma_stack": ma.get("stack"),
        "ma_values": ma_vals if ma_vals else None,
        "source": "detect_emiten_regime",
        "_note": (
            "Rezim EMITEN deterministik (MA/ADX/ROC). "
            "Agent AI jangan menghitung ulang — pakai field ini apa adanya. "
            "Terpisah dari market_regime = IHSG."
        ),
    }


def _pe_class(pe, p=ENGINE_PARAMS) -> str:
    if pe is None:
        return "UNKNOWN"
    if pe < p["pe_low"]:
        return "LOW"
    if pe <= p["pe_high"]:
        return "MODERATE"
    return "HIGH"


def _pb_class(pb, p=ENGINE_PARAMS) -> str:
    if pb is None:
        return "UNKNOWN"
    if pb < p["pb_low"]:
        return "LOW"
    if pb <= p["pb_high"]:
        return "MODERATE"
    return "HIGH"


def _roe_class(roe, p=ENGINE_PARAMS) -> str:
    if roe is None:
        return "UNKNOWN"
    if roe < p["roe_weak"]:
        return "WEAK"
    if roe < p["roe_strong"]:
        return "MODERATE"
    return "STRONG"


def _normalize_market_regime(label: str) -> str:
    u = (label or "UNKNOWN").upper()
    if u in ("UNKNOWN", ""):
        return "UNKNOWN"
    if "BULL" in u:
        return "BULLISH"
    if "BEAR" in u:
        return "BEARISH"
    if "NEUTRAL" in u or "SIDE" in u or "RANGE" in u:
        return "NEUTRAL"
    return "UNKNOWN"


def build_signals(facts: dict, p: dict | None = None) -> dict[str, Any]:
    """Layer SIGNAL — deterministic dari FACT."""
    p = p or ENGINE_PARAMS
    trend = _trend_from_ma(facts)
    roc = facts.get("roc")
    adx = facts.get("adx")
    if roc is not None and roc > 0:
        momentum = "POSITIVE"
    elif roc is not None and roc < 0:
        momentum = "NEGATIVE"
    else:
        momentum = "UNKNOWN"

    fib_state, fib_dist_atr = _classify_fib_distance(
        facts.get("close"), facts.get("fib382"), facts.get("atr"), p
    )
    fib_ref = "fib382"
    if fib_state is None and facts.get("fib618") is not None:
        fib_state, fib_dist_atr = _classify_fib_distance(
            facts.get("close"), facts.get("fib618"), facts.get("atr"), p
        )
        fib_ref = "fib618"

    # Interpretasi Fibo: AT_LEVEL ≠ resistance rejection
    if fib_state == "AT_LEVEL":
        fib_note = (
            f"Harga praktis di level ({fib_ref}); jarak {fib_dist_atr}×ATR. "
            "BUKAN otomatis resistance/weakness."
        )
    elif fib_state == "NEAR_LEVEL":
        fib_note = f"Dekat {fib_ref} ({fib_dist_atr}×ATR); level masih relevan."
    elif fib_state == "AWAY":
        fib_note = f"Menjauh dari {fib_ref} ({fib_dist_atr}×ATR)."
    elif fib_state == "FAR":
        fib_note = f"Jauh dari {fib_ref}; level kurang relevan untuk setup immediate."
    else:
        fib_note = "Data Fibo/ATR tidak cukup untuk klasifikasi."

    retest = _retest_status(facts)
    retest_note = {
        "NO_RETEST": "Belum ada uji ulang level yang terkonfirmasi — netral, bukan gagal.",
        "VALID_RETEST": "Retest hold / konfirmasi positif.",
        "FAILED_RETEST": "Uji level gagal (breakdown) — negatif.",
        "UNKNOWN": "Field retest tidak ada di report strategi ini — jangan diasumsikan gagal.",
    }.get(retest, "")

    market = _normalize_market_regime(str(facts.get("ihsg_regime_raw") or "UNKNOWN"))
    market_note = (
        "IHSG belum/tidak tersedia → netral (bukan bearish)."
        if market == "UNKNOWN"
        else (
            "Headwind indeks (bukan death-cross emiten)."
            if market == "BEARISH"
            else (
                "Tailwind indeks."
                if market == "BULLISH"
                else "Indeks netral/sideways."
            )
        )
    )

    pe_c = _pe_class(facts.get("pe"), p)
    pb_c = _pb_class(facts.get("pb"), p)
    roe_c = _roe_class(facts.get("roe"), p)

    # Jangan pernah label OVERVALUED tanpa peer/historical di JSON
    if pb_c == "HIGH" and pe_c in ("MODERATE", "LOW", "UNKNOWN"):
        val_sig = "PBV_PREMIUM"
    elif pe_c == "HIGH" and pb_c == "HIGH":
        val_sig = "PREMIUM_BOTH"
    elif pe_c == "LOW" and pb_c in ("LOW", "MODERATE"):
        val_sig = "VALUE_LEAN"
    elif pe_c == "UNKNOWN" and pb_c == "UNKNOWN":
        val_sig = "UNKNOWN"
    else:
        val_sig = "MIXED"
    val_note = (
        f"PE={pe_c}, PBV={pb_c}, ROE={roe_c}. "
        f"Signal={val_sig}. Bukan OVERVALUED tanpa peer/historical."
    )

    w1, w2 = p["rr_tp1_weight"], p["rr_tp2_weight"]
    rr1, rr2 = facts.get("rr_tp1"), facts.get("rr_tp2")
    if rr1 is not None and rr2 is not None:
        expected_rr = round(w1 * rr1 + w2 * rr2, 2)
    elif rr1 is not None:
        expected_rr = rr1
    else:
        expected_rr = facts.get("rr_screener")

    # Risk breakdown
    atr = facts.get("atr")
    entry = facts.get("entry")
    sl = facts.get("sl")
    vol_risk = None
    if atr and entry and entry > 0:
        vol_risk = round(atr / entry * 100, 2)
    acct_risk = facts.get("account_risk_pct")

    return {
        "trend_signal": trend,
        "momentum_signal": momentum,
        "adx": adx,
        "fib_signal": fib_state or "UNKNOWN",
        "fib_ref": fib_ref,
        "fib_distance_atr": fib_dist_atr,
        "fib_note": fib_note,
        "retest_status": retest,
        "retest_note": retest_note,
        "market_regime": market,
        "market_note": market_note,
        "pe_class": pe_c,
        "pb_class": pb_c,
        "roe_class": roe_c,
        "valuation_signal": val_sig,
        "valuation_note": val_note,
        "forbid_overvalued_label": True,
        "rr_tp1": rr1,
        "rr_tp2": rr2,
        "expected_rr": expected_rr,
        "rr_headline": (
            f"TP1={rr1}R" + (f" | TP2={rr2}R" if rr2 is not None else "")
            + (f" | Expected={expected_rr}R" if expected_rr is not None else "")
            if rr1 is not None
            else None
        ),
        "account_risk_pct": acct_risk,
        "volatility_risk_pct": vol_risk,
        "risk_note": (
            f"Account risk={acct_risk}% vs cfg; volatility (ATR/price)={vol_risk}%. "
            "Critic fokus setup/market; Risk role fokus account+fee."
        ),
        # --- Ticker regime (emiten) vs IHSG ---
        "ticker_regime": facts.get("ticker_regime") or "UNKNOWN",
        "ticker_regime_strength": facts.get("ticker_regime_strength") or "UNKNOWN",
        "ticker_regime_summary": facts.get("ticker_regime_summary") or "",
        "ticker_regime_notes": facts.get("ticker_regime_notes") or [],
        "regime_alignment": (
            "ALIGNED"
            if (facts.get("ticker_regime") in ("BULLISH", "BEARISH")
                and market == facts.get("ticker_regime"))
            else (
                "CONFLICT"
                if (
                    facts.get("ticker_regime") in ("BULLISH", "BEARISH")
                    and market in ("BULLISH", "BEARISH")
                    and market != facts.get("ticker_regime")
                )
                else "NEUTRAL"
            )
        ),
        "regime_alignment_note": (
            "Rezim emiten searah IHSG — tailwind konteks."
            if (
                facts.get("ticker_regime") in ("BULLISH", "BEARISH")
                and market == facts.get("ticker_regime")
            )
            else (
                "Rezim emiten berlawanan IHSG — size lebih hati-hati / butuh konfirmasi."
                if (
                    facts.get("ticker_regime") in ("BULLISH", "BEARISH")
                    and market in ("BULLISH", "BEARISH")
                    and market != facts.get("ticker_regime")
                )
                else "Alignment netral (salah satu UNKNOWN/NEUTRAL)."
            )
        ),
    }


def build_scores(signals: dict, facts: dict, p: dict | None = None) -> dict[str, Any]:
    """Layer SCORE — angka 0–100 + breakdown."""
    p = p or ENGINE_PARAMS
    tech = 50.0
    if signals["trend_signal"] == "BULLISH":
        tech += 15
    elif signals["trend_signal"] == "BEARISH":
        tech -= 15
    if signals["momentum_signal"] == "POSITIVE":
        tech += 10
    elif signals["momentum_signal"] == "NEGATIVE":
        tech -= 10
    adx = signals.get("adx")
    if adx is not None and adx >= 25:
        tech += 8
    elif adx is not None and adx < 18:
        tech -= 5
    # Fibo: AT_LEVEL netral/positif ringan; FAR sedikit negatif untuk setup fib
    fs = signals.get("fib_signal")
    if fs == "AT_LEVEL":
        tech += 5
    elif fs == "NEAR_LEVEL":
        tech += 2
    elif fs == "FAR":
        tech -= 3
    # Retest: NO_RETEST = 0; VALID +; FAILED -
    rs = signals.get("retest_status")
    if rs == "VALID_RETEST":
        tech += 10
    elif rs == "FAILED_RETEST":
        tech -= 12
    # NO_RETEST / UNKNOWN → 0
    tech = max(0, min(100, tech))

    fund = 50.0
    if signals["pe_class"] == "LOW":
        fund += 8
    elif signals["pe_class"] == "HIGH":
        fund -= 5
    if signals["pb_class"] == "HIGH":
        fund -= 5  # premium, bukan overvalued crash
    elif signals["pb_class"] == "LOW":
        fund += 5
    if signals["roe_class"] == "STRONG":
        fund += 12
    elif signals["roe_class"] == "WEAK":
        fund -= 8
    if signals["valuation_signal"] == "PBV_PREMIUM":
        fund -= 3
    conf = str(facts.get("fund_confidence") or "").lower()
    if conf in ("low", "medium") or str(facts.get("fund_source") or "").lower() == "yfinance":
        fund = 40 + (fund - 50) * 0.5  # compress toward 40-60
    fund = max(0, min(100, fund))

    risk_s = 50.0
    arp = signals.get("account_risk_pct")
    cfg = facts.get("risk_pct_cfg") or 1.0
    if arp is not None:
        if arp <= cfg:
            risk_s += 15
        elif arp <= cfg * 1.5:
            risk_s += 0
        else:
            risk_s -= 15
    err = signals.get("expected_rr")
    if err is not None:
        if err >= 2.0:
            risk_s += 15
        elif err >= 1.5:
            risk_s += 8
        elif err < 1.2:
            risk_s -= 12
    risk_s = max(0, min(100, risk_s))

    # Market IHSG: UNKNOWN → 0 adjustment (score 50)
    mr = signals.get("market_regime")
    if mr == "BULLISH":
        mkt = 65.0
    elif mr == "BEARISH":
        mkt = 35.0
    elif mr == "NEUTRAL":
        mkt = 50.0
    else:
        mkt = 50.0  # UNKNOWN

    # Alignment emiten vs IHSG (penyesuaian ringan, bukan driver utama)
    align = signals.get("regime_alignment")
    if align == "ALIGNED":
        mkt = min(100.0, mkt + 5)
    elif align == "CONFLICT":
        mkt = max(0.0, mkt - 8)

    catalyst = 50.0  # news not scored hard without structured tone

    final = (
        p["score_tech_w"] * tech
        + p["score_fund_w"] * fund
        + p["score_risk_w"] * risk_s
        + p["score_market_w"] * mkt
        + p["score_catalyst_w"] * catalyst
    )
    final = round(max(0, min(100, final)), 1)

    return {
        "technical_score": round(tech, 1),
        "fundamental_score": round(fund, 1),
        "risk_score": round(risk_s, 1),
        "market_score": round(mkt, 1),
        "catalyst_score": round(catalyst, 1),
        "final_score": final,
        "weights": {
            "technical": p["score_tech_w"],
            "fundamental": p["score_fund_w"],
            "risk": p["score_risk_w"],
            "market": p["score_market_w"],
            "catalyst": p["score_catalyst_w"],
        },
    }


def derive_stance(scores: dict, signals: dict, facts: dict) -> str:
    """Deterministic stance rules."""
    if not facts.get("close") and not facts.get("entry"):
        return "INSUFFICIENT_DATA"
    final = scores.get("final_score") or 0
    risk_ok = (signals.get("account_risk_pct") is None) or (
        signals.get("account_risk_pct") <= (facts.get("risk_pct_cfg") or 1.0) * 1.25
    )
    if signals.get("retest_status") == "FAILED_RETEST" and final < 70:
        return "AVOID"
    if final >= 80 and risk_ok:
        return "STRONG_SETUP"
    if final >= 70 and risk_ok:
        return "SETUP"
    if final >= 55:
        return "WATCH"
    if final >= 40:
        return "WAIT"
    return "AVOID"


def data_quality_report(facts: dict, signals: dict) -> dict[str, str]:
    return {
        "technical": "COMPLETE" if facts.get("ma_available") or facts.get("close") else "PARTIAL",
        "fundamental": (
            "COMPLETE"
            if facts.get("fund_confidence") == "high"
            else ("PARTIAL" if facts.get("pe") or facts.get("pb") else "UNKNOWN")
        ),
        "market_regime": (
            "COMPLETE"
            if signals.get("market_regime") not in (None, "UNKNOWN")
            else "UNKNOWN"
        ),
        "overall": "PARTIAL",
    }


def render_engine_scorecard(engine: dict, ticker: str = "TICKER") -> str:
    """Scorecard teks wajib dipakai Chief (tidak diganti model)."""
    sig = engine.get("signals") or {}
    sc = engine.get("scores") or {}
    t = (ticker or "TICKER").upper()

    def _dir(trend: str) -> str:
        u = (trend or "").upper()
        if "BULL" in u:
            return "Bullish"
        if "BEAR" in u:
            return "Bearish"
        return "Netral"

    lines = [
        f"### SCORECARD ENGINE — {t} (deterministik, jangan diubah)",
        f"- Teknikal: {_dir(sig.get('trend_signal'))} {sc.get('technical_score')}%",
        f"- Rezim emiten: {sig.get('ticker_regime')} ({sig.get('ticker_regime_strength')}) — {sig.get('ticker_regime_summary')}",
        f"- Fundamental: {_dir('BULLISH' if sig.get('roe_class')=='STRONG' and sig.get('valuation_signal')!='PREMIUM_BOTH' else 'NEUTRAL')} {sc.get('fundamental_score')}%",
        f"- Pasar IHSG: {sig.get('market_regime')} {sc.get('market_score')}% — {sig.get('market_note')}",
        f"- Alignment emiten↔IHSG: {sig.get('regime_alignment')} — {sig.get('regime_alignment_note')}",
        f"- **Final: {sc.get('final_score')}/100** · stance: **{engine.get('stance')}**",
        f"- R:R headline: {sig.get('rr_headline')}",
        f"- Fibo: {sig.get('fib_signal')} (dist {sig.get('fib_distance_atr')} ATR) — {sig.get('fib_note')}",
        f"- Retest: {sig.get('retest_status')} — {sig.get('retest_note')}",
        f"- Valuasi: {sig.get('valuation_signal')} — {sig.get('valuation_note')}",
        f"- Risk: account {sig.get('account_risk_pct')}% | vol ATR {sig.get('volatility_risk_pct')}%",
        f"- Breakdown skor: tech={sc.get('technical_score')} fund={sc.get('fundamental_score')} "
        f"risk={sc.get('risk_score')} mkt={sc.get('market_score')} "
        f"(weights {sc.get('weights')})",
    ]
    return "\n".join(lines)


def run_analysis_engine(context: dict, params: dict | None = None) -> dict[str, Any]:
    """Pipeline penuh: facts → signals → scores → stance (+ debug)."""
    p = {**ENGINE_PARAMS, **(params or {})}
    facts = build_facts(context)
    signals = build_signals(facts, p)
    scores = build_scores(signals, facts, p)
    stance = derive_stance(scores, signals, facts)
    dq = data_quality_report(facts, signals)
    eng = {
        "facts": facts,
        "signals": signals,
        "scores": scores,
        "stance": stance,
        "data_quality": dq,
        "engine_version": "v3-spec-p0.2",
        "instructions_for_llm": (
            "engine.* = kebenaran deterministik. Jangan ubah signal/skor/stance. "
            "NO_RETEST ≠ gagal. fib AT_LEVEL ≠ resistance rejection. "
            "market UNKNOWN ≠ bearish. PBV_PREMIUM ≠ OVERVALUED. "
            "Pakai scorecard_text apa adanya di awal jawaban Chief."
        ),
    }
    eng["scorecard_text"] = render_engine_scorecard(
        eng, str(facts.get("ticker") or context.get("ticker") or "TICKER")
    )
    return eng



# Tesis + invalidation per strategy — critic/chief harus koheren dengan ini
STRATEGY_PLAYBOOK = {
    "v2": {
        "name": "Breakout V2",
        "thesis": "Breakout resistance dengan konfirmasi volume/momentum; ideal ada sweep di resistance lalu reclaim.",
        "must_have": ["level breakout jelas", "entry dekat break", "RR masuk akal"],
        "invalidation": [
            "Close kembali di bawah level breakout (failed breakout)",
            "Volume break lemah vs rata-rata",
            "Tidak ada follow-through 1–3 bar setelah BO",
            "SL terlalu ketat di noise / terlalu jauh tanpa struktur",
        ],
        "not_red_flag": [
            "Tidak ada field retest (UNKNOWN) — normal di V2",
            "Fibo tidak wajib untuk V2",
            "Fundamental mahal tidak membatalkan breakout harian",
        ],
        "critic_focus": "false breakout, reclaim gagal, volume, jarak SL vs ATR",
    },
    "v3": {
        "name": "Breakout+Fibo/Retest V3",
        "thesis": "Breakout atau zona Fibo dengan kualitas retest, ADX, freshness DaysSinceBO.",
        "must_have": ["struktur BO/retest", "RR ≥ floor", "score quality"],
        "invalidation": [
            "FAILED_RETEST / breakdown level",
            "Setup basi (DaysSinceBO besar) + ADX lemah",
            "Harga jauh dari zona Fibo relevan (FAR) tanpa momentum",
            "RiskATR ekstrem atau SL di luar napas volatilitas",
        ],
        "not_red_flag": [
            "NO_RETEST = belum uji, bukan gagal",
            "AT_LEVEL Fibo ≠ resistance rejection otomatis",
        ],
        "critic_focus": "retest status, freshness BO, ADX, Fibo distance ATR, SL breathing room",
    },
    "v4": {
        "name": "SMC Order Block V4",
        "thesis": "Mitigasi order block + BOS menuju target likuiditas; entry di zona OB.",
        "must_have": ["OB valid", "BOS/struktur", "target liquidity", "RR floor"],
        "invalidation": [
            "Close menembus invalidasi OB (di bawah OB bottom / SL struktural)",
            "OB terlalu tua tanpa reaksi harga",
            "Target liquidity sudah di-sweep / tidak relevan",
            "SL jauh dari struktur tanpa ATR floor — risk tidak proporsional",
            "Mitigasi > toleransi (harga tidak menghargai OB)",
        ],
        "not_red_flag": [
            "Tidak ada RetestOK field (bukan V3)",
            "Fundamental premium tidak membatalkan setup SMC harian",
            "Tanpa sweep low tidak selalu fatal jika BOS kuat",
        ],
        "critic_focus": "validitas OB, usia OB, jarak entry–OB, target liquidity, RiskATR/SL source",
    },
    "v5": {
        "name": "SMC CHOCH + Discount V5",
        "thesis": "Change of character bullish lalu entry di zona discount Fibo menuju peak.",
        "must_have": ["CHOCH fresh", "harga di discount", "RR ≥ 1.5", "target peak"],
        "invalidation": [
            "CHOCH basi (age tinggi) tanpa struktur baru",
            "Harga keluar discount / breakdown di bawah lowest_low",
            "RR tipis setelah fee pada risk lebar",
            "Tidak ada sweep low saat require_sweep aktif / skor mengandalkan sweep",
            "Peak target sudah jauh / liquidity sudah diambil",
        ],
        "not_red_flag": [
            "NO_RETEST / tanpa field retest — normal V5",
            "Deep discount di Fibo50 borderline bukan otomatis gagal",
            "Overvalued fundamental — sekunder untuk swing SMC pendek",
        ],
        "critic_focus": "CHOCH age, posisi vs Fibo discount, sweep low, lebar SL vs RR, peak target",
    },
    "intraday": {
        "name": "Intraday",
        "thesis": "Setup satu hari: momentum/level dengan RR konservatif tercapai intraday; patuhi ARA/ARB.",
        "must_have": ["target realistis 1 hari", "SL dalam range harian", "likuiditas"],
        "invalidation": [
            "Target di luar jangkauan range/ATR harian",
            "SL terlalu lebar untuk holding intraday",
            "Volume/likuiditas tipis (slippage)",
            "Melawan bias sesi tanpa edge",
        ],
        "not_red_flag": [
            "Fundamental jangka panjang kurang relevan",
            "Tidak ada weekly retest",
        ],
        "critic_focus": "keterjangkauan TP 1 hari, fee vs edge, likuiditas, ARA/ARB",
    },
    "highbeta": {
        "name": "High Beta",
        "thesis": "Saham beta/volatilitas tinggi dengan ekspansi range; size kecil wajib.",
        "must_have": ["volatilitas terkonfirmasi", "RR setelah gap risk", "size disiplin"],
        "invalidation": [
            "Gap risk menghapus SL",
            "Size terlalu besar vs ATR%",
            "Likuiditas tidak mendukung exit cepat",
            "Tidak ada katalis/momentum saat vol tinggi (chop)",
        ],
        "not_red_flag": [
            "Fundamental lemah sering normal di high-beta spekulatif",
            "RR screener tinggi bisa ilusi jika SL ketat noise",
        ],
        "critic_focus": "ATR%, gap, size, likuiditas exit, false RR",
    },
    "accumulation": {
        "name": "Accumulation",
        "thesis": "Fase akumulasi mendekati selesai: range ketat, spring/sweep low, siap ekspansi.",
        "must_have": ["tanda akhir akumulasi", "support/sweep", "risiko breakdown range"],
        "invalidation": [
            "Breakdown di bawah range akumulasi",
            "Volume distribusi (supply) bukan demand",
            "Belum selesai akumulasi — terlalu dini",
            "Tidak ada sweep/spring saat thesis mengandalkannya",
        ],
        "not_red_flag": [
            "Momentum ROC rendah di dalam range — sering normal",
            "Belum breakout — thesis-nya pra-break",
        ],
        "critic_focus": "validitas fase akumulasi, spring/sweep, risiko breakdown, timing terlalu dini",
    },
}


def _normalize_strategy_key(version: str | None) -> str:
    v = str(version or "").strip().lower().replace(" ", "")
    aliases = {
        "v2": "v2",
        "breakout": "v2",
        "breakoutv2": "v2",
        "v3": "v3",
        "v4": "v4",
        "v4_smc": "v4",
        "smc": "v4",
        "orderblock": "v4",
        "v5": "v5",
        "v5_smc": "v5",
        "choch": "v5",
        "intraday": "intraday",
        "vintraday": "intraday",
        "highbeta": "highbeta",
        "hb": "highbeta",
        "accumulation": "accumulation",
        "acc": "accumulation",
    }
    if v in aliases:
        return aliases[v]
    for key in ("v2", "v3", "v4", "v5", "intraday", "highbeta", "accumulation"):
        if key in v:
            return key
    return v or "unknown"


def get_strategy_playbook(version: str | None) -> dict:
    key = _normalize_strategy_key(version)
    base = STRATEGY_PLAYBOOK.get(key)
    if base:
        return {"key": key, **base}
    return {
        "key": key or "unknown",
        "name": str(version or "unknown"),
        "thesis": "Strategy tidak dikenali — kritik hanya dari engine + screener_row.",
        "must_have": [],
        "invalidation": ["Data setup tidak lengkap", "RR/SL tidak koheren"],
        "not_red_flag": [],
        "critic_focus": "koherensi entry-SL-TP, data quality, alignment rezim",
    }


def build_critic_system_prompt(version: str | None) -> str:
    """Prompt critic + lampiran playbook strategy aktif."""
    pb = get_strategy_playbook(version)
    base = SYSTEM_PROMPTS["critic"]
    extra = f"""

STRATEGY AKTIF: {pb.get('name')} (key={pb.get('key')})
Thesis: {pb.get('thesis')}
Must-have: {', '.join(pb.get('must_have') or []) or '-'}
Invalidation spesifik:
""" + "\n".join(f"- {x}" for x in (pb.get("invalidation") or [])) + f"""

BUKAN red flag untuk strategy ini:
""" + "\n".join(f"- {x}" for x in (pb.get("not_red_flag") or [])) + f"""

Fokus critic: {pb.get('critic_focus')}
JANGAN menerapkan invalidation strategy lain (mis. tuntut CHOCH saat V2, atau RetestOK saat V5).
"""
    return base + extra


SYSTEM_PROMPTS = {
    "technical": """Kamu adalah analis teknikal IDX. Hanya data JSON. Jangan mengarang.

UTAMAKAN field **engine.signals** dan **engine.facts** (deterministik):
- trend_signal, momentum_signal, fib_signal, fib_distance_atr, retest_status
- RR: engine.signals.rr_tp1, rr_tp2, expected_rr (bukan hanya TP1)
- retest_status: NO_RETEST | VALID_RETEST | FAILED_RETEST | UNKNOWN
  * NO_RETEST = belum diuji, BUKAN kegagalan, tanpa penalti naratif
  * FAILED_RETEST saja yang boleh disebut gagal
- fib_signal AT_LEVEL / NEAR_LEVEL: jangan red-flag hanya karena close < fib382
  jika fib_distance_atr kecil (dekat level)

REZIM (sudah dihitung sistem — JANGAN hitung ulang dari raw MA):
A) **Rezim emiten** = context.ticker_regime / engine.signals.ticker_regime (BULLISH|BEARISH|NEUTRAL)
   + strength + summary — pakai apa adanya
B) **Regime IHSG** = engine.signals.market_regime (last_close = indeks, bukan harga saham)
C) **Alignment** = ticker_regime.alignment_with_ihsg / engine.signals.regime_alignment

LARANGAN: menghitung golden/death cross sendiri; PE/PB; menyamakan MA IHSG dengan emiten.
Jelaskan setup entry/SL/TP + WHY dari engine; sebut rezim hanya dari field di atas.
Bahasa Indonesia, poin. Bukan saran investasi.""",
    "fundamental": """Analis fundamental IDX. Angka hanya canonical_valuation / engine.facts.

Gunakan engine.signals:
- pe_class, pb_class, roe_class, valuation_signal
- valuation_signal PBV_PREMIUM ≠ OVERVALUED (tanpa peer/historical)
- Jangan tulis OVERVALUED kecuali ada pembanding di JSON

Sebut source + confidence. yfinance = ketidakpastian.
Bahasa Indonesia, poin. Bukan saran investasi.""",
    "risk": """Risk manager IDX. Fokus lots, account_risk_pct (engine), SL vs ATR,
fee/pajak, expected_rr (TP1+TP2).

engine.signals.market_regime BEARISH = headwind konteks, bukan invalidasi SL.
UNKNOWN market = netral (bukan negatif).
Bahasa Indonesia, singkat. Bukan saran investasi.""",
    "critic": """Devil's advocate IDX. Kritik HANYA dari engine + screener_row + strategy_playbook.

ATURAN UMUM (semua strategy):
- Kritik harus koheren dengan **screener_version** / strategy_playbook — jangan pakai kriteria strategy lain
- Jangan sebut retest gagal jika retest_status = NO_RETEST / VALID_RETEST / UNKNOWN
- Jangan samakan regime IHSG dengan rezim/death-cross emiten
- CONFLICT alignment = risiko konteks, bukan otomatis invalidasi setup
- PBV_PREMIUM ≠ OVERVALUED tanpa peer
- market_regime UNKNOWN ≠ bearish
- Fokus: invalidation setup, false signal, SL quality, RR tipis, data quality

FORMAT:
1) Premis strategy (1 kalimat dari strategy_playbook.thesis)
2) 3–5 titik gagal spesifik strategy ini (bukan generic)
3) Apa yang BUKAN red flag untuk strategy ini
4) Verdict critic: WEAK / ACCEPTABLE_WITH_CAVEATS / STRONG_CAVEATS — selaras engine.stance

Bahasa Indonesia, poin. Bukan saran investasi.""",
    "chief": """Head trader. AI MENJELASKAN hasil engine; JANGAN mengarang skor baru.

WAJIB pakai engine.scores dan engine.stance sebagai sumber skor:
- technical_score, fundamental_score, risk_score, market_score, final_score
- stance: STRONG_SETUP | SETUP | WATCH | WAIT | AVOID | INSUFFICIENT_DATA

SCORECARD (salin angka engine, mapping arah dari signals):

### SCORECARD BIAS — {TICKER}
- Teknikal: {Bullish|Bearish|Netral} {engine.scores.technical_score}%
- Rezim emiten: {engine.signals.ticker_regime} ({engine.signals.ticker_regime_strength})
- Fundamental: {arah dari valuation/ROE} {engine.scores.fundamental_score}%
- Pasar IHSG: {engine.signals.market_regime} {engine.scores.market_score}%
- Alignment: {engine.signals.regime_alignment}
- **Final (engine): {engine.scores.final_score}/100** · stance: {engine.stance}
- Fundamental confidence: ...
- Link Sectors berita + profil

Lalu jelaskan:
1) Breakdown singkat kenapa final_score terbentuk (weights di engine.scores)
2) Rezim emiten vs IHSG (alignment) — jangan dicampur
3) Tiga syarat sebelum entry
4) Key positives / key risks (pisah emiten vs IHSG)
5) Retest + Fibo state dari engine.signals (NO_RETEST bukan gagal)
6) Kesimpulan selaras stance engine

Bahasa Indonesia. Bukan financial advice.""",
}


def _search_dirs() -> list[str]:
    dirs = [".", os.getcwd()]
    try:
        here = os.path.dirname(os.path.abspath(__file__))
        if here and here not in dirs:
            dirs.append(here)
    except Exception:
        pass
    for extra in ["/home/workdir", "/home/workdir/artifacts", "/home/workdir/attachments"]:
        if os.path.isdir(extra) and extra not in dirs:
            dirs.append(extra)
    return dirs


def find_report(version: str) -> str | None:
    files: list[str] = []
    for d in _search_dirs():
        files.extend(glob.glob(os.path.join(d, f"idx_report_{version}_*.csv")))
    if not files:
        return None
    return max(files, key=os.path.getmtime)


def load_report_df(version: str) -> pd.DataFrame:
    path = find_report(version)
    if not path:
        return pd.DataFrame()
    try:
        df = pd.read_csv(path)
        if "Ticker" in df.columns:
            df["Ticker"] = df["Ticker"].astype(str).str.upper().str.strip()
        return df
    except Exception:
        return pd.DataFrame()


def list_tickers_in_report(version: str) -> list[str]:
    df = load_report_df(version)
    if df.empty or "Ticker" not in df.columns:
        return []
    return sorted(df["Ticker"].dropna().unique().tolist())


def _row_to_dict(row: pd.Series) -> dict:
    out = {}
    for k, v in row.items():
        if pd.isna(v):
            out[k] = None
        elif hasattr(v, "item"):
            try:
                out[k] = v.item()
            except Exception:
                out[k] = str(v)
        else:
            out[k] = v if not isinstance(v, (pd.Timestamp,)) else str(v)
    return out


def _cross_events(
    short: "pd.Series",
    long: "pd.Series",
    *,
    lookback: int = 60,
) -> dict[str, Any]:
    """Deteksi golden/death cross terbaru antara dua MA."""
    s = short.dropna()
    l = long.dropna()
    common = s.index.intersection(l.index)
    if len(common) < 5:
        return {"status": "unknown", "days_since": None, "direction": None}
    s = s.loc[common]
    l = l.loc[common]
    above = s > l
    # cari perubahan status dari belakang
    days_since = None
    direction = None
    for i in range(len(above) - 1, 0, -1):
        if bool(above.iloc[i]) != bool(above.iloc[i - 1]):
            days_since = len(above) - 1 - i
            direction = "golden" if bool(above.iloc[i]) else "death"
            break
        if (len(above) - 1 - i) >= lookback:
            break
    status = "short_above_long" if bool(above.iloc[-1]) else "short_below_long"
    return {
        "status": status,
        "direction_last_cross": direction,
        "days_since_cross": days_since,
        "cross_fresh": days_since is not None and days_since <= 10,
        "cross_recent": days_since is not None and days_since <= 30,
    }


def fetch_ma_structure(ticker: str, days: int = 200) -> dict[str, Any]:
    """
    MA signifikan emiten: MA20/50/100/200, posisi harga vs MA,
    golden/death cross, seberapa baru.
    """
    out: dict[str, Any] = {
        "ticker": str(ticker).upper().strip(),
        "available": False,
        "source": "yfinance",
    }
    try:
        import yfinance as yf

        df = yf.download(
            f"{out['ticker']}.JK",
            period=f"{max(days, 220)}d",
            interval="1d",
            progress=False,
            auto_adjust=True,
            multi_level_index=False,
        )
        if df is None or df.empty or len(df) < 30:
            out["error"] = "data historis tidak cukup"
            return out
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        c = pd.to_numeric(df["Close"], errors="coerce")
        last = float(c.iloc[-1])
        ma20 = c.rolling(20).mean()
        ma50 = c.rolling(50).mean()
        ma100 = c.rolling(100).mean()
        ma200 = c.rolling(200).mean()

        def _last(series):
            v = series.iloc[-1]
            return None if pd.isna(v) else round(float(v), 2)

        m20, m50, m100, m200 = _last(ma20), _last(ma50), _last(ma100), _last(ma200)

        def _pos(price, ma):
            if ma is None or ma == 0:
                return None
            return {
                "above": price > ma,
                "dist_pct": round((price / ma - 1) * 100, 2),
            }

        stack_notes = []
        if m20 and m50 and m100:
            if m20 > m50 > m100:
                stack_notes.append("susunan bullish: MA20 > MA50 > MA100")
            elif m20 < m50 < m100:
                stack_notes.append("susunan bearish: MA20 < MA50 < MA100")
            else:
                stack_notes.append("susunan MA campur / rotasi")

        cross_20_50 = _cross_events(ma20, ma50)
        cross_50_100 = _cross_events(ma50, ma100) if m100 is not None else {}
        cross_50_200 = _cross_events(ma50, ma200) if m200 is not None else {}

        # slope kasar MA20 (5 hari)
        slope_ma20 = None
        if len(ma20.dropna()) >= 6:
            a = float(ma20.dropna().iloc[-1])
            b = float(ma20.dropna().iloc[-6])
            if b:
                slope_ma20 = round((a / b - 1) * 100, 2)

        out.update(
            {
                "available": True,
                "last_close": round(last, 2),
                "bars": int(len(df)),
                "ma": {
                    "MA20": m20,
                    "MA50": m50,
                    "MA100": m100,
                    "MA200": m200,
                },
                "price_vs_ma": {
                    "vs_MA20": _pos(last, m20),
                    "vs_MA50": _pos(last, m50),
                    "vs_MA100": _pos(last, m100),
                    "vs_MA200": _pos(last, m200),
                },
                "ma20_slope_5d_pct": slope_ma20,
                "stack": stack_notes,
                "crosses": {
                    "MA20_vs_MA50": cross_20_50,
                    "MA50_vs_MA100": cross_50_100,
                    "MA50_vs_MA200": cross_50_200,
                },
                "interpretation_hints": [],
            }
        )
        hints = out["interpretation_hints"]
        for label, pos in (out["price_vs_ma"] or {}).items():
            if not pos:
                continue
            side = "di atas" if pos["above"] else "di bawah"
            hints.append(f"Harga {side} {label.replace('vs_', '')} (jarak {pos['dist_pct']}%).")
        for name, cx in (out["crosses"] or {}).items():
            if not cx or not cx.get("direction_last_cross"):
                continue
            d = cx["direction_last_cross"]
            days = cx.get("days_since_cross")
            age = (
                "baru saja (≤10 hari)"
                if cx.get("cross_fresh")
                else (
                    "relatif baru (≤30 hari)"
                    if cx.get("cross_recent")
                    else f"sudah lama ({days} hari lalu)" if days is not None else "usia tidak jelas"
                )
            )
            kind = "Golden cross" if d == "golden" else "Death cross"
            hints.append(f"{kind} pada {name}: {age}.")
        return out
    except Exception as e:
        out["error"] = str(e)
        return out


def _price_snapshot(ticker: str, days: int = 60) -> dict | None:
    try:
        import yfinance as yf

        df = yf.download(
            f"{ticker}.JK",
            period=f"{days}d",
            interval="1d",
            progress=False,
            auto_adjust=True,
            multi_level_index=False,
        )
        if df is None or df.empty or len(df) < 10:
            return None
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        c = df["Close"]
        last = float(c.iloc[-1])
        prev = float(c.iloc[-2])
        high20 = float(df["High"].tail(20).max())
        low20 = float(df["Low"].tail(20).min())
        vol = df["Volume"] if "Volume" in df.columns else None
        avg_vol = float(vol.tail(20).mean()) if vol is not None else None
        return {
            "last_close": round(last, 2),
            "chg_1d_pct": round((last / prev - 1) * 100, 2),
            "high_20": round(high20, 2),
            "low_20": round(low20, 2),
            "range_20_pct": round((high20 - low20) / last * 100, 2),
            "dist_high_20_pct": round((high20 - last) / last * 100, 2),
            "avg_vol_20": int(avg_vol) if avg_vol else None,
        }
    except Exception:
        return None


def _safe_float(x: Any) -> float | None:
    try:
        if x is None or (isinstance(x, float) and pd.isna(x)):
            return None
        return float(x)
    except Exception:
        return None


def _classify_valuation(pe: float | None, pb: float | None) -> str:
    notes = []
    if pe is not None and pe > 0:
        if pe < 10:
            notes.append("PE relatif rendah")
        elif pe <= 20:
            notes.append("PE moderat")
        else:
            notes.append("PE relatif tinggi")
    elif pe is not None and pe <= 0:
        notes.append("PE tidak bermakna (negatif/0)")
    if pb is not None and pb > 0:
        if pb < 1:
            notes.append("PB di bawah 1")
        elif pb <= 3:
            notes.append("PB moderat")
        else:
            notes.append("PB relatif tinggi")
    return "; ".join(notes) if notes else "valuasi tidak dapat disimpulkan dari data"


def _sectors_urls(ticker: str) -> dict[str, str]:
    """Deep-link Sectors (tanpa scrape). Profil: /idx/{ticker_lower}."""
    t = str(ticker).upper().strip().replace(".JK", "")
    slug = t.lower()
    return {
        "sectors_company_url": f"https://sectors.app/idx/{slug}",
        "sectors_news_url": f"https://sectors.app/indonesia/news?nticker={t}.JK",
    }



def _parse_metric_number(val) -> float | None:
    """Parse angka kotor dari export Screener: '1,031.63 B', '6.40%', '-', 'N/A'."""
    if val is None:
        return None
    try:
        if isinstance(val, (int, float)):
            if val != val:  # NaN
                return None
            return float(val)
    except Exception:
        pass
    s = str(val).strip()
    if not s or s in ("-", "—", "N/A", "n/a", "NA", "null", "None", "#N/A"):
        return None
    s = s.replace(",", "").replace(" ", "")
    mult = 1.0
    su = s.upper()
    if su.endswith("%"):
        s = s[:-1]
        try:
            return float(s)
        except Exception:
            return None
    if su.endswith("T"):
        mult = 1e12
        s = s[:-1]
    elif su.endswith("B"):
        mult = 1e9
        s = s[:-1]
    elif su.endswith("M"):
        mult = 1e6
        s = s[:-1]
    elif su.endswith("K"):
        mult = 1e3
        s = s[:-1]
    try:
        return float(s) * mult
    except Exception:
        return None


def _iter_fundamentals_override_frames() -> list[tuple[pd.DataFrame, str]]:
    """Kumpulkan semua sumber override (gdrive + setiap file lokal yang ada)."""
    frames: list[tuple[pd.DataFrame, str]] = []
    seen_paths: set[str] = set()

    try:
        from idx_gdrive_data import load_fundamentals_override_df

        df, source_label = load_fundamentals_override_df()
        if df is not None and not df.empty:
            frames.append((df, source_label or "gdrive"))
    except Exception:
        pass

    candidates: list[str] = []
    try:
        here = os.path.dirname(os.path.abspath(__file__))
        candidates.append(os.path.join(here, "fundamentals_override.csv"))
    except Exception:
        pass
    for d in _search_dirs():
        candidates.append(os.path.join(d, "fundamentals_override.csv"))
    candidates.extend(
        [
            os.path.join(os.getcwd(), "fundamentals_override.csv"),
            "fundamentals_override.csv",
            "/home/workdir/artifacts/fundamentals_override.csv",
            "/home/workdir/attachments/fundamentals_override.csv",
        ]
    )
    for path in candidates:
        try:
            key = os.path.abspath(path)
        except Exception:
            key = path
        if key in seen_paths or not os.path.isfile(path):
            continue
        seen_paths.add(key)
        try:
            df = pd.read_csv(path, index_col=False)
            if df is not None and not df.empty:
                frames.append((df, f"local:{path}"))
        except Exception:
            continue
    return frames


def _load_fundamentals_override(ticker: str) -> dict[str, Any] | None:
    """
    Baca fundamentals override dari SEMUA sumber sampai ticker ketemu.
    Prioritas: Google Drive (jika secrets) lalu file lokal.
    Kolom ticker yang dikenali: Ticker, ticker, Kode, Symbol, Code, Emiten.
    """
    t = str(ticker).upper().strip().replace(".JK", "")
    frames = _iter_fundamentals_override_frames()
    if not frames:
        return None

    def _find_row(df: pd.DataFrame):
        df = df.copy()
        df.columns = [str(c).strip() for c in df.columns]
        colmap = {str(c).strip().lower(): c for c in df.columns}
        tcol = (
            colmap.get("ticker")
            or colmap.get("kode")
            or colmap.get("symbol")
            or colmap.get("code")
            or colmap.get("emiten")
            or colmap.get("saham")
        )
        if not tcol:
            return None, colmap
        series = (
            df[tcol]
            .astype(str)
            .str.upper()
            .str.replace(".JK", "", regex=False)
            .str.strip()
        )
        hit = df[series == t]
        if hit.empty:
            return None, colmap
        return hit.iloc[0], colmap

    row = None
    colmap: dict = {}
    source_label = "none"
    for df, label in frames:
        found, cmap = _find_row(df)
        if found is not None:
            row, colmap, source_label = found, cmap, label
            break
    if row is None:
        return None

    def _cell(row, *keys):
        for k in keys:
            c = colmap.get(k.lower())
            if c is not None and c in row.index:
                v = row[c]
                if pd.isna(v):
                    continue
                return v
        return None
    pe = _parse_metric_number(
        _cell(
            row,
            "pe",
            "pe_ratio",
            "trailing_pe",
            "current pe ratio (ttm)",
            "current pe ratio",
            "pe ratio",
        )
    )
    pb = _parse_metric_number(
        _cell(
            row,
            "pb",
            "pb_ratio",
            "price_to_book",
            "current price to book value",
            "price to book value",
            "pbv",
        )
    )
    eps = _parse_metric_number(
        _cell(row, "eps", "current eps (ttm)", "current eps", "eps (ttm)")
    )
    roe = _parse_metric_number(
        _cell(
            row,
            "roe",
            "average (roe 5 yr)",
            "average roe 5 yr",
            "roe 5 yr",
            "return on equity",
        )
    )
    mcap = _parse_metric_number(
        _cell(row, "marketcap", "market_cap", "market cap")
    )
    div_y = _parse_metric_number(
        _cell(
            row,
            "dividendyield",
            "dividend_yield",
            "dividend_yield_pct",
            "dividend yield",
        )
    )
    debt = _parse_metric_number(
        _cell(
            row,
            "debttoequity",
            "debt_to_equity",
            "debt to equity ratio (quarter)",
            "debt to equity ratio",
            "debt to equity",
        )
    )
    pm = _parse_metric_number(
        _cell(
            row,
            "profitmargin",
            "profit_margin",
            "profit_margin_pct",
            "net profit margin (ttm)(%)",
            "net profit margin (ttm)",
            "net profit margin",
        )
    )
    urls = _sectors_urls(t)
    src = str(_cell(row, "source") or "fundamentals_override").strip()
    if not src or src.lower() in ("nan", "none"):
        src = "fundamentals_override"
    conf_note = (
        f"Override dari {source_label}."
        if source_label.startswith("gdrive")
        else "Data dari fundamentals_override.csv (kurasi manual)."
    )
    out: dict[str, Any] = {
        "ticker": t,
        "available": True,
        "source": src if not source_label.startswith("gdrive") else f"{src}|gdrive",
        "confidence": "high",
        "confidence_note": conf_note,
        "name": _cell(row, "name", "company_name"),
        "sector": _cell(row, "sector"),
        "industry": _cell(row, "industry"),
        "currency": "IDR",
        "market_cap": int(mcap) if mcap else None,
        "pe_ratio": round(pe, 2) if pe is not None else None,
        "pb_ratio": round(pb, 2) if pb is not None else None,
        "eps": round(eps, 2) if eps is not None else None,
        "roe": round(roe, 2) if roe is not None else None,
        "profit_margin_pct": round(pm, 2) if pm is not None else None,
        "debt_to_equity": round(debt, 2) if debt is not None else None,
        "dividend_yield_pct": round(div_y, 2) if div_y is not None else None,
        "as_of": _cell(row, "asof", "as_of", "date"),
        "notes": _cell(row, "notes", "note"),
        "valuation_note": _classify_valuation(pe, pb),
        "override_file": source_label,
        **urls,
        "disclaimer": "Override manual/Drive — pastikan AsOf dan angka sesuai laporan terkini.",
    }
    return out


def sanitize_fundamental(fund: dict[str, Any] | None) -> dict[str, Any]:
    """
    Bersihkan metrik tidak masuk akal (sering dari yfinance IDX) dan
    bangun canonical_valuation yang wajib dipakai semua role.
    """
    if not fund or not isinstance(fund, dict):
        return {
            "available": False,
            "source": "none",
            "confidence": "low",
            "rejected_metrics": {},
            "canonical_valuation": {},
        }

    f = dict(fund)
    rejected: dict[str, str] = {}

    def _reject(key: str, reason: str) -> None:
        if f.get(key) is not None:
            rejected[key] = f"{f.get(key)} ({reason})"
            f[key] = None

    pe = _safe_float(f.get("pe_ratio"))
    pb = _safe_float(f.get("pb_ratio"))
    div_y = _safe_float(f.get("dividend_yield_pct"))
    roe = _safe_float(f.get("roe"))
    eps = _safe_float(f.get("eps"))

    # PB absurd (contoh 17397) = error data, bukan overvalued
    if pb is not None and (pb < 0 or pb > 25):
        _reject("pb_ratio", "di luar rentang wajar IDX 0–25")
        pb = None
    if pe is not None and (pe > 120 or pe < -50):
        _reject("pe_ratio", "di luar rentang wajar")
        pe = None
    # Dividend: kadang 1.65 (=165%) atau 165
    if div_y is not None:
        if div_y > 30:
            # coba skala jika terlihat 100x
            if 30 < div_y <= 300:
                rejected["dividend_yield_pct"] = f"{div_y} (mencurigakan; diabaikan)"
                f["dividend_yield_pct"] = None
                div_y = None
            else:
                _reject("dividend_yield_pct", "tidak wajar >30%")
                div_y = None
    if roe is not None and abs(roe) > 100:
        _reject("roe", "tidak wajar |roe|>100")
        roe = None

    src = str(f.get("source") or "unknown")
    conf = str(f.get("confidence") or "low")
    # Override / Stockbit = high
    if any(x in src.lower() for x in ("override", "stockbit", "laporan", "idx_fundamental")):
        conf = f.get("confidence") or "high"
        f["confidence"] = conf

    canonical = {
        "ticker": f.get("ticker"),
        "source": src,
        "confidence": f.get("confidence") or conf,
        "pe_ratio": pe if pe is not None else f.get("pe_ratio"),
        "pb_ratio": pb if pb is not None else f.get("pb_ratio"),
        "eps": eps if eps is not None else f.get("eps"),
        "roe": roe if roe is not None else f.get("roe"),
        "dividend_yield_pct": div_y if div_y is not None else f.get("dividend_yield_pct"),
        "profit_margin_pct": f.get("profit_margin_pct"),
        "debt_to_equity": f.get("debt_to_equity"),
        "market_cap": f.get("market_cap"),
        "valuation_note": f.get("valuation_note"),
        "as_of": f.get("as_of"),
        "sectors_company_url": f.get("sectors_company_url"),
        "sectors_news_url": f.get("sectors_news_url"),
        "instruction": (
            "SEMUA role wajib memakai angka valuasi HANYA dari objek ini. "
            "Abaikan PE/PB yang disebut role lain jika berbeda."
        ),
    }
    # sinkron field utama yang di-null
    if "pb_ratio" in rejected:
        canonical["pb_ratio"] = None
        f["pb_ratio"] = None
    if "pe_ratio" in rejected:
        canonical["pe_ratio"] = None
        f["pe_ratio"] = None
    if "dividend_yield_pct" in rejected:
        canonical["dividend_yield_pct"] = None
    if "roe" in rejected:
        canonical["roe"] = None
        f["roe"] = None

    f["rejected_metrics"] = rejected
    f["canonical_valuation"] = canonical
    if rejected:
        note = f.get("confidence_note") or ""
        f["confidence_note"] = (
            (note + " " if note else "")
            + "Metrik ditolak (data invalid): "
            + ", ".join(rejected.keys())
        ).strip()
    return f


def fetch_fundamental(ticker: str) -> dict[str, Any]:
    """Ambil fundamental: override CSV → idx_fundamental → yfinance."""
    ticker = str(ticker).upper().strip().replace(".JK", "")

    # 1) CSV override (prioritas tertinggi)
    try:
        ov = _load_fundamentals_override(ticker)
        if ov:
            return sanitize_fundamental(ov)
    except Exception:
        pass

    # 2) Modul lokal idx_fundamental
    try:
        from idx_fundamental import get_fundamental_snapshot  # type: ignore

        snap = get_fundamental_snapshot(ticker)
        if isinstance(snap, dict) and snap:
            snap = dict(snap)
            snap.setdefault("source", "idx_fundamental")
            snap.setdefault("confidence", "high")
            urls = _sectors_urls(ticker)
            snap.setdefault("sectors_company_url", urls["sectors_company_url"])
            snap.setdefault("sectors_news_url", urls["sectors_news_url"])
            return sanitize_fundamental(snap)
    except Exception:
        pass

    out: dict[str, Any] = {
        "ticker": ticker,
        "source": "yfinance",
        "available": False,
    }
    try:
        import yfinance as yf

        t = yf.Ticker(f"{ticker}.JK")
        info: dict = {}
        try:
            info = t.info or {}
        except Exception:
            info = {}
        try:
            fi = getattr(t, "fast_info", None)
            if fi:
                for k in ("market_cap", "last_price", "year_high", "year_low"):
                    if not info.get(k):
                        try:
                            info[k] = fi.get(k) if hasattr(fi, "get") else getattr(fi, k, None)
                        except Exception:
                            pass
        except Exception:
            pass

        pe = _safe_float(info.get("trailingPE") or info.get("forwardPE"))
        pb = _safe_float(info.get("priceToBook"))
        eps = _safe_float(info.get("trailingEps") or info.get("epsTrailingTwelveMonths"))
        mcap = _safe_float(info.get("marketCap") or info.get("market_cap"))
        roe = _safe_float(info.get("returnOnEquity"))
        profit_m = _safe_float(info.get("profitMargins"))
        debt_eq = _safe_float(info.get("debtToEquity"))
        div_y = _safe_float(info.get("dividendYield"))
        if div_y is not None and div_y < 1:
            div_y_pct = round(div_y * 100, 2)
        else:
            div_y_pct = div_y

        roe_out = None
        if roe is not None:
            roe_out = round(roe * 100, 2) if abs(roe) <= 1 else round(roe, 2)

        filled = sum(
            1
            for x in (pe, pb, eps, mcap, roe)
            if x is not None
        )
        confidence = "low"
        if filled >= 4:
            confidence = "medium"
        # yfinance IDX tetap max medium — jangan high

        urls = _sectors_urls(ticker)
        sectors_company_url = urls["sectors_company_url"]
        sectors_news_url = urls["sectors_news_url"]

        out.update(
            {
                "available": bool(info),
                "source": "yfinance",
                "confidence": confidence,
                "confidence_note": (
                    "Sumber yfinance sering tidak akurat untuk emiten IDX. "
                    "Verifikasi PE/PB/EPS di Sectors atau laporan resmi sebelum diandalkan."
                ),
                "sectors_company_url": sectors_company_url,
                "sectors_news_url": sectors_news_url,
                "name": info.get("longName") or info.get("shortName"),
                "sector": info.get("sector"),
                "industry": info.get("industry"),
                "currency": info.get("currency") or "IDR",
                "market_cap": int(mcap) if mcap else None,
                "pe_ratio": round(pe, 2) if pe is not None else None,
                "pb_ratio": round(pb, 2) if pb is not None else None,
                "eps": round(eps, 2) if eps is not None else None,
                "roe": roe_out,
                "profit_margin_pct": round(profit_m * 100, 2) if profit_m is not None else None,
                "debt_to_equity": round(debt_eq, 2) if debt_eq is not None else None,
                "dividend_yield_pct": div_y_pct,
                "52w_high": _safe_float(info.get("fiftyTwoWeekHigh") or info.get("year_high")),
                "52w_low": _safe_float(info.get("fiftyTwoWeekLow") or info.get("year_low")),
                "valuation_note": _classify_valuation(pe, pb),
                "disclaimer": "Data Yahoo/yfinance bisa telat atau tidak lengkap untuk emiten IDX.",
            }
        )
    except Exception as e:
        out["error"] = str(e)
    return sanitize_fundamental(out)




def _normalize_news_url(url: str | None) -> str | None:
    """Rapikan URL agar bisa diklik/verifikasi."""
    if not url:
        return None
    u = str(url).strip()
    if not u or u.lower() in ("none", "null", "-"):
        return None
    # buang tracking fragment berlebih yang tidak perlu
    if u.startswith("//"):
        u = "https:" + u
    if u.startswith("http://") or u.startswith("https://"):
        return u
    if u.startswith("www."):
        return "https://" + u
    return None


def _fallback_search_url(ticker: str, title: str | None = None) -> str:
    """URL pencarian Google News sebagai fallback verifikasi."""
    import urllib.parse

    q = f"{ticker} saham"
    if title:
        # ambil beberapa kata judul agar tetap relevan
        words = " ".join(str(title).split()[:8])
        q = f"{ticker} {words}"
    return (
        "https://news.google.com/search?"
        + urllib.parse.urlencode({"q": q, "hl": "id", "gl": "ID", "ceid": "ID:id"})
    )


def _extract_yf_link(n: dict) -> str | None:
    """Ambil URL terbaik dari struktur yfinance news (beragam versi)."""
    if not isinstance(n, dict):
        return None
    candidates = []
    for key in ("link", "url", "canonicalUrl", "clickThroughUrl", "relatedTickers"):
        v = n.get(key)
        if isinstance(v, str):
            candidates.append(v)
        elif isinstance(v, dict):
            candidates.append(v.get("url") or v.get("webUrl") or v.get("clickThroughUrl") or "")
    content = n.get("content")
    if isinstance(content, dict):
        for key in ("canonicalUrl", "clickThroughUrl", "previewUrl", "link", "url"):
            v = content.get(key)
            if isinstance(v, str):
                candidates.append(v)
            elif isinstance(v, dict):
                candidates.append(
                    v.get("url") or v.get("webUrl") or v.get("clickThroughUrl") or ""
                )
        # provider sometimes nests
        for nest in ("provider", "finance"):
            p = content.get(nest)
            if isinstance(p, dict):
                candidates.append(p.get("url") or "")
    for c in candidates:
        nu = _normalize_news_url(c)
        if nu:
            return nu
    return None




def _parse_rss_datetime(pub_date: str):
    """Return timezone-aware or naive datetime for sorting; None if fail."""
    from email.utils import parsedate_to_datetime
    if not pub_date:
        return None
    try:
        return parsedate_to_datetime(pub_date)
    except Exception:
        pass
    for fmt in ("%Y-%m-%d", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%d %H:%M:%S"):
        try:
            return datetime.strptime(str(pub_date)[:19], fmt)
        except Exception:
            continue
    return None


def _fetch_google_news_rss(query: str, max_items: int = 10, label: str = "google_news") -> list[dict]:
    """Ambil item dari Google News RSS (real-time relatif, locale ID)."""
    import xml.etree.ElementTree as ET
    import urllib.request
    import urllib.parse

    items: list[dict] = []
    q = urllib.parse.quote(query)
    rss_url = f"https://news.google.com/rss/search?q={q}&hl=id&gl=ID&ceid=ID:id"
    req = urllib.request.Request(
        rss_url,
        headers={
            "User-Agent": "Mozilla/5.0 (compatible; IDXAssistant/1.1; +local)"
        },
        method="GET",
    )
    with urllib.request.urlopen(req, timeout=14) as r:
        xml_bytes = r.read()
    root = ET.fromstring(xml_bytes)
    channel = root.find("channel")
    rss_items = channel.findall("item") if channel is not None else root.findall(".//item")
    for it in rss_items:
        if len(items) >= max_items:
            break
        title = (it.findtext("title") or "").strip()
        link = (it.findtext("link") or "").strip()
        if not link:
            guid = it.find("guid")
            if guid is not None and (guid.text or "").startswith("http"):
                link = guid.text.strip()
        pub_date = (it.findtext("pubDate") or "").strip()
        source_el = it.find("source")
        publisher = (source_el.text or "").strip() if source_el is not None else None
        dt = _parse_rss_datetime(pub_date)
        date_s = dt.strftime("%Y-%m-%d %H:%M") if dt else (pub_date[:16] if pub_date else None)
        url = _normalize_news_url(link)
        items.append(
            {
                "title": title[:240],
                "publisher": (publisher[:80] if publisher else None),
                "url": url,
                "date": date_s,
                "date_ts": dt.timestamp() if dt else 0.0,
                "source": label,
            }
        )
    return items


def fetch_news_intel(
    ticker: str,
    max_items: int = 8,
    *,
    hours_lookback: int = 72,
) -> dict[str, Any]:
    """
    Berita real-time / near real-time terkait ticker.

    Sumber (digabung, diurutkan terbaru):
      1) yfinance Ticker.news
      2) Google News RSS — query ticker (when:7d)
      3) Google News RSS — site berita ID (CNBC/Kontan/Bisnis) + ticker

    Setiap item diusahakan punya URL klikabel.
    """
    import urllib.parse

    ticker = str(ticker).upper().strip()
    raw: list[dict] = []
    sources_ok: list[str] = []
    errors: list[str] = []
    now = datetime.now()
    fetched_at = now.strftime("%Y-%m-%d %H:%M:%S")

    def _push(title, publisher, link, date_s, date_ts, source):
        title = (title or "").strip()
        if not title:
            return
        if any(title[:80].lower() == (x.get("title") or "")[:80].lower() for x in raw):
            return
        url = _normalize_news_url(link)
        if not url:
            url = _fallback_search_url(ticker, title)
            link_kind = "search_fallback"
        else:
            link_kind = "direct"
        raw.append(
            {
                "title": title[:240],
                "publisher": (str(publisher)[:80] if publisher else None),
                "url": url,
                "link_kind": link_kind,
                "date": date_s,
                "date_ts": float(date_ts or 0),
                "source": source,
            }
        )

    # --- 1) yfinance ---
    try:
        import yfinance as yf

        t = yf.Ticker(f"{ticker}.JK")
        for n in (getattr(t, "news", None) or [])[: max_items + 5]:
            if not isinstance(n, dict):
                continue
            content = n.get("content") if isinstance(n.get("content"), dict) else None
            date_ts = 0.0
            if content:
                title = content.get("title") or content.get("summary") or ""
                pub = content.get("provider") if isinstance(content.get("provider"), dict) else {}
                publisher = (
                    (pub or {}).get("displayName")
                    or content.get("publisher")
                    or n.get("publisher")
                    or ""
                )
                ts = content.get("pubDate") or content.get("displayTime") or ""
                dt = _parse_rss_datetime(str(ts)) if ts else None
                if dt:
                    date_ts = dt.timestamp()
                    date_s = dt.strftime("%Y-%m-%d %H:%M")
                else:
                    date_s = str(ts)[:32]
            else:
                title = n.get("title") or ""
                publisher = n.get("publisher") or ""
                ts = n.get("providerPublishTime") or 0
                if isinstance(ts, (int, float)) and ts > 0:
                    try:
                        date_ts = float(ts)
                        date_s = datetime.utcfromtimestamp(int(ts)).strftime("%Y-%m-%d %H:%M")
                    except Exception:
                        date_s = str(ts)
                else:
                    date_s = None
            _push(title, publisher, _extract_yf_link(n), date_s, date_ts, "yfinance")
        if any(x.get("source") == "yfinance" for x in raw):
            sources_ok.append("yfinance")
    except Exception as e:
        errors.append(f"yfinance_news: {e}")

    # --- 2) Google News general (prioritas 7 hari) ---
    try:
        q = f"{ticker} (saham OR emiten OR IDX) when:7d"
        for it in _fetch_google_news_rss(q, max_items=max_items + 4, label="google_news"):
            _push(
                it.get("title"),
                it.get("publisher"),
                it.get("url"),
                it.get("date"),
                it.get("date_ts"),
                "google_news",
            )
        if any(x.get("source") == "google_news" for x in raw):
            sources_ok.append("google_news")
    except Exception as e:
        errors.append(f"google_news: {e}")

    # --- 3) Google News fokus media ID ---
    try:
        q2 = (
            f"{ticker} (saham OR IHSG) "
            f"(site:cnbcindonesia.com OR site:kontan.co.id OR site:bisnis.com "
            f"OR site:market.bisnis.com OR site:investasi.kontan.co.id) when:14d"
        )
        for it in _fetch_google_news_rss(q2, max_items=6, label="media_id"):
            _push(
                it.get("title"),
                it.get("publisher"),
                it.get("url"),
                it.get("date"),
                it.get("date_ts"),
                "media_id",
            )
        if any(x.get("source") == "media_id" for x in raw):
            sources_ok.append("media_id")
    except Exception as e:
        errors.append(f"media_id: {e}")

    # Sort terbaru dulu
    raw.sort(key=lambda x: float(x.get("date_ts") or 0), reverse=True)

    # Filter lookback opsional (keep items without date)
    cutoff = now.timestamp() - hours_lookback * 3600
    filtered = []
    for it in raw:
        ts = float(it.get("date_ts") or 0)
        if ts <= 0 or ts >= cutoff:
            filtered.append(it)
    if not filtered:
        filtered = raw

    items = filtered[:max_items]

    pos_kw = (
        "naik", "tumbuh", "laba", "untung", "kontrak", "dividen", "buyback",
        "ekspansi", "rekor", "positif", "upgrade", "net profit", "menguat",
    )
    neg_kw = (
        "turun", "rugi", "gagal", "sanksi", "default", "downgrade", "suspend",
        "fraud", "korupsi", "investigasi", "bangkrut", "negatif", "melemah",
    )
    pos = neg = 0
    for it in items:
        tl = (it.get("title") or "").lower()
        if any(k in tl for k in pos_kw):
            pos += 1
        if any(k in tl for k in neg_kw):
            neg += 1
    if not items:
        tone = "unknown"
    elif pos > neg + 1:
        tone = "lean_positive"
    elif neg > pos + 1:
        tone = "lean_negative"
    else:
        tone = "mixed_or_neutral"

    # strip date_ts from public payload (internal sort only) — keep for debug optional
    public_items = []
    for it in items:
        public_items.append({k: v for k, v in it.items() if k != "date_ts"})

    # Tautan Sectors (filter emiten) — tanpa scraping, hanya deep-link UI
    sectors_ticker = f"{ticker}.JK"
    sectors_news_url = f"https://sectors.app/indonesia/news?nticker={sectors_ticker}"

    return {
        "ticker": ticker,
        "headlines": public_items,
        "headline_count": len(public_items),
        "sources": sources_ok,
        "tone_hint": tone,
        "tone_counts": {"positive_kw": pos, "negative_kw": neg},
        "verify_search_url": _fallback_search_url(ticker),
        "sectors_news_url": sectors_news_url,
        "fetched_at": fetched_at,
        "hours_lookback": hours_lookback,
        "realtime": True,
        "errors": errors,
        "disclaimer": (
            "Berita near real-time dari agregator (Yahoo/Google News/media ID). "
            "Link Sectors hanya deep-link UI (bukan scrape). "
            "Bisa delay atau tidak lengkap. Verifikasi sumber resmi BEI/emiten. "
            "Bukan rekomendasi investasi."
        ),
    }


def fetch_market_news(max_items: int = 10) -> dict[str, Any]:
    """Berita pasar umum IHSG / BEI untuk tab Kondisi Pasar."""
    errors: list[str] = []
    raw: list[dict] = []
    sources_ok: list[str] = []
    fetched_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    queries = [
        ("IHSG OR \"Bursa Efek\" OR BEI when:3d", "google_ihsg"),
        (
            "(IHSG OR saham) (site:cnbcindonesia.com OR site:kontan.co.id OR site:bisnis.com) when:5d",
            "media_id",
        ),
    ]
    for q, label in queries:
        try:
            for it in _fetch_google_news_rss(q, max_items=max_items, label=label):
                title = (it.get("title") or "").strip()
                if not title:
                    continue
                if any(title[:80].lower() == (x.get("title") or "")[:80].lower() for x in raw):
                    continue
                url = _normalize_news_url(it.get("url")) or (
                    "https://news.google.com/search?q=" + title[:40]
                )
                raw.append(
                    {
                        "title": title[:240],
                        "publisher": it.get("publisher"),
                        "url": url,
                        "link_kind": "direct" if it.get("url") else "search_fallback",
                        "date": it.get("date"),
                        "date_ts": float(it.get("date_ts") or 0),
                        "source": label,
                    }
                )
            sources_ok.append(label)
        except Exception as e:
            errors.append(f"{label}: {e}")

    raw.sort(key=lambda x: float(x.get("date_ts") or 0), reverse=True)
    items = [{k: v for k, v in it.items() if k != "date_ts"} for it in raw[:max_items]]

    return {
        "scope": "market",
        "headlines": items,
        "headline_count": len(items),
        "sources": sources_ok,
        "fetched_at": fetched_at,
        "verify_search_url": (
            "https://news.google.com/search?q=IHSG&hl=id&gl=ID&ceid=ID:id"
        ),
        "errors": errors,
        "disclaimer": (
            "Agregasi berita pasar near real-time. Bukan rekomendasi investasi."
        ),
    }


def build_context(


    ticker: str,
    version: str,
    *,
    regime: dict | None = None,
    account_size: float = 50_000_000,
    risk_pct: float = 1.0,
    broker_buy_pct: float = 0.15,
    broker_sell_pct: float = 0.25,
    include_price_snapshot: bool = True,
    include_fundamental: bool = True,
    include_news: bool = True,
    include_llmquant: bool = True,
) -> dict[str, Any]:
    ticker = str(ticker).upper().strip()
    df = load_report_df(version)
    row_dict: dict = {}
    if not df.empty and "Ticker" in df.columns:
        hit = df[df["Ticker"] == ticker]
        if not hit.empty:
            row_dict = _row_to_dict(hit.iloc[0])

    csv_fund = {}
    for k in list(row_dict.keys()):
        lk = str(k).lower()
        if any(x in lk for x in ("pe", "pb", "eps", "roe", "value", "fundamental", "sector")):
            csv_fund[k] = row_dict[k]

    ctx: dict[str, Any] = {
        "as_of": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "ticker": ticker,
        "screener_version": version,
        "strategy_playbook": get_strategy_playbook(version),
        "screener_row": row_dict,
        "account": {
            "size_rp": account_size,
            "risk_per_trade_pct": risk_pct,
        },
        "fees": {
            "broker_buy_pct": broker_buy_pct,
            "broker_sell_pct": broker_sell_pct,
            "notes": "Tambahan tipikal IDX: PPN 12% atas fee, levy ~0.043%, PPh final 0.1% pada jual",
        },
        "regime": regime or {},
        "disclaimer": "Bukan rekomendasi investasi. Hanya asisten analisa berbasis data screener.",
    }
    if include_price_snapshot:
        snap = _price_snapshot(ticker)
        if snap:
            ctx["price_snapshot"] = snap
        # MA signifikan + golden/death cross (data terpisah, horizon lebih panjang)
        try:
            ctx["ma_structure"] = fetch_ma_structure(ticker)
        except Exception as e:
            ctx["ma_structure"] = {
                "ticker": ticker,
                "available": False,
                "error": str(e),
            }
        # Rezim emiten dihitung SEBELUM agent (hemat prompt / token)
        try:
            ctx["ticker_regime"] = detect_emiten_regime(
                ticker,
                screener_row=row_dict,
                ma_structure=ctx.get("ma_structure"),
                ihsg_regime=regime,
                fetch_ma=False,
            )
        except Exception as e:
            ctx["ticker_regime"] = {
                "ticker": ticker,
                "regime": "UNKNOWN",
                "strength": "UNKNOWN",
                "summary": "UNKNOWN",
                "error": str(e),
                "source": "detect_emiten_regime",
            }
    if include_fundamental:
        fund = fetch_fundamental(ticker)
        # Jangan campur metrik CSV screener yang bisa bentrok / field salah
        # (mis. kolom mengandung 'pe' tapi bukan PE ratio).
        if csv_fund:
            fund["from_screener_csv_raw"] = csv_fund
        fund = sanitize_fundamental(fund)
        ctx["fundamental"] = fund
        ctx["canonical_valuation"] = fund.get("canonical_valuation") or {}
        # Petunjuk kontrak data untuk semua role
        ctx["data_contract"] = {
            "valuation_source": "canonical_valuation only",
            "technical_roles_forbid_valuation": True,
            "regime_is_ihsg_index_only": True,
            "regime_last_close_is_ihsg_index": True,
            "ticker_trend_source": "ma_structure + screener_row (bukan regime.reason)",
            "ticker_regime_source": "engine.signals.ticker_regime (derive_ticker_regime)",
            "ihsg_vs_ticker_must_stay_separate": True,
            "setup_truth_source": "screener_row (RetestOK, Alasan, Fibo, Score)",
            "do_not_merge_ihsg_ma_with_ticker_ma": True,
            "rejected_metrics": fund.get("rejected_metrics") or {},
        }
        # Alias eksplisit agar model tidak salah baca
        if regime:
            ctx["market_regime"] = {
                **regime,
                "_note": (
                    "Ini kondisi INDEKS IHSG. last_close = level IHSG. "
                    "reason tentang MA mengacu ke IHSG, bukan emiten."
                ),
            }
    # Engine deterministik — berlaku semua strategy (v2–v5, intra, hb, acc)
    try:
        ctx["engine"] = run_analysis_engine(ctx)
        # Ekspos rezim emiten di root context agar semua role mudah baca
        eng = ctx["engine"]
        sig = (eng or {}).get("signals") or {}
        facts = (eng or {}).get("facts") or {}
        # Jangan buang hasil detect_emiten_regime — lengkapi dari engine bila perlu
        pre = ctx.get("ticker_regime") or {}
        ctx["ticker_regime"] = {
            **pre,
            "regime": pre.get("regime")
            or sig.get("ticker_regime")
            or facts.get("ticker_regime"),
            "strength": pre.get("strength")
            or sig.get("ticker_regime_strength")
            or facts.get("ticker_regime_strength"),
            "summary": pre.get("summary")
            or sig.get("ticker_regime_summary")
            or facts.get("ticker_regime_summary"),
            "notes": pre.get("notes")
            or sig.get("ticker_regime_notes")
            or facts.get("ticker_regime_notes")
            or [],
            "alignment_with_ihsg": pre.get("alignment_with_ihsg")
            or sig.get("regime_alignment"),
            "alignment_note": pre.get("alignment_note")
            or sig.get("regime_alignment_note"),
            "source": pre.get("source") or "detect_emiten_regime",
            "_note": (
                "Rezim EMITEN precomputed. Agent AI jangan hitung ulang MA/cross. "
                "Terpisah dari market_regime = IHSG."
            ),
        }
    except Exception as e:
        ctx["engine"] = {
            "error": str(e),
            "stance": "INSUFFICIENT_DATA",
            "instructions_for_llm": "Engine gagal; jangan mengarang skor.",
        }
        ctx["ticker_regime"] = {"regime": "UNKNOWN", "error": str(e)}
    if include_news:
        try:
            ctx["news_intel"] = fetch_news_intel(ticker)
        except Exception as e:
            ctx["news_intel"] = {
                "ticker": ticker,
                "headlines": [],
                "headline_count": 0,
                "errors": [str(e)],
                "tone_hint": "unknown",
            }
    if include_llmquant:
        try:
            from idx_llmquant import build_llmquant_context_for_ticker

            strategy_hint = None
            ver = str(version).lower()
            if "v4" in ver or "ob" in ver:
                strategy_hint = "order block fair value gap institutional imbalance smart money concepts"
            elif "v5" in ver or "choch" in ver:
                strategy_hint = "change of character break of structure market structure shift liquidity"
            elif "v2" in ver or "break" in ver:
                strategy_hint = "breakout false breakout liquidity sweep momentum continuation"
            elif "accum" in ver:
                strategy_hint = "accumulation spring liquidity grab range compression volume"
            ctx["llmquant"] = build_llmquant_context_for_ticker(
                ticker, strategy_hint=strategy_hint
            )
        except Exception as e:
            ctx["llmquant"] = {
                "available": False,
                "errors": [str(e)],
                "skipped": "idx_llmquant tidak tersedia atau gagal",
            }
    return ctx


def _get_api_key() -> str | None:
    return (
        os.environ.get("OPENAI_API_KEY")
        or os.environ.get("XAI_API_KEY")
        or os.environ.get("LLM_API_KEY")
    )


def llm_available() -> bool:
    return bool(_get_api_key())


def _format_llm_error(exc: BaseException) -> str:
    parts = [f"{type(exc).__name__}: {exc}"]
    cause = getattr(exc, "__cause__", None) or getattr(exc, "__context__", None)
    if cause:
        parts.append(f"cause={type(cause).__name__}: {cause}")
    status = getattr(exc, "status_code", None)
    if status:
        parts.append(f"status={status}")
    return " | ".join(parts)


def _is_retryable_error(exc: BaseException) -> bool:
    msg = str(exc).lower()
    hints = (
        "timeout",
        "timed out",
        "connection",
        "reset",
        "temporary",
        "503",
        "502",
        "429",
        "ssl",
        "network",
    )
    if any(h in msg for h in hints):
        return True
    code = getattr(exc, "code", None)
    return code in (429, 500, 502, 503, 504)


def _extract_message_content(data: dict) -> str:
    choices = data.get("choices") or []
    if not choices:
        raise RuntimeError(f"Response tanpa choices: {str(data)[:300]}")
    msg = choices[0].get("message") or {}
    content = msg.get("content")
    if content is None:
        content = choices[0].get("text")
    if content is None:
        raise RuntimeError(
            "Model mengembalikan content kosong/null "
            f"(finish_reason={choices[0].get('finish_reason')})"
        )
    text = str(content).strip()
    if not text:
        raise RuntimeError("Model mengembalikan string kosong")
    return text


def _uses_max_completion_tokens(model: str) -> bool:
    """
    Model baru OpenAI (gpt-5*, o1*, o3*, o4*) menolak max_tokens;
    wajib max_completion_tokens.
    """
    m = (model or "").lower().strip()
    if not m:
        return False
    prefixes = (
        "gpt-5",
        "o1",
        "o3",
        "o4",
        "chatgpt-5",
    )
    return any(m.startswith(p) for p in prefixes)


def _build_chat_body(
    *,
    model: str,
    system: str,
    user: str,
    temperature: float,
    max_tokens: int,
    use_max_completion_tokens: bool | None = None,
) -> dict:
    body: dict = {
        "model": model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
    }
    # reasoning models sering menolak temperature custom
    if not _uses_max_completion_tokens(model):
        body["temperature"] = temperature
    else:
        # gpt-5 / o-series: hanya set temperature jika env mengizinkan
        if os.environ.get("OPENAI_FORCE_TEMPERATURE", "").strip() in ("1", "true", "yes"):
            body["temperature"] = temperature

    use_mct = (
        use_max_completion_tokens
        if use_max_completion_tokens is not None
        else _uses_max_completion_tokens(model)
    )
    if use_mct:
        body["max_completion_tokens"] = int(max_tokens)
    else:
        body["max_tokens"] = int(max_tokens)
    return body



def _parse_retry_after_seconds(err_body: str, default: float = 3.0) -> float:
    """Ambil 'Please try again in X.Ys' dari body 429 OpenAI."""
    import re
    if not err_body:
        return default
    m = re.search(r"try again in\s+([0-9.]+)\s*s", err_body, re.I)
    if m:
        try:
            return max(1.0, float(m.group(1)) + 0.5)
        except Exception:
            pass
    m2 = re.search(r"retry.after[\"']?\s*[:=]\s*[\"']?([0-9.]+)", err_body, re.I)
    if m2:
        try:
            return max(1.0, float(m2.group(1)))
        except Exception:
            pass
    return default


def _compact_context_for_role(role: str, context: dict) -> dict:
    """
    Kurangi payload JSON ke LLM (hemat TPM).
    Full context sering >6–8k token; compact ~1.5–3k.
    """
    eng = context.get("engine") or {}
    facts = eng.get("facts") or {}
    signals = eng.get("signals") or {}
    scores = eng.get("scores") or {}
    row = context.get("screener_row") or {}
    # Field screener penting saja
    row_keys = (
        "Ticker", "Close", "Entry", "EntryBreakout", "StopLoss",
        "Target1", "Target", "Target(Peak)", "Target(Liquidity)", "Target2",
        "RR_Ratio", "Score", "Alasan", "RetestOK", "Sweep", "SweepLow",
        "CHOCH_Level", "CHOCH_Age", "Fibo_50", "Fibo_618", "Fibo_786",
        "OB_TOP", "OB_BOTTOM", "ATR", "ADX", "ROC(10)", "DaysSinceBO",
        "Lots", "Strategy", "SetupType",
    )
    slim_row = {k: row[k] for k in row_keys if k in row and row[k] is not None}

    fund = context.get("fundamental") or {}
    fund_keys = (
        "source", "confidence", "pe_ratio", "pb_ratio", "roe", "eps",
        "dividend_yield_pct", "market_cap", "sector", "valuation_note",
        "sectors_company_url", "sectors_news_url",
    )
    slim_fund = {k: fund[k] for k in fund_keys if k in fund and fund[k] is not None}

    ma = context.get("ma_structure") or {}
    slim_ma = {
        "available": ma.get("available"),
        "ma": ma.get("ma"),
        "stack": ma.get("stack"),
        "crosses": ma.get("crosses"),
        "interpretation_hints": (ma.get("interpretation_hints") or [])[:6],
    }

    news = context.get("news_intel") or {}
    heads = news.get("headlines") or []
    slim_news = {
        "tone_hint": news.get("tone_hint"),
        "headline_count": news.get("headline_count"),
        "headlines": [
            {"title": (h.get("title") or "")[:120], "source": h.get("source")}
            for h in heads[:4]
            if isinstance(h, dict)
        ],
        "sectors_news_url": news.get("sectors_news_url"),
    }

    base = {
        "ticker": context.get("ticker"),
        "screener_version": context.get("screener_version"),
        "strategy_playbook": context.get("strategy_playbook"),
        "data_contract": context.get("data_contract"),
        "screener_row": slim_row,
        "engine": {
            "facts": facts,
            "signals": signals,
            "scores": scores,
            "stance": eng.get("stance"),
            "scorecard_text": eng.get("scorecard_text"),
            "instructions_for_llm": eng.get("instructions_for_llm"),
        },
        "ticker_regime": context.get("ticker_regime"),
        "market_regime": context.get("market_regime") or context.get("regime"),
        "account": context.get("account"),
        "fees": context.get("fees"),
        "disclaimer": context.get("disclaimer"),
    }

    if role in ("fundamental", "chief", "critic"):
        base["fundamental"] = slim_fund
        base["canonical_valuation"] = context.get("canonical_valuation") or {
            k: slim_fund.get(k) for k in ("pe_ratio", "pb_ratio", "roe")
        }
        base["news_intel"] = slim_news
    # MA mentah hanya jika rezim UNKNOWN (opsional debug); default cukup ticker_regime
    tr = context.get("ticker_regime") or {}
    if role in ("technical", "chief", "critic", "risk"):
        if str(tr.get("regime") or "UNKNOWN") == "UNKNOWN":
            base["ma_structure"] = slim_ma
        else:
            base["ma_structure"] = {
                "available": slim_ma.get("available"),
                "stack": slim_ma.get("stack"),
                "note": "Detail MA di ticker_regime; jangan hitung ulang.",
            }
    if role == "chief":
        # analyses diisi di run_agents
        if "analyses" in context:
            # potong panjang per role
            an = {}
            for k, v in (context.get("analyses") or {}).items():
                s = str(v or "")
                an[k] = s if len(s) <= 1800 else s[:1800] + "\n…[truncated]"
            base["analyses"] = an
        base["engine_scorecard_locked"] = context.get("engine_scorecard_locked") or eng.get(
            "scorecard_text"
        )
    if role == "risk":
        base["fundamental"] = {
            k: slim_fund.get(k) for k in ("source", "confidence", "market_cap")
        }
    return base


def _chat(
    system: str,
    user: str,
    *,
    model: str | None = None,
    base_url: str | None = None,
    temperature: float = 0.3,
    max_tokens: int = 900,
) -> str:
    api_key = _get_api_key()
    if not api_key:
        raise RuntimeError(
            "API key tidak ditemukan. Set OPENAI_API_KEY atau XAI_API_KEY."
        )

    model = model or _default_model()
    base_url = (base_url or _default_base_url()).rstrip("/")
    timeout_sec = float(os.environ.get("OPENAI_TIMEOUT", "180"))
    # Default lebih longgar untuk 429 TPM (tier rendah sering 30k TPM)
    max_retries = int(os.environ.get("OPENAI_MAX_RETRIES", "5"))

    import urllib.request
    import urllib.error

    body = _build_chat_body(
        model=model,
        system=system,
        user=user,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    payload = json.dumps(body).encode("utf-8")

    last_err: Exception | None = None
    attempts = max(1, max_retries + 1)

    for attempt in range(1, attempts + 1):
        try:
            req = urllib.request.Request(
                f"{base_url}/chat/completions",
                data=payload,
                headers={
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {api_key}",
                },
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=timeout_sec) as r:
                raw = r.read().decode("utf-8")
            data = json.loads(raw)
            return _extract_message_content(data)
        except urllib.error.HTTPError as he:
            err_body = he.read().decode("utf-8", errors="ignore")
            last_err = RuntimeError(f"HTTPError {he.code}: {err_body}")

            # Auto-fallback: max_tokens → max_completion_tokens (model baru)
            if he.code == 400 and "max_tokens" in err_body and "max_completion_tokens" in err_body:
                body = _build_chat_body(
                    model=model,
                    system=system,
                    user=user,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    use_max_completion_tokens=True,
                )
                payload = json.dumps(body).encode("utf-8")
                if attempt < attempts:
                    continue
                # coba sekali lagi meski attempt terakhir
                try:
                    req2 = urllib.request.Request(
                        f"{base_url}/chat/completions",
                        data=payload,
                        headers={
                            "Content-Type": "application/json",
                            "Authorization": f"Bearer {api_key}",
                        },
                        method="POST",
                    )
                    with urllib.request.urlopen(req2, timeout=timeout_sec) as r2:
                        raw2 = r2.read().decode("utf-8")
                    return _extract_message_content(json.loads(raw2))
                except Exception as e2:
                    last_err = e2

            # Auto-fallback: buang temperature jika model menolak
            if he.code == 400 and "temperature" in err_body.lower():
                body.pop("temperature", None)
                if "max_tokens" in body and _uses_max_completion_tokens(model):
                    body["max_completion_tokens"] = body.pop("max_tokens")
                payload = json.dumps(body).encode("utf-8")
                if attempt < attempts:
                    continue

            if he.code == 429 and attempt < attempts:
                wait = _parse_retry_after_seconds(err_body, default=float(2 ** attempt))
                wait = min(max(wait, 1.5), 60.0)
                time.sleep(wait)
                continue
            if he.code in (500, 502, 503, 504) and attempt < attempts:
                time.sleep(min(2 ** attempt, 12))
                continue
            raise last_err from he
        except Exception as e:
            last_err = e
            if attempt < attempts and _is_retryable_error(e):
                msg = str(e)
                wait = _parse_retry_after_seconds(msg, default=float(2 ** attempt))
                time.sleep(min(max(wait, 1.5), 60.0))
                continue
            raise RuntimeError(_format_llm_error(e)) from e

    raise RuntimeError(_format_llm_error(last_err or RuntimeError("unknown")))


def _offline_fallback(role: str, context: dict) -> str:
    row = context.get("screener_row") or {}
    fund = context.get("fundamental") or {}
    ticker = context.get("ticker", "?")
    rr = row.get("RR_Ratio") or row.get("RR")
    score = row.get("Score")
    sweep = row.get("Sweep") or row.get("SweepLow")
    entry = row.get("Entry") or row.get("EntryBreakout") or row.get("Close")
    sl = row.get("StopLoss")
    tp = (
        row.get("Target1")
        or row.get("Target(Peak)")
        or row.get("Target")
        or row.get("Target(Liquidity)")
    )

    if role == "technical":
        return "\n".join(
            [
                f"[Offline] Technical — {ticker}",
                f"- Entry/Close: {entry} | SL: {sl} | TP: {tp}",
                f"- RR: {rr} | Score: {score} | Sweep: {sweep}",
            ]
        )
    if role == "fundamental":
        news = context.get("news_intel") or {}
        heads = news.get("headlines") or []
        top = "; ".join((h.get("title") or "")[:60] for h in heads[:3] if h.get("title"))
        return "\n".join(
            [
                f"[Offline] Fundamental — {ticker}",
                f"- Source: {fund.get('source')}",
                f"- PE: {fund.get('pe_ratio')} | PB: {fund.get('pb_ratio')} | EPS: {fund.get('eps')}",
                f"- Sektor: {fund.get('sector')} | MktCap: {fund.get('market_cap')}",
                f"- Note: {fund.get('valuation_note')}",
                f"- News tone: {news.get('tone_hint')} | n={news.get('headline_count', 0)}",
                f"- Headlines: {top or 'tidak ada'}",
            ]
        )
    if role == "risk":
        acc = context.get("account") or {}
        fees = context.get("fees") or {}
        return "\n".join(
            [
                f"[Offline] Risk — {ticker}",
                f"- Modal: Rp {acc.get('size_rp', 0):,.0f} | Risk: {acc.get('risk_per_trade_pct')}%",
                f"- Fee: {fees.get('broker_buy_pct')}% / {fees.get('broker_sell_pct')}%",
                f"- Lots: {row.get('Lots') or row.get('SuggestedLots')}",
            ]
        )
    if role == "critic":
        pb = context.get("strategy_playbook") or get_strategy_playbook(
            context.get("screener_version")
        )
        inv = pb.get("invalidation") or []
        lines = [
            f"[Offline] Critic — {ticker} | {pb.get('name')}",
            f"- Thesis: {pb.get('thesis')}",
            f"- Fokus: {pb.get('critic_focus')}",
        ]
        for x in inv[:4]:
            lines.append(f"- Invalidation: {x}")
        lines.append("- Alignment CONFLICT / SL vs ATR / data quality tetap dicek.")
        return "\n".join(lines)
    bias = "watch"
    try:
        if rr is not None and float(rr) >= 2 and score is not None and float(score) >= 70:
            bias = "consider"
        if rr is not None and float(rr) < 1.3:
            bias = "avoid"
    except Exception:
        pass
    return "\n".join(
        [
            f"[Offline] Chief — {ticker}",
            f"- Skor kasar: {score} | Bias: {bias}",
            f"- Fundamental: {fund.get('valuation_note', 'n/a')}",
            "- Bukan rekomendasi investasi.",
        ]
    )


def run_role(role: str, context: dict, *, model: str | None = None) -> str:
    if role not in SYSTEM_PROMPTS:
        raise ValueError(f"Role tidak dikenal: {role}")
    if not llm_available():
        return _offline_fallback(role, context)
    # Pastikan playbook strategy ada di context (critic/chief)
    ver = context.get("screener_version") or context.get("version")
    if "strategy_playbook" not in context:
        context = {**context, "strategy_playbook": get_strategy_playbook(ver)}
    compact = _compact_context_for_role(role, context)
    user_payload = json.dumps(compact, ensure_ascii=False, separators=(",", ":"))
    max_tokens = ROLE_MAX_TOKENS.get(role, 700)
    system = (
        build_critic_system_prompt(ver)
        if role == "critic"
        else SYSTEM_PROMPTS[role]
    )
    return _chat(
        system,
        user_payload,
        model=model,
        max_tokens=max_tokens,
    )


def _ensure_sectors_link_in_chief(chief_text: str, context: dict) -> str:
    """Sisipkan deep-link Sectors (berita + profil) setelah Overall di scorecard Chief."""
    text = chief_text or ""
    t = str(context.get("ticker") or "TICKER").upper().strip()
    news = context.get("news_intel") or {}
    fund = context.get("fundamental") or {}

    news_url = news.get("sectors_news_url") or fund.get("sectors_news_url")
    if not news_url:
        news_url = f"https://sectors.app/indonesia/news?nticker={t}.JK"
    company_url = fund.get("sectors_company_url")
    if not company_url:
        company_url = _sectors_urls(t)["sectors_company_url"]

    conf = fund.get("confidence") or (
        "low" if str(fund.get("source", "")).lower() == "yfinance" else None
    )
    src = fund.get("source") or "unknown"
    conf_line = None
    if conf or src:
        conf_line = f"- Fundamental confidence: {conf or 'n/a'} ({src})"

    line_news = f"- Berita Sectors (filter emiten): {news_url}"
    line_co = f"- Profil/fundamental Sectors (verifikasi manual): {company_url}"

    lines = text.splitlines()
    out: list[str] = []
    inserted = False
    for ln in lines:
        # drop old auto lines so we can re-insert cleanly
        if (
            "Berita Sectors" in ln
            or "Profil/fundamental Sectors" in ln
            or "Fundamental confidence:" in ln
        ):
            continue
        out.append(ln)
        if not inserted and "Overall:" in ln:
            if conf_line:
                out.append(conf_line)
            out.append(line_news)
            out.append(line_co)
            inserted = True
    if not inserted:
        block = ([conf_line] if conf_line else []) + [line_news, line_co, ""]
        out = block + out
    return "\n".join(out)



def run_agents(
    context: dict,
    *,
    model: str | None = None,
    roles: list[str] | None = None,
) -> dict[str, Any]:
    roles = roles or list(ROLE_ORDER)
    offline = not llm_available()
    analyses: dict[str, str] = {}
    errors: dict[str, str] = {}

    non_chief = [r for r in roles if r != "chief"]
    for i, role in enumerate(non_chief):
        if i > 0 and not offline:
            time.sleep(ROLE_GAP_SEC)
        try:
            analyses[role] = run_role(role, context, model=model)
        except Exception as e:
            errors[role] = _format_llm_error(e)
            analyses[role] = f"[Gagal] {errors[role]}"
            # cooldown ekstra setelah 429
            if "429" in str(e) or "rate_limit" in str(e).lower():
                time.sleep(max(ROLE_GAP_SEC, 5.0))

    if "chief" in roles:
        ok = {k: v for k, v in analyses.items() if not str(v).startswith("[Gagal]")}
        eng = context.get("engine") or {}
        scorecard = eng.get("scorecard_text") or ""
        if not ok and not scorecard:
            analyses["chief"] = (
                "[Gagal] Tidak ada analisa role yang berhasil; chief dilewati. "
                + ("; ".join(f"{k}: {v}" for k, v in errors.items()) or "")
            )
            errors["chief"] = analyses["chief"]
        else:
            try:
                if not offline:
                    time.sleep(ROLE_GAP_SEC)
                analyses["chief"] = run_role(
                    "chief",
                    {
                        **context,
                        "analyses": ok,
                        "engine_scorecard_locked": scorecard,
                    },
                    model=model,
                )
            except Exception as e:
                errors["chief"] = _format_llm_error(e)
                analyses["chief"] = f"[Gagal] {errors['chief']}"
            else:
                # Paksa scorecard engine di atas (model tidak boleh menimpa angka)
                body = analyses["chief"] or ""
                if scorecard:
                    # Buang scorecard lama model jika ada, prepend locked card
                    lines = body.splitlines()
                    filtered = []
                    skip = False
                    for ln in lines:
                        if "SCORECARD" in ln.upper() and "ENGINE" not in ln.upper():
                            skip = True
                            continue
                        if skip and (
                            ln.startswith("- ")
                            or ln.startswith("**")
                            or "Overall" in ln
                            or "stance" in ln.lower()
                        ):
                            continue
                        skip = False
                        filtered.append(ln)
                    analyses["chief"] = scorecard + "\n\n" + "\n".join(filtered).strip()
                analyses["chief"] = _ensure_sectors_link_in_chief(
                    analyses["chief"], context
                )

    return {
        "ticker": context.get("ticker"),
        "version": context.get("screener_version"),
        "offline": offline,
        "analyses": analyses,
        "errors": errors,
        "partial": bool(errors),
        "engine": context.get("engine"),
        "context": context,
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }


def analyze_ticker(
    ticker: str,
    version: str,
    *,
    regime: dict | None = None,
    account_size: float = 50_000_000,
    risk_pct: float = 1.0,
    broker_buy_pct: float = 0.15,
    broker_sell_pct: float = 0.25,
    model: str | None = None,
    include_fundamental: bool = True,
    include_news: bool = True,
    include_llmquant: bool = True,
) -> dict[str, Any]:
    ctx = build_context(
        ticker,
        version,
        regime=regime,
        account_size=account_size,
        risk_pct=risk_pct,
        broker_buy_pct=broker_buy_pct,
        broker_sell_pct=broker_sell_pct,
        include_fundamental=include_fundamental,
        include_news=include_news,
        include_llmquant=include_llmquant,
    )
    if not ctx.get("screener_row"):
        return {
            "ticker": ticker,
            "version": version,
            "error": f"Ticker {ticker} tidak ada di report {version}.",
            "analyses": {},
            "offline": not llm_available(),
        }
    return run_agents(ctx, model=model)


def analyze_many(tickers: list[str], version: str, **kwargs) -> list[dict]:
    return [analyze_ticker(t, version, **kwargs) for t in tickers]


def save_analysis_to_txt(
    result: dict[str, Any],
    path: str | Path | None = None,
    *,
    output_dir: str | Path = "ai_analysis_output",
    include_errors: bool = True,
    include_context_summary: bool = True,
) -> str:
    """
    Simpan hasil analyse_ticker / run_agents ke file .txt.

    Parameters
    ----------
    result : dict
        Output dari analyze_ticker() atau run_agents().
    path : optional
        Path file tujuan. Jika None, dibuat otomatis:
        {output_dir}/{TICKER}_{version}_{YYYYMMDD_HHMMSS}.txt
    output_dir : str
        Folder default bila path tidak diberikan.
    include_errors : bool
        Sertakan section errors bila ada.
    include_context_summary : bool
        Ringkas screener_row / fundamental / regime (bukan full JSON).

    Returns
    -------
    str
        Path absolut file yang ditulis.
    """
    if not isinstance(result, dict):
        raise TypeError("result harus dict (output analyze_ticker/run_agents)")

    ticker = str(result.get("ticker") or "UNKNOWN").upper().replace(".JK", "")
    version = str(result.get("version") or result.get("screener_version") or "na")
    generated = str(result.get("generated_at") or datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

    if path is None:
        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = out_dir / f"{ticker}_{version}_{stamp}.txt"
    else:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

    lines: list[str] = []
    lines.append("=" * 72)
    lines.append("IDX AI TRADER ASSISTANT — ANALISA")
    lines.append("=" * 72)
    lines.append(f"Ticker      : {ticker}")
    lines.append(f"Screener    : {version}")
    lines.append(f"Generated   : {generated}")
    lines.append(f"Offline     : {result.get('offline', False)}")
    lines.append(f"Partial     : {result.get('partial', False)}")
    if result.get("error"):
        lines.append(f"Error       : {result.get('error')}")
    lines.append("")

    eng = result.get("engine") or (result.get("context") or {}).get("engine")
    if eng and isinstance(eng, dict) and not eng.get("error"):
        lines.append("-" * 72)
        lines.append("ENGINE (deterministik)")
        lines.append("-" * 72)
        sig = eng.get("signals") or {}
        sc = eng.get("scores") or {}
        lines.append(f"  trend={sig.get('trend_signal')} momentum={sig.get('momentum_signal')}")
        lines.append(
            f"  fib={sig.get('fib_signal')} dist_atr={sig.get('fib_distance_atr')} "
            f"retest={sig.get('retest_status')}"
        )
        lines.append(
            f"  market={sig.get('market_regime')} valuation={sig.get('valuation_signal')}"
        )
        lines.append(
            f"  RR_tp1={sig.get('rr_tp1')} RR_tp2={sig.get('rr_tp2')} "
            f"expected_RR={sig.get('expected_rr')}"
        )
        lines.append(
            f"  scores: tech={sc.get('technical_score')} fund={sc.get('fundamental_score')} "
            f"risk={sc.get('risk_score')} mkt={sc.get('market_score')} "
            f"final={sc.get('final_score')}"
        )
        lines.append(f"  stance={eng.get('stance')} quality={eng.get('data_quality')}")
        lines.append("")

    # Ringkasan konteks (opsional)
    ctx = result.get("context") or {}
    if include_context_summary and ctx:
        lines.append("-" * 72)
        lines.append("RINGKASAN KONTEKS")
        lines.append("-" * 72)
        row = ctx.get("screener_row") or {}
        if row:
            lines.append("[Screener row]")
            for k in (
                "Ticker",
                "SetupType",
                "Close",
                "EntryBreakout",
                "Entry",
                "StopLoss",
                "Target1",
                "Target",
                "RR_Ratio",
                "Score",
                "Sweep",
                "Alasan",
            ):
                if k in row and row[k] is not None:
                    lines.append(f"  {k}: {row[k]}")
            lines.append("")
        fund = ctx.get("fundamental") or {}
        if fund:
            lines.append("[Fundamental]")
            for k in (
                "source",
                "confidence",
                "pe_ratio",
                "pb_ratio",
                "eps",
                "roe",
                "valuation_note",
                "as_of",
            ):
                if fund.get(k) is not None:
                    lines.append(f"  {k}: {fund.get(k)}")
            lines.append("")
        tr = ctx.get("ticker_regime") or {}
        if tr:
            lines.append("[Rezim Emiten]")
            for k in ("regime", "strength", "summary", "alignment_with_ihsg", "alignment_note"):
                if tr.get(k) is not None:
                    lines.append(f"  {k}: {tr.get(k)}")
            notes = tr.get("notes") or []
            if notes:
                lines.append(f"  notes: {'; '.join(str(n) for n in notes[:6])}")
            lines.append("")
        regime = ctx.get("regime") or ctx.get("market_regime") or {}
        if regime:
            lines.append("[Regime IHSG]")
            for k, v in list(regime.items())[:12]:
                if str(k).startswith("_"):
                    continue
                lines.append(f"  {k}: {v}")
            lines.append("")

    # Analisa per role
    analyses = result.get("analyses") or {}
    role_order = list(ROLE_ORDER)
    for role in role_order:
        if role not in analyses:
            continue
        lines.append("=" * 72)
        lines.append(f"ROLE: {role.upper()}")
        lines.append("=" * 72)
        text = analyses.get(role) or ""
        lines.append(str(text).rstrip())
        lines.append("")

    # Role lain di luar urutan standar
    for role, text in analyses.items():
        if role in role_order:
            continue
        lines.append("=" * 72)
        lines.append(f"ROLE: {role.upper()}")
        lines.append("=" * 72)
        lines.append(str(text or "").rstrip())
        lines.append("")

    errors = result.get("errors") or {}
    if include_errors and errors:
        lines.append("-" * 72)
        lines.append("ERRORS")
        lines.append("-" * 72)
        for k, v in errors.items():
            lines.append(f"  [{k}] {v}")
        lines.append("")

    lines.append("=" * 72)
    lines.append("Disclaimer: Bukan rekomendasi investasi. Verifikasi manual wajib.")
    lines.append("=" * 72)
    lines.append("")

    path.write_text("\n".join(lines), encoding="utf-8")
    return str(path.resolve())


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(description="IDX AI Trader Assistant")
    p.add_argument("ticker", help="Kode emiten")
    p.add_argument("--version", default="v2")
    p.add_argument("--model", default=None)
    p.add_argument(
        "--save-txt",
        nargs="?",
        const="",
        default=None,
        help="Simpan analisa ke .txt (path opsional; default folder ai_analysis_output/)",
    )
    args = p.parse_args()

    out = analyze_ticker(args.ticker, args.version, model=args.model)
    if args.save_txt is not None:
        save_path = args.save_txt.strip() or None
        written = save_analysis_to_txt(out, path=save_path)
        print(f"Analisa disimpan: {written}")
    print(
        json.dumps(
            {k: v for k, v in out.items() if k != "context"},
            ensure_ascii=False,
            indent=2,
        )
    )
    if out.get("analyses"):
        for role, text in out["analyses"].items():
            print("\n" + "=" * 60)
            print(role.upper())
            print("=" * 60)
            print(text)