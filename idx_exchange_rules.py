"""
IDX Exchange Rules — shared tick size + ARA/ARB (BEII-A update Sep 2026)

Efektif:
  - 28 Sep 2026: harga minimum Rp1 (sebelumnya Rp50)
  - 28 Sep – 31 Des 2026: ARB asimetris 15% untuk harga > Rp10;
    band Rp1–10: ARA/ARB = ±Rp1 nominal
  - 1 Jan 2027+: ARB simetris dengan ARA per rentang harga

Referensi: SK Direksi BEI Kep-00136/BEI/09-2026 (Perubahan II-A)
"""

from __future__ import annotations

from datetime import date, datetime
from typing import Optional, Tuple, Union

import pandas as pd

# ---------------------------------------------------------------------------
# Tanggal efektif
# ---------------------------------------------------------------------------
FLOOR_PRICE_EFFECTIVE = date(2026, 9, 28)  # min price Rp50 → Rp1
ARB_SYMMETRIC_EFFECTIVE = date(2027, 1, 1)  # ARB = ARA% untuk band >10

MIN_PRICE_FLOOR = 1.0  # sejak 28 Sep 2026
MIN_PRICE_FLOOR_LEGACY = 50.0

# Lantai internal default untuk screener (bukan aturan bursa)
# — hindari noise penny stock di setup day/swing
DEFAULT_SCREENER_MIN_PRICE = 50.0


def _as_date(d: Optional[Union[date, datetime, str]] = None) -> date:
    if d is None:
        return date.today()
    if isinstance(d, datetime):
        return d.date()
    if isinstance(d, date):
        return d
    if isinstance(d, str):
        return datetime.strptime(d[:10], "%Y-%m-%d").date()
    return date.today()


def is_post_floor_change(asof: Optional[Union[date, datetime, str]] = None) -> bool:
    return _as_date(asof) >= FLOOR_PRICE_EFFECTIVE


def is_arb_symmetric(asof: Optional[Union[date, datetime, str]] = None) -> bool:
    return _as_date(asof) >= ARB_SYMMETRIC_EFFECTIVE


# ---------------------------------------------------------------------------
# Tick size (aturan fraksi harga BEI yang umum dipakai di screener)
# ---------------------------------------------------------------------------
def round_to_idx_tick(price: float) -> int:
    """
    Bulatkan ke fraksi harga IDX yang dipakai di pipeline screener.
    Untuk harga < 50: 1 rupiah (cocok rezim post-floor Rp1).
    """
    if price is None or (isinstance(price, float) and pd.isna(price)) or price <= 0:
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


# Alias kompatibilitas
_round_tick = round_to_idx_tick


# ---------------------------------------------------------------------------
# ARA / ARB
# ---------------------------------------------------------------------------
def get_ara_arb_limits(
    prev_close: float,
    *,
    asof: Optional[Union[date, datetime, str]] = None,
) -> Tuple[float, float, str]:
    """
    Hitung batas ARA (atas) dan ARB (bawah) absolut dari prev_close.

    Returns:
        (ara_price, arb_price, rule_label)
    """
    pc = float(prev_close or 0)
    if pc <= 0:
        return 0.0, 0.0, "invalid"

    asof_d = _as_date(asof)
    symmetric = is_arb_symmetric(asof_d)

    # Band Rp1 – Rp10: nominal ±Rp1 (sejak floor change; sebelum itu band ini tidak ada di reguler)
    if pc <= 10:
        if is_post_floor_change(asof_d):
            ara = round_to_idx_tick(pc + 1)
            arb = max(MIN_PRICE_FLOOR, round_to_idx_tick(pc - 1))
            return float(ara), float(arb), "nominal_Rp1_band_1_10"
        # legacy: treat as <200 style if somehow needed
        ara = round_to_idx_tick(pc * 1.35)
        arb = round_to_idx_tick(pc * 0.85)  # historical ARB often 15%
        return float(ara), float(arb), "legacy_sub10"

    # Band >10 – 200
    if pc <= 200:
        ara_pct = 0.35
        arb_pct = 0.35 if symmetric else 0.15
        label = "band_11_200_sym" if symmetric else "band_11_200_arb15"
    # Band >200 – 5000
    elif pc <= 5000:
        ara_pct = 0.25
        arb_pct = 0.25 if symmetric else 0.15
        label = "band_201_5000_sym" if symmetric else "band_201_5000_arb15"
    # Band >5000
    else:
        ara_pct = 0.20
        arb_pct = 0.20 if symmetric else 0.15
        label = "band_gt5000_sym" if symmetric else "band_gt5000_arb15"

    ara = round_to_idx_tick(pc * (1.0 + ara_pct))
    arb = round_to_idx_tick(pc * (1.0 - arb_pct))
    # Jangan di bawah lantai harga
    floor = MIN_PRICE_FLOOR if is_post_floor_change(asof_d) else MIN_PRICE_FLOOR_LEGACY
    arb = max(float(arb), floor)
    return float(ara), float(arb), label


def apply_ara_arb_limits(
    price: float,
    prev_close: float,
    is_target: bool,
    *,
    asof: Optional[Union[date, datetime, str]] = None,
) -> float:
    """
    Clamp harga ke batas ARA (jika target) atau ARB (jika stop/SL).

    Kompatibel dengan signature lama:
        apply_ara_arb_limits(price, prev_close, is_target=True/False)
    """
    if price is None or prev_close is None:
        return float(price or 0)
    try:
        price_f = float(price)
        pc = float(prev_close)
    except (TypeError, ValueError):
        return float(price or 0)

    if pc <= 0:
        return round_to_idx_tick(price_f)

    ara, arb, _ = get_ara_arb_limits(pc, asof=asof)
    if is_target:
        return float(min(round_to_idx_tick(price_f), ara))
    return float(max(round_to_idx_tick(price_f), arb))


def clamp_stop_to_arb(
    stop: float,
    prev_close: float,
    *,
    asof: Optional[Union[date, datetime, str]] = None,
) -> float:
    """Pastikan SL tidak di bawah ARB harian (order tidak ditolak / gap risk jelas)."""
    return apply_ara_arb_limits(stop, prev_close, is_target=False, asof=asof)


def clamp_target_to_ara(
    target: float,
    prev_close: float,
    *,
    asof: Optional[Union[date, datetime, str]] = None,
) -> float:
    """Pastikan TP tidak di atas ARA harian."""
    return apply_ara_arb_limits(target, prev_close, is_target=True, asof=asof)


def describe_rules(asof: Optional[Union[date, datetime, str]] = None) -> str:
    """Ringkasan human-readable untuk dashboard / log."""
    d = _as_date(asof)
    floor = MIN_PRICE_FLOOR if is_post_floor_change(d) else MIN_PRICE_FLOOR_LEGACY
    if is_arb_symmetric(d):
        arb_note = "ARB simetris dengan ARA (35/25/20% per band; Rp1–10: ±Rp1)"
    elif is_post_floor_change(d):
        arb_note = "Transisi: ARB 15% (harga >10); Rp1–10: ±Rp1 nominal"
    else:
        arb_note = "Legacy: min Rp50; ARB asimetris 15%"
    return (
        f"asof={d.isoformat()} | floor=Rp{floor:g} | {arb_note} | "
        f"sym_from={ARB_SYMMETRIC_EFFECTIVE.isoformat()}"
    )


__all__ = [
    "FLOOR_PRICE_EFFECTIVE",
    "ARB_SYMMETRIC_EFFECTIVE",
    "MIN_PRICE_FLOOR",
    "MIN_PRICE_FLOOR_LEGACY",
    "DEFAULT_SCREENER_MIN_PRICE",
    "round_to_idx_tick",
    "get_ara_arb_limits",
    "apply_ara_arb_limits",
    "clamp_stop_to_arb",
    "clamp_target_to_ara",
    "is_post_floor_change",
    "is_arb_symmetric",
    "describe_rules",
]
