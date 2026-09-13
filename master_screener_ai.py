"""
IDX MASTER SCREENER AI — ADAPTIVE MARKET REGIME
============================================================
Orchestrator:
  1. Analisa rezim IHSG + aktivitas pasar (RAMAI/NORMAL/SEPI)
  2. Jalankan strategi yang sesuai (V2–V5)
  3. Enrich hasil CSV per versi:
     - Trailing Stop
     - Fundamental + valuasi
     - Biaya broker + pajak Indonesia (nett P/L)

Rezim → Strategi:
  BULLISH_STRONG     → V2
  BULLISH_PULLBACK   → V3
  SIDEWAYS           → V3, V4
  BEARISH_WEAK/STRONG→ V5

[UPDATE]
- Aktivitas pasar: Volume ^JKSE sering 0 di Yahoo → validasi dulu;
  fallback ke range 5D/20D + ATR ratio (bukan 0x palsu).
- Indentasi blok activity diperbaiki.
- Return lengkap untuk dashboard (activity, vol_ratio, range, atr_ratio).
"""

from __future__ import annotations

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime
import importlib
import os
import glob

# --- Trailing Stop ---
try:
    from idx_trailing_stop import enrich_with_trailing_stop
    TRAILING_AVAILABLE = True
except ImportError:
    TRAILING_AVAILABLE = False
    print("[WARNING] idx_trailing_stop.py tidak ditemukan. Trailing stop dilewati.")

# --- Fundamental + Valuasi ---
try:
    from idx_fundamental import enrich_with_fundamental
    FUNDAMENTAL_AVAILABLE = True
except ImportError:
    FUNDAMENTAL_AVAILABLE = False
    print("[WARNING] idx_fundamental.py tidak ditemukan. Fundamental dilewati.")

# --- Biaya + Pajak ---
try:
    from idx_cost_tax import enrich_dataframe_with_costs
    COST_AVAILABLE = True
except ImportError:
    COST_AVAILABLE = False
    print("[WARNING] idx_cost_tax.py tidak ditemukan. Biaya/pajak dilewati.")


def _search_dirs():
    dirs = ["."]
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


def _find_today_report(version: str, day: str | None = None) -> str | None:
    """Cari idx_report_{version}_YYYY-MM-DD.csv di beberapa folder."""
    day = day or datetime.now().strftime("%Y-%m-%d")
    name = f"idx_report_{version}_{day}.csv"
    for d in _search_dirs():
        path = os.path.join(d, name)
        if os.path.exists(path):
            return path
    files = []
    for d in _search_dirs():
        files.extend(glob.glob(os.path.join(d, f"idx_report_{version}_*.csv")))
    if not files:
        return None
    return max(files, key=os.path.getmtime)


def _compute_atr(df: pd.DataFrame, period: int = 14) -> pd.Series | None:
    if not {"High", "Low", "Close"}.issubset(df.columns):
        return None
    h, l, c = df["High"], df["Low"], df["Close"]
    prev = c.shift(1)
    tr = pd.concat([h - l, (h - prev).abs(), (l - prev).abs()], axis=1).max(axis=1)
    return tr.rolling(period).mean()


def _volume_is_valid(vol: pd.Series, min_nonzero: int = 10) -> bool:
    """Yahoo sering isi Volume=0 untuk ^JKSE — anggap invalid."""
    if vol is None or vol.empty:
        return False
    v = vol.tail(20).replace(0, np.nan).dropna()
    return len(v) >= min_nonzero and float(v.mean()) > 0


def analyze_ihsg_regime(lookback_days: int = 180) -> dict:
    """
    Analisa rezim IHSG (5 state) + aktivitas RAMAI/NORMAL/SEPI.

    Aktivitas:
      - Volume dipakai HANYA jika valid (bukan deret 0).
      - Jika volume invalid → proxy range 5D vs 20D + ATR5/ATR14.
    """
    print("\n" + "=" * 80)
    print("MENGANALISA REZIM PASAR IHSG SAAT INI (^JKSE)")
    print("=" * 80)

    try:
        df = yf.download(
            "^JKSE",
            period=f"{lookback_days}d",
            interval="1d",
            progress=False,
            auto_adjust=True,
            multi_level_index=False,
        )
    except Exception as e:
        print(f"Gagal mengambil data IHSG: {e}")
        return {
            "regime": "UNKNOWN",
            "reason": ["Data IHSG tidak tersedia."],
            "strategies": [],
            "activity": "UNKNOWN",
            "activity_reason": [],
        }

    if df is None or df.empty:
        return {
            "regime": "UNKNOWN",
            "reason": ["Data IHSG kosong."],
            "strategies": [],
            "activity": "UNKNOWN",
            "activity_reason": [],
        }

    if isinstance(df.columns, pd.MultiIndex):
        try:
            df.columns = df.columns.get_level_values(0)
        except Exception:
            try:
                df = df.droplevel(-1, axis=1)
            except Exception:
                pass

    if "Close" not in df.columns:
        return {
            "regime": "UNKNOWN",
            "reason": ["Kolom Close tidak ditemukan."],
            "strategies": [],
            "activity": "UNKNOWN",
            "activity_reason": [],
        }

    # Pakai bar yang punya Close; jaga High/Low/Volume sejajar
    df = df.copy()
    df = df[df["Close"].notna()]
    close = df["Close"]
    if len(close) < 100:
        return {
            "regime": "UNKNOWN",
            "reason": ["Data tidak cukup (<100 bar)."],
            "strategies": [],
            "activity": "UNKNOWN",
            "activity_reason": [],
        }

    ma20 = close.rolling(20).mean()
    ma50 = close.rolling(50).mean()
    ma100 = close.rolling(100).mean()

    last_close = float(close.iloc[-1])
    last_ma20 = float(ma20.iloc[-1])
    last_ma50 = float(ma50.iloc[-1])
    last_ma100 = float(ma100.iloc[-1])

    if any(pd.isna(x) for x in [last_ma20, last_ma50, last_ma100]):
        return {
            "regime": "UNKNOWN",
            "reason": ["MA mengandung NaN."],
            "strategies": [],
            "activity": "UNKNOWN",
            "activity_reason": [],
        }

    ma20_prev = float(ma20.iloc[-6]) if len(ma20) >= 6 else last_ma20
    ma50_prev = float(ma50.iloc[-6]) if len(ma50) >= 6 else last_ma50
    ma20_rising = last_ma20 > ma20_prev
    ma50_rising = last_ma50 > ma50_prev
    ma20_falling = last_ma20 < ma20_prev
    ma50_falling = last_ma50 < ma50_prev

    # High/Low lebih akurat dari kolom High/Low jika ada
    if "High" in df.columns and "Low" in df.columns:
        high_20 = float(df["High"].tail(20).max())
        low_20 = float(df["Low"].tail(20).min())
        high_60 = float(df["High"].tail(60).max()) if len(df) >= 60 else high_20
        low_60 = float(df["Low"].tail(60).min()) if len(df) >= 60 else low_20
        high_5 = float(df["High"].tail(5).max())
        low_5 = float(df["Low"].tail(5).min())
    else:
        high_20 = float(close.tail(20).max())
        low_20 = float(close.tail(20).min())
        high_60 = float(close.tail(60).max()) if len(close) >= 60 else high_20
        low_60 = float(close.tail(60).min()) if len(close) >= 60 else low_20
        high_5 = float(close.tail(5).max())
        low_5 = float(close.tail(5).min())

    dist_to_high_20 = (high_20 - last_close) / last_close * 100
    dist_to_low_20 = (last_close - low_20) / low_20 * 100
    dist_to_high_60 = (high_60 - last_close) / last_close * 100
    dist_to_low_60 = (last_close - low_60) / low_60 * 100
    range_20_pct = (high_20 - low_20) / last_close * 100
    range_5_pct = (high_5 - low_5) / last_close * 100

    bullish_stack = last_close > last_ma20 > last_ma50 > last_ma100
    above_ma50 = last_close > last_ma50
    above_ma100 = last_close > last_ma100
    below_ma50 = last_close < last_ma50
    below_ma20 = last_close < last_ma20

    # =================================================================
    # REZIM
    # =================================================================
    regime = "SIDEWAYS"
    reason: list[str] = []
    strategies: list[str] = []

    if bullish_stack and ma20_rising and ma50_rising and dist_to_high_20 <= 5.0:
        regime = "BULLISH_STRONG"
        reason += [
            "Uptrend kuat: Close > MA20 > MA50 > MA100.",
            "MA20 & MA50 sedang naik (momentum positif).",
            f"Harga dekat high 20 hari (jarak {dist_to_high_20:.1f}%).",
        ]
        strategies = ["V2"]
    elif above_ma50 and above_ma100 and (last_ma50 > last_ma100) and below_ma20:
        regime = "BULLISH_PULLBACK"
        reason += [
            "Struktur menengah masih bullish (Close > MA50 > MA100).",
            "Harga sedang pullback di bawah MA20 (koreksi wajar).",
        ]
        if ma50_rising:
            reason.append("MA50 masih naik — pullback dalam uptrend.")
        strategies = ["V3"]
    elif (
        below_ma20
        and below_ma50
        and (ma20_falling or ma50_falling)
        and dist_to_low_20 <= 5.0
    ):
        regime = "BEARISH_STRONG"
        reason += [
            "Downtrend: Close < MA20 & MA50.",
            "MA sedang turun (momentum negatif).",
            f"Harga dekat low 20 hari (jarak {dist_to_low_20:.1f}%).",
        ]
        strategies = ["V5"]
    elif below_ma50 and below_ma20:
        regime = "BEARISH_WEAK"
        reason.append("Harga di bawah MA20 & MA50 (tekanan jual).")
        reason.append("Belum di zona low ekstrem — downtrend / distribusi.")
        if dist_to_low_60 < 8.0:
            reason.append(f"Mendekati low 60 hari (jarak {dist_to_low_60:.1f}%).")
        strategies = ["V5"]
    else:
        regime = "SIDEWAYS"
        if range_20_pct <= 6.0:
            reason.append(f"Range 20 hari sempit ({range_20_pct:.1f}%) — konsolidasi.")
        elif above_ma50 and not bullish_stack:
            reason.append("Di atas MA50 tapi struktur MA belum rapi — rotasi / mixed.")
        elif below_ma50 and not below_ma20:
            reason.append("Di sekitar MA — tanpa tren ekstrem.")
        else:
            reason.append("Tidak ada tren bullish/bearish yang jelas (rotasi sektor).")
        strategies = ["V3", "V4"]

    # =================================================================
    # AKTIVITAS PASAR (RAMAI / NORMAL / SEPI)
    # Volume ^JKSE sering 0 di Yahoo → jangan pakai angka 0x palsu
    # =================================================================
    activity = "NORMAL"
    activity_reason: list[str] = []
    vol_ratio_20 = None
    vol_ratio_5 = None
    atr_ratio = None
    score = 0

    # --- Volume (opsional) ---
    vol_valid = False
    if "Volume" in df.columns:
        vol = pd.to_numeric(df["Volume"], errors="coerce").fillna(0.0)
        if _volume_is_valid(vol):
            vol_valid = True
            avg_vol_20 = float(vol.tail(20).replace(0, np.nan).mean())
            avg_vol_5 = float(vol.tail(5).replace(0, np.nan).mean())
            last_vol = float(vol.iloc[-1]) if float(vol.iloc[-1]) > 0 else avg_vol_5
            if avg_vol_20 > 0:
                vol_ratio_20 = last_vol / avg_vol_20
                vol_ratio_5 = avg_vol_5 / avg_vol_20

    if not vol_valid:
        activity_reason.append(
            "Volume ^JKSE tidak valid (sering 0 di Yahoo) — aktivitas memakai range/ATR."
        )
    else:
        if vol_ratio_20 is not None and vol_ratio_20 >= 1.25:
            score += 2
            activity_reason.append(
                f"Volume terakhir tinggi ({vol_ratio_20:.2f}x avg20)."
            )
        elif vol_ratio_5 is not None and vol_ratio_5 >= 1.20:
            score += 1
            activity_reason.append(
                f"Volume 5 hari di atas rata-rata ({vol_ratio_5:.2f}x avg20)."
            )
        elif vol_ratio_20 is not None and vol_ratio_20 <= 0.70:
            score -= 2
            activity_reason.append(
                f"Volume lemah ({vol_ratio_20:.2f}x avg20)."
            )
        elif vol_ratio_20 is not None and vol_ratio_20 <= 0.85:
            score -= 1
            activity_reason.append(
                f"Volume di bawah rata-rata ({vol_ratio_20:.2f}x avg20)."
            )
        else:
            activity_reason.append(
                f"Volume relatif normal ({vol_ratio_20:.2f}x avg20)."
            )

    # --- Range proxy ---
    # range 5 hari relatif aktif vs skala range 20 hari
    range_expanding = range_5_pct >= max(1.5, range_20_pct * 0.55)
    range_tight = range_5_pct <= 1.2 and range_20_pct <= 5.0

    if range_expanding:
        score += 1
        activity_reason.append(
            f"Range 5D melebar ({range_5_pct:.2f}% vs range20 {range_20_pct:.2f}%)."
        )
    elif range_tight:
        score -= 1
        activity_reason.append(
            f"Pergerakan sempit (range5 {range_5_pct:.2f}%, range20 {range_20_pct:.2f}%)."
        )

    # --- ATR proxy ---
    atr14 = _compute_atr(df, 14)
    atr5 = _compute_atr(df, 5)
    if atr14 is not None and atr5 is not None:
        a14 = float(atr14.iloc[-1]) if not pd.isna(atr14.iloc[-1]) else 0.0
        a5 = float(atr5.iloc[-1]) if not pd.isna(atr5.iloc[-1]) else 0.0
        if a14 > 0 and a5 > 0:
            atr_ratio = a5 / a14
            if atr_ratio >= 1.20:
                score += 1
                activity_reason.append(
                    f"Volatilitas naik (ATR5/ATR14={atr_ratio:.2f})."
                )
            elif atr_ratio <= 0.80:
                score -= 1
                activity_reason.append(
                    f"Volatilitas turun (ATR5/ATR14={atr_ratio:.2f})."
                )

    if score >= 2:
        activity = "RAMAI"
    elif score <= -2:
        activity = "SEPI"
    else:
        activity = "NORMAL"

    # --- Output terminal ---
    print(f"Index Terakhir : {last_close:,.2f}")
    print(
        f"Status MA      : MA20={last_ma20:,.0f} | "
        f"MA50={last_ma50:,.0f} | MA100={last_ma100:,.0f}"
    )
    print(
        f"Slope MA       : MA20={'naik' if ma20_rising else 'turun'} | "
        f"MA50={'naik' if ma50_rising else 'turun'}"
    )
    print(f"Dist High20/60 : {dist_to_high_20:.1f}% / {dist_to_high_60:.1f}%")
    print(f"Dist Low20/60  : {dist_to_low_20:.1f}% / {dist_to_low_60:.1f}%")
    print(f"Range 5/20 hari: {range_5_pct:.2f}% / {range_20_pct:.2f}%")
    if vol_valid and vol_ratio_20 is not None:
        print(f"Vol ratio      : {vol_ratio_20:.2f}x avg20 (valid)")
    else:
        print("Vol ratio      : n/a (volume indeks tidak valid)")
    if atr_ratio is not None:
        print(f"ATR5/ATR14     : {atr_ratio:.2f}")
    # =================================================================
    # BIAS OPERASIONAL (arah kerja day/swing trader)
    # Gabungan rezim struktur + aktivitas (bukan sinyal emiten)
    # =================================================================
    bias = "NETRAL"
    bias_score = 0
    bias_reason: list[str] = []

    if regime == "BULLISH_STRONG":
        bias_score += 2
        bias_reason.append("Struktur bullish kuat (stack MA + dekat high).")
    elif regime == "BULLISH_PULLBACK":
        bias_score += 1
        bias_reason.append("Pullback dalam struktur menengah bullish.")
    elif regime == "BEARISH_STRONG":
        bias_score -= 2
        bias_reason.append("Downtrend kuat / dekat low — bias defensif.")
    elif regime == "BEARISH_WEAK":
        bias_score -= 1
        bias_reason.append("Tekanan jual di bawah MA20/MA50.")
    else:
        bias_reason.append("Sideways / rotasi — tidak ada edge arah yang jelas.")

    if activity == "RAMAI":
        # Aktivitas tinggi memperkuat arah yang sudah ada; di sideways = waspada chop
        if bias_score > 0:
            bias_score += 1
            bias_reason.append("Aktivitas RAMAI memperkuat bias naik.")
        elif bias_score < 0:
            bias_score -= 1
            bias_reason.append("Aktivitas RAMAI memperkuat tekanan turun.")
        else:
            bias_reason.append("Aktivitas RAMAI di sideways — waspada false breakout.")
    elif activity == "SEPI":
        if bias_score > 0:
            bias_score -= 1
            bias_reason.append("Pasar SEPI mengurangi keyakinan long agresif.")
        elif bias_score < 0:
            bias_score += 1
            bias_reason.append("Pasar SEPI mengurangi keyakinan short/panic.")
        else:
            bias_reason.append("Pasar SEPI + sideways — fokus selektif / tunggu trigger.")

    if bias_score >= 2:
        bias = "BULLISH"
    elif bias_score == 1:
        bias = "SEDIKIT_BULLISH"
    elif bias_score <= -2:
        bias = "BEARISH"
    elif bias_score == -1:
        bias = "SEDIKIT_BEARISH"
    else:
        bias = "NETRAL"

    print(f"Rezim Pasar    : ** {regime} **")
    for r in reason:
        print(f"  - {r}")
    print(f"Aktivitas      : ** {activity} ** (score={score})")
    for r in activity_reason:
        print(f"  - {r}")
    print(f"Bias           : ** {bias} ** (score={bias_score})")
    for r in bias_reason:
        print(f"  - {r}")
    print(f"Strategi       : {', '.join(strategies)}")

    return {
        "regime": regime,
        "strategies": strategies,
        "last_close": last_close,
        "ma20": last_ma20,
        "ma50": last_ma50,
        "ma100": last_ma100,
        "reason": reason,
        "activity": activity,
        "activity_reason": activity_reason,
        "vol_ratio_20": round(vol_ratio_20, 2) if vol_ratio_20 is not None else None,
        "range_5_pct": round(range_5_pct, 2),
        "range_20_pct": round(range_20_pct, 2),
        "atr_ratio": round(atr_ratio, 2) if atr_ratio is not None else None,
        "activity_score": score,
        "bias": bias,
        "bias_score": bias_score,
        "bias_reason": bias_reason,
    }


def apply_enrichment_to_latest_reports(
    broker_buy_pct: float = 0.15,
    broker_sell_pct: float = 0.25,
):
    """Trailing + fundamental + biaya/pajak pada report versi yang ada."""
    print("\n" + "=" * 80)
    print("ENRICHMENT: Trailing + Fundamental + Biaya/Pajak")
    print("=" * 80)
    print(f"  Fee beli={broker_buy_pct}% | Fee jual={broker_sell_pct}%")

    for ver in ["v2", "v3", "v4", "v5", "intraday", "highbeta", "accumulation"]:
        fpath = _find_today_report(ver)
        if not fpath:
            print(f"  • {ver.upper()}: file tidak ada, dilewati.")
            continue
        try:
            df = pd.read_csv(fpath)
            if df.empty:
                print(f"  • {ver.upper()}: kosong.")
                continue

            if TRAILING_AVAILABLE:
                df = enrich_with_trailing_stop(df)

            if FUNDAMENTAL_AVAILABLE:
                df = enrich_with_fundamental(df)

            if COST_AVAILABLE:
                df = enrich_dataframe_with_costs(df, broker_buy_pct, broker_sell_pct)

            df.to_csv(fpath, index=False)
            print(f"  • {os.path.basename(fpath)} → OK ({len(df)} baris) @ {fpath}")
        except Exception as e:
            print(f"  • Gagal {ver}: {e}")

    print("Selesai enrichment.")


def run_orchestrator(
    account_size: float = None,
    risk_pct: float = None,
    broker_buy_pct: float = 0.15,
    broker_sell_pct: float = 0.25,
):
    print("\n" + "=" * 80)
    print("IDX MASTER SCREENER AI - SETUP MODAL, RISIKO & FEE")
    print("=" * 80)

    if account_size is None or risk_pct is None:
        try:
            raw_modal = input(
                "Masukkan Total Modal Trading Anda (Contoh: 50000000) [Default: 50000000]: "
            ).strip()
            account_size = (
                float(raw_modal.replace(",", "").replace(".", ""))
                if raw_modal
                else 50_000_000.0
            )

            raw_risk = input(
                "Masukkan Toleransi Risiko per Posisi (%) (Contoh: 1.0) [Default: 1.0]: "
            ).strip()
            risk_pct = float(raw_risk) if raw_risk else 1.0

            raw_buy = input("Fee Beli broker % [Default: 0.15]: ").strip()
            if raw_buy:
                broker_buy_pct = float(raw_buy)

            raw_sell = input("Fee Jual broker % [Default: 0.25]: ").strip()
            if raw_sell:
                broker_sell_pct = float(raw_sell)
        except ValueError:
            print("Input tidak valid! Menggunakan default.")
            account_size = 50_000_000.0
            risk_pct = 1.0
            broker_buy_pct = 0.15
            broker_sell_pct = 0.25

    if account_size is None or account_size <= 0:
        account_size = 50_000_000.0
    if risk_pct is None or risk_pct <= 0 or risk_pct > 10:
        risk_pct = 1.0
    if broker_buy_pct is None or broker_buy_pct < 0:
        broker_buy_pct = 0.15
    if broker_sell_pct is None or broker_sell_pct < 0:
        broker_sell_pct = 0.25

    print(
        f"\n[SETUP] Modal: Rp {account_size:,.0f} | "
        f"Risiko/Trade: {risk_pct}% "
        f"(Maks Rugi: Rp {account_size * risk_pct / 100:,.0f})"
    )
    print(
        f"[SETUP] Fee beli: {broker_buy_pct}% | Fee jual: {broker_sell_pct}% "
        f"(+ PPN 12% atas fee, levy ~0.043%, PPh Final 0.1% jual)"
    )

    market_status = analyze_ihsg_regime()
    strategies = market_status.get("strategies", [])

    if not strategies or market_status.get("regime") == "UNKNOWN":
        print("\nTidak dapat menentukan strategi yang valid. Skrip dihentikan.")
        return

    print("\n" + "=" * 80)
    print(f"REKOMENDASI AI: MENJALANKAN STRATEGI {', '.join(strategies)}")
    print(
        f"Rezim={market_status.get('regime')} | "
        f"Aktivitas={market_status.get('activity')}"
    )
    print("=" * 80)

    user_params = {
        "account_size": float(account_size),
        "risk_per_trade_pct": float(risk_pct),
        "broker_buy_pct": float(broker_buy_pct),
        "broker_sell_pct": float(broker_sell_pct),
    }

    for strat in strategies:
        try:
            if strat == "V2":
                print("\n>>> Memulai IDX Breakout Screener V2...")
                module = importlib.import_module("idx_breakout_screener_v2")
                if hasattr(module, "PARAMS") and isinstance(module.PARAMS, dict):
                    module.PARAMS.update(user_params)
                module.run_screener(params=getattr(module, "PARAMS", user_params))

            elif strat == "V3":
                print("\n>>> Memulai IDX Fibo Retest Screener V3...")
                module = importlib.import_module("idx_breakout_screener_v3")
                if hasattr(module, "PARAMS") and isinstance(module.PARAMS, dict):
                    module.PARAMS.update(user_params)
                module.run_screener(params=getattr(module, "PARAMS", user_params))

            elif strat == "V4":
                print("\n>>> Memulai IDX SMC Screener V4...")
                module = importlib.import_module("idx_breakout_screener_v4_smc")
                module.run_screener_v4(user_params=user_params)

            elif strat == "V5":
                print("\n>>> Memulai IDX SMC Screener V5...")
                module = importlib.import_module("idx_breakout_screener_v5_smc")
                module.run_screener_v5(user_params=user_params)

        except Exception as e:
            print(f"Error saat menjalankan strategi {strat}: {e}")
            print("Pastikan file skrip strategi berada di folder yang sama.")

    apply_enrichment_to_latest_reports(
        broker_buy_pct=float(broker_buy_pct),
        broker_sell_pct=float(broker_sell_pct),
    )

    print("\n" + "=" * 80)
    print("SELESAI — Strategi + Trailing + Fundamental + Biaya/Pajak diproses.")
    print("=" * 80)


if __name__ == "__main__":
    run_orchestrator()
