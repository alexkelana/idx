"""
IDX MASTER SCREENER AI — ADAPTIVE MARKET REGIME
============================================================
Skrip utama (Orchestrator) membaca kondisi makro IHSG, memilih
strategi yang sesuai, menjalankan screener, lalu menambahkan:
  1. Saran Trailing Stop berbasis volatilitas
  2. Data Fundamental + status valuasi (Undervalued / Fair / Overvalued)

Logika Pemilihan Strategi:
1. BULLISH (Uptrend Kuat)      -> V2 (Breakout)
2. SIDEWAYS (Konsolidasi)      -> V3 & V4 (Retest Fibo & Order Block)
3. BEARISH (Downtrend/Crash)   -> V5 (CHOCH / Bottom Fishing)
"""

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
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

# --- Report schema (nama file klasik/smc) ---
try:
    from idx_report_schema import get_report_filename
    SCHEMA_AVAILABLE = True
except ImportError:
    SCHEMA_AVAILABLE = False
    def get_report_filename(group: str) -> str:
        today = datetime.now().strftime("%Y-%m-%d")
        return f"idx_master_report_{group}_{today}.csv"


def analyze_ihsg_regime(lookback_days: int = 180) -> dict:
    """
    Menganalisis rezim pasar IHSG secara lebih realistis.

    Rezim:
      - BULLISH_STRONG   → uptrend kuat, dekat high
      - BULLISH_PULLBACK → uptrend masih valid, sedang koreksi
      - SIDEWAYS         → tanpa arah jelas / rotasi
      - BEARISH_WEAK     → tekanan jual, belum ekstrem
      - BEARISH_STRONG   → downtrend / dekat low
      - UNKNOWN          → data tidak cukup

    Strategi yang direkomendasikan menyesuaikan rezim di atas.
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
        return {"regime": "UNKNOWN", "reason": "Data IHSG tidak tersedia.", "strategies": []}

    if df is None or df.empty:
        return {"regime": "UNKNOWN", "reason": "Data IHSG kosong.", "strategies": []}

    if isinstance(df.columns, pd.MultiIndex):
        try:
            df.columns = df.columns.get_level_values(0)
        except Exception:
            df = df.droplevel(-1, axis=1)

    if "Close" not in df.columns:
        return {"regime": "UNKNOWN", "reason": "Kolom Close tidak ditemukan.", "strategies": []}

    close = df["Close"].dropna()
    if len(close) < 100:
        return {"regime": "UNKNOWN", "reason": "Data tidak cukup (<100 bar).", "strategies": []}

    # --- Moving Averages ---
    ma20 = close.rolling(20).mean()
    ma50 = close.rolling(50).mean()
    ma100 = close.rolling(100).mean()

    last_close = float(close.iloc[-1])
    last_ma20 = float(ma20.iloc[-1])
    last_ma50 = float(ma50.iloc[-1])
    last_ma100 = float(ma100.iloc[-1])

    if any(pd.isna(x) for x in [last_ma20, last_ma50, last_ma100]):
        return {"regime": "UNKNOWN", "reason": "MA mengandung NaN.", "strategies": []}

    # Slope MA (bandingkan dengan 5 bar lalu)
    ma20_prev = float(ma20.iloc[-6]) if len(ma20) >= 6 else last_ma20
    ma50_prev = float(ma50.iloc[-6]) if len(ma50) >= 6 else last_ma50
    ma20_rising = last_ma20 > ma20_prev
    ma50_rising = last_ma50 > ma50_prev
    ma20_falling = last_ma20 < ma20_prev
    ma50_falling = last_ma50 < ma50_prev

    # High / Low
    high_20 = float(close.tail(20).max())
    low_20 = float(close.tail(20).min())
    high_60 = float(close.tail(60).max()) if len(close) >= 60 else high_20
    low_60 = float(close.tail(60).min()) if len(close) >= 60 else low_20

    dist_to_high_20 = (high_20 - last_close) / last_close * 100
    dist_to_low_20 = (last_close - low_20) / low_20 * 100
    dist_to_high_60 = (high_60 - last_close) / last_close * 100
    dist_to_low_60 = (last_close - low_60) / low_60 * 100

    # Lebar range 20 hari (untuk deteksi sideways)
    range_20_pct = (high_20 - low_20) / last_close * 100

    # Struktur MA
    bullish_stack = last_close > last_ma20 > last_ma50 > last_ma100
    bearish_stack = last_close < last_ma20 < last_ma50  # MA100 opsional
    above_ma50 = last_close > last_ma50
    above_ma100 = last_close > last_ma100
    below_ma50 = last_close < last_ma50
    below_ma20 = last_close < last_ma20

    # =================================================================
    # DECISION ENGINE
    # =================================================================
    regime = "SIDEWAYS"
    reason = []
    strategies = []

    # 1) BULLISH STRONG
    #    Uptrend sempurna + momentum naik + tidak jauh dari high
    if bullish_stack and ma20_rising and ma50_rising and dist_to_high_20 <= 5.0:
        regime = "BULLISH_STRONG"
        reason.append("Uptrend kuat: Close > MA20 > MA50 > MA100.")
        reason.append("MA20 & MA50 sedang naik (momentum positif).")
        reason.append(f"Harga dekat high 20 hari (jarak {dist_to_high_20:.1f}%).")
        strategies = ["V2"]

    # 2) BULLISH PULLBACK
    #    Struktur jangka menengah masih naik, tapi harga koreksi di bawah MA20
    elif above_ma50 and above_ma100 and (last_ma50 > last_ma100) and below_ma20:
        regime = "BULLISH_PULLBACK"
        reason.append("Struktur menengah masih bullish (Close > MA50 > MA100).")
        reason.append("Harga sedang pullback di bawah MA20 (koreksi wajar).")
        if ma50_rising:
            reason.append("MA50 masih naik — pullback dalam uptrend.")
        strategies = ["V3"]  # Retest / buy on weakness; V2 opsional tidak dipaksa

    # 3) BEARISH STRONG
    #    Di bawah MA pendek & menengah + dekat low + momentum turun
    elif below_ma20 and below_ma50 and (ma20_falling or ma50_falling) and dist_to_low_20 <= 5.0:
        regime = "BEARISH_STRONG"
        reason.append("Downtrend: Close < MA20 & MA50.")
        reason.append("MA sedang turun (momentum negatif).")
        reason.append(f"Harga dekat low 20 hari (jarak {dist_to_low_20:.1f}%).")
        strategies = ["V5"]

    # 4) BEARISH WEAK
    #    Di bawah MA50 tapi belum ekstrem di low
    elif below_ma50 and below_ma20:
        regime = "BEARISH_WEAK"
        reason.append("Harga di bawah MA20 & MA50 (tekanan jual).")
        reason.append("Belum di zona low ekstrem — downtrend / distribusi.")
        if dist_to_low_60 < 8.0:
            reason.append(f"Mendekati low 60 hari (jarak {dist_to_low_60:.1f}%).")
        strategies = ["V5"]  # tetap prioritaskan bottom-fishing / hati-hati

    # 5) SIDEWAYS
    #    Range sempit, atau MA saling silang tanpa arah jelas
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

    # --- Output terminal ---
    print(f"Index Terakhir : {last_close:,.2f}")
    print(f"Status MA      : MA20={last_ma20:,.0f} | MA50={last_ma50:,.0f} | MA100={last_ma100:,.0f}")
    print(f"Slope MA       : MA20={'naik' if ma20_rising else 'turun'} | MA50={'naik' if ma50_rising else 'turun'}")
    print(f"Dist High20/60 : {dist_to_high_20:.1f}% / {dist_to_high_60:.1f}%")
    print(f"Dist Low20/60  : {dist_to_low_20:.1f}% / {dist_to_low_60:.1f}%")
    print(f"Range 20 hari  : {range_20_pct:.1f}%")
    print(f"Rezim Pasar    : ** {regime} **")
    for r in reason:
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
    }


def apply_enrichment_to_latest_reports():
    """
    Setelah semua strategi selesai:
    1. Tambahkan Trailing Stop
    2. Tambahkan Fundamental + Valuasi (Undervalued / Fair / Overvalued)
    """
    print("\n" + "=" * 80)
    print("MENAMBAHKAN TRAILING STOP + FUNDAMENTAL / VALUASI")
    print("=" * 80)

    for group in ["klasik", "smc"]:
        fpath = get_report_filename(group)

        if not os.path.exists(fpath):
            print(f"  • {group.upper()}: file tidak ditemukan, dilewati.")
            continue

        try:
            df = pd.read_csv(fpath)
            if df.empty:
                print(f"  • {group.upper()}: kosong, dilewati.")
                continue

            n_before = len(df.columns)

            # 1. Trailing Stop
            if TRAILING_AVAILABLE:
                df = enrich_with_trailing_stop(df)
                print(f"  • {group.upper()}: trailing stop ditambahkan.")
            else:
                print(f"  • {group.upper()}: trailing stop dilewati (modul tidak ada).")

            # 2. Fundamental + Valuasi
            if FUNDAMENTAL_AVAILABLE:
                df = enrich_with_fundamental(df)
                print(f"  • {group.upper()}: fundamental + valuasi ditambahkan.")
            else:
                print(f"  • {group.upper()}: fundamental dilewati (modul tidak ada).")

            df.to_csv(fpath, index=False)
            print(f"  • {os.path.basename(fpath)} → selesai ({len(df)} baris, {len(df.columns)} kolom).")

        except Exception as e:
            print(f"  • Gagal memproses {group}: {e}")

    print("Selesai enrichment.")


def run_orchestrator(account_size: float = None, risk_pct: float = None):
    print("\n" + "=" * 80)
    print("IDX MASTER SCREENER AI - SETUP MODAL & RISIKO")
    print("=" * 80)

    if account_size is None or risk_pct is None:
        try:
            raw_modal = input(
                "Masukkan Total Modal Trading Anda (Contoh: 50000000) [Default: 50000000]: "
            ).strip()
            account_size = float(raw_modal.replace(",", "").replace(".", "")) if raw_modal else 50_000_000.0

            raw_risk = input(
                "Masukkan Toleransi Risiko per Posisi (%) (Contoh: 1.0) [Default: 1.0]: "
            ).strip()
            risk_pct = float(raw_risk) if raw_risk else 1.0
        except ValueError:
            print("Input tidak valid! Menggunakan default (Modal: Rp50 Jt, Risiko: 1%).")
            account_size = 50_000_000.0
            risk_pct = 1.0

    if account_size <= 0:
        account_size = 50_000_000.0
    if risk_pct <= 0 or risk_pct > 10:
        risk_pct = 1.0

    print(
        f"\n[SETUP] Modal: Rp {account_size:,.0f} | "
        f"Risiko/Trade: {risk_pct}% "
        f"(Maks Rugi: Rp {account_size * risk_pct / 100:,.0f})"
    )

    market_status = analyze_ihsg_regime()
    strategies = market_status.get("strategies", [])

    if not strategies or market_status.get("regime") == "UNKNOWN":
        print("\nTidak dapat menentukan strategi yang valid. Skrip dihentikan.")
        return

    print("\n" + "=" * 80)
    print(f"REKOMENDASI AI: MENJALANKAN STRATEGI {', '.join(strategies)}")
    print("=" * 80)

    user_params = {
        "account_size": account_size,
        "risk_per_trade_pct": risk_pct,
    }

    # Eksekusi strategi
    for strat in strategies:
        try:
            if strat == "V2":
                print("\n>>> Memulai IDX Breakout Screener V2 (Momentum Play)...")
                module = importlib.import_module("idx_breakout_screener_v2")
                if hasattr(module, "PARAMS") and isinstance(module.PARAMS, dict):
                    module.PARAMS.update(user_params)
                module.run_screener(params=getattr(module, "PARAMS", user_params))

            elif strat == "V3":
                print("\n>>> Memulai IDX Fibo Retest Screener V3 (Buy on Weakness)...")
                module = importlib.import_module("idx_breakout_screener_v3")
                if hasattr(module, "PARAMS") and isinstance(module.PARAMS, dict):
                    module.PARAMS.update(user_params)
                module.run_screener(params=getattr(module, "PARAMS", user_params))

            elif strat == "V4":
                print("\n>>> Memulai IDX SMC Screener V4 (Order Block Mitigation)...")
                module = importlib.import_module("idx_breakout_screener_v4_smc")
                module.run_screener_v4(user_params=user_params)

            elif strat == "V5":
                print("\n>>> Memulai IDX SMC Screener V5 (CHOCH Bottom Fishing)...")
                module = importlib.import_module("idx_breakout_screener_v5_smc")
                module.run_screener_v5(user_params=user_params)

        except Exception as e:
            print(f"Error saat menjalankan strategi {strat}: {e}")
            print("Pastikan file skrip strategi tersebut berada di folder yang sama.")

    # Setelah semua strategi → Trailing Stop + Fundamental/Valuasi
    apply_enrichment_to_latest_reports()

    print("\n" + "=" * 80)
    print("SELESAI — Strategi + Trailing Stop + Fundamental telah diproses.")
    print("=" * 80)


if __name__ == "__main__":
    run_orchestrator()