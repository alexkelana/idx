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


def analyze_ihsg_regime(lookback_days: int = 150) -> dict:
    """Menganalisis rezim pasar IHSG."""
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
        return {"regime": "UNKNOWN", "reason": "Data tidak cukup.", "strategies": []}

    ma20 = close.rolling(20).mean().iloc[-1]
    ma50 = close.rolling(50).mean().iloc[-1]
    ma100 = close.rolling(100).mean().iloc[-1]
    last_close = close.iloc[-1]

    if pd.isna(ma20) or pd.isna(ma50) or pd.isna(ma100):
        return {"regime": "UNKNOWN", "reason": "MA mengandung NaN.", "strategies": []}

    recent_20d = close.tail(20)
    high_20d = recent_20d.max()
    low_20d = recent_20d.min()
    dist_to_high_pct = (high_20d - last_close) / last_close * 100
    dist_to_low_pct = (last_close - low_20d) / low_20d * 100

    regime = "SIDEWAYS / PULLBACK"
    reason = []
    recommended_strategy = []

    if last_close > ma20 and ma20 > ma50 and ma50 > ma100 and dist_to_high_pct < 2.0:
        regime = "BULLISH"
        reason.append("IHSG berada dalam uptrend kuat (Close > MA20 > MA50 > MA100).")
        reason.append("Harga menempel di area Resistance tertinggi bulanan.")
        recommended_strategy = ["V2"]
    elif last_close < ma20 and last_close < ma50 and dist_to_low_pct < 2.0:
        regime = "BEARISH"
        reason.append("IHSG berada dalam tekanan jual / downtrend (Close < MA20 & MA50).")
        reason.append("Harga berada di dekat titik terendah bulanan.")
        recommended_strategy = ["V5"]
    else:
        regime = "SIDEWAYS / PULLBACK"
        if last_close > ma50 and last_close < ma20:
            reason.append("IHSG sedang mengalami koreksi wajar (Pullback ke MA50).")
        else:
            reason.append("IHSG bergerak tanpa tren yang ekstrem (Rotasi Sektor).")
        recommended_strategy = ["V3", "V4"]

    print(f"Index Terakhir : {last_close:,.2f}")
    print(f"Status MA      : MA20={ma20:,.0f} | MA50={ma50:,.0f} | MA100={ma100:,.0f}")
    print(f"Rezim Pasar    : ** {regime} **")
    for r in reason:
        print(f"  - {r}")

    return {
        "regime": regime,
        "strategies": recommended_strategy,
        "last_close": float(last_close),
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