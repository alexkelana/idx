"""
IDX LIQUIDITY SCANNER — Fixed Version
============================================================
[FIXED]
- MultiIndex handling + auto_adjust
- Hapus duplikat fallback ticker
- dropna + guard data lebih ketat
- Cari file Excel di beberapa lokasi
- Optional simple in-memory cache
- Progress print lebih aman
- Timeout / exception handling lebih bersih
"""

import pandas as pd
import yfinance as yf
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta
import time
from pathlib import Path

class IdxLiquidityScanner:
    def __init__(
        self,
        min_avg_value_rp: float = 10_000_000_000,
        min_avg_volume: float = 1_000_000,
        lookback_days: int = 20,
        max_workers: int = 15,
        use_cache: bool = True,
    ):
        """
        Scanner untuk mencari saham IDX yang memenuhi syarat likuiditas secara dinamis.
        """
        self.min_avg_value_rp = min_avg_value_rp
        self.min_avg_volume = min_avg_volume
        self.lookback_days = lookback_days
        self.max_workers = max_workers
        self.use_cache = use_cache

        self.raw_tickers: list[str] = []
        self.liquid_tickers: list[str] = []

        # [FIX] Simple in-memory cache (berlaku selama proses Python hidup)
        # Key: (ticker, lookback, min_value, min_vol) → hasil True/False
        self._cache: dict = {}

    # ------------------------------------------------------------------
    # 1. Ambil daftar emiten
    # ------------------------------------------------------------------
    def _fetch_all_idx_tickers(self):
        """Mengambil daftar emiten IDX dari file Excel lokal (prioritas) atau fallback."""
        print("1. Mengambil daftar seluruh emiten IDX...")

        # [FIX] Cari file Excel di beberapa lokasi umum
        search_dirs = [
            Path("."),
            Path(__file__).parent if "__file__" in globals() else Path("."),
            Path.home() / "Downloads",
            Path("/home/workdir/attachments"),
            Path("/home/workdir"),
        ]

        excel_file = None
        for directory in search_dirs:
            if not directory.exists():
                continue
            try:
                for file in directory.iterdir():
                    name = file.name
                    if name.startswith("Daftar Saham") and name.lower().endswith((".xlsx", ".xls")):
                        excel_file = file
                        break
            except Exception:
                continue
            if excel_file:
                break

        if excel_file:
            try:
                print(f"   Membaca data dari file lokal: {excel_file}")
                df = pd.read_excel(excel_file)

                # Cari kolom 'Kode' (case-insensitive)
                kode_col = None
                for col in df.columns:
                    if str(col).strip().lower() in ("kode", "code", "ticker", "symbol"):
                        kode_col = col
                        break

                if kode_col is not None:
                    tickers = df[kode_col].dropna().astype(str).tolist()
                    cleaned = []
                    for t in tickers:
                        t = t.upper().strip()
                        # Terima 3–4 huruf (beberapa emiten lama 3 huruf)
                        if 3 <= len(t) <= 4 and t.isalpha():
                            cleaned.append(t)

                    self.raw_tickers = sorted(set(cleaned))
                    if len(self.raw_tickers) > 50:
                        print(f"   Berhasil memuat {len(self.raw_tickers)} emiten dari file Excel.")
                        return
                    else:
                        print("   Jumlah ticker dari Excel terlalu sedikit, pakai fallback.")
            except Exception as e:
                print(f"   Gagal membaca file Excel: {e}")

        # [FIX] Fallback statis — sudah di-unique & diurutkan
        print("   File Excel tidak ditemukan / gagal. Menggunakan daftar fallback (unique).")
        fallback = [
            "BBCA", "BBRI", "BMRI", "BBNI", "BRIS", "ARTO", "TLKM", "EXCL", "ISAT", "MTEL",
            "ASII", "UNTR", "AUTO", "ANTM", "MDKA", "INCO", "TINS", "PSAB", "ADRO", "PTBA",
            "ITMG", "BUMI", "HRUM", "INDY", "BYAN", "ICBP", "INDF", "MYOR", "CPIN", "JPFA",
            "UNVR", "GGRM", "HMSP", "SMGR", "INTP", "SMBR", "PGAS", "MEDC", "PGEO", "AKRA",
            "ELSA", "KLBF", "SIDO", "SILO", "MIKA", "PWON", "BSDE", "CTRA", "SMRA", "GOTO",
            "BUKA", "EMTK", "MNCN", "SCMA", "AMMN", "BRPT", "TPIA", "ESSA", "BRMS", "PANI",
            "CUAN", "RATU", "DSSA", "AVIA", "FILM", "MAPA", "MAPI", "ACES", "ERAA", "BTPS",
            "TOWR", "TBIG", "JSMR", "WIKA", "PTPP", "ADHI", "WSKT", "WEGE", "SSIA", "SRTG",
            "TKIM", "BMTR", "LSIP", "INKP", "MAIN", "SMSM", "ASRI", "KIJA", "DILD", "LPKR",
            "APLN", "BEST", "VKTR",
        ]
        self.raw_tickers = sorted(set(fallback))
        print(f"   Fallback: {len(self.raw_tickers)} ticker unik.")

    # ------------------------------------------------------------------
    # 2. Cek likuiditas satu ticker
    # ------------------------------------------------------------------
    def _check_liquidity(self, ticker: str):
        """Worker: cek apakah satu saham memenuhi syarat likuiditas."""
        # [FIX] Cache key
        cache_key = (ticker, self.lookback_days, self.min_avg_value_rp, self.min_avg_volume)
        if self.use_cache and cache_key in self._cache:
            return ticker if self._cache[cache_key] else None

        symbol = f"{ticker}.JK"
        try:
            # [FIX] Gunakan period + auto_adjust + multi_level_index=False
            df = yf.download(
                symbol,
                period=f"{int(self.lookback_days * 1.8)}d",
                interval="1d",
                progress=False,
                auto_adjust=True,
                multi_level_index=False,
                threads=False,
            )

            if df is None or df.empty:
                self._cache[cache_key] = False
                return None

            # [FIX] MultiIndex fallback (jika parameter diabaikan yfinance)
            if isinstance(df.columns, pd.MultiIndex):
                try:
                    df.columns = df.columns.get_level_values(0)
                except Exception:
                    try:
                        df = df.droplevel(-1, axis=1)
                    except Exception:
                        self._cache[cache_key] = False
                        return None

            if "Close" not in df.columns or "Volume" not in df.columns:
                self._cache[cache_key] = False
                return None

            # [FIX] Drop NaN lalu ambil lookback
            df = df[["Close", "Volume"]].dropna()
            if len(df) < max(10, self.lookback_days - 8):
                self._cache[cache_key] = False
                return None

            df = df.tail(self.lookback_days)

            avg_volume = df["Volume"].mean()
            daily_value = df["Close"] * df["Volume"]
            avg_value = daily_value.mean()

            # Hindari NaN
            if pd.isna(avg_volume) or pd.isna(avg_value):
                self._cache[cache_key] = False
                return None

            is_liquid = (avg_volume >= self.min_avg_volume) and (avg_value >= self.min_avg_value_rp)
            self._cache[cache_key] = is_liquid

            return ticker if is_liquid else None

        except Exception:
            self._cache[cache_key] = False
            return None

    # ------------------------------------------------------------------
    # 3. Main entry
    # ------------------------------------------------------------------
    def get_liquid_universe(self) -> list[str]:
        """
        Menjalankan scanning multi-threading.
        Mengembalikan list ticker yang lolos filter likuiditas.
        """
        self._fetch_all_idx_tickers()

        if not self.raw_tickers:
            print("   Tidak ada ticker yang bisa dipindai.")
            return []

        print(
            f"\n2. Memindai likuiditas {len(self.raw_tickers)} saham "
            f"(Target: Avg Value > Rp{self.min_avg_value_rp/1e9:.0f}M & "
            f"Avg Vol > {self.min_avg_volume/1e6:.1f}Jt)"
        )
        print(f"   Multi-threading ({self.max_workers} workers). Estimasi 1–2 menit...")

        start_time = time.time()
        valid_tickers: list[str] = []
        completed_count = 0
        total = len(self.raw_tickers)

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_ticker = {
                executor.submit(self._check_liquidity, t): t for t in self.raw_tickers
            }

            for future in as_completed(future_to_ticker):
                completed_count += 1

                # [FIX] Progress lebih jarang & aman
                if completed_count % 40 == 0 or completed_count == total:
                    print(f"   Progress: {completed_count}/{total} saham diproses...", end="\r")

                try:
                    result = future.result(timeout=30)
                    if result:
                        valid_tickers.append(result)
                except Exception:
                    pass

        print()  # baris baru setelah progress

        elapsed = time.time() - start_time
        print(f"3. Selesai dalam {elapsed:.1f} detik.")
        print(f"   Ditemukan {len(valid_tickers)} saham yang sangat likuid saat ini.")

        self.liquid_tickers = sorted(set(valid_tickers))
        return self.liquid_tickers

    def clear_cache(self):
        """Hapus cache manual jika diperlukan."""
        self._cache.clear()


# ------------------------------------------------------------------
# Testing mandiri
# ------------------------------------------------------------------
if __name__ == "__main__":
    print("=== TEST IDX LIQUIDITY SCANNER (Fixed) ===")
    scanner = IdxLiquidityScanner(
        min_avg_value_rp=10_000_000_000,
        min_avg_volume=1_000_000,
        lookback_days=20,
        max_workers=15,
        use_cache=True,
    )
    liquid_stocks = scanner.get_liquid_universe()

    print("\nDaftar Saham Likuid:")
    print(liquid_stocks)
    print(f"\nTotal: {len(liquid_stocks)} saham")