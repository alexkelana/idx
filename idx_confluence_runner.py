"""
IDX CONFLUENCE RUNNER
============================================================
1. Menjalankan semua strategy screener yang tersedia
2. Membaca idx_report_* hari ini
3. Menghitung confluence antar strategi
4. Menampilkan Top 5 ticker dengan overlap terkuat
"""

from __future__ import annotations

import os
import glob
import importlib
import traceback
from collections import defaultdict
from datetime import datetime

import pandas as pd

# -----------------------------------------------------------------------
# DAFTAR STRATEGY
# module_path → (callable_name, report_version_key, label)
# -----------------------------------------------------------------------
STRATEGIES = [
    ("idx_breakout_screener_v2", "run_screener", "v2", "V2 Breakout"),
    ("idx_breakout_screener_v3", "run_screener", "v3", "V3 Retest"),
    ("idx_breakout_screener_v4_smc", "run_screener_v4", "v4", "V4 OrderBlock"),
    ("idx_breakout_screener_v5_smc", "run_screener_v5", "v5", "V5 CHOCH"),
    ("idx_intraday_screener", "run_intraday_screener", "intraday", "Intraday"),
    ("idx_highbeta_screener", "run_highbeta_screener", "highbeta", "HighBeta"),
    ("idx_accumulation_screener", "run_accumulation_screener", "accumulation", "Accumulation"),
]

DEFAULT_PARAMS = {
    "account_size": 50_000_000,
    "risk_per_trade_pct": 1.0,
}


def _search_dirs() -> list[str]:
    dirs = [os.getcwd(), "."]
    try:
        here = os.path.dirname(os.path.abspath(__file__))
        if here not in dirs:
            dirs.append(here)
    except Exception:
        pass
    for extra in ["/home/workdir", "/home/workdir/artifacts", "/home/workdir/attachments"]:
        if os.path.isdir(extra) and extra not in dirs:
            dirs.append(extra)
    return dirs


def find_report(version: str, today: str | None = None) -> str | None:
    """Cari file report terbaru untuk version."""
    files: list[str] = []
    for d in _search_dirs():
        files.extend(glob.glob(os.path.join(d, f"idx_report_{version}_*.csv")))
    if today:
        today_files = [f for f in files if today in os.path.basename(f)]
        if today_files:
            return max(today_files, key=os.path.getmtime)
    if not files:
        return None
    return max(files, key=os.path.getmtime)


def run_one_strategy(module_name: str, fn_name: str, label: str, params: dict) -> bool:
    print("\n" + "-" * 70)
    print(f"▶ Menjalankan {label} ({module_name}.{fn_name})")
    print("-" * 70)
    try:
        mod = importlib.import_module(module_name)
        fn = getattr(mod, fn_name, None)
        if fn is None:
            print(f"  [SKIP] Fungsi {fn_name} tidak ada.")
            return False

        # Beberapa skrip pakai params / user_params / PARAMS global
        if fn_name == "run_screener":
            if hasattr(mod, "PARAMS") and isinstance(mod.PARAMS, dict):
                mod.PARAMS.update(params)
                fn(params=mod.PARAMS)
            else:
                try:
                    fn(params=params)
                except TypeError:
                    fn(user_params=params)
        else:
            try:
                fn(user_params=params)
            except TypeError:
                try:
                    fn(params=params)
                except TypeError:
                    fn()
        print(f"  [OK] {label} selesai.")
        return True
    except Exception as e:
        print(f"  [ERROR] {label}: {e}")
        traceback.print_exc()
        return False


def load_report(version: str, label: str) -> pd.DataFrame:
    path = find_report(version)
    if not path or not os.path.exists(path):
        print(f"  • {label}: file tidak ditemukan")
        return pd.DataFrame()
    try:
        df = pd.read_csv(path)
        if df.empty or "Ticker" not in df.columns:
            print(f"  • {label}: kosong / tanpa kolom Ticker")
            return pd.DataFrame()
        df = df.copy()
        df["Ticker"] = df["Ticker"].astype(str).str.upper().str.strip()
        df["_strategy"] = label
        df["_version"] = version
        df["_source"] = os.path.basename(path)
        print(f"  • {label}: {len(df)} baris ← {os.path.basename(path)}")
        return df
    except Exception as e:
        print(f"  • {label}: gagal baca ({e})")
        return pd.DataFrame()


def _pick_score(row: pd.Series) -> float:
    for col in ("Score", "RR_Ratio", "RR"):
        if col in row.index and pd.notna(row[col]):
            try:
                return float(row[col])
            except Exception:
                pass
    return 0.0


def build_confluence(frames: list[pd.DataFrame], top_n: int = 5) -> pd.DataFrame:
    """
    Confluence = ticker muncul di >= 2 strategi.
    Rank: jumlah strategi ↓, total score ↓, avg RR ↓.
    """
    if not frames:
        return pd.DataFrame()

    # ticker → list record per strategy
    bucket: dict[str, list[dict]] = defaultdict(list)

    for df in frames:
        if df.empty:
            continue
        # 1 baris per ticker per strategy
        sub = df.drop_duplicates(subset=["Ticker"], keep="first")
        for _, row in sub.iterrows():
            t = row["Ticker"]
            bucket[t].append(
                {
                    "strategy": row["_strategy"],
                    "version": row["_version"],
                    "score": _pick_score(row),
                    "rr": float(row["RR_Ratio"]) if "RR_Ratio" in row and pd.notna(row["RR_Ratio"]) else None,
                    "entry": row.get("Entry", row.get("Close")),
                    "sl": row.get("StopLoss"),
                    "target": row.get("Target(Liquidity)", row.get("Target(Peak)", row.get("Target"))),
                    "close": row.get("Close"),
                }
            )

    rows = []
    for ticker, items in bucket.items():
        n = len(items)
        if n < 2:
            continue  # butuh minimal 2 strategi
        strategies = sorted({x["strategy"] for x in items})
        scores = [x["score"] for x in items]
        rrs = [x["rr"] for x in items if x["rr"] is not None]
        rows.append(
            {
                "Ticker": ticker,
                "N_Strategies": n,
                "Strategies": " + ".join(strategies),
                "TotalScore": round(sum(scores), 1),
                "AvgScore": round(sum(scores) / n, 1),
                "AvgRR": round(sum(rrs) / len(rrs), 2) if rrs else None,
                "Close": items[0].get("close"),
                "Entries": ", ".join(
                    f"{x['strategy']}:{x['entry']}" for x in items if x.get("entry") is not None
                ),
                "Stops": ", ".join(
                    f"{x['strategy']}:{x['sl']}" for x in items if x.get("sl") is not None
                ),
                "Targets": ", ".join(
                    f"{x['strategy']}:{x['target']}" for x in items if x.get("target") is not None
                ),
            }
        )

    if not rows:
        return pd.DataFrame()

    out = pd.DataFrame(rows)
    out = out.sort_values(
        by=["N_Strategies", "TotalScore", "AvgRR"],
        ascending=[False, False, False],
    ).reset_index(drop=True)
    return out.head(top_n)


def run_confluence(
    params: dict | None = None,
    top_n: int = 5,
    skip_run: bool = False,
    only_versions: list[str] | None = None,
) -> pd.DataFrame:
    """
    skip_run=True → hanya baca CSV yang sudah ada (tidak menjalankan ulang screener).
    only_versions → mis. ["v2","v4","accumulation"]
    """
    p = DEFAULT_PARAMS.copy()
    if params:
        p.update(params)

    today = datetime.now().strftime("%Y-%m-%d")
    print("=" * 70)
    print(f"IDX CONFLUENCE RUNNER — {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print("=" * 70)

    # 1) Jalankan semua strategy
    if not skip_run:
        for mod_name, fn_name, ver, label in STRATEGIES:
            if only_versions and ver not in only_versions:
                continue
            run_one_strategy(mod_name, fn_name, label, p)
    else:
        print("(skip_run=True — tidak menjalankan ulang screener)")

    # 2) Baca semua hasil
    print("\n" + "=" * 70)
    print("MEMBACA HASIL REPORT")
    print("=" * 70)
    frames = []
    for mod_name, fn_name, ver, label in STRATEGIES:
        if only_versions and ver not in only_versions:
            continue
        df = load_report(ver, label)
        if not df.empty:
            frames.append(df)

    if not frames:
        print("\nTidak ada report yang bisa dibaca. Jalankan screener dulu.")
        return pd.DataFrame()

    # 3) Confluence top N
    print("\n" + "=" * 70)
    print(f"TOP {top_n} TICKER BER-CONFLUENCE (≥2 strategi)")
    print("=" * 70)
    top = build_confluence(frames, top_n=top_n)

    if top.empty:
        print("Tidak ada ticker yang muncul di ≥2 strategi hari ini.")
        return top

    print(top.to_string(index=False))
    print("=" * 70)

    out_name = f"idx_report_confluence_{today}.csv"
    top.to_csv(out_name, index=False)
    print(f"Disimpan: {out_name}")
    return top


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="IDX Confluence Runner")
    parser.add_argument("--skip-run", action="store_true", help="Hanya baca CSV existing")
    parser.add_argument("--top", type=int, default=5, help="Jumlah ticker top confluence")
    parser.add_argument("--modal", type=float, default=50_000_000)
    parser.add_argument("--risk", type=float, default=1.0)
    args = parser.parse_args()

    run_confluence(
        params={"account_size": args.modal, "risk_per_trade_pct": args.risk},
        top_n=args.top,
        skip_run=args.skip_run,
    )