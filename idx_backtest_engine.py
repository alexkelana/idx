"""
IDX BACKTEST ENGINE
============================================================
Uji efektivitas setup dari output screener (V2–V5, Intra, HB, ACC, …)
per emiten — signal-forward simulation pada data harian.

Alur:
  1. Baca baris setup (CSV report / DataFrame / dict)
  2. Tentukan tanggal sinyal (BreakoutDay / signal_date / file date)
  3. Ambil OHLCV ke depan (horizon_bars)
  4. Simulasi exit: SL / TP1 / TP2 / time-stop / trailing (opsional)
  5. Hitung metrik per trade + agregat per strategy / ticker

Model eksekusi (realistis BEI):
  - Entry: Close hari sinyal ATAU Open bar berikutnya (konfigurasi)
  - Intrabar sama hari: asumsi konservatif → SL dicek dulu jika Low<=SL & High>=TP
  - Fee beli/jual + PPh final jual 0.1% (opsional)
  - Lot 100, tick diabaikan pada fill historis (harga bar mentah)

Disclaimer: backtest ≠ jaminan hasil live. Slippage/gap/ARA tidak dimodel penuh.
"""

from __future__ import annotations

import glob
import re
import os
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Iterable

import numpy as np
try:
    from idx_exchange_rules import round_to_idx_tick as _round_tick
except ImportError:
    def _legacy_round_tick_UNUSED(price: float) -> float:
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

import pandas as pd

try:
    import yfinance as yf
except ImportError:
    yf = None


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
DEFAULT_BT_PARAMS = {
    "horizon_bars": 20,          # max holding (hari bursa)
    "entry_mode": "next_open",   # next_open | signal_close
    "intrabar_priority": "sl_first",  # sl_first | tp_first
    "use_tp2": False,            # jika True, TP2 setelah TP1 partial (sederhana: full TP2)
    "tp_level": "tp1",           # tp1 | tp2 | trailing
    "trailing_activate_r": 1.0,  # aktif trail setelah +1R (jika tp_level=trailing)
    "trailing_atr_mult": 1.5,
    "lot_size": 100,
    "account_size": 50_000_000,
    "risk_per_trade_pct": 1.0,
    "buy_fee_pct": 0.15,         # %
    "sell_fee_pct": 0.25,        # % (belum PPh)
    "pph_sell_pct": 0.10,        # PPh final jual 0.1%
    "include_pph": True,
    "min_bars_after_signal": 1,
}


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------
@dataclass
class TradeResult:
    ticker: str
    strategy: str
    signal_date: str
    entry_date: str
    exit_date: str
    entry: float
    stop_loss: float
    target: float
    exit_price: float
    exit_reason: str  # SL | TP1 | TP2 | TIME | TRAIL
    bars_held: int
    r_multiple: float
    pnl_gross: float
    pnl_net: float
    fees: float
    lots: int
    planned_rr: float | None = None
    risk_per_share: float = 0.0
    notes: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class BacktestSummary:
    n_trades: int = 0
    n_wins: int = 0
    n_losses: int = 0
    win_rate: float = 0.0
    avg_r: float = 0.0
    expectancy_r: float = 0.0
    profit_factor: float = 0.0
    total_pnl_net: float = 0.0
    max_win_r: float = 0.0
    max_loss_r: float = 0.0
    avg_bars_held: float = 0.0
    avg_planned_rr: float | None = None
    by_exit_reason: dict = field(default_factory=dict)
    by_strategy: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


# ---------------------------------------------------------------------------
# Helpers: load reports, parse rows
# ---------------------------------------------------------------------------
def _search_dirs() -> list[str]:
    dirs = [os.getcwd(), "."]
    try:
        here = os.path.dirname(os.path.abspath(__file__))
        if here not in dirs:
            dirs.append(here)
    except Exception:
        pass
    for extra in ("/home/workdir", "/home/workdir/artifacts", "/home/workdir/attachments"):
        if os.path.isdir(extra) and extra not in dirs:
            dirs.append(extra)
    return dirs


def find_report(version: str, date: str | None = None) -> str | None:
    files: list[str] = []
    for d in _search_dirs():
        files.extend(glob.glob(os.path.join(d, f"idx_report_{version}_*.csv")))
        # beberapa versi menyimpan nama sedikit berbeda
        files.extend(glob.glob(os.path.join(d, f"idx_report*{version}*.csv")))
    files = sorted(set(files))
    if not files:
        return None
    if date:
        today_files = [f for f in files if date in os.path.basename(f)]
        if today_files:
            return max(today_files, key=os.path.getmtime)
    return max(files, key=os.path.getmtime)


def extract_report_date(path: str | None) -> str | None:
    """Ambil YYYY-MM-DD dari nama file report atau mtime file."""
    if not path:
        return None
    base = os.path.basename(path)
    m = re.search(r"(20\d{2}-\d{2}-\d{2})", base)
    if m:
        return m.group(1)
    try:
        ts = datetime.fromtimestamp(os.path.getmtime(path))
        return ts.strftime("%Y-%m-%d")
    except Exception:
        return datetime.now().strftime("%Y-%m-%d")


def load_screener_setups(
    path: str | None = None,
    version: str | None = None,
    tickers: list[str] | None = None,
) -> pd.DataFrame:
    """Muat CSV report screener."""
    if path is None and version:
        path = find_report(version)
    if path is None or not os.path.isfile(path):
        raise FileNotFoundError(f"Report tidak ditemukan: path={path} version={version}")
    df = pd.read_csv(path)
    if "Ticker" not in df.columns:
        raise ValueError("CSV harus punya kolom Ticker")
    df["Ticker"] = df["Ticker"].astype(str).str.upper().str.replace(".JK", "", regex=False).str.strip()
    if tickers:
        want = {str(t).upper().replace(".JK", "").strip() for t in tickers}
        df = df[df["Ticker"].isin(want)].copy()
    return df.reset_index(drop=True)


def _row_get(row: dict | pd.Series, *keys, default=None):
    for k in keys:
        if k in row and row[k] is not None and str(row[k]).strip() not in ("", "nan", "None"):
            try:
                if pd.isna(row[k]):
                    continue
            except Exception:
                pass
            return row[k]
    return default


def _f(x, default=None) -> float | None:
    try:
        if x is None or (isinstance(x, float) and x != x):
            return default
        return float(x)
    except Exception:
        return default


def normalize_setup(row: dict | pd.Series, report_date: str | None = None) -> dict[str, Any]:
    """Canonical fields dari report CSV **atau** sinyal Phase 2 (lowercase).

    Report: Ticker, Entry, StopLoss, BreakoutDay, ...
    Phase2: ticker, entry, stop_loss, tp1, signal_date, ...
    """
    if isinstance(row, pd.Series):
        row = row.to_dict()

    # Sudah canonical (dari find_signals_*)
    if row.get("entry") is not None and row.get("stop_loss") is not None and (
        row.get("signal_date") or row.get("ticker")
    ):
        ticker = str(row.get("ticker") or row.get("Ticker") or "").upper().replace(".JK", "").strip()
        signal_date = row.get("signal_date") or report_date
        if signal_date is not None:
            signal_date = str(signal_date).strip()[:10].replace("/", "-")
        return {
            "ticker": ticker,
            "entry": _f(row.get("entry")),
            "stop_loss": _f(row.get("stop_loss")),
            "tp1": _f(row.get("tp1") or row.get("target")),
            "tp2": _f(row.get("tp2")),
            "atr": _f(row.get("atr")),
            "strategy": str(row.get("strategy") or row.get("Strategy") or "unknown"),
            "planned_rr": _f(row.get("planned_rr") or row.get("RR_Ratio")),
            "signal_date": signal_date,
            "raw": row.get("raw", row),
        }

    ticker = str(
        _row_get(row, "Ticker", "ticker", default="")
    ).upper().replace(".JK", "").strip()
    entry = _f(
        _row_get(
            row,
            "Entry",
            "entry",
            "EntryBreakout",
            "EntryRetest",
            "Close",
        )
    )
    sl = _f(_row_get(row, "StopLoss", "stop_loss", "SL"))
    tp1 = _f(
        _row_get(
            row,
            "Target1",
            "tp1",
            "Target1(Peak)",
            "Target(Liquidity)",
            "Target(Peak)",
            "Target",
            "target",
            "Target2",
        )
    )
    tp2 = _f(_row_get(row, "Target2(Ext)", "tp2", "Target2", "Target(Ext)"))
    atr = _f(_row_get(row, "ATR", "atr", "ATR14"))
    strategy = str(_row_get(row, "Strategy", "strategy", default="unknown"))
    planned_rr = _f(_row_get(row, "RR_Ratio", "planned_rr", "RR"))
    signal_date = _row_get(
        row,
        "signal_date",
        "BreakoutDay",
        "SignalDate",
        "Date",
        "Signal_Date",
        "AsOf",
        default=None,
    )
    if signal_date is None or str(signal_date).strip() in ("", "nan", "None"):
        signal_date = report_date
    if signal_date is not None:
        signal_date = str(signal_date).strip()[:10]
        signal_date = signal_date.replace("/", "-")
    return {
        "ticker": ticker,
        "entry": entry,
        "stop_loss": sl,
        "tp1": tp1,
        "tp2": tp2,
        "atr": atr,
        "strategy": strategy,
        "planned_rr": planned_rr,
        "signal_date": signal_date,
        "raw": row,
    }


# ---------------------------------------------------------------------------
# Market data
# ---------------------------------------------------------------------------
def fetch_ohlcv(
    ticker: str,
    start: str,
    end: str | None = None,
    buffer_days: int = 5,
) -> pd.DataFrame:
    if yf is None:
        raise RuntimeError("yfinance tidak terpasang")
    sym = str(ticker).upper().replace(".JK", "").strip() + ".JK"
    start_dt = pd.Timestamp(start) - timedelta(days=buffer_days)
    end_dt = pd.Timestamp(end) + timedelta(days=buffer_days) if end else None
    # end None → sampai data terbaru; pastikan start tidak di masa depan
    if end_dt is None:
        end_dt = pd.Timestamp.now() + timedelta(days=2)
    if start_dt > pd.Timestamp.now():
        start_dt = pd.Timestamp.now() - timedelta(days=30)
    df = yf.download(
        sym,
        start=start_dt.strftime("%Y-%m-%d"),
        end=(end_dt + timedelta(days=1)).strftime("%Y-%m-%d"),
        interval="1d",
        progress=False,
        auto_adjust=True,
        multi_level_index=False,
        threads=False,
    )
    if df is None or df.empty:
        return pd.DataFrame()
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df.dropna(subset=["Open", "High", "Low", "Close"])
    df.index = pd.to_datetime(df.index).tz_localize(None)
    return df


def _position_size(entry: float, sl: float, params: dict) -> tuple[int, float]:
    risk_ps = entry - sl
    if risk_ps <= 0:
        return 0, 0.0
    risk_rp = params["account_size"] * params["risk_per_trade_pct"] / 100.0
    lots = int((risk_rp / risk_ps) // params["lot_size"])
    return max(lots, 0), risk_ps


def _fees_round_trip(entry: float, exit_p: float, shares: int, params: dict) -> float:
    buy = entry * shares * (params["buy_fee_pct"] / 100.0)
    sell = exit_p * shares * (params["sell_fee_pct"] / 100.0)
    pph = exit_p * shares * (params["pph_sell_pct"] / 100.0) if params.get("include_pph") else 0.0
    return buy + sell + pph


# ---------------------------------------------------------------------------
# Core simulation (one setup)
# ---------------------------------------------------------------------------
def simulate_trade(
    setup: dict,
    ohlcv: pd.DataFrame,
    params: dict | None = None,
) -> TradeResult | None:
    """
    Forward-test satu setup long-only.
    """
    p = {**DEFAULT_BT_PARAMS, **(params or {})}
    ticker = setup["ticker"]
    entry_plan = setup["entry"]
    sl = setup["stop_loss"]
    tp1 = setup["tp1"]
    tp2 = setup.get("tp2")
    atr = setup.get("atr") or 0.0
    signal_date = setup.get("signal_date")

    if not ticker or entry_plan is None or sl is None or tp1 is None:
        return None
    if entry_plan <= sl:
        return None
    if ohlcv is None or ohlcv.empty:
        return None

    # Lokasi bar sinyal
    if signal_date:
        try:
            sig_ts = pd.Timestamp(signal_date)
        except Exception:
            return None
        # bar pada atau setelah signal_date; jika sinyal di masa depan data,
        # pakai bar terakhir yang tersedia (report "hari ini")
        future_idx = ohlcv.index[ohlcv.index >= sig_ts]
        if len(future_idx) == 0:
            past_idx = ohlcv.index[ohlcv.index <= sig_ts]
            if len(past_idx) == 0:
                return None
            sig_loc = ohlcv.index.get_loc(past_idx[-1])
        else:
            sig_loc = ohlcv.index.get_loc(future_idx[0])
        if isinstance(sig_loc, slice):
            sig_loc = sig_loc.start or 0
    else:
        # tanpa tanggal → anggap sinyal di bar terakhir
        sig_loc = len(ohlcv) - 1

    # Entry bar
    mode = p.get("entry_mode", "next_open")
    if mode == "signal_close":
        entry_loc = sig_loc
        entry_price = float(ohlcv["Close"].iloc[entry_loc])
    else:
        # next_open; jika belum ada bar berikutnya (sinyal hari terakhir),
        # fallback ke close bar sinyal agar tetap bisa disimulasikan
        entry_loc = sig_loc + 1
        if entry_loc >= len(ohlcv):
            entry_loc = sig_loc
            entry_price = float(ohlcv["Close"].iloc[entry_loc])
            mode = "signal_close_fallback"
        else:
            entry_price = float(ohlcv["Open"].iloc[entry_loc])

    # Optional: gunakan planned entry dari screener jika dekat open/close
    # (default: harga bar aktual agar realistis)
    entry_date = ohlcv.index[entry_loc].strftime("%Y-%m-%d")
    risk_ps = entry_price - sl
    if risk_ps <= 0:
        # adjust SL if entry gap below planned
        return None

    lots, _ = _position_size(entry_price, sl, p)
    shares = lots * p["lot_size"]
    if shares <= 0:
        # tetap catat trade 1 lot untuk metrik R (size=0 → pnl 0, R tetap)
        shares = 0

    horizon = int(p["horizon_bars"])
    end_loc = min(entry_loc + horizon, len(ohlcv) - 1)

    # Trailing state
    use_trail = str(p.get("tp_level", "tp1")).lower() == "trailing"
    trail_stop = None
    activated = False
    target = tp2 if (p.get("use_tp2") and tp2) else tp1
    if p.get("tp_level") == "tp2" and tp2:
        target = tp2

    exit_price = None
    exit_reason = "TIME"
    exit_loc = end_loc

    for i in range(entry_loc + 1, end_loc + 1):
        hi = float(ohlcv["High"].iloc[i])
        lo = float(ohlcv["Low"].iloc[i])
        cl = float(ohlcv["Close"].iloc[i])

        # Trailing activation
        if use_trail and atr and atr > 0:
            r_now = (cl - entry_price) / risk_ps if risk_ps else 0
            if not activated and r_now >= float(p["trailing_activate_r"]):
                activated = True
                trail_stop = cl - float(p["trailing_atr_mult"]) * atr
            if activated:
                trail_stop = max(trail_stop or sl, cl - float(p["trailing_atr_mult"]) * atr)
                stop_use = max(sl, trail_stop)
            else:
                stop_use = sl
        else:
            stop_use = sl

        hit_sl = lo <= stop_use
        hit_tp = hi >= target if target else False

        if hit_sl and hit_tp:
            if p["intrabar_priority"] == "tp_first":
                exit_price, exit_reason, exit_loc = target, "TP1", i
            else:
                exit_price, exit_reason, exit_loc = stop_use, "SL", i
            break
        if hit_sl:
            exit_price, exit_reason, exit_loc = stop_use, ("TRAIL" if activated and stop_use > sl else "SL"), i
            break
        if hit_tp:
            exit_price, exit_reason, exit_loc = target, ("TP2" if target == tp2 else "TP1"), i
            break

    if exit_price is None:
        exit_price = float(ohlcv["Close"].iloc[end_loc])
        exit_reason = "TIME"
        exit_loc = end_loc

    exit_date = ohlcv.index[exit_loc].strftime("%Y-%m-%d")
    bars_held = exit_loc - entry_loc
    r_mult = (exit_price - entry_price) / risk_ps if risk_ps else 0.0

    pnl_gross = (exit_price - entry_price) * shares
    fees = _fees_round_trip(entry_price, exit_price, shares, p) if shares else 0.0
    pnl_net = pnl_gross - fees

    return TradeResult(
        ticker=ticker,
        strategy=str(setup.get("strategy") or "unknown"),
        signal_date=str(signal_date or ""),
        entry_date=entry_date,
        exit_date=exit_date,
        entry=round(entry_price, 2),
        stop_loss=round(float(sl), 2),
        target=round(float(target or 0), 2),
        exit_price=round(float(exit_price), 2),
        exit_reason=exit_reason,
        bars_held=int(bars_held),
        r_multiple=round(r_mult, 3),
        pnl_gross=round(pnl_gross, 0),
        pnl_net=round(pnl_net, 0),
        fees=round(fees, 0),
        lots=lots,
        planned_rr=_f(setup.get("planned_rr")),
        risk_per_share=round(risk_ps, 2),
        notes=f"entry_mode={p['entry_mode']}; priority={p['intrabar_priority']}",
    )


# ---------------------------------------------------------------------------
# Batch runner
# ---------------------------------------------------------------------------
def run_backtest_on_setups(
    setups: pd.DataFrame | list[dict],
    params: dict | None = None,
    report_date: str | None = None,
    progress: bool = True,
) -> tuple[pd.DataFrame, BacktestSummary]:
    """
    Backtest banyak setup dari screener output.
    """
    p = {**DEFAULT_BT_PARAMS, **(params or {})}
    if isinstance(setups, pd.DataFrame):
        rows = [normalize_setup(r, report_date) for _, r in setups.iterrows()]
    else:
        rows = [normalize_setup(r, report_date) for r in setups]

    trades: list[TradeResult] = []
    skips: list[str] = []
    # pastikan selalu ada fallback tanggal
    if not report_date:
        report_date = datetime.now().strftime("%Y-%m-%d")

    for i, setup in enumerate(rows, 1):
        if progress:
            print(f"  [{i}/{len(rows)}] {setup.get('ticker')}...", end="\r")
        if not setup.get("ticker"):
            skips.append("(empty ticker)")
            continue
        if not setup.get("signal_date"):
            setup["signal_date"] = report_date
        if setup.get("entry") is None or setup.get("stop_loss") is None or setup.get("tp1") is None:
            skips.append(f"{setup.get('ticker')}: level Entry/SL/TP kosong")
            continue
        sig = setup["signal_date"]
        horizon = int(p["horizon_bars"])
        try:
            end = (pd.Timestamp(sig) + timedelta(days=horizon * 2 + 14)).strftime("%Y-%m-%d")
            ohlcv = fetch_ohlcv(setup["ticker"], start=sig, end=end)
            if ohlcv is None or ohlcv.empty:
                # coba period lebih longgar ke belakang
                ohlcv = fetch_ohlcv(
                    setup["ticker"],
                    start=(pd.Timestamp(sig) - timedelta(days=40)).strftime("%Y-%m-%d"),
                    end=end,
                )
            if ohlcv is None or ohlcv.empty:
                skips.append(f"{setup.get('ticker')}: OHLCV kosong (yfinance)")
                continue
            tr = simulate_trade(setup, ohlcv, p)
            if tr:
                trades.append(tr)
            else:
                skips.append(
                    f"{setup.get('ticker')}: simulasi None "
                    f"(signal={sig}, bars={len(ohlcv)})"
                )
        except Exception as e:
            skips.append(f"{setup.get('ticker')}: {e}")
            if progress:
                print(f"\n  [skip] {setup.get('ticker')}: {e}")
            continue
    if progress and skips and not trades:
        print("Alasan skip:")
        for s in skips[:15]:
            print(f"  - {s}")
    if progress:
        print(" " * 50, end="\r")

    df = pd.DataFrame([t.to_dict() for t in trades]) if trades else pd.DataFrame()
    summary = summarize_trades(df)
    return df, summary


def summarize_trades(df: pd.DataFrame) -> BacktestSummary:
    s = BacktestSummary()
    if df is None or df.empty:
        return s
    s.n_trades = len(df)
    s.n_wins = int((df["r_multiple"] > 0).sum())
    s.n_losses = int((df["r_multiple"] <= 0).sum())
    s.win_rate = round(s.n_wins / s.n_trades * 100, 1) if s.n_trades else 0.0
    s.avg_r = round(float(df["r_multiple"].mean()), 3)
    s.expectancy_r = s.avg_r  # R expectancy per trade
    gains = df.loc[df["pnl_net"] > 0, "pnl_net"].sum()
    losses = df.loc[df["pnl_net"] < 0, "pnl_net"].sum()
    s.profit_factor = round(float(gains / abs(losses)), 2) if losses < 0 else float("inf")
    s.total_pnl_net = round(float(df["pnl_net"].sum()), 0)
    s.max_win_r = round(float(df["r_multiple"].max()), 3)
    s.max_loss_r = round(float(df["r_multiple"].min()), 3)
    s.avg_bars_held = round(float(df["bars_held"].mean()), 1)
    if "planned_rr" in df.columns and df["planned_rr"].notna().any():
        s.avg_planned_rr = round(float(df["planned_rr"].mean()), 2)
    s.by_exit_reason = df["exit_reason"].value_counts().to_dict()
    if "strategy" in df.columns:
        by = {}
        for st, g in df.groupby("strategy"):
            by[str(st)] = {
                "n": len(g),
                "win_rate": round(float((g["r_multiple"] > 0).mean() * 100), 1),
                "avg_r": round(float(g["r_multiple"].mean()), 3),
                "pnl_net": round(float(g["pnl_net"].sum()), 0),
            }
        s.by_strategy = by
    return s


def backtest_ticker_from_report(
    ticker: str,
    version: str,
    *,
    path: str | None = None,
    params: dict | None = None,
    report_date: str | None = None,
) -> tuple[pd.DataFrame, BacktestSummary]:
    """Convenience: satu emiten dari report versi tertentu."""
    df = load_screener_setups(path=path, version=version, tickers=[ticker])
    if report_date is None:
        report_date = extract_report_date(path) or datetime.now().strftime("%Y-%m-%d")
    return run_backtest_on_setups(df, params=params, report_date=report_date)


def backtest_report(
    version: str,
    *,
    path: str | None = None,
    tickers: list[str] | None = None,
    params: dict | None = None,
    save: bool = True,
) -> tuple[pd.DataFrame, BacktestSummary, str | None]:
    """
    Backtest semua (atau subset) setup di report screener.
    """
    path = path or find_report(version)
    df = load_screener_setups(path=path, version=version, tickers=tickers)
    report_date = extract_report_date(path) or datetime.now().strftime("%Y-%m-%d")
    print("=" * 70)
    print(
        f"IDX BACKTEST ENGINE — {version} | setups={len(df)} | "
        f"signal_date fallback={report_date}"
    )
    print("=" * 70)
    trades, summary = run_backtest_on_setups(df, params=params, report_date=report_date)
    print_summary(summary)
    out_path = None
    if save and not trades.empty:
        out_path = f"idx_backtest_{version}_{datetime.now().strftime('%Y-%m-%d')}.csv"
        trades.to_csv(out_path, index=False)
        print(f"Trades disimpan: {out_path}")
    return trades, summary, out_path


def print_summary(s: BacktestSummary) -> None:
    print("-" * 70)
    print(f"Trades     : {s.n_trades}  (W {s.n_wins} / L {s.n_losses})")
    print(f"Win rate   : {s.win_rate}%")
    print(f"Avg R      : {s.avg_r}   | Expectancy R: {s.expectancy_r}")
    print(f"Profit factor: {s.profit_factor}")
    print(f"Total PnL net: Rp {s.total_pnl_net:,.0f}")
    print(f"Max W/L R  : {s.max_win_r} / {s.max_loss_r}")
    print(f"Avg hold   : {s.avg_bars_held} bars")
    if s.avg_planned_rr is not None:
        print(f"Avg planned RR (screener): {s.avg_planned_rr}")
    if s.by_exit_reason:
        print(f"Exit reason: {s.by_exit_reason}")
    if s.by_strategy:
        print("By strategy:")
        for k, v in s.by_strategy.items():
            print(f"  {k}: {v}")
    print("-" * 70)


# ---------------------------------------------------------------------------
# Historical signal replay (optional): scan past N days with same rules
# ---------------------------------------------------------------------------
def walk_forward_hint() -> str:
    return (
        "Walk-forward penuh (ulang rule V2–V5 tiap hari historis) butuh adapter "
        "per strategy. Versi engine ini fokus **signal-forward**: ambil setup "
        "dari report screener, uji fill SL/TP ke depan. Untuk walk-forward, "
        "panggil screener logic pada rolling window (fase 2)."
    )



# ===========================================================================
# PHASE 2 — Historical strategy adapters (per-ticker effectiveness)
# ===========================================================================
# Satu engine; adapter find_signals() per strategi.
# V3 diimplementasi penuh; V2/V4/V5: breakout-proxy sederhana (bisa diperkaya).

def _prep_ohlcv_df(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()
    if isinstance(df.columns, pd.MultiIndex):
        try:
            df = df.copy()
            df.columns = df.columns.get_level_values(0)
        except Exception:
            pass
    need = ["Open", "High", "Low", "Close", "Volume"]
    if any(c not in df.columns for c in need):
        return pd.DataFrame()
    df = df.dropna(subset=need).copy()
    df.index = pd.to_datetime(df.index).tz_localize(None)
    return df


def _download_history(ticker: str, lookback_days: int = 500) -> pd.DataFrame:
    if yf is None:
        raise RuntimeError("yfinance tidak terpasang")
    sym = str(ticker).upper().replace(".JK", "").strip() + ".JK"
    df = yf.download(
        sym,
        period=f"{int(lookback_days)}d",
        interval="1d",
        progress=False,
        auto_adjust=True,
        multi_level_index=False,
        threads=False,
    )
    return _prep_ohlcv_df(df)



def _atr_series(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high, low, close = df["High"], df["Low"], df["Close"]
    prev = close.shift(1)
    tr = pd.concat(
        [high - low, (high - prev).abs(), (low - prev).abs()],
        axis=1,
    ).max(axis=1)
    return tr.rolling(period).mean()


def _v3_regime_ok(df_slice: pd.DataFrame, mode: str = "relaxed") -> bool:
    """Weekly regime check on slice ending at signal bar (approx)."""
    if len(df_slice) < 80:
        return False
    try:
        weekly = (
            df_slice.resample("W")
            .agg(
                {
                    "Open": "first",
                    "High": "max",
                    "Low": "min",
                    "Close": "last",
                    "Volume": "sum",
                }
            )
            .dropna()
        )
        if len(weekly) < 30:
            return True  # jangan terlalu ketat di history
        ma10 = weekly["Close"].rolling(10).mean()
        ma30 = weekly["Close"].rolling(30).mean()
        w_c = float(weekly["Close"].iloc[-1])
        w10 = float(ma10.iloc[-1])
        w30 = float(ma30.iloc[-1])
        if any(x != x for x in (w10, w30)):
            return True
        clear_bear = w_c < w10 < w30
        mode = (mode or "relaxed").lower()
        if mode == "strict":
            return (w_c > w10 > w30) and (w10 >= float(ma10.iloc[-2]))
        if mode == "relaxed":
            return not clear_bear
        return (w_c > w10) and (w10 >= w30 * 0.98)
    except Exception:
        return True


def find_signals_v3(
    ticker: str,
    *,
    lookback_days: int = 400,
    max_signals: int = 40,
    min_gap_bars: int = 5,
    regime_mode: str = "relaxed",
    atr_sl_mult: float = 1.35,
    min_rr: float = 1.2,
) -> list[dict]:
    """
    Scan history: tiap bar di zona Fibo setelah breakout volume = 1 sinyal V3.
    Return list setup canonical (signal_date, entry, stop_loss, tp1, ...).
    """
    df = _download_history(ticker, lookback_days)
    if df.empty or len(df) < 160:
        return []

    atr_s = _atr_series(df, 14)
    signals: list[dict] = []
    last_sig_i = -999
    n = len(df)

    # i = candidate signal bar (bukan breakout bar)
    for i in range(120, n - 1):
        if i - last_sig_i < min_gap_bars:
            continue
        if not _v3_regime_ok(df.iloc[: i + 1], regime_mode):
            continue

        # cari breakout terbaru dalam 15 bar sebelum i
        breakout_idx = -1
        breakout_price = 0.0
        for j in range(i - 1, max(i - 16, 25), -1):
            past_high = float(df["High"].iloc[j - 20 : j].max())
            past_vol = float(df["Volume"].iloc[j - 20 : j].mean())
            if past_vol <= 0:
                continue
            cl = float(df["Close"].iloc[j])
            hi = float(df["High"].iloc[j])
            lo = float(df["Low"].iloc[j])
            br = max(hi - lo, 1e-9)
            strong = cl >= lo + 0.45 * br
            if cl > past_high and float(df["Volume"].iloc[j]) >= past_vol * 1.4 and strong:
                breakout_idx = j
                breakout_price = past_high
                break
        if breakout_idx < 0:
            continue

        swing_low = float(df["Low"].iloc[max(0, breakout_idx - 20) : breakout_idx].min())
        peak = float(df["High"].iloc[breakout_idx : i + 1].max())
        range_up = peak - swing_low
        if range_up <= 0:
            continue
        fib_236 = peak - 0.236 * range_up
        fib_382 = peak - 0.382 * range_up
        fib_618 = peak - 0.618 * range_up

        last_close = float(df["Close"].iloc[i])
        if not (fib_618 <= last_close <= fib_236):
            continue
        if last_close < breakout_price * 0.995:
            continue

        atr = float(atr_s.iloc[i]) if not pd.isna(atr_s.iloc[i]) else last_close * 0.02
        entry = _round_tick(last_close)
        # hybrid SL mirip V3
        stop_bo = breakout_price * 0.98
        stop_swing = swing_low * 0.995
        stop_atr = entry - atr_sl_mult * atr
        stop_raw = min(stop_bo, stop_swing, stop_atr)
        atr_floor = entry - 1.0 * atr
        if stop_raw > atr_floor:
            stop_raw = atr_floor
        max_depth = entry * 0.91
        if stop_raw < max_depth:
            stop_raw = max_depth
        sl = _round_tick(stop_raw)
        if sl >= entry:
            continue
        risk = entry - sl
        if risk <= 0 or (atr > 0 and risk / atr < 0.8):
            continue
        tp1 = _round_tick(max(peak, entry + risk * 1.5))
        if tp1 <= entry:
            continue
        rr = (tp1 - entry) / risk
        if rr < min_rr:
            continue
        tp2 = _round_tick(max(peak + 0.272 * range_up, entry + risk * 2.5))

        sig_date = df.index[i].strftime("%Y-%m-%d")
        signals.append(
            {
                "ticker": str(ticker).upper().replace(".JK", "").strip(),
                "entry": entry,
                "stop_loss": sl,
                "tp1": tp1,
                "tp2": tp2,
                "atr": round(atr, 1),
                "strategy": "V3 (Retest Fibo)",
                "planned_rr": round(rr, 2),
                "signal_date": sig_date,
                "raw": {
                    "BreakoutDay": df.index[breakout_idx].strftime("%Y-%m-%d"),
                    "Fibo382": _round_tick(fib_382),
                    "Fibo618": _round_tick(fib_618),
                },
            }
        )
        last_sig_i = i
        if len(signals) >= max_signals:
            break

    return signals



def find_signals_v2(
    ticker: str,
    *,
    lookback_days: int = 400,
    max_signals: int = 40,
    min_gap_bars: int = 8,
) -> list[dict]:
    """
    V2 Breakout+Sweep+Retest (historis).
    Sinyal: dekat resistance konsolidasi + volume + (opsional) close di atas resistance
    atau retest hold; SL di bawah swing/ATR.
    """
    df = _download_history(ticker, lookback_days)
    if df.empty or len(df) < 60:
        return []
    atr_s = _atr_series(df, 14)
    signals: list[dict] = []
    last_sig = -999
    n = len(df)
    win = 20
    for i in range(50, n - 1):
        if i - last_sig < min_gap_bars:
            continue
        sub = df.iloc[: i + 1]
        close = sub["Close"]
        high = sub["High"]
        low = sub["Low"]
        volume = sub["Volume"]
        last_close = float(close.iloc[-1])
        recent_high = float(high.tail(win).max())
        recent_low = float(low.tail(win).min())
        range_pct = (recent_high - recent_low) / last_close * 100 if last_close else 999
        dist_to_high = (recent_high - last_close) / last_close * 100 if last_close else 999
        if range_pct > 25 or dist_to_high > 8:
            continue
        avg_vol20 = float(volume.tail(20).mean())
        if avg_vol20 <= 0:
            continue
        vol_ratio = float(volume.tail(5).mean()) / avg_vol20
        last_vol_ratio = float(volume.iloc[-1]) / avg_vol20
        if vol_ratio < 1.1 and last_vol_ratio < 1.15:
            continue
        ma20 = float(close.rolling(20).mean().iloc[-1])
        if last_close < ma20 * 0.98:
            continue
        # RSI kasar
        delta = close.diff()
        gain = delta.clip(lower=0).rolling(14).mean()
        loss = (-delta.clip(upper=0)).rolling(14).mean()
        rs = gain / loss.replace(0, np.nan)
        rsi = float((100 - (100 / (1 + rs))).iloc[-1]) if not pd.isna(rs.iloc[-1]) else 50
        if rsi > 78 or rsi < 35:
            continue
        atr = float(atr_s.iloc[i]) if not pd.isna(atr_s.iloc[i]) else last_close * 0.02
        # Sweep: wick di atas resistance dalam 12 bar
        has_sweep = False
        for k in range(max(0, i - 12), i + 1):
            hi = float(high.iloc[k])
            lo = float(low.iloc[k])
            cl = float(close.iloc[k])
            op = float(sub["Open"].iloc[k])
            rng = max(hi - lo, 1e-9)
            upper_wick = hi - max(op, cl)
            if hi >= recent_high * 0.998 and (upper_wick / rng >= 0.25 or hi > recent_high):
                has_sweep = True
                break
        # Breakout atau dekat BO
        broke = last_close > recent_high * 0.998 or float(high.iloc[-1]) >= recent_high
        if not (broke or has_sweep):
            continue
        entry = _round_tick(last_close)
        swing = float(low.tail(15).min())
        sl = _round_tick(min(entry - 1.5 * atr, swing * 0.995))
        if sl >= entry:
            sl = _round_tick(entry - atr)
        risk = entry - sl
        if risk <= 0:
            continue
        tp1 = _round_tick(max(recent_high * 1.02, entry + risk * 1.5))
        tp2 = _round_tick(entry + risk * 2.5)
        rr = (tp1 - entry) / risk
        if rr < 1.2:
            continue
        signals.append(
            {
                "ticker": str(ticker).upper().replace(".JK", "").strip(),
                "entry": entry,
                "stop_loss": sl,
                "tp1": tp1,
                "tp2": tp2,
                "atr": round(atr, 1),
                "strategy": "V2 (Breakout+Sweep)",
                "planned_rr": round(rr, 2),
                "signal_date": df.index[i].strftime("%Y-%m-%d"),
                "raw": {"Resistance": recent_high, "Sweep": has_sweep},
            }
        )
        last_sig = i
        if len(signals) >= max_signals:
            break
    return signals


def _detect_ob_zones(df: pd.DataFrame, lookback: int = 60) -> list[dict]:
    """Simplified bullish OB: down-candle before BOS up."""
    zones = []
    n = len(df)
    start = max(5, n - lookback)
    for i in range(start, n - 2):
        o = float(df["Open"].iloc[i])
        c = float(df["Close"].iloc[i])
        h = float(df["High"].iloc[i])
        l = float(df["Low"].iloc[i])
        if c >= o:
            continue  # need bearish/impulse candle as OB candidate
        # BOS: later close breaks above this high
        for j in range(i + 1, min(i + 15, n)):
            if float(df["Close"].iloc[j]) > h:
                zones.append(
                    {
                        "ob_high": h,
                        "ob_low": l,
                        "ob_idx": i,
                        "bos_idx": j,
                    }
                )
                break
    return zones


def find_signals_v4(
    ticker: str,
    *,
    lookback_days: int = 400,
    max_signals: int = 40,
    min_gap_bars: int = 6,
) -> list[dict]:
    """V4 SMC Order Block — mitigasi OB terbaru, dist ≤3%, RR≥1.5."""
    df = _download_history(ticker, lookback_days)
    if df.empty or len(df) < 50:
        return []
    signals: list[dict] = []
    last_sig = -999
    n = len(df)
    for i in range(45, n - 1):
        if i - last_sig < min_gap_bars:
            continue
        sub = df.iloc[: i + 1]
        zones = _detect_ob_zones(sub, lookback=60)
        if not zones:
            continue
        ob = zones[-1]
        last_close = float(sub["Close"].iloc[-1])
        last_low = float(sub["Low"].iloc[-1])
        ob_top = float(ob["ob_high"])
        ob_bottom = float(ob["ob_low"])
        if ob_top <= 0:
            continue
        dist = (last_close - ob_top) / ob_top * 100
        mitigated = last_low <= ob_top and last_close >= ob_bottom
        if dist > 3.0 or not mitigated:
            continue
        entry = _round_tick(last_close)
        sl = _round_tick(ob_bottom * 0.985)
        if sl >= entry:
            continue
        risk = entry - sl
        if risk <= 0:
            continue
        peak = float(sub["High"].iloc[ob["bos_idx"] :].max())
        tp1 = _round_tick(max(peak, entry + risk * 1.5))
        if tp1 <= entry:
            continue
        rr = (tp1 - entry) / risk
        if rr < 1.5:
            continue
        tp2 = _round_tick(entry + risk * 2.5)
        signals.append(
            {
                "ticker": str(ticker).upper().replace(".JK", "").strip(),
                "entry": entry,
                "stop_loss": sl,
                "tp1": tp1,
                "tp2": tp2,
                "atr": 0.0,
                "strategy": "V4 (SMC Order Block)",
                "planned_rr": round(rr, 2),
                "signal_date": df.index[i].strftime("%Y-%m-%d"),
                "raw": {"OB_Top": ob_top, "OB_Bottom": ob_bottom},
            }
        )
        last_sig = i
        if len(signals) >= max_signals:
            break
    return signals


def find_signals_v5(
    ticker: str,
    *,
    lookback_days: int = 400,
    max_signals: int = 40,
    min_gap_bars: int = 6,
) -> list[dict]:
    """
    V5 CHOCH + discount zone.
    CHOCH: swing low broken then reclaimed / structure shift; entry di zona fib 50-78.6.
    """
    df = _download_history(ticker, lookback_days)
    if df.empty or len(df) < 60:
        return []
    atr_s = _atr_series(df, 14)
    signals: list[dict] = []
    last_sig = -999
    n = len(df)
    for i in range(50, n - 1):
        if i - last_sig < min_gap_bars:
            continue
        sub = df.iloc[: i + 1]
        # swing points simple
        highs = sub["High"].values
        lows = sub["Low"].values
        closes = sub["Close"].values
        # find recent swing high then lower low (CHOCH bear→bull when reclaim)
        choch_level = None
        swing_low_idx = None
        for j in range(len(sub) - 3, 20, -1):
            if lows[j] < lows[j - 1] and lows[j] < lows[j + 1]:
                # subsequent close reclaim above a prior short-term high
                local_high = float(highs[max(0, j - 10) : j].max())
                if float(closes[-1]) > local_high and float(closes[j]) <= local_high:
                    choch_level = local_high
                    swing_low_idx = j
                    break
        if choch_level is None or swing_low_idx is None:
            continue
        swing_low = float(lows[swing_low_idx])
        peak = float(highs[swing_low_idx:].max())
        range_up = peak - swing_low
        if range_up <= 0:
            continue
        fib50 = peak - 0.5 * range_up
        fib618 = peak - 0.618 * range_up
        fib786 = peak - 0.786 * range_up
        last_close = float(closes[-1])
        # discount zone
        if not (fib786 <= last_close <= fib50):
            continue
        atr = float(atr_s.iloc[i]) if not pd.isna(atr_s.iloc[i]) else last_close * 0.02
        entry = _round_tick(last_close)
        sl = _round_tick(min(swing_low * 0.99, entry - 1.2 * atr))
        if sl >= entry:
            continue
        risk = entry - sl
        if risk <= 0:
            continue
        tp1 = _round_tick(max(peak, entry + risk * 1.5))
        rr = (tp1 - entry) / risk
        if rr < 1.3:
            continue
        # optional sweep low
        has_sweep = False
        for k in range(max(0, len(sub) - 15), len(sub)):
            lo = float(lows[k])
            hi = float(highs[k])
            cl = float(closes[k])
            op = float(sub["Open"].iloc[k])
            rng = max(hi - lo, 1e-9)
            lower_wick = min(op, cl) - lo
            if lo < swing_low * 1.002 and lower_wick / rng >= 0.25 and cl > swing_low:
                has_sweep = True
                break
        signals.append(
            {
                "ticker": str(ticker).upper().replace(".JK", "").strip(),
                "entry": entry,
                "stop_loss": sl,
                "tp1": tp1,
                "tp2": _round_tick(entry + risk * 2.5),
                "atr": round(atr, 1),
                "strategy": "V5 (SMC CHOCH)",
                "planned_rr": round(rr, 2),
                "signal_date": df.index[i].strftime("%Y-%m-%d"),
                "raw": {
                    "CHOCH": choch_level,
                    "Fibo50": fib50,
                    "Fibo618": fib618,
                    "SweepLow": has_sweep,
                },
            }
        )
        last_sig = i
        if len(signals) >= max_signals:
            break
    return signals


def find_signals_intraday(
    ticker: str,
    *,
    lookback_days: int = 400,
    max_signals: int = 40,
    min_gap_bars: int = 5,
) -> list[dict]:
    """Intraday screener rules on daily bars (same filters as idx_intraday_screener)."""
    df = _download_history(ticker, lookback_days)
    if df.empty or len(df) < 35:
        return []
    atr_s = _atr_series(df, 14)
    signals: list[dict] = []
    last_sig = -999
    n = len(df)
    for i in range(30, n - 1):
        if i - last_sig < min_gap_bars:
            continue
        sub = df.iloc[: i + 1]
        close = sub["Close"]
        high = sub["High"]
        low = sub["Low"]
        vol = sub["Volume"]
        last_close = float(close.iloc[-1])
        if last_close < 50 or last_close > 20000:
            continue
        avg_vol = float(vol.tail(20).mean())
        if avg_vol <= 0:
            continue
        high20 = float(high.tail(20).max())
        low20 = float(low.tail(20).min())
        range20 = (high20 - low20) / last_close * 100
        if range20 > 22:
            continue
        atr = float(atr_s.iloc[i]) if not pd.isna(atr_s.iloc[i]) else 0
        if atr <= 0:
            continue
        atr_pct = atr / last_close * 100
        if atr_pct < 0.8 or atr_pct > 6.0:
            continue
        ma10 = float(close.rolling(10).mean().iloc[-1])
        ma20 = float(close.rolling(20).mean().iloc[-1])
        vol_ratio = float(vol.tail(5).mean()) / avg_vol
        dist_high = (high20 - last_close) / last_close * 100
        delta = close.diff()
        gain = delta.clip(lower=0).rolling(14).mean()
        loss = (-delta.clip(upper=0)).rolling(14).mean()
        rs = gain / loss.replace(0, np.nan)
        rsi = float((100 - (100 / (1 + rs))).iloc[-1]) if len(rs) and not pd.isna(rs.iloc[-1]) else 50
        score = 0
        if last_close > ma10 > ma20:
            score += 20
        elif last_close > ma10:
            score += 10
        if dist_high <= 4:
            score += 20
        if vol_ratio >= 1.2:
            score += 20
        elif vol_ratio >= 1.1:
            score += 8
        if 40 <= rsi <= 70:
            score += 15
        elif rsi > 72:
            score -= 10
        if 1.2 <= atr_pct <= 4.0:
            score += 10
        if score < 45:
            continue
        entry = _round_tick(last_close)
        stop_raw = max(entry - 1.0 * atr, entry * 0.985)
        sl = _round_tick(stop_raw)
        if sl >= entry:
            continue
        risk = entry - sl
        if risk <= 0 or risk / entry > 0.025:
            continue
        tp = _round_tick(min(entry + 1.5 * atr, entry * 1.03, high20 * 1.01 if high20 > entry else entry * 1.03))
        if tp <= entry:
            continue
        rr = (tp - entry) / risk
        if rr < 1.0:
            continue
        signals.append(
            {
                "ticker": str(ticker).upper().replace(".JK", "").strip(),
                "entry": entry,
                "stop_loss": sl,
                "tp1": tp,
                "tp2": _round_tick(entry + risk * 2.0),
                "atr": round(atr, 1),
                "strategy": "Intraday",
                "planned_rr": round(rr, 2),
                "signal_date": df.index[i].strftime("%Y-%m-%d"),
                "raw": {"Score": score, "ATR%": round(atr_pct, 2)},
            }
        )
        last_sig = i
        if len(signals) >= max_signals:
            break
    return signals


def find_signals_highbeta(
    ticker: str,
    *,
    lookback_days: int = 400,
    max_signals: int = 40,
    min_gap_bars: int = 5,
) -> list[dict]:
    """HighBeta liquid — range/ATR spekulatif + vol spike + RSI band."""
    df = _download_history(ticker, lookback_days)
    if df.empty or len(df) < 35:
        return []
    atr_s = _atr_series(df, 14)
    signals: list[dict] = []
    last_sig = -999
    n = len(df)
    for i in range(30, n - 1):
        if i - last_sig < min_gap_bars:
            continue
        sub = df.iloc[: i + 1]
        c = sub["Close"]
        h = sub["High"]
        l = sub["Low"]
        v = sub["Volume"]
        last = float(c.iloc[-1])
        prev = float(c.iloc[-2])
        high20 = float(h.tail(20).max())
        low20 = float(l.tail(20).min())
        range20 = (high20 - low20) / last * 100 if last else 0
        if range20 < 8 or range20 > 45:
            continue
        atr = float(atr_s.iloc[i]) if not pd.isna(atr_s.iloc[i]) else 0
        if atr <= 0:
            continue
        atr_pct = atr / last * 100
        if atr_pct < 2.0 or atr_pct > 12:
            continue
        avg_vol = float(v.tail(20).mean())
        if avg_vol <= 0:
            continue
        vol_ratio = float(v.tail(5).mean()) / avg_vol
        if vol_ratio < 1.15:
            continue
        delta = c.diff()
        gain = delta.clip(lower=0).rolling(14).mean()
        loss = (-delta.clip(upper=0)).rolling(14).mean()
        rs = gain / loss.replace(0, np.nan)
        rsi = float((100 - (100 / (1 + rs))).iloc[-1]) if not pd.isna(rs.iloc[-1]) else 50
        if not (30 <= rsi <= 75):
            continue
        ma10 = float(c.rolling(10).mean().iloc[-1])
        if last < ma10:
            continue
        score = 40
        if vol_ratio >= 1.5:
            score += 15
        if 12 <= range20 <= 28:
            score += 15
        if 3 <= atr_pct <= 7:
            score += 10
        dist_high = (high20 - last) / last * 100
        if dist_high <= 5:
            score += 10
        if score < 55:
            continue
        entry = _round_tick(last)
        sl = _round_tick(entry - 1.5 * atr)
        if sl >= entry:
            sl = _round_tick(entry * 0.97)
        risk = entry - sl
        if risk <= 0:
            continue
        tp = _round_tick(min(entry + 2.0 * atr, high20 * 1.02 if high20 > entry else entry + 2 * atr))
        rr = (tp - entry) / risk if risk else 0
        if rr < 1.2:
            continue
        signals.append(
            {
                "ticker": str(ticker).upper().replace(".JK", "").strip(),
                "entry": entry,
                "stop_loss": sl,
                "tp1": tp,
                "tp2": _round_tick(entry + risk * 2.5),
                "atr": round(atr, 1),
                "strategy": "HighBeta Liquid",
                "planned_rr": round(rr, 2),
                "signal_date": df.index[i].strftime("%Y-%m-%d"),
                "raw": {"Range20%": round(range20, 1), "ATR%": round(atr_pct, 2)},
            }
        )
        last_sig = i
        if len(signals) >= max_signals:
            break
    return signals


def find_signals_accumulation(
    ticker: str,
    *,
    lookback_days: int = 400,
    max_signals: int = 40,
    min_gap_bars: int = 10,
) -> list[dict]:
    """Late accumulation: base sideways, vol kontraksi, posisi atas range."""
    df = _download_history(ticker, lookback_days)
    if df.empty or len(df) < 70:
        return []
    atr_s = _atr_series(df, 14)
    signals: list[dict] = []
    last_sig = -999
    n = len(df)
    base_w = 50
    for i in range(base_w + 5, n - 1):
        if i - last_sig < min_gap_bars:
            continue
        sub = df.iloc[: i + 1]
        window = sub.tail(base_w)
        last = float(window["Close"].iloc[-1])
        hi = float(window["High"].max())
        lo = float(window["Low"].min())
        if lo <= 0:
            continue
        range_pct = (hi - lo) / last * 100
        if range_pct < 6 or range_pct > 28:
            continue
        pos = (last - lo) / (hi - lo) if hi > lo else 0
        if pos < 0.55:
            continue
        vol = window["Volume"]
        vol_early = float(vol.head(25).mean())
        vol_late = float(vol.tail(15).mean())
        if vol_early <= 0:
            continue
        contraction = vol_late / vol_early
        if contraction > 0.95:
            continue
        atr_now = float(atr_s.iloc[i]) if not pd.isna(atr_s.iloc[i]) else 0
        atr_prev = float(atr_s.iloc[i - 20]) if i >= 20 and not pd.isna(atr_s.iloc[i - 20]) else atr_now
        if atr_prev > 0 and atr_now / atr_prev > 1.05:
            # volatilitas belum kontraksi — longgar masih boleh jika pos tinggi
            if pos < 0.7:
                continue
        # higher low check
        half = base_w // 2
        low1 = float(window["Low"].head(half).min())
        low2 = float(window["Low"].tail(half).min())
        higher_low = low2 >= low1 * 0.98
        if not higher_low and pos < 0.7:
            continue
        entry = _round_tick(last)
        sl = _round_tick(lo * 0.985)
        if sl >= entry:
            continue
        risk = entry - sl
        if risk <= 0:
            continue
        tp = _round_tick(max(hi * 1.02, entry + risk * 1.5))
        rr = (tp - entry) / risk
        if rr < 1.3:
            continue
        # sweep low spring optional score
        has_sweep = False
        for k in range(max(0, len(sub) - 20), len(sub)):
            lk = float(sub["Low"].iloc[k])
            ck = float(sub["Close"].iloc[k])
            hk = float(sub["High"].iloc[k])
            ok = float(sub["Open"].iloc[k])
            rng = max(hk - lk, 1e-9)
            if lk < lo * 1.001 and ck > lo and (min(ok, ck) - lk) / rng >= 0.25:
                has_sweep = True
                break
        signals.append(
            {
                "ticker": str(ticker).upper().replace(".JK", "").strip(),
                "entry": entry,
                "stop_loss": sl,
                "tp1": tp,
                "tp2": _round_tick(entry + risk * 2.5),
                "atr": round(atr_now, 1) if atr_now else 0.0,
                "strategy": "Accumulation",
                "planned_rr": round(rr, 2),
                "signal_date": df.index[i].strftime("%Y-%m-%d"),
                "raw": {
                    "Range%": round(range_pct, 1),
                    "PosInRange": round(pos, 2),
                    "VolContract": round(contraction, 2),
                    "SweepLow": has_sweep,
                },
            }
        )
        last_sig = i
        if len(signals) >= max_signals:
            break
    return signals


STRATEGY_ADAPTERS = {
    "v3": {
        "label": "V3 Retest Fibo",
        "finder": lambda t, **kw: find_signals_v3(t, **kw),
        "full": True,
    },
    "v2": {
        "label": "V2 Breakout+Sweep",
        "finder": lambda t, **kw: find_signals_v2(t, **kw),
        "full": True,
    },
    "v4": {
        "label": "V4 Order Block",
        "finder": lambda t, **kw: find_signals_v4(t, **kw),
        "full": True,
    },
    "v5": {
        "label": "V5 CHOCH",
        "finder": lambda t, **kw: find_signals_v5(t, **kw),
        "full": True,
    },
    "intraday": {
        "label": "Intraday",
        "finder": lambda t, **kw: find_signals_intraday(t, **kw),
        "full": True,
    },
    "highbeta": {
        "label": "HighBeta Liquid",
        "finder": lambda t, **kw: find_signals_highbeta(t, **kw),
        "full": True,
    },
    "accumulation": {
        "label": "Accumulation",
        "finder": lambda t, **kw: find_signals_accumulation(t, **kw),
        "full": True,
    },
}


def find_historical_signals(
    ticker: str,
    strategy: str,
    *,
    lookback_days: int = 400,
    max_signals: int = 40,
) -> list[dict]:
    key = str(strategy).lower().replace(" ", "")
    if key.startswith("v"):
        key = key[:2] if key[:2] in STRATEGY_ADAPTERS else key
    # normalize keys
    for k in STRATEGY_ADAPTERS:
        if key == k or key.startswith(k):
            key = k
            break
    adapter = STRATEGY_ADAPTERS.get(key)
    if not adapter:
        raise ValueError(f"Strategy tidak dikenal: {strategy}. Pilihan: {list(STRATEGY_ADAPTERS)}")
    return adapter["finder"](ticker, lookback_days=lookback_days, max_signals=max_signals)


def backtest_ticker_strategy(
    ticker: str,
    strategy: str,
    *,
    lookback_days: int = 400,
    max_signals: int = 40,
    params: dict | None = None,
    progress: bool = True,
) -> tuple[pd.DataFrame, BacktestSummary, list[dict]]:
    """
    Phase 2: efektivitas strategi S pada emiten X.
    1) find_historical_signals
    2) simulate tiap sinyal
    """
    if progress:
        print(f"[phase2] Scan sinyal {strategy} pada {ticker} (lookback={lookback_days})...")
    signals = find_historical_signals(
        ticker, strategy, lookback_days=lookback_days, max_signals=max_signals
    )
    if progress:
        print(f"[phase2] Ditemukan {len(signals)} sinyal historis")
    if not signals:
        return pd.DataFrame(), summarize_trades(pd.DataFrame()), []

    trades, summary = run_backtest_on_setups(
        signals,
        params=params,
        report_date=None,
        progress=progress,
    )
    return trades, summary, signals


def backtest_tickers_strategy(
    tickers: list[str],
    strategy: str,
    *,
    lookback_days: int = 400,
    max_signals: int = 30,
    params: dict | None = None,
) -> tuple[pd.DataFrame, BacktestSummary]:
    """Batch phase-2 untuk beberapa ticker."""
    all_trades = []
    for t in tickers:
        tr, _, _ = backtest_ticker_strategy(
            t,
            strategy,
            lookback_days=lookback_days,
            max_signals=max_signals,
            params=params,
            progress=True,
        )
        if tr is not None and not tr.empty:
            all_trades.append(tr)
    if not all_trades:
        return pd.DataFrame(), summarize_trades(pd.DataFrame())
    df = pd.concat(all_trades, ignore_index=True)
    return df, summarize_trades(df)



# Public API (dashboard / CLI)
__all__ = [
    "DEFAULT_BT_PARAMS",
    "find_report",
    "load_screener_setups",
    "normalize_setup",
    "simulate_trade",
    "run_backtest_on_setups",
    "summarize_trades",
    "backtest_report",
    "backtest_ticker_from_report",
    "STRATEGY_ADAPTERS",
    "find_historical_signals",
    "backtest_ticker_strategy",
    "backtest_tickers_strategy",
]



if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="IDX Backtest Engine")
    ap.add_argument("--version", default="v3", help="v2|v3|v4|v5|intraday|...")
    ap.add_argument("--path", default=None, help="Path CSV report")
    ap.add_argument("--ticker", default=None, help="Filter / target ticker")
    ap.add_argument("--horizon", type=int, default=20)
    ap.add_argument("--entry-mode", default="next_open", choices=["next_open", "signal_close"])
    ap.add_argument("--no-save", action="store_true")
    ap.add_argument(
        "--mode",
        default="report",
        choices=["report", "historical"],
        help="report=setup CSV hari ini; historical=phase2 scan rule di history",
    )
    ap.add_argument("--lookback", type=int, default=400, help="Hari lookback phase2")
    args = ap.parse_args()

    params = {
        "horizon_bars": args.horizon,
        "entry_mode": args.entry_mode,
    }
    if args.mode == "historical":
        if not args.ticker:
            raise SystemExit("--ticker wajib untuk mode historical")
        trades, summary, sigs = backtest_ticker_strategy(
            args.ticker,
            args.version,
            lookback_days=args.lookback,
            params=params,
        )
        print_summary(summary)
        print(f"Sinyal historis: {len(sigs)}")
        if not args.no_save and trades is not None and not trades.empty:
            out = f"idx_backtest_hist_{args.version}_{args.ticker}_{datetime.now().strftime('%Y-%m-%d')}.csv"
            trades.to_csv(out, index=False)
            print(f"Trades: {out}")
    else:
        tickers = [args.ticker] if args.ticker else None
        backtest_report(
            args.version,
            path=args.path,
            tickers=tickers,
            params=params,
            save=not args.no_save,
        )
