"""
IDX external data dari Google Drive / URL
=========================================
Baca CSV daftar saham & fundamentals override dari URL
yang dikonfigurasi di Streamlit secrets atau environment.

secrets.toml contoh
-------------------
[gdrive]
# File harus di-share "Anyone with the link" (viewer)
fundamentals_override_url = "https://drive.google.com/file/d/FILE_ID/view?usp=sharing"
ticker_universe_url      = "https://drive.google.com/file/d/FILE_ID/view?usp=sharing"
# opsional: liquidity / watchlist
liquidity_list_url       = "https://drive.google.com/uc?export=download&id=FILE_ID"

# Cache lokal (detik). Default 3600.
cache_ttl_sec = 3600

Environment (fallback lokal / Cloud tanpa nested table)
-------------------------------------------------------
GDRIVE_FUNDAMENTALS_OVERRIDE_URL=...
GDRIVE_TICKER_UNIVERSE_URL=...
GDRIVE_LIQUIDITY_LIST_URL=...
GDRIVE_CACHE_TTL_SEC=3600
"""

from __future__ import annotations

import io
import os
import re
import time
import urllib.error
import urllib.request
from typing import Any

import pandas as pd

_CACHE: dict[str, tuple[float, bytes]] = {}


def _secrets_get(key: str, default: Any = None) -> Any:
    """Baca Streamlit secrets lalu environment."""
    # nested [gdrive] key
    try:
        import streamlit as st

        g = st.secrets.get("gdrive", None)
        if g is not None:
            try:
                if key in g:
                    return g[key]
            except Exception:
                pass
            try:
                return g.get(key, default)
            except Exception:
                pass
        # flat key
        if key in st.secrets:
            return st.secrets[key]
    except Exception:
        pass

    env_map = {
        "fundamentals_override_url": "GDRIVE_FUNDAMENTALS_OVERRIDE_URL",
        "ticker_universe_url": "GDRIVE_TICKER_UNIVERSE_URL",
        "liquidity_list_url": "GDRIVE_LIQUIDITY_LIST_URL",
        "cache_ttl_sec": "GDRIVE_CACHE_TTL_SEC",
    }
    env_k = env_map.get(key, key.upper())
    return os.environ.get(env_k, default)


def to_direct_download_url(url: str) -> str:
    """
    Ubah link Google Drive share menjadi direct download.
    Mendukung:
      - https://drive.google.com/file/d/ID/view?...
      - https://drive.google.com/open?id=ID
      - https://drive.google.com/uc?export=download&id=ID  (sudah direct)
      - URL mentah non-Drive (dikembalikan apa adanya)
    """
    if not url or not str(url).strip():
        return ""
    url = str(url).strip()

    m = re.search(r"/file/d/([a-zA-Z0-9_-]+)", url)
    if m:
        return f"https://drive.google.com/uc?export=download&id={m.group(1)}"

    m = re.search(r"[?&]id=([a-zA-Z0-9_-]+)", url)
    if m and "drive.google.com" in url:
        return f"https://drive.google.com/uc?export=download&id={m.group(1)}"

    return url


def _ttl() -> float:
    try:
        return float(_secrets_get("cache_ttl_sec", 3600) or 3600)
    except Exception:
        return 3600.0


def download_bytes(url: str, *, timeout: float = 45.0, use_cache: bool = True) -> bytes:
    direct = to_direct_download_url(url)
    if not direct:
        raise ValueError("URL kosong")

    now = time.time()
    if use_cache and direct in _CACHE:
        ts, blob = _CACHE[direct]
        if now - ts < _ttl():
            return blob

    req = urllib.request.Request(
        direct,
        headers={
            "User-Agent": "Mozilla/5.0 (compatible; IDX-Master-Screener/1.0)",
            "Accept": "text/csv,text/plain,application/octet-stream,*/*",
        },
        method="GET",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        data = resp.read()
        # Google kadang mengembalikan HTML confirm untuk file besar
        ctype = (resp.headers.get("Content-Type") or "").lower()
        if "text/html" in ctype and b"confirm=" in data[:2000]:
            # coba ekstrak link confirm sederhana
            m = re.search(rb'href="(/uc\?[^"]+confirm=[^"]+)"', data)
            if m:
                conf = "https://drive.google.com" + m.group(1).decode("utf-8").replace("&amp;", "&")
                req2 = urllib.request.Request(
                    conf,
                    headers={"User-Agent": "Mozilla/5.0 (compatible; IDX-Master-Screener/1.0)"},
                )
                with urllib.request.urlopen(req2, timeout=timeout) as resp2:
                    data = resp2.read()

    if use_cache:
        _CACHE[direct] = (now, data)
    return data


def read_csv_url(url: str, **read_csv_kwargs) -> pd.DataFrame:
    """Download URL (Drive/HTTP) lalu pd.read_csv."""
    raw = download_bytes(url)
    # strip BOM
    if raw.startswith(b"\xef\xbb\xbf"):
        raw = raw[3:]
    kwargs = {"index_col": False}
    kwargs.update(read_csv_kwargs)
    return pd.read_csv(io.BytesIO(raw), **kwargs)


def load_fundamentals_override_df() -> tuple[pd.DataFrame | None, str]:
    """
    Prioritas:
      1) URL di secrets/env
      2) File lokal fundamentals_override.csv
    Return (df|None, source_label).
    """
    url = _secrets_get("fundamentals_override_url") or ""
    if url:
        try:
            df = read_csv_url(str(url))
            if df is not None and not df.empty:
                df.columns = [str(c).strip() for c in df.columns]
                return df, f"gdrive:{to_direct_download_url(str(url))[:60]}"
        except Exception as e:
            # jangan gagal total — fallback lokal
            err = str(e)
        else:
            err = None
    else:
        err = None

    candidates = [
        "fundamentals_override.csv",
        os.path.join(os.getcwd(), "fundamentals_override.csv"),
    ]
    try:
        here = os.path.dirname(os.path.abspath(__file__))
        candidates.insert(0, os.path.join(here, "fundamentals_override.csv"))
    except Exception:
        pass
    for extra in ("/home/workdir/artifacts", "/home/workdir"):
        candidates.append(os.path.join(extra, "fundamentals_override.csv"))

    for p in candidates:
        if os.path.isfile(p):
            try:
                df = pd.read_csv(p, index_col=False)
                df.columns = [str(c).strip() for c in df.columns]
                return df, f"local:{p}"
            except Exception:
                continue

    if err:
        return None, f"error:{err[:120]}"
    return None, "none"


def load_ticker_list_from_url(url: str | None = None) -> list[str]:
    """
    Baca daftar ticker dari CSV Drive/URL.
    Kolom yang dikenali (case-insensitive):
      Ticker, Symbol, Code, Kode, Emiten, Saham
    Jika tidak ketemu → pakai kolom pertama.
    """
    url = url or _secrets_get("ticker_universe_url") or _secrets_get("liquidity_list_url")
    if not url:
        return []

    df = read_csv_url(str(url))
    if df is None or df.empty:
        return []

    df.columns = [str(c).strip() for c in df.columns]
    colmap = {c.lower(): c for c in df.columns}
    tcol = (
        colmap.get("ticker")
        or colmap.get("symbol")
        or colmap.get("code")
        or colmap.get("kode")  # format lama IDX
        or colmap.get("emiten")
        or colmap.get("saham")
    )
    if tcol:
        series = df[tcol]
    else:
        series = df.iloc[:, 0]

    out: list[str] = []
    seen: set[str] = set()
    skip_headers = {
        "TICKER", "SYMBOL", "CODE", "KODE", "EMITEN", "SAHAM", "NAN", "NONE",
    }
    for v in series.astype(str):
        t = v.strip().upper().replace(".JK", "")
        if not t or t in skip_headers:
            continue
        if t not in seen:
            seen.add(t)
            out.append(t)
    return out


def get_ticker_universe(fallback: list[str] | None = None) -> list[str]:
    """Universe dari Drive jika ada, else fallback."""
    try:
        remote = load_ticker_list_from_url()
        if remote:
            return remote
    except Exception:
        pass
    return list(fallback or [])


def clear_cache() -> None:
    _CACHE.clear()


def resolve_screener_universe(fallback_fn=None) -> list:
    """
    Universe final untuk screener.

    Google Drive hanya mengisi *kandidat* di dalam IdxLiquidityScanner
    (menggantikan hardcode), lalu tetap difilter likuiditas.

    Jika fallback_fn diberikan (biasanya scanner.get_liquid_universe),
    panggil itu — jangan return daftar Drive mentah.
    """
    if callable(fallback_fn):
        try:
            fb = fallback_fn()
            if fb:
                print(f"[universe] liquid: {len(fb)} ticker")
                return list(fb)
        except Exception as e:
            print(f"[universe] scanner error: {e}")
        return []
    # Tanpa scanner: Drive mentah (hanya debug / kasus khusus)
    try:
        remote = get_ticker_universe(fallback=[])
        if remote:
            print(f"[universe] Google Drive mentah (tanpa filter likuiditas): {len(remote)}")
            return list(remote)
    except Exception as e:
        print(f"[universe] GDrive skip: {e}")
    return []
