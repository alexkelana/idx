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
    OPENAI_MAX_RETRIES (default 2)
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
    "technical": 1000,
    "fundamental": 1000,
    "risk": 900,
    "critic": 1100,
    "chief": 1600,
}

SYSTEM_PROMPTS = {
    "technical": """Kamu adalah analis teknikal pasar saham Indonesia (IDX).
Hanya gunakan data JSON yang diberikan. Jangan mengarang harga atau indikator.
Fokus: kualitas setup (entry, SL, TP, RR), struktur, risiko false breakout/sweep,
apakah level masuk akal untuk karakter BEI (tick, volatilitas).
Rujuk angka di screener_row dan price_snapshot secara eksplisit.
Jika ada llmquant.quant_wiki, boleh pakai sebagai kerangka istilah (OB, FVG, liquidity) tanpa mengarang level harga.
Jawab dalam Bahasa Indonesia, terstruktur, poin-poin. Bukan saran investasi.""",
    "fundamental": """Kamu adalah analis fundamental saham Indonesia (IDX).
Hanya gunakan field "fundamental", "news_intel", "llmquant", dan "screener_row" pada JSON.
Jika data terbatas/null, katakan secara eksplisit — jangan mengarang.
Fokus:
- Valuasi kasar (PE, PB, EPS, market cap) vs konteks sektor bila ada
- Undervalued / fair / premium secara kualitatif dari angka yang ada
- Risiko bisnis singkat (sektor, profitabilitas bila ada)
- Ringkas berita/headline relevan di news_intel (katalis positif/negatif, corporate action)
- Jika ada llmquant.quant_wiki / quant_papers: pakai hanya sebagai kerangka konsep quant (bukan fakta harga emiten)
- Apakah fundamental + berita mendukung ATAU bertentangan dengan setup teknikal jangka pendek
Jawab Bahasa Indonesia, poin-poin. Bukan saran investasi. Horizon: swing pendek–menengah.""",
    "risk": """Kamu adalah risk manager trading IDX.
Hanya gunakan data JSON yang diberikan.
Fokus: posisi size (lots), risiko vs account, jarak SL, dampak fee beli/jual +
perkiraan pajak Indonesia (PPN atas fee, levy, PPh final jual 0.1%) secara kualitatif,
likuiditas jika ada di data.
Jawab Bahasa Indonesia, singkat, poin. Bukan saran investasi.""",
    "critic": """Kamu adalah devil's advocate / risk skeptic.
Tugas: cari alasan setup ini GAGAL atau sebaiknya dihindari.
Gabungkan celah teknikal, fundamental, berita (news_intel), dan konteks makro/konsep di llmquant bila ada:
valuasi mahal, data kosong, sektor lemah, headline negatif, makro global tidak mendukung risk-on.
Sebut invalidation, skenario worst-case, dan red flags dari data JSON saja.
Jangan mengarang berita di luar news_intel. Jangan memuji setup.
Bahasa Indonesia, poin. Bukan saran investasi.""",
    "chief": """Kamu adalah head trader assistant yang merangkum analisa
technical, fundamental, risk, critic, berita (news_intel), dan llmquant (makro/wiki quant) bila ada.
Dari JSON gabungan tersebut, buat:
1) Skor setup 1-10 (teknikal+risk dasar; fundamental/berita/makro mendukung = bonus, bertentangan = penalti)
2) Bias: avoid | watch | consider  (jangan bilang "wajib beli/jual")
3) Tiga syarat sebelum entry
4) Tiga red flags (fundamental, berita, atau makro global bila relevan)
5) Catatan fundamental + berita + makro 2-3 kalimat
6) Satu kalimat kesimpulan netral
Hanya berdasarkan data yang ada; jangan mengarang headline.
Bahasa Indonesia. Bukan financial advice.
Format output jelas dengan heading singkat.""",
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


def fetch_fundamental(ticker: str) -> dict[str, Any]:
    """Ambil fundamental: idx_fundamental (jika ada) lalu yfinance."""
    ticker = str(ticker).upper().strip()

    try:
        from idx_fundamental import get_fundamental_snapshot  # type: ignore

        snap = get_fundamental_snapshot(ticker)
        if isinstance(snap, dict) and snap:
            snap = dict(snap)
            snap.setdefault("source", "idx_fundamental")
            return snap
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

        out.update(
            {
                "available": bool(info),
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
    return out




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

    return {
        "ticker": ticker,
        "headlines": public_items,
        "headline_count": len(public_items),
        "sources": sources_ok,
        "tone_hint": tone,
        "tone_counts": {"positive_kw": pos, "negative_kw": neg},
        "verify_search_url": _fallback_search_url(ticker),
        "fetched_at": fetched_at,
        "hours_lookback": hours_lookback,
        "realtime": True,
        "errors": errors,
        "disclaimer": (
            "Berita near real-time dari agregator (Yahoo/Google News/media ID). "
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
    if include_fundamental:
        fund = fetch_fundamental(ticker)
        if csv_fund:
            fund["from_screener_csv"] = csv_fund
        ctx["fundamental"] = fund
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
    max_retries = int(os.environ.get("OPENAI_MAX_RETRIES", "2"))

    import urllib.request
    import urllib.error

    payload = json.dumps(
        {
            "model": model,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        }
    ).encode("utf-8")

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
            if he.code in (429, 500, 502, 503, 504) and attempt < attempts:
                time.sleep(min(2 ** attempt, 8))
                continue
            raise last_err from he
        except Exception as e:
            last_err = e
            if attempt < attempts and _is_retryable_error(e):
                time.sleep(min(2 ** attempt, 8))
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
        return "\n".join(
            [
                f"[Offline] Critic — {ticker}",
                "- Data screener bisa basi; fundamental Yahoo bisa tidak lengkap.",
                "- Rezim tidak align / SL terlalu lebar = red flag.",
            ]
        )
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
    user_payload = json.dumps(context, ensure_ascii=False, indent=2)
    max_tokens = ROLE_MAX_TOKENS.get(role, 900)
    return _chat(
        SYSTEM_PROMPTS[role],
        user_payload,
        model=model,
        max_tokens=max_tokens,
    )


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

    for role in [r for r in roles if r != "chief"]:
        try:
            analyses[role] = run_role(role, context, model=model)
        except Exception as e:
            errors[role] = _format_llm_error(e)
            analyses[role] = f"[Gagal] {errors[role]}"

    if "chief" in roles:
        ok = {k: v for k, v in analyses.items() if not str(v).startswith("[Gagal]")}
        if not ok:
            analyses["chief"] = (
                "[Gagal] Tidak ada analisa role yang berhasil; chief dilewati. "
                + ("; ".join(f"{k}: {v}" for k, v in errors.items()) or "")
            )
            errors["chief"] = analyses["chief"]
        else:
            try:
                analyses["chief"] = run_role(
                    "chief", {**context, "analyses": ok}, model=model
                )
            except Exception as e:
                errors["chief"] = _format_llm_error(e)
                analyses["chief"] = f"[Gagal] {errors['chief']}"

    return {
        "ticker": context.get("ticker"),
        "version": context.get("screener_version"),
        "offline": offline,
        "analyses": analyses,
        "errors": errors,
        "partial": bool(errors),
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


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(description="IDX AI Trader Assistant")
    p.add_argument("ticker", help="Kode emiten")
    p.add_argument("--version", default="v2")
    p.add_argument("--model", default=None)
    args = p.parse_args()

    out = analyze_ticker(args.ticker, args.version, model=args.model)
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
