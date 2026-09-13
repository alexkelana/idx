"""
IDX × LLMQuant Data — pelengkap pengetahuan quant + makro global
==============================================================
Tidak menggantikan data OHLCV emiten .JK (tetap yfinance).
Digunakan untuk:
  - context AI Assistant (wiki quant + makro)
  - narasi pasar di dashboard (makro global)

Path REST (dari client resmi @llmquant/data-mcp):
  GET /api/macro/snapshot?indicator=...
  GET /api/macro/indicators
  GET /api/wiki/search?query=...
  GET /api/paper/search?query=...

Env:
  LLMQUANT_API_KEY   (wajib untuk live call)
  LLMQUANT_BASE_URL  (default https://api.llmquantdata.com)
  LLMQUANT_TIMEOUT   (default 20 detik)
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime
from typing import Any


def _base_url() -> str:
    return (os.environ.get("LLMQUANT_BASE_URL") or "https://api.llmquantdata.com").rstrip("/")


def _api_key() -> str | None:
    return os.environ.get("LLMQUANT_API_KEY") or os.environ.get("LLMQUANT_KEY")


def available() -> bool:
    return bool(_api_key())


def _timeout() -> float:
    try:
        return float(os.environ.get("LLMQUANT_TIMEOUT", "20"))
    except Exception:
        return 20.0


def _short_err(msg: str, limit: int = 180) -> str:
    """Hindari dump HTML Next.js di UI."""
    s = (msg or "").replace("\n", " ").strip()
    if "<html" in s.lower() or "<!doctype" in s.lower():
        # ambil hanya status/path jika ada
        if "HTTP " in s:
            return s.split(":", 1)[0][:limit] + " (HTML 404 — path/indicator salah)"
        return "HTTP error (HTML body, bukan JSON API)"
    return s[:limit]


def _get(path: str, params: dict | None = None) -> dict[str, Any]:
    """GET JSON dari LLMQuant REST. Raise RuntimeError jika gagal."""
    key = _api_key()
    if not key:
        raise RuntimeError("LLMQUANT_API_KEY belum di-set")

    q = urllib.parse.urlencode({k: v for k, v in (params or {}).items() if v is not None})
    url = f"{_base_url()}{path}"
    if q:
        url = f"{url}?{q}"

    req = urllib.request.Request(
        url,
        headers={
            "Authorization": f"Bearer {key}",
            "Accept": "application/json",
            "User-Agent": "IDX-Master-Screener/1.1",
        },
        method="GET",
    )
    try:
        with urllib.request.urlopen(req, timeout=_timeout()) as r:
            body = r.read().decode("utf-8")
        if not body:
            return {}
        try:
            return json.loads(body)
        except json.JSONDecodeError as e:
            raise RuntimeError(f"{path}: response bukan JSON ({_short_err(body)})") from e
    except urllib.error.HTTPError as e:
        err = e.read().decode("utf-8", errors="ignore")
        raise RuntimeError(f"HTTP {e.code} {path}: {_short_err(err)}") from e
    except Exception as e:
        raise RuntimeError(f"{path}: {e}") from e


# Alias resmi dari catalog MCP (bukan nama tebakan)
# Contoh di tool description: us.cpi.headline, us.unemployment_rate,
# us.rates.fed_funds, us.yield.10y, us.pce.core, us.gdp.real
DEFAULT_MACRO_INDICATORS = [
    "us.rates.fed_funds",
    "us.unemployment_rate",
    "us.cpi.headline",
    "us.yield.10y",
    "us.pce.core",
]

# Fallback series_id FRED jika alias gagal
INDICATOR_SERIES_FALLBACK = {
    "us.rates.fed_funds": "FEDFUNDS",
    "us.unemployment_rate": "UNRATE",
    "us.cpi.headline": "CPIAUCSL",
    "us.yield.10y": "DGS10",
    "us.pce.core": "PCEPILFE",
    "us.gdp.real": "GDPC1",
}


def _parse_macro_payload(data: dict, requested: str) -> dict[str, Any]:
    payload = data.get("data") if isinstance(data.get("data"), dict) else data
    if not isinstance(payload, dict):
        payload = {}
    latest = payload.get("latest") or {}
    previous = payload.get("previous") or {}
    if not isinstance(latest, dict):
        latest = {}
    if not isinstance(previous, dict):
        previous = {}
    return {
        "indicator": payload.get("indicator") or requested,
        "series_id": payload.get("series_id") or payload.get("seriesId"),
        "title": payload.get("title") or requested,
        "units": payload.get("units"),
        "frequency": payload.get("frequency"),
        "latest_value": latest.get("value") if latest else payload.get("value"),
        "latest_date": latest.get("date") if latest else payload.get("date"),
        "previous_value": previous.get("value"),
        "delta_abs": payload.get("delta_abs") if "delta_abs" in payload else payload.get("deltaAbs"),
        "delta_pct": payload.get("delta_pct") if "delta_pct" in payload else payload.get("deltaPct"),
        "attribution": payload.get("attribution"),
    }


def fetch_macro_snapshot(
    indicators: list[str] | None = None,
) -> dict[str, Any]:
    """
    Ambil snapshot makro global via GET /api/macro/snapshot.
    """
    indicators = indicators or DEFAULT_MACRO_INDICATORS
    out: dict[str, Any] = {
        "available": available(),
        "fetched_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "items": [],
        "errors": [],
        "source": "llmquant",
        "disclaimer": (
            "Makro global (terutama AS via FRED/LLMQuant). Bukan data BEI. "
            "Untuk konteks risk-on/off saja — bukan sinyal beli/jual emiten IDX."
        ),
    }
    if not available():
        out["errors"].append("LLMQUANT_API_KEY belum di-set")
        return out

    path = "/api/macro/snapshot"
    for ind in indicators:
        item = None
        last_err = None
        # 1) alias indicator
        try:
            data = _get(path, {"indicator": ind})
            item = _parse_macro_payload(data, ind)
            item["query"] = f"indicator={ind}"
        except Exception as e:
            last_err = str(e)
            # 2) series_id FRED
            sid = INDICATOR_SERIES_FALLBACK.get(ind)
            if sid:
                try:
                    data = _get(path, {"series_id": sid})
                    item = _parse_macro_payload(data, ind)
                    item["query"] = f"series_id={sid}"
                    last_err = None
                except Exception as e2:
                    last_err = f"{last_err}; series_id={sid}: {e2}"

        if item and item.get("latest_value") is not None:
            out["items"].append(item)
        else:
            out["errors"].append(f"{ind}: {_short_err(last_err or 'no data')}")

    out["narrative_hint"] = _macro_narrative_hint(out["items"])
    return out


def _macro_narrative_hint(items: list[dict]) -> str:
    if not items:
        return "Data makro LLMQuant tidak tersedia."
    bits = []
    for it in items:
        title = it.get("title") or it.get("indicator")
        lv = it.get("latest_value")
        d = it.get("delta_abs")
        if lv is None:
            continue
        if d is None:
            bits.append(f"{title}={lv}")
        else:
            try:
                df = float(d)
                arrow = "↑" if df > 0 else ("↓" if df < 0 else "→")
            except Exception:
                arrow = "→"
            bits.append(f"{title}={lv} ({arrow}{d})")
    return "Makro global: " + "; ".join(bits[:6]) if bits else "Makro: nilai terbatas."


def search_quant_wiki(query: str, limit: int = 5) -> dict[str, Any]:
    """Semantic search Quant Wiki — GET /api/wiki/search"""
    out: dict[str, Any] = {
        "available": available(),
        "query": query,
        "fetched_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "items": [],
        "errors": [],
        "source": "llmquant",
    }
    if not available():
        out["errors"].append("LLMQUANT_API_KEY belum di-set")
        return out
    if not query or not str(query).strip():
        out["errors"].append("query kosong")
        return out

    try:
        data = _get(
            "/api/wiki/search",
            {"query": query.strip(), "limit": str(limit)},
        )
    except Exception as e:
        out["errors"].append(_short_err(str(e)))
        return out

    rows = data.get("data") if isinstance(data, dict) else None
    if not isinstance(rows, list):
        rows = data if isinstance(data, list) else []

    for row in rows[:limit]:
        if not isinstance(row, dict):
            continue
        scores = row.get("scores") if isinstance(row.get("scores"), dict) else {}
        out["items"].append(
            {
                "id": row.get("wikiItemId") or row.get("id") or row.get("slug"),
                "title": row.get("title"),
                "summary": (row.get("summary") or "")[:500],
                "tags": row.get("tags") or [],
                "score": scores.get("combined") or row.get("score"),
            }
        )
    return out


def search_quant_papers(query: str, limit: int = 3) -> dict[str, Any]:
    """Semantic search papers — GET /api/paper/search"""
    out: dict[str, Any] = {
        "available": available(),
        "query": query,
        "fetched_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "items": [],
        "errors": [],
        "source": "llmquant",
    }
    if not available():
        out["errors"].append("LLMQUANT_API_KEY belum di-set")
        return out

    try:
        data = _get(
            "/api/paper/search",
            {"query": query.strip(), "limit": str(limit)},
        )
    except Exception as e:
        out["errors"].append(_short_err(str(e)))
        return out

    rows = data.get("data") if isinstance(data, dict) else None
    if not isinstance(rows, list):
        rows = data if isinstance(data, list) else []

    for row in rows[:limit]:
        if not isinstance(row, dict):
            continue
        out["items"].append(
            {
                "id": row.get("paperId") or row.get("id") or row.get("slug"),
                "title": row.get("title"),
                "summary": (row.get("summary") or row.get("abstract") or "")[:500],
            }
        )
    return out


def build_llmquant_context_for_ticker(
    ticker: str,
    *,
    strategy_hint: str | None = None,
    include_macro: bool = True,
    include_wiki: bool = True,
) -> dict[str, Any]:
    """Paket context untuk AI Assistant."""
    ticker = str(ticker).upper().strip()
    ctx: dict[str, Any] = {
        "provider": "llmquant",
        "ticker_note": (
            f"{ticker} adalah emiten IDX; data harga/fundamental lokal tetap dari screener/yfinance. "
            "Blok ini hanya pengetahuan quant generik + makro global."
        ),
        "fetched_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "available": available(),
    }
    if not available():
        ctx["skipped"] = "LLMQUANT_API_KEY tidak ada — blok dilompati"
        return ctx

    if include_macro:
        ctx["macro"] = fetch_macro_snapshot()

    if include_wiki:
        q = strategy_hint or (
            "order block fair value gap liquidity sweep breakout "
            "change of character risk management position sizing"
        )
        ctx["quant_wiki"] = search_quant_wiki(q, limit=4)
        ctx["quant_papers"] = search_quant_papers(
            "momentum breakout mean reversion liquidity risk", limit=2
        )

    return ctx


def build_market_macro_pack() -> dict[str, Any]:
    """Untuk tab Kondisi Pasar — fokus makro global."""
    pack = fetch_macro_snapshot()
    pack["role"] = "market_narrative"
    return pack


def _heuristic_macro_ihsg_conclusion(macro: dict, regime: dict | None = None) -> str:
    """Fallback tanpa LLM: narasi korelasi generik makro AS ↔ IHSG."""
    items = (macro or {}).get("items") or []
    regime = regime or {}
    ihsg_reg = regime.get("regime") or "belum dicek"
    activity = regime.get("activity") or "-"
    last = regime.get("last_close")

    lines = [
        "**Kesimpulan (heuristik, tanpa LLM)**",
        (
            f"- Rezim IHSG saat ini: **{ihsg_reg}** · aktivitas: **{activity}**"
            + (f" · level ≈ {last}" if last else "")
        ),
        "",
        "Korelasi tipikal makro global → IHSG (bukan jaminan):",
        "- **Suku bunga AS naik / yield naik** → cenderung menekan risk asset emerging (termasuk IHSG) lewat outflow & penguatan USD.",
        "- **Suku bunga AS turun / pause dovish** → mendukung risk-on; IHSG sering lebih responsif jika likuiditas domestik ikut longgar.",
        "- **Inflasi AS (CPI/PCE) tinggi** → ekspektasi suku bunga ketat → bias hati-hati pada IHSG cyclical.",
        "- **Pengangguran AS naik tajam** → sinyal pelemahan global → sektor ekspor/komoditas BEI bisa tertekan, defensif relatif lebih aman.",
        "",
    ]

    if items:
        lines.append("Bacaan data terbaru:")
        for it in items:
            title = it.get("title") or it.get("indicator")
            lv = it.get("latest_value")
            d = it.get("delta_abs")
            lines.append(f"- {title}: {lv}" + (f" (Δ {d})" if d is not None else ""))
        lines.append("")

    bias = "netral / campuran"
    for it in items:
        ind = str(it.get("indicator") or "").lower()
        d = it.get("delta_abs")
        try:
            df = float(d) if d is not None else None
        except Exception:
            df = None
        if df is None:
            continue
        if "fed_funds" in ind or "yield" in ind or "10y" in ind:
            if df > 0:
                bias = "hati-hati (tekanan yield/suku bunga naik)"
            elif df < 0:
                bias = "sedikit mendukung risk-on (yield/suku bunga turun)"
        if "unemployment" in ind and df > 0.1:
            bias = "waspada pelemahan global (pengangguran naik)"

    lines.append(f"**Implikasi singkat untuk IHSG:** bias **{bias}**.")
    lines.append(
        "Tetap utamakan struktur harga IHSG & likuiditas domestik; makro AS hanya konteks, "
        "bukan trigger entry emiten individual."
    )
    lines.append("_Bukan saran investasi._")
    return "\n".join(lines)


def conclude_macro_vs_ihsg(
    macro: dict | None = None,
    regime: dict | None = None,
    *,
    use_llm: bool = True,
    model: str | None = None,
) -> dict[str, Any]:
    """
    Kesimpulan korelasi makro global (LLMQuant) terhadap pergerakan/rezim IHSG.
    Menggunakan LLM bila OPENAI/XAI key ada; else heuristik.
    """
    macro = macro or fetch_macro_snapshot()
    regime = regime or {}
    result: dict[str, Any] = {
        "fetched_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "mode": "heuristic",
        "conclusion": "",
        "error": None,
    }

    payload = {
        "macro_items": macro.get("items") or [],
        "macro_narrative": macro.get("narrative_hint"),
        "macro_errors": macro.get("errors") or [],
        "ihsg_regime": regime.get("regime"),
        "ihsg_activity": regime.get("activity"),
        "ihsg_last_close": regime.get("last_close"),
        "ihsg_reason": regime.get("reason") or [],
        "ihsg_strategies": regime.get("strategies") or [],
        "vol_ratio_20": regime.get("vol_ratio_20"),
        "range_20_pct": regime.get("range_20_pct"),
    }

    if use_llm:
        try:
            import idx_ai_assistant as ai

            if ai.llm_available():
                system = (
                    "Anda analis makro-pasar Indonesia. "
                    "Tugas: simpulkan bagaimana kondisi makro global (terutama AS) "
                    "berkorelasi dengan prospek pergerakan IHSG dalam beberapa hari–minggu ke depan. "
                    "Gunakan HANYA data JSON yang diberikan. Jangan mengarang angka. "
                    "Jelaskan mekanisme transmisi singkat (suku bunga/USD/risk appetite/komoditas). "
                    "Sebutkan skenario mendukung vs menekan IHSG. "
                    "Akui ketidakpastian dan bahwa korelasi historis bisa putus. "
                    "Bahasa Indonesia, terstruktur, 6–12 kalimat. Bukan saran investasi."
                )
                user = (
                    "Data makro + rezim IHSG (JSON):\n"
                    + json.dumps(payload, ensure_ascii=False, default=str)[:6000]
                    + "\n\nTulis kesimpulan dengan heading: "
                    "1) Bacaan makro 2) Transmisi ke IHSG 3) Bias arah 4) Yang perlu diawasi."
                )
                text_out = ai._chat(
                    system, user, model=model, temperature=0.25, max_tokens=700
                )
                result["mode"] = "llm"
                result["conclusion"] = text_out.strip()
                return result
        except Exception as e:
            result["error"] = str(e)

    result["conclusion"] = _heuristic_macro_ihsg_conclusion(macro, regime)
    if result.get("error"):
        result["mode"] = "heuristic_fallback"
    return result
