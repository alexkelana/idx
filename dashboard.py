"""
IDX Master Screener AI — Streamlit Dashboard
- Panel rezim IHSG (on-demand)
- Jalankan AI adaptive / V2–V5 / Intraday / HighBeta / Accumulation / Confluence
- Tab hasil per strategi
- Komisi + pajak (enrich lewat master / fallback)
- Status run terakhir
- Download CSV + freeze Ticker
- AI Trader Assistant (auto-load .env + Streamlit secrets)
"""

import os
import glob
import io
import sys
import importlib
from datetime import datetime
from pathlib import Path

import pandas as pd


def _load_local_env() -> None:
    """Muat .env lokal ke os.environ (tidak override env yang sudah ada)."""
    candidates = [
        Path.cwd() / ".env",
        Path(__file__).resolve().parent / ".env",
        Path(__file__).resolve().parent.parent / ".env",
    ]
    try:
        from dotenv import load_dotenv

        for p in candidates:
            if p.is_file():
                load_dotenv(p, override=False)
        return
    except ImportError:
        pass

    for p in candidates:
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
        except Exception:
            continue


_load_local_env()

import streamlit as st

# Inject Streamlit secrets → env (Cloud / secrets.toml lokal)
def _inject_streamlit_secrets() -> None:
    try:
        secrets = st.secrets
    except Exception:
        return
    for key in (
        "OPENAI_API_KEY",
        "XAI_API_KEY",
        "LLM_API_KEY",
        "OPENAI_BASE_URL",
        "XAI_BASE_URL",
        "OPENAI_MODEL",
        "XAI_MODEL",
        "LLMQUANT_API_KEY",
        "LLMQUANT_BASE_URL",
    ):
        try:
            if key in secrets and key not in os.environ:
                os.environ[key] = str(secrets[key])
        except Exception:
            continue


_inject_streamlit_secrets()

st.set_page_config(
    page_title="IDX Master Screener Dashboard",
    page_icon="📈",
    layout="wide",
)

st.title("📈 IDX Master Screener AI Dashboard")
st.markdown(
    "Analisa IHSG + screener Breakout, Retest, OB, CHOCH, Intraday, "
    "HighBeta, Accumulation, Confluence + **AI Trader Assistant**."
)

# Semua kunci report yang dikenali dashboard


def _colorize_bias_text(md: str) -> str:
    """Warnai kata Bullish/Bearish/Netral di teks analisa (HTML untuk Streamlit)."""
    if not md:
        return md
    import re as _re

    s = str(md)

    def repl_bull(m):
        return f'<span style="color:#16a34a;font-weight:600">{m.group(0)}</span>'

    def repl_bear(m):
        return f'<span style="color:#dc2626;font-weight:600">{m.group(0)}</span>'

    def repl_net(m):
        return f'<span style="color:#ca8a04;font-weight:600">{m.group(0)}</span>'

    s = _re.sub(r"(?i)\b(sedikit\s+bullish|bullish)\b", repl_bull, s)
    s = _re.sub(r"(?i)\b(sedikit\s+bearish|bearish)\b", repl_bear, s)
    s = _re.sub(r"(?i)\b(netral)\b", repl_net, s)
    return s


def _md_bias(md: str) -> None:
    """Render markdown dengan highlight bias bullish/bearish."""
    try:
        st.markdown(_colorize_bias_text(md), unsafe_allow_html=True)
    except Exception:
        st.markdown(md)


REPORT_VERSIONS = [
    "v2",
    "v3",
    "v4",
    "v5",
    "intraday",
    "highbeta",
    "accumulation",
    "confluence",
]

# =====================================================================
# SIDEBAR
# =====================================================================
st.sidebar.header("⚙️ Modal & Risiko")
account_size = st.sidebar.number_input(
    "Total Modal (Rp)",
    min_value=1_000_000,
    value=4_000_000,
    step=1_000_000,
    format="%d",
)
risk_pct = st.sidebar.number_input(
    "Risiko per posisi (%)",
    min_value=0.1,
    max_value=10.0,
    value=1.0,
    step=0.1,
)

st.sidebar.markdown("---")
st.sidebar.header("💸 Komisi & Pajak")
broker_buy = st.sidebar.number_input(
    "Fee Beli broker (%)",
    min_value=0.0,
    max_value=1.0,
    value=0.15,
    step=0.01,
    help="Komisi beli murni (belum termasuk PPN & levy)",
)
broker_sell = st.sidebar.number_input(
    "Fee Jual broker (%)",
    min_value=0.0,
    max_value=1.0,
    value=0.25,
    step=0.01,
    help="Komisi jual murni (PPh Final 0,1% dihitung terpisah)",
)
st.sidebar.caption(
    "Otomatis ditambah: PPN 12% atas fee · Levy ~0,043% · "
    "PPh Final 0,1% (hanya jual)"
)

st.sidebar.markdown("---")
st.sidebar.header("🚀 Jalankan Screener")
mode = st.sidebar.radio(
    "Mode",
    [
        "🤖 AI Adaptive",
        "V2 — Breakout",
        "V3 — Retest Fibo",
        "V4 — Order Block",
        "V5 — CHOCH",
        "⚡ Intraday — Confluence",
        "🔥 HighBeta — Spekulatif Likuid",
        "📦 Accumulation — Late Base",
        "🎯 Confluence — Top Overlap",
    ],
)

if mode.startswith("🎯"):
    st.sidebar.caption(
        "Menjalankan semua screener lalu mengambil Top 5 ticker "
        "yang muncul di ≥2 strategi. Memakan waktu lebih lama."
    )
    skip_cf = st.sidebar.checkbox("Hanya baca CSV existing (skip run)", value=False)
else:
    skip_cf = False

run_button = st.sidebar.button("▶️ JALANKAN", width="stretch", type="primary")


# =====================================================================
# HELPERS — RUN SCREENER
# =====================================================================
def capture_run(fn, *args, **kwargs):
    old = sys.stdout
    buf = io.StringIO()
    sys.stdout = buf
    ok, err = False, None
    try:
        fn(*args, **kwargs)
        ok = True
    except Exception as e:
        err = str(e)
    finally:
        sys.stdout = old
    return ok, err, buf.getvalue()


def run_v2(p):
    m = importlib.import_module("idx_breakout_screener_v2")
    if hasattr(m, "PARAMS") and isinstance(m.PARAMS, dict):
        m.PARAMS.update(p)
    m.run_screener(params=getattr(m, "PARAMS", p))


def run_v3(p):
    m = importlib.import_module("idx_breakout_screener_v3")
    if hasattr(m, "PARAMS") and isinstance(m.PARAMS, dict):
        m.PARAMS.update(p)
    m.run_screener(params=getattr(m, "PARAMS", p))


def run_v4(p):
    m = importlib.import_module("idx_breakout_screener_v4_smc")
    m.run_screener_v4(user_params=p)


def run_v5(p):
    m = importlib.import_module("idx_breakout_screener_v5_smc")
    m.run_screener_v5(user_params=p)


def run_ai(account_size, risk_pct, broker_buy_pct, broker_sell_pct):
    import master_screener_ai

    master_screener_ai.run_orchestrator(
        account_size=float(account_size),
        risk_pct=float(risk_pct),
        broker_buy_pct=float(broker_buy_pct),
        broker_sell_pct=float(broker_sell_pct),
    )


def run_intraday(p):
    m = importlib.import_module("idx_intraday_screener")
    m.run_intraday_screener(user_params=p)


def run_highbeta(p):
    m = importlib.import_module("idx_highbeta_screener")
    m.run_highbeta_screener(user_params=p)


def run_accumulation(p):
    m = importlib.import_module("idx_accumulation_screener")
    m.run_accumulation_screener(user_params=p)


def run_confluence(p, skip_run=False):
    m = importlib.import_module("idx_confluence_runner")
    m.run_confluence(params=p, top_n=5, skip_run=skip_run)


def enrich_all_version_csvs(broker_buy_pct, broker_sell_pct):
    """Enrich report hari ini: trailing + fundamental + biaya/pajak."""
    try:
        import master_screener_ai

        if hasattr(master_screener_ai, "apply_enrichment_to_latest_reports"):
            master_screener_ai.apply_enrichment_to_latest_reports(
                broker_buy_pct=float(broker_buy_pct),
                broker_sell_pct=float(broker_sell_pct),
            )
            return
    except Exception:
        pass

    try:
        from idx_cost_tax import enrich_dataframe_with_costs
    except ImportError:
        return

    today = datetime.now().strftime("%Y-%m-%d")
    # confluence = agregat ranking; skip enrich biaya per-setup
    for ver in ["v2", "v3", "v4", "v5", "intraday", "highbeta", "accumulation"]:
        path = None
        for d in _search_dirs():
            cand = os.path.join(d, f"idx_report_{ver}_{today}.csv")
            if os.path.exists(cand):
                path = cand
                break
        if not path:
            continue
        try:
            df = pd.read_csv(path)
            if df.empty:
                continue
            try:
                from idx_trailing_stop import enrich_with_trailing_stop

                df = enrich_with_trailing_stop(df)
            except Exception:
                pass
            try:
                from idx_fundamental import enrich_with_fundamental

                df = enrich_with_fundamental(df)
            except Exception:
                pass
            df = enrich_dataframe_with_costs(df, broker_buy_pct, broker_sell_pct)
            df.to_csv(path, index=False)
        except Exception:
            continue


# =====================================================================
# HELPERS — FILE & TABEL
# =====================================================================
def _search_dirs():
    dirs = ["."]
    try:
        here = os.path.dirname(os.path.abspath(__file__))
        if here and here not in dirs:
            dirs.append(here)
    except Exception:
        pass
    for extra in [
        "/home/workdir",
        "/home/workdir/artifacts",
        "/home/workdir/attachments",
    ]:
        if os.path.isdir(extra) and extra not in dirs:
            dirs.append(extra)
    return dirs


def find_report(version: str):
    files = []
    for d in _search_dirs():
        files.extend(glob.glob(os.path.join(d, f"idx_report_{version}_*.csv")))
    if not files:
        return None
    return max(files, key=os.path.getmtime)


def count_report_rows(version: str) -> int:
    path = find_report(version)
    if not path or not os.path.exists(path):
        return 0
    try:
        return len(pd.read_csv(path))
    except Exception:
        return 0


def stockbit_url(ticker: str) -> str:
    """https://stockbit.com/symbol/BBRI"""
    t = str(ticker or "").upper().replace(".JK", "").strip()
    if not t or t in ("NAN", "NONE", "-"):
        return ""
    return f"https://stockbit.com/symbol/{t}"


def show_report(version: str, title: str):
    path = find_report(version)
    if not path:
        st.info(
            f"Belum ada data **{title}**. "
            f"Jalankan screener terkait atau **AI Adaptive** / **Confluence**."
        )
        return

    mtime = datetime.fromtimestamp(os.path.getmtime(path))
    st.caption(f"`{os.path.basename(path)}` • {mtime.strftime('%Y-%m-%d %H:%M')}")

    try:
        df = pd.read_csv(path)
        if df.empty:
            st.warning("File kosong.")
            return

        with open(path, "rb") as f:
            st.download_button(
                label=f"⬇️ Download {os.path.basename(path)}",
                data=f.read(),
                file_name=os.path.basename(path),
                mime="text/csv",
                key=f"dl_{version}",
            )

        view = df.copy()
        if "Ticker" in view.columns:
            view = view.drop_duplicates(subset=["Ticker"], keep="first")
            view["Ticker"] = (
                view["Ticker"].astype(str).str.upper().str.replace(".JK", "", regex=False).str.strip()
            )
            # Kolom hyperlink Stockbit (klikable di tabel)
            view.insert(
                0,
                "Stockbit",
                view["Ticker"].map(stockbit_url),
            )
            view = view.set_index("Ticker")

            st.dataframe(
                view,
                width="stretch",
                height=480,
                column_config={
                    "Stockbit": st.column_config.LinkColumn(
                        "Stockbit",
                        help="Buka halaman emiten di Stockbit",
                        display_text="📈 Stockbit",
                        max_chars=40,
                    )
                },
            )
        else:
            st.dataframe(view, width="stretch", height=480)

        st.caption(
            f"Total: {len(view)} baris • Kolom Ticker di-freeze di kiri • "
            "Klik **Stockbit** untuk buka https://stockbit.com/symbol/…"
        )
    except Exception as e:
        st.error(f"Gagal baca: {e}")


# =====================================================================
# MAIN TABS: Kondisi Pasar | Screener | AI Assistant | Backtest
# =====================================================================
tab_pasar, tab_screener, tab_ai, tab_bt = st.tabs(
    ["📡 Kondisi Pasar", "📊 Screener", "🤖 AI Assistant", "📉 Backtest"]
)

# ---------- TAB 1: KONDISI PASAR ----------
with tab_pasar:
    st.subheader("📡 Kondisi Pasar IHSG")

    cek_rezim = st.button("🔍 Cek Rezim Sekarang", width="stretch", key="btn_cek_rezim")

    if cek_rezim:
        with st.spinner("Mengambil data IHSG..."):
            try:
                import master_screener_ai

                st.session_state["ihsg_regime"] = master_screener_ai.analyze_ihsg_regime()
            except Exception as e:
                st.session_state["ihsg_regime"] = {
                    "regime": "UNKNOWN",
                    "reason": [str(e)],
                    "strategies": [],
                    "last_close": None,
                }

    if "ihsg_regime" not in st.session_state:
        st.session_state["ihsg_regime"] = {
            "regime": "Belum dicek",
            "reason": ["Klik tombol **Cek Rezim Sekarang** untuk menganalisa IHSG."],
            "strategies": [],
            "last_close": None,
        }

    reg = st.session_state.get("ihsg_regime", {})
    regime = reg.get("regime", "UNKNOWN")

    if "BULLISH" in str(regime):
        st.success(f"**Rezim: {regime}**")
    elif "BEARISH" in str(regime):
        st.error(f"**Rezim: {regime}**")
    elif "SIDEWAYS" in str(regime):
        st.warning(f"**Rezim: {regime}**")
    else:
        st.info(f"**Rezim: {regime}**")

    activity = reg.get("activity", "Belum dicek")
    if activity == "RAMAI":
        st.success(f"**Aktivitas pasar: {activity} (bergairah)**")
    elif activity == "SEPI":
        st.warning(f"**Aktivitas pasar: {activity} (lesu)**")
    elif activity == "NORMAL":
        st.info(f"**Aktivitas pasar: {activity}**")

    for r in reg.get("activity_reason") or []:
        st.caption(r)

    # --- Bias operasional (arah kerja) ---
    bias = reg.get("bias")
    if bias:
        bias_label = {
            "BULLISH": "🟢 BULLISH — prioritaskan setup long / breakout (sesuai strategi)",
            "SEDIKIT_BULLISH": "🟡 SEDIKIT BULLISH — long selektif, konfirmasi wajib",
            "NETRAL": "⚪ NETRAL — range/rotasi; hindari FOMO, tunggu trigger jelas",
            "SEDIKIT_BEARISH": "🟠 SEDIKIT BEARISH — kurangi agresivitas long; defensif",
            "BEARISH": "🔴 BEARISH — bias hati-hati / cash / setup khusus downtrend",
        }.get(str(bias), str(bias))
        if "BULLISH" in str(bias) and "SEDIKIT" not in str(bias):
            st.success(f"**Bias pasar: {bias_label}**")
        elif "BEARISH" in str(bias) and "SEDIKIT" not in str(bias):
            st.error(f"**Bias pasar: {bias_label}**")
        elif "SEDIKIT_BULLISH" in str(bias):
            st.success(f"**Bias pasar: {bias_label}**")
        elif "SEDIKIT_BEARISH" in str(bias):
            st.warning(f"**Bias pasar: {bias_label}**")
        else:
            st.info(f"**Bias pasar: {bias_label}**")
        if reg.get("bias_score") is not None:
            st.caption(f"Bias score: {reg.get('bias_score')}")
        for r in reg.get("bias_reason") or []:
            st.caption(f"• {r}")
    elif str(regime) not in ("Belum dicek", "UNKNOWN", ""):
        st.caption(
            "Bias belum tersedia di hasil rezim ini — jalankan ulang **Cek Rezim** "
            "setelah update `master_screener_ai.py`."
        )

    if reg.get("vol_ratio_20") is not None:
        st.caption(
            f"Vol IHSG: {reg['vol_ratio_20']}x avg20 · "
            f"Range5: {reg.get('range_5_pct', '-')}% · "
            f"Range20: {reg.get('range_20_pct', '-')}%"
        )
    elif reg.get("activity_reason"):
        st.caption("Vol IHSG: n/a (data volume indeks sering tidak valid di Yahoo)")

    mc = st.columns(4)
    if reg.get("last_close"):
        mc[0].metric("IHSG", f"{reg['last_close']:,.2f}")
    if reg.get("ma20"):
        mc[1].metric("MA20", f"{reg['ma20']:,.0f}")
    if reg.get("ma50"):
        mc[2].metric("MA50", f"{reg['ma50']:,.0f}")
    if reg.get("ma100"):
        mc[3].metric("MA100", f"{reg['ma100']:,.0f}")

    for r in reg.get("reason") or []:
        st.markdown(f"- {r}")
    if reg.get("strategies"):
        st.markdown(f"**Strategi disarankan:** `{', '.join(reg['strategies'])}`")

    st.markdown("---")
    st.markdown("##### 📰 Berita pasar (near real-time)")
    col_n1, col_n2 = st.columns([1, 3])
    with col_n1:
        refresh_news = st.button("🔄 Muat berita IHSG", width="stretch", key="btn_market_news")
    with col_n2:
        st.caption("Sumber: Google News / media ID (CNBC, Kontan, Bisnis). Bukan feed BEI resmi.")

    if refresh_news or st.session_state.get("market_news"):
        if refresh_news or not st.session_state.get("market_news"):
            with st.spinner("Mengambil berita pasar..."):
                try:
                    import idx_ai_assistant as _ai_news

                    if not hasattr(_ai_news, "fetch_market_news"):
                        st.session_state["market_news"] = {
                            "headlines": [],
                            "errors": [
                                "Modul idx_ai_assistant di Cloud belum punya fetch_market_news. "
                                "Push versi terbaru file tersebut ke GitHub lalu reboot app."
                            ],
                            "fetched_at": None,
                        }
                    else:
                        st.session_state["market_news"] = _ai_news.fetch_market_news(max_items=10)
                except Exception as e:
                    st.session_state["market_news"] = {
                        "headlines": [],
                        "errors": [str(e)],
                        "fetched_at": None,
                    }

        mn = st.session_state.get("market_news") or {}
        if mn.get("fetched_at"):
            st.caption(f"Update: {mn.get('fetched_at')} · sumber: {', '.join(mn.get('sources') or [])}")
        heads = mn.get("headlines") or []
        if not heads:
            st.info("Belum ada headline. Klik **Muat berita IHSG**.")
            if mn.get("errors"):
                st.caption("Error: " + "; ".join(mn["errors"][:3]))
        else:
            vurl = mn.get("verify_search_url")
            if vurl:
                st.markdown(f"[Buka pencarian IHSG di Google News]({vurl})")
            for h in heads:
                title = h.get("title") or "-"
                url = h.get("url") or ""
                pub = h.get("publisher") or h.get("source") or ""
                date = h.get("date") or ""
                meta = " · ".join(x for x in [pub, date] if x)
                if url:
                    st.markdown(f"- [{title}]({url})" + (f" — _{meta}_" if meta else ""))
                else:
                    st.markdown(f"- **{title}**" + (f" — _{meta}_" if meta else ""))
            if mn.get("disclaimer"):
                st.caption(mn["disclaimer"])

    st.markdown("---")
    st.markdown("##### 🌍 Makro global (LLMQuant)")
    st.caption(
        "Pelengkap narasi risk-on/off (data AS/global). "
        "Bukan pengganti IHSG / data emiten BEI."
    )
    col_m1, col_m2 = st.columns([1, 3])
    with col_m1:
        load_macro = st.button("🔄 Muat makro LLMQuant", width="stretch", key="btn_llmquant_macro")
    with col_m2:
        try:
            import idx_llmquant as _lq

            if _lq.available():
                st.success("LLMQUANT_API_KEY terdeteksi")
            else:
                st.info("Set `LLMQUANT_API_KEY` di Secrets / .env untuk mengaktifkan.")
        except Exception:
            st.warning("Modul `idx_llmquant.py` belum ada di project.")

    if load_macro or st.session_state.get("llmquant_macro"):
        if load_macro or not st.session_state.get("llmquant_macro"):
            with st.spinner("Mengambil makro dari LLMQuant..."):
                try:
                    import idx_llmquant as _lq

                    st.session_state["llmquant_macro"] = _lq.build_market_macro_pack()
                except Exception as e:
                    st.session_state["llmquant_macro"] = {
                        "items": [],
                        "errors": [str(e)],
                        "fetched_at": None,
                        "narrative_hint": "",
                    }
        mp = st.session_state.get("llmquant_macro") or {}
        if mp.get("fetched_at"):
            st.caption(f"Update: {mp.get('fetched_at')}")
        if mp.get("narrative_hint"):
            st.markdown(f"**Ringkas:** {mp['narrative_hint']}")
        items = mp.get("items") or []
        if items:
            for it in items:
                title = it.get("title") or it.get("indicator")
                lv = it.get("latest_value")
                ld = it.get("latest_date") or ""
                dlt = it.get("delta_abs")
                line = f"- **{title}**: {lv}"
                if dlt is not None:
                    line += f" (Δ {dlt})"
                if ld:
                    line += f" · _{ld}_"
                st.markdown(line)
        elif mp.get("errors"):
            st.warning("Gagal memuat makro: " + "; ".join(str(x) for x in mp["errors"][:3]))
            st.caption(
                "Cek API key, kredit LLMQuant, dan apakah path REST masih sesuai docs."
            )
        if mp.get("disclaimer"):
            st.caption(mp["disclaimer"])

        st.markdown("---")
        st.markdown("##### 🧠 Kesimpulan AI: makro global ↔ IHSG")
        st.caption(
            "Menilai transmisi suku bunga/inflasi/yield AS ke risk appetite IHSG. "
            "Butuh data makro di atas; hasil lebih baik jika rezim IHSG sudah dicek."
        )
        gen_c = st.button(
            "✍️ Buat kesimpulan korelasi IHSG",
            width="stretch",
            key="btn_macro_ihsg_conclusion",
            disabled=not (mp.get("items") or mp.get("narrative_hint")),
        )
        if gen_c:
            with st.spinner("Menyusun kesimpulan (LLM bila tersedia)..."):
                try:
                    import idx_llmquant as _lq

                    conc = _lq.conclude_macro_vs_ihsg(
                        macro=mp,
                        regime=st.session_state.get("ihsg_regime") or {},
                        use_llm=True,
                    )
                    st.session_state["macro_ihsg_conclusion"] = conc
                except Exception as e:
                    st.session_state["macro_ihsg_conclusion"] = {
                        "mode": "error",
                        "conclusion": "",
                        "error": str(e),
                    }

        conc = st.session_state.get("macro_ihsg_conclusion")
        if conc:
            mode = conc.get("mode") or "-"
            if mode == "llm":
                st.success(f"Mode: LLM · {conc.get('fetched_at', '')}")
            elif mode in ("heuristic", "heuristic_fallback"):
                st.info(f"Mode: heuristik · {conc.get('fetched_at', '')}")
                if conc.get("error"):
                    st.caption(f"LLM gagal → fallback. Detail: {conc['error'][:200]}")
            elif mode == "error":
                st.error(conc.get("error") or "Gagal")
            if conc.get("conclusion"):
                _md_bias(conc["conclusion"])
            st.caption("Bukan saran investasi. Korelasi makro–IHSG bersifat kontekstual dan dapat berubah.")

# ---------- TAB 2: SCREENER ----------
with tab_screener:
    st.subheader("📊 Screener")
    st.caption("Pilih mode di sidebar, lalu klik **JALANKAN**. Hasil tampil per strategi di bawah.")

    # Jalankan screener (tombol di sidebar)
    if run_button:
        params = {
            "account_size": float(account_size),
            "risk_per_trade_pct": float(risk_pct),
            "broker_buy_pct": float(broker_buy),
            "broker_sell_pct": float(broker_sell),
        }
        label = mode
        started_at = datetime.now()

        spinner_msg = f"Menjalankan {label}..."
        if mode.startswith("🎯") and not skip_cf:
            spinner_msg += " (semua strategi — bisa 5–15 menit)"
        else:
            spinner_msg += " (1–3 menit)"

        with st.spinner(spinner_msg):
            if mode.startswith("🤖"):
                ok, err, log = capture_run(
                    run_ai, account_size, risk_pct, broker_buy, broker_sell
                )
                try:
                    import master_screener_ai

                    st.session_state["ihsg_regime"] = (
                        master_screener_ai.analyze_ihsg_regime()
                    )
                except Exception:
                    pass
            elif mode.startswith("V2"):
                ok, err, log = capture_run(run_v2, params)
            elif mode.startswith("V3"):
                ok, err, log = capture_run(run_v3, params)
            elif mode.startswith("V4"):
                ok, err, log = capture_run(run_v4, params)
            elif mode.startswith("V5"):
                ok, err, log = capture_run(run_v5, params)
            elif mode.startswith("⚡"):
                ok, err, log = capture_run(run_intraday, params)
            elif mode.startswith("🔥") or "HighBeta" in mode:
                ok, err, log = capture_run(run_highbeta, params)
            elif mode.startswith("📦") or "Accumulation" in mode:
                ok, err, log = capture_run(run_accumulation, params)
            elif mode.startswith("🎯") or "Confluence" in mode:
                ok, err, log = capture_run(run_confluence, params, skip_cf)
            else:
                ok, err, log = False, "Mode tidak dikenal", ""

            if ok and not mode.startswith("🤖"):
                enrich_all_version_csvs(broker_buy, broker_sell)

        finished_at = datetime.now()
        duration_sec = (finished_at - started_at).total_seconds()
        counts = {ver: count_report_rows(ver) for ver in REPORT_VERSIONS}

        st.session_state["last_run_status"] = {
            "ok": ok,
            "mode": label,
            "error": err,
            "log": log or "",
            "started_at": started_at.strftime("%Y-%m-%d %H:%M:%S"),
            "finished_at": finished_at.strftime("%Y-%m-%d %H:%M:%S"),
            "duration_sec": round(duration_sec, 1),
            "counts": counts,
            "account_size": float(account_size),
            "risk_pct": float(risk_pct),
            "broker_buy": float(broker_buy),
            "broker_sell": float(broker_sell),
        }

        if ok:
            st.success(f"✅ {label} selesai.")
        else:
            st.error(f"❌ {err}")

        with st.expander("📋 Log", expanded=not ok):
            st.text(log if log and log.strip() else "(kosong)")

    # Status run terakhir
    st.markdown("##### 📌 Status Run Terakhir")
    status = st.session_state.get("last_run_status")

    if not status:
        st.info("Belum ada screener yang dijalankan di sesi ini.")
    else:
        if status.get("ok"):
            st.success("Status: **Berhasil**")
        else:
            st.error(f"Status: **Gagal** — {status.get('error') or '-'}")

        st.caption(
            f"**Mode:** {status.get('mode', '-')} · "
            f"**Durasi:** {status.get('duration_sec', 0)} dtk · "
            f"**Mulai:** {status.get('started_at', '-')} · "
            f"**Selesai:** {status.get('finished_at', '-')}"
        )
        st.caption(
            f"Modal: Rp {status.get('account_size', 0):,.0f} · "
            f"Risiko: {status.get('risk_pct', 0)}% · "
            f"Fee beli/jual: {status.get('broker_buy', 0)}% / {status.get('broker_sell', 0)}%"
        )

        counts = status.get("counts") or {}
        if counts:
            st.caption(
                f"Setup — V2:**{counts.get('v2', 0)}** · "
                f"V3:**{counts.get('v3', 0)}** · "
                f"V4:**{counts.get('v4', 0)}** · "
                f"V5:**{counts.get('v5', 0)}** · "
                f"Intra:**{counts.get('intraday', 0)}** · "
                f"HB:**{counts.get('highbeta', 0)}** · "
                f"Acc:**{counts.get('accumulation', 0)}** · "
                f"CF:**{counts.get('confluence', 0)}**"
            )

        with st.expander("📋 Log run terakhir", expanded=False):
            st.text(status.get("log") or "(kosong)")

    st.markdown("---")
    st.markdown("##### Hasil per strategi")

    sub_tabs = st.tabs(
        [
            "V2 Breakout",
            "V3 Retest Fibo",
            "V4 Order Block",
            "V5 CHOCH",
            "⚡ Intraday",
            "🔥 HighBeta",
            "📦 Accumulation",
            "🎯 Confluence",
        ]
    )

    with sub_tabs[0]:
        show_report("v2", "V2 Breakout")
    with sub_tabs[1]:
        show_report("v3", "V3 Retest Fibo")
    with sub_tabs[2]:
        show_report("v4", "V4 Order Block")
    with sub_tabs[3]:
        show_report("v5", "V5 CHOCH")
    with sub_tabs[4]:
        show_report("intraday", "Intraday Confluence")
    with sub_tabs[5]:
        show_report("highbeta", "HighBeta Liquid")
    with sub_tabs[6]:
        show_report("accumulation", "Accumulation Late/Early")
    with sub_tabs[7]:
        show_report("confluence", "Confluence Top Overlap")

# ---------- TAB 3: AI ASSISTANT ----------
with tab_ai:
    st.subheader("🤖 AI Trader Assistant")
    st.caption(
        "Pilih ticker dari hasil screener → multi-agent "
        "(technical / fundamental / risk / critic / chief). "
        "Bukan rekomendasi investasi."
    )

    try:
        import idx_ai_assistant as ai_asst

        AI_OK = True
    except ImportError:
        AI_OK = False
        ai_asst = None

    if not AI_OK:
        st.warning(
            "Modul `idx_ai_assistant.py` tidak ditemukan di folder project. "
            "Tambahkan file tersebut lalu restart Streamlit."
        )
    else:
        offline = not ai_asst.llm_available()
        if offline:
            st.info(
                "Mode **offline** (heuristik). Set secret/env "
                "`OPENAI_API_KEY` atau `XAI_API_KEY` "
                "(opsional `OPENAI_BASE_URL`, `OPENAI_MODEL`) untuk analisa LLM penuh."
            )
        else:
            st.success("LLM API terdeteksi — agent akan memanggil model.")

        ver_labels = {
            "v2": "V2 Breakout",
            "v3": "V3 Retest",
            "v4": "V4 Order Block",
            "v5": "V5 CHOCH",
            "intraday": "Intraday",
            "highbeta": "HighBeta",
            "accumulation": "Accumulation",
            "confluence": "Confluence",
        }
        ai_col1, ai_col2 = st.columns([1, 2])
        with ai_col1:
            ai_version = st.selectbox(
                "Sumber report",
                options=list(ver_labels.keys()),
                format_func=lambda v: ver_labels.get(v, v),
                key="ai_version",
            )
        tickers_avail = ai_asst.list_tickers_in_report(ai_version)
        with ai_col2:
            if not tickers_avail:
                st.selectbox(
                    "Ticker",
                    options=["(tidak ada data — jalankan screener dulu)"],
                    disabled=True,
                    key="ai_ticker_dummy",
                )
                selected_tickers = []
            else:
                selected_tickers = st.multiselect(
                    "Ticker (maks. 3 per analisa)",
                    options=tickers_avail,
                    default=tickers_avail[:1],
                    max_selections=3,
                    key="ai_tickers",
                )

        ai_model = st.text_input(
            "Model (opsional, kosongkan = default env)",
            value="",
            key="ai_model",
            help="Contoh: gpt-4o-mini, grok-2-latest",
        )

        auto_save_txt = st.checkbox(
            "Simpan otomatis ke file .txt setelah analisa",
            value=True,
            key="ai_auto_save_txt",
            help="Menyimpan ke folder ai_analysis_output/ di server (lokal/Cloud).",
        )

        run_ai_btn = st.button(
            "🧠 Analisa dengan AI",
            type="primary",
            width="stretch",
            disabled=not selected_tickers,
            key="ai_run_btn",
        )

        if run_ai_btn and selected_tickers:
            regime_ctx = st.session_state.get("ihsg_regime") or {}
            regime_slim = {
                k: regime_ctx.get(k)
                for k in (
                    "regime",
                    "activity",
                    "bias",
                    "bias_score",
                    "bias_reason",
                    "strategies",
                    "last_close",
                    "reason",
                    "activity_reason",
                    "vol_ratio_20",
                    "range_5_pct",
                    "range_20_pct",
                )
                if k in regime_ctx
            }
            results_ai = []
            with st.spinner(
                f"Menjalankan agent untuk {', '.join(selected_tickers)}..."
            ):
                for t in selected_tickers:
                    try:
                        out = ai_asst.analyze_ticker(
                            t,
                            ai_version,
                            regime=regime_slim,
                            account_size=float(account_size),
                            risk_pct=float(risk_pct),
                            broker_buy_pct=float(broker_buy),
                            broker_sell_pct=float(broker_sell),
                            model=ai_model.strip() or None,
                        )
                    except Exception as e:
                        out = {
                            "ticker": t,
                            "version": ai_version,
                            "error": str(e),
                            "analyses": {},
                        }
                    if auto_save_txt and (
                        out.get("analyses") or out.get("error")
                    ):
                        try:
                            txt_path = ai_asst.save_analysis_to_txt(out)
                            out["saved_txt_path"] = txt_path
                            try:
                                with open(txt_path, "rb") as fh:
                                    out["saved_txt_bytes"] = fh.read()
                            except Exception:
                                out["saved_txt_bytes"] = None
                        except Exception as se:
                            out["saved_txt_error"] = str(se)
                    results_ai.append(out)
            st.session_state["ai_assistant_results"] = results_ai
            saved_ok = [
                r.get("saved_txt_path")
                for r in results_ai
                if r.get("saved_txt_path")
            ]
            if saved_ok:
                st.success(
                    "Analisa disimpan ke .txt:\n"
                    + "\n".join(f"- `{p}`" for p in saved_ok)
                )

        for out in st.session_state.get("ai_assistant_results") or []:
            t = out.get("ticker", "?")
            _sb = stockbit_url(t)
            if _sb:
                st.markdown(
                    f"### {t} · `{out.get('version', '')}` · "
                    f"[📈 Stockbit]({_sb})"
                )
            else:
                st.markdown(f"### {t} · `{out.get('version', '')}`")
            if out.get("error") and not (out.get("analyses") or {}):
                st.error(out["error"])
                continue
            if out.get("offline"):
                st.caption("Hasil mode offline / heuristik")
            if out.get("partial"):
                st.warning(
                    "Sebagian role gagal (sering timeout/koneksi klien meski OpenAI "
                    "sudah memproses). Role yang sukses tetap ditampilkan di bawah."
                )
                err_map = out.get("errors") or {}
                if err_map:
                    with st.expander("Detail error per role", expanded=False):
                        for rk, rv in err_map.items():
                            st.text(f"{rk}: {rv}")

            # Tombol unduh / simpan ulang TXT
            c_dl1, c_dl2 = st.columns([1, 1])
            with c_dl1:
                if out.get("saved_txt_bytes"):
                    fname = (
                        f"{t}_{out.get('version', 'na')}_"
                        f"{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
                    )
                    st.download_button(
                        label=f"⬇️ Unduh analisa {t} (.txt)",
                        data=out["saved_txt_bytes"],
                        file_name=fname,
                        mime="text/plain",
                        key=f"ai_dl_{t}_{out.get('generated_at', '')}",
                    )
                elif out.get("analyses"):
                    # generate on the fly if belum auto-save
                    try:
                        _p = ai_asst.save_analysis_to_txt(out)
                        with open(_p, "rb") as _f:
                            _b = _f.read()
                        st.download_button(
                            label=f"⬇️ Unduh analisa {t} (.txt)",
                            data=_b,
                            file_name=f"{t}_{out.get('version', 'na')}.txt",
                            mime="text/plain",
                            key=f"ai_dl_live_{t}_{out.get('generated_at', id(out))}",
                        )
                        out["saved_txt_path"] = _p
                    except Exception as se:
                        st.caption(f"Gagal siapkan unduhan: {se}")
            with c_dl2:
                if out.get("saved_txt_path"):
                    st.caption(f"Server: `{out['saved_txt_path']}`")
                if out.get("saved_txt_error"):
                    st.caption(f"Gagal simpan: {out['saved_txt_error']}")

            analyses = out.get("analyses") or {}
            if analyses.get("chief"):
                st.markdown("**Chief (ringkasan)**")
                _md_bias(analyses["chief"])
            for role in ("technical", "fundamental", "risk", "critic"):
                if analyses.get(role):
                    with st.expander(f"{role.capitalize()}", expanded=(role == "technical")):
                        _md_bias(analyses[role])
            ctx = out.get("context") or {}
            ma = ctx.get("ma_structure") or {}
            if ma.get("available"):
                with st.expander("Struktur MA (data live)", expanded=False):
                    m = ma.get("ma") or {}
                    st.caption(
                        f"Close {ma.get('last_close')} · "
                        f"MA20={m.get('MA20')} · MA50={m.get('MA50')} · "
                        f"MA100={m.get('MA100')} · MA200={m.get('MA200')}"
                    )
                    for h in ma.get("interpretation_hints") or []:
                        st.markdown(f"- {h}")
                    for name, cx in (ma.get("crosses") or {}).items():
                        if not cx:
                            continue
                        st.caption(
                            f"{name}: {cx.get('status')} · "
                            f"cross={cx.get('direction_last_cross')} · "
                            f"hari sejak={cx.get('days_since_cross')}"
                        )
            news = ctx.get("news_intel") or {}
            if news.get("headlines") or news.get("verify_search_url") or news.get("sectors_news_url"):
                with st.expander(
                    f"Berita / web intel ({news.get('headline_count', 0)}) · tone={news.get('tone_hint', '?')}"
                + (f" · {news.get('fetched_at')}" if news.get("fetched_at") else ""),
                    expanded=False,
                ):
                    vurl = news.get("verify_search_url")
                    if vurl:
                        st.markdown(f"[Cari semua berita terkait di Google News]({vurl})")
                    surl = news.get("sectors_news_url")
                    if surl:
                        st.markdown(f"[Berita Sectors (filter emiten)]({surl})")
                    fund_ctx = ctx.get("fundamental") or {}
                    curl = fund_ctx.get("sectors_company_url")
                    if curl:
                        st.markdown(
                            f"[Profil/fundamental Sectors (verifikasi manual)]({curl})"
                        )
                    conf = fund_ctx.get("confidence")
                    src = fund_ctx.get("source")
                    if conf or src:
                        st.caption(
                            f"Fundamental confidence: {conf or 'n/a'} ({src or '?'})"
                        )
                    for h in news.get("headlines") or []:
                        title = h.get("title") or "-"
                        pub = h.get("publisher") or h.get("source") or ""
                        date = h.get("date") or ""
                        url = h.get("url") or ""
                        kind = h.get("link_kind") or ""
                        meta = " · ".join(x for x in [pub, date, kind] if x)
                        if url:
                            st.markdown(f"- [{title}]({url})" + (f" — _{meta}_" if meta else ""))
                        else:
                            st.markdown(f"- **{title}**" + (f" — _{meta}_" if meta else ""))
                    if news.get("disclaimer"):
                        st.caption(news["disclaimer"])
            lq = ctx.get("llmquant") or {}
            if lq and not lq.get("skipped"):
                with st.expander("LLMQuant (makro + quant wiki)", expanded=False):
                    if (lq.get("macro") or {}).get("narrative_hint"):
                        st.markdown(f"**Makro:** {lq['macro']['narrative_hint']}")
                    wiki_items = (lq.get("quant_wiki") or {}).get("items") or []
                    if wiki_items:
                        st.markdown("**Quant Wiki:**")
                        for w in wiki_items[:4]:
                            st.markdown(
                                f"- **{w.get('title') or '-'}** — {(w.get('summary') or '')[:180]}"
                            )
                    errs = list((lq.get("macro") or {}).get("errors") or []) + list(
                        (lq.get("quant_wiki") or {}).get("errors") or []
                    )
                    if errs:
                        st.caption("Note: " + "; ".join(str(e) for e in errs[:3]))
            with st.expander("Konteks JSON (debug)", expanded=False):
                st.json(ctx)
            st.markdown("---")


# ---------- TAB 4: BACKTEST ----------
with tab_bt:
    st.subheader("📉 Backtest Setup Screener")
    st.caption(
        "Pilih ticker dari hasil report strategi → simulasi SL/TP ke depan (signal-forward). "
        "Data history: Yahoo Finance (yfinance). Bukan jaminan hasil live."
    )

    try:
        import idx_backtest_engine as bt_eng

        BT_OK = True
    except ImportError:
        BT_OK = False
        bt_eng = None

    if not BT_OK:
        st.warning(
            "Modul `idx_backtest_engine.py` tidak ditemukan di folder project. "
            "Salin file tersebut lalu restart Streamlit."
        )
    else:
        ver_labels_bt = {
            "v2": "V2 Breakout",
            "v3": "V3 Retest",
            "v4": "V4 Order Block",
            "v5": "V5 CHOCH",
            "intraday": "Intraday",
            "highbeta": "HighBeta",
            "accumulation": "Accumulation",
            "confluence": "Confluence",
        }

        bt_c1, bt_c2 = st.columns([1, 2])
        with bt_c1:
            bt_version = st.selectbox(
                "Strategi (sumber report)",
                options=list(ver_labels_bt.keys()),
                format_func=lambda v: ver_labels_bt.get(v, v),
                key="bt_version",
            )

        # Ambil daftar ticker dari report (reuse AI helper jika ada)
        tickers_bt = []
        report_path = None
        try:
            report_path = bt_eng.find_report(bt_version)
        except Exception:
            report_path = None

        try:
            import idx_ai_assistant as _ai_for_list

            tickers_bt = _ai_for_list.list_tickers_in_report(bt_version) or []
        except Exception:
            tickers_bt = []

        if not tickers_bt and report_path:
            try:
                _df_bt = bt_eng.load_screener_setups(path=report_path, version=bt_version)
                tickers_bt = (
                    _df_bt["Ticker"].astype(str).str.upper().unique().tolist()
                    if not _df_bt.empty
                    else []
                )
            except Exception:
                tickers_bt = []

        with bt_c2:
            if not tickers_bt:
                st.selectbox(
                    "Ticker",
                    options=["(tidak ada data — jalankan screener dulu)"],
                    disabled=True,
                    key="bt_ticker_dummy",
                )
                selected_bt = []
            else:
                selected_bt = st.multiselect(
                    "Ticker untuk di-backtest (maks. 10)",
                    options=tickers_bt,
                    default=tickers_bt[: min(3, len(tickers_bt))],
                    max_selections=10,
                    key="bt_tickers",
                )

        if report_path:
            st.caption(f"Report: `{report_path}`")
        else:
            st.caption("Report CSV belum ditemukan untuk versi ini.")

        bt_mode = st.radio(
            "Mode backtest",
            options=["report", "historical"],
            format_func=lambda x: (
                "📄 Setup dari report (1 sinyal / baris CSV)"
                if x == "report"
                else "📜 Historis strategi pada ticker (Phase 2)"
            ),
            horizontal=True,
            key="bt_mode",
            help=(
                "Report: uji Entry/SL/TP di CSV. "
                "Historis: scan ulang rule strategi di history ticker → win rate emiten×strategi."
            ),
        )
        if bt_mode == "historical":
            adapter_meta = getattr(bt_eng, "STRATEGY_ADAPTERS", {})
            info = adapter_meta.get(bt_version, {})
            if info.get("full"):
                st.caption(f"Adapter **{bt_version}** = rule lengkap.")
            else:
                st.caption(
                    f"Adapter **{bt_version}** masih **proxy breakout** "
                    "(bukan rule SMC/intra penuh). V3 = rule Fibo lengkap."
                )

        st.markdown("**Parameter simulasi**")
        p1, p2, p3, p4 = st.columns(4)
        with p1:
            bt_horizon = st.number_input(
                "Horizon (hari)",
                min_value=3,
                max_value=60,
                value=20,
                step=1,
                key="bt_horizon",
            )
        with p2:
            bt_entry_mode = st.selectbox(
                "Entry mode",
                options=["next_open", "signal_close"],
                format_func=lambda x: (
                    "Open bar berikutnya" if x == "next_open" else "Close hari sinyal"
                ),
                key="bt_entry_mode",
            )
        with p3:
            bt_priority = st.selectbox(
                "Intrabar priority",
                options=["sl_first", "tp_first"],
                format_func=lambda x: (
                    "SL dulu (konservatif)" if x == "sl_first" else "TP dulu (optimis)"
                ),
                key="bt_priority",
            )
        with p4:
            if bt_mode == "historical":
                bt_lookback = st.number_input(
                    "Lookback history (hari)",
                    min_value=120,
                    max_value=800,
                    value=400,
                    step=20,
                    key="bt_lookback",
                )
            else:
                bt_tp_level = st.selectbox(
                    "Target",
                    options=["tp1", "tp2"],
                    format_func=lambda x: "TP1" if x == "tp1" else "TP2 (jika ada)",
                    key="bt_tp_level",
                )
                bt_lookback = 400

        if bt_mode == "historical":
            bt_tp_level = st.selectbox(
                "Target",
                options=["tp1", "tp2"],
                format_func=lambda x: "TP1" if x == "tp1" else "TP2 (jika ada)",
                key="bt_tp_level_hist",
            )

        # Historical: boleh input ticker manual jika tidak ada di report
        if bt_mode == "historical":
            extra_t = st.text_input(
                "Tambah ticker manual (pisah koma)",
                value="",
                key="bt_extra_tickers",
                help="Contoh: MARK,BBCA — untuk uji emiten meski tidak ada di report hari ini",
            )
            if extra_t.strip():
                for x in extra_t.split(","):
                    x = x.strip().upper().replace(".JK", "")
                    if x and x not in selected_bt:
                        selected_bt = list(selected_bt) + [x]

        run_bt = st.button(
            "▶️ Jalankan Backtest",
            type="primary",
            width="stretch",
            disabled=not selected_bt,
            key="bt_run_btn",
        )

        if run_bt and selected_bt:
            bt_params = {
                "horizon_bars": int(bt_horizon),
                "entry_mode": bt_entry_mode,
                "intrabar_priority": bt_priority,
                "tp_level": bt_tp_level,
                "account_size": float(account_size),
                "risk_per_trade_pct": float(risk_pct),
                "buy_fee_pct": float(broker_buy),
                "sell_fee_pct": float(broker_sell),
            }
            with st.spinner(
                f"Backtest [{bt_mode}] {', '.join(selected_bt)} · {bt_version}..."
            ):
                try:
                    if bt_mode == "historical":
                        # Kompatibel: file lama mungkin belum punya backtest_tickers_strategy
                        batch_fn = getattr(bt_eng, "backtest_tickers_strategy", None)
                        single_fn = getattr(bt_eng, "backtest_ticker_strategy", None)
                        if batch_fn is None and single_fn is None:
                            raise AttributeError(
                                "idx_backtest_engine belum berisi Phase 2. "
                                "Update file idx_backtest_engine.py (salin dari artifacts), "
                                "lalu restart Streamlit / redeploy Cloud."
                            )
                        if batch_fn is not None:
                            trades, summary = batch_fn(
                                selected_bt,
                                bt_version,
                                lookback_days=int(bt_lookback),
                                max_signals=30,
                                params=bt_params,
                            )
                        else:
                            # fallback: loop per ticker
                            import pandas as _pd_hist

                            _parts = []
                            for _t in selected_bt:
                                _tr, _sm, _ = single_fn(
                                    _t,
                                    bt_version,
                                    lookback_days=int(bt_lookback),
                                    max_signals=30,
                                    params=bt_params,
                                    progress=True,
                                )
                                if _tr is not None and not _tr.empty:
                                    _parts.append(_tr)
                            if _parts:
                                trades = _pd_hist.concat(_parts, ignore_index=True)
                                summary = bt_eng.summarize_trades(trades)
                            else:
                                trades = _pd_hist.DataFrame()
                                summary = bt_eng.summarize_trades(trades)
                        out_path = None
                        if trades is not None and not trades.empty:
                            out_path = (
                                f"idx_backtest_hist_{bt_version}_"
                                f"{datetime.now().strftime('%Y-%m-%d')}.csv"
                            )
                            trades.to_csv(out_path, index=False)
                    else:
                        trades, summary, out_path = bt_eng.backtest_report(
                            bt_version,
                            path=report_path,
                            tickers=selected_bt,
                            params=bt_params,
                            save=True,
                        )
                    st.session_state["bt_results"] = {
                        "trades": (
                            trades.to_dict(orient="records")
                            if trades is not None and not trades.empty
                            else []
                        ),
                        "summary": summary.to_dict() if summary else {},
                        "path": out_path,
                        "version": bt_version,
                        "tickers": selected_bt,
                        "params": bt_params,
                        "mode": bt_mode,
                    }
                except Exception as e:
                    st.session_state["bt_results"] = {
                        "error": str(e),
                        "trades": [],
                        "summary": {},
                    }

        bt_res = st.session_state.get("bt_results")
        if bt_res:
            if bt_res.get("error"):
                st.error(bt_res["error"])
            else:
                sm = bt_res.get("summary") or {}
                mode_lbl = bt_res.get("mode") or "report"
                st.markdown(
                    f"### Ringkasan "
                    f"({'Historis Phase 2' if mode_lbl == 'historical' else 'Setup report'})"
                )
                m1, m2, m3, m4 = st.columns(4)
                m1.metric("Trades", sm.get("n_trades", 0))
                m2.metric("Win rate", f"{sm.get('win_rate', 0)}%")
                m3.metric("Avg R", sm.get("avg_r", 0))
                m4.metric("Expectancy R", sm.get("expectancy_r", 0))

                m5, m6, m7, m8 = st.columns(4)
                m5.metric("Profit factor", sm.get("profit_factor", 0))
                m6.metric("PnL net (Rp)", f"{sm.get('total_pnl_net', 0):,.0f}")
                m7.metric("Avg hold (bars)", sm.get("avg_bars_held", 0))
                m8.metric(
                    "Max W / L R",
                    f"{sm.get('max_win_r', 0)} / {sm.get('max_loss_r', 0)}",
                )

                if sm.get("by_exit_reason"):
                    st.caption(f"Exit reason: {sm.get('by_exit_reason')}")
                if sm.get("avg_planned_rr") is not None:
                    st.caption(
                        f"Rata planned RR screener: {sm.get('avg_planned_rr')} "
                        f"vs realized avg R: {sm.get('avg_r')}"
                    )

                trades_list = bt_res.get("trades") or []
                if trades_list:
                    import pandas as _pd_bt

                    tdf = _pd_bt.DataFrame(trades_list)
                    st.markdown("### Detail trade")
                    if "ticker" in tdf.columns:
                        tdf = tdf.copy()
                        tdf.insert(
                            0,
                            "Stockbit",
                            tdf["ticker"].map(stockbit_url),
                        )
                    show_cols = [
                        c
                        for c in [
                            "Stockbit",
                            "ticker",
                            "strategy",
                            "signal_date",
                            "entry_date",
                            "exit_date",
                            "entry",
                            "stop_loss",
                            "target",
                            "exit_price",
                            "exit_reason",
                            "bars_held",
                            "r_multiple",
                            "pnl_net",
                            "lots",
                            "planned_rr",
                        ]
                        if c in tdf.columns
                    ]
                    st.dataframe(
                        tdf[show_cols],
                        width="stretch",
                        hide_index=True,
                        column_config={
                            "Stockbit": st.column_config.LinkColumn(
                                "Stockbit",
                                display_text="📈",
                                help="Buka di Stockbit",
                            )
                        },
                    )

                    # Download CSV
                    csv_bytes = tdf.to_csv(index=False).encode("utf-8")
                    st.download_button(
                        "⬇️ Unduh hasil backtest (.csv)",
                        data=csv_bytes,
                        file_name=(
                            f"idx_backtest_{bt_res.get('version', 'x')}_"
                            f"{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
                        ),
                        mime="text/csv",
                        key="bt_download_csv",
                    )
                    if bt_res.get("path"):
                        st.caption(f"Server: `{bt_res['path']}`")
                else:
                    mode_lbl = bt_res.get("mode") or "report"
                    if mode_lbl == "historical":
                        st.info(
                            "Tidak ada trade historis. Kemungkinan: "
                            "tidak ketemu sinyal rule pada lookback, "
                            "level Entry/SL/TP kosong setelah normalize, "
                            "atau data yfinance gagal. Coba ticker lain / perbesar lookback."
                        )
                    else:
                        st.info(
                            "Tidak ada trade yang bisa disimulasikan. "
                            "Cek BreakoutDay/signal date di report dan data yfinance."
                        )

                with st.expander("Parameter run", expanded=False):
                    st.json(bt_res.get("params") or {})

st.caption("IDX Master Screener AI • Bukan rekomendasi investasi")

