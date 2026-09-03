"""
IDX Master Screener AI — Streamlit Dashboard
- Panel rezim IHSG (on-demand)
- Jalankan AI adaptive / V2–V5 / Intraday / HighBeta / Accumulation / Confluence
- Tab hasil per strategi
- Komisi + pajak (enrich lewat master / fallback)
- Status run terakhir
- Download CSV + freeze Ticker
"""

import streamlit as st
import pandas as pd
import os
import glob
import io
import sys
import importlib
from datetime import datetime

st.set_page_config(
    page_title="IDX Master Screener Dashboard",
    page_icon="📈",
    layout="wide",
)

st.title("📈 IDX Master Screener AI Dashboard")
st.markdown(
    "Analisa IHSG + screener Breakout, Retest, OB, CHOCH, Intraday, "
    "HighBeta, Accumulation, Confluence."
)

# Semua kunci report yang dikenali dashboard
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
    value=50_000_000,
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
            view = view.set_index("Ticker")

        st.dataframe(view, width="stretch", height=480)
        st.caption(f"Total: {len(view)} baris • Kolom Ticker di-freeze di kiri")
    except Exception as e:
        st.error(f"Gagal baca: {e}")


# =====================================================================
# PANEL REZIM IHSG (on-demand)
# =====================================================================
st.subheader("📡 Kondisi Pasar IHSG")

cek_rezim = st.button("🔍 Cek Rezim Sekarang", width="stretch")

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

# =====================================================================
# JALANKAN SCREENER
# =====================================================================
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

        # Enrich setelah run non-AI (termasuk setelah confluence menjalankan sub-screener)
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

# =====================================================================
# STATUS RUN TERAKHIR
# =====================================================================
st.subheader("📌 Status Run Terakhir")
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

# =====================================================================
# HASIL PER STRATEGI
# =====================================================================
st.subheader("📊 Hasil Screener per Strategi")

tabs = st.tabs(
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

with tabs[0]:
    show_report("v2", "V2 Breakout")
with tabs[1]:
    show_report("v3", "V3 Retest Fibo")
with tabs[2]:
    show_report("v4", "V4 Order Block")
with tabs[3]:
    show_report("v5", "V5 CHOCH")
with tabs[4]:
    show_report("intraday", "Intraday Confluence")
with tabs[5]:
    show_report("highbeta", "HighBeta Liquid")
with tabs[6]:
    show_report("accumulation", "Accumulation Late/Early")
with tabs[7]:
    show_report("confluence", "Confluence Top Overlap")

st.markdown("---")
st.caption("IDX Master Screener AI • Bukan rekomendasi investasi")
