"""
IDX Master Screener AI — Streamlit Dashboard
- Panel rezim IHSG (on-demand, tidak auto-hit setiap refresh)
- Jalankan AI adaptive / V2 / V3 / V4 / V5
- Tab hasil per strategi (CSV terpisah idx_report_v2..v5)
- Cari file di beberapa path
- Download CSV
- Freeze kolom Ticker
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
st.markdown("Analisa IHSG + screener Breakout, Retest Fibo, Order Block, CHOCH.")

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
st.sidebar.header("🚀 Jalankan Screener")
mode = st.sidebar.radio(
    "Mode",
    [
        "🤖 AI Adaptive",
        "V2 — Breakout",
        "V3 — Retest Fibo",
        "V4 — Order Block",
        "V5 — CHOCH",
    ],
)
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


def run_ai(account_size, risk_pct):
    import master_screener_ai
    master_screener_ai.run_orchestrator(
        account_size=float(account_size),
        risk_pct=float(risk_pct),
    )


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
    for extra in ["/home/workdir", "/home/workdir/artifacts", "/home/workdir/attachments"]:
        if os.path.isdir(extra) and extra not in dirs:
            dirs.append(extra)
    return dirs


def find_report(version: str):
    """Cari idx_report_v{N}_*.csv di beberapa folder; ambil yang terbaru."""
    files = []
    for d in _search_dirs():
        files.extend(glob.glob(os.path.join(d, f"idx_report_{version}_*.csv")))
    if not files:
        return None
    return max(files, key=os.path.getmtime)


def show_report(version: str, title: str):
    path = find_report(version)
    if not path:
        st.info(
            f"Belum ada data **{title}**. "
            f"Jalankan screener **{version.upper()}** atau **AI Adaptive**."
        )
        return

    mtime = datetime.fromtimestamp(os.path.getmtime(path))
    st.caption(f"`{os.path.basename(path)}` • {mtime.strftime('%Y-%m-%d %H:%M')}")

    try:
        df = pd.read_csv(path)
        if df.empty:
            st.warning("File kosong.")
            return

        # Download
        with open(path, "rb") as f:
            st.download_button(
                label=f"⬇️ Download {os.path.basename(path)}",
                data=f.read(),
                file_name=os.path.basename(path),
                mime="text/csv",
                key=f"dl_{version}",
            )

        # Freeze Ticker
        view = df.copy()
        if "Ticker" in view.columns:
            view = view.drop_duplicates(subset=["Ticker"], keep="first")
            view = view.set_index("Ticker")

        st.dataframe(view, width="stretch", height=480)
        st.caption(f"Total: {len(view)} baris • Kolom Ticker di-freeze di kiri")
    except Exception as e:
        st.error(f"Gagal baca: {e}")


# =====================================================================
# PANEL REZIM IHSG (on-demand saja)
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

# Placeholder jika belum pernah dicek
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
elif regime == "Belum dicek":
    st.info(f"**Rezim: {regime}**")
else:
    st.info(f"**Rezim: {regime}**")

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
    }
    label = mode
    started_at = datetime.now()

    with st.spinner(f"Menjalankan {label}... (1–3 menit)"):
        if mode.startswith("🤖"):
            ok, err, log = capture_run(run_ai, account_size, risk_pct)
            try:
                import master_screener_ai
                st.session_state["ihsg_regime"] = master_screener_ai.analyze_ihsg_regime()
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
        else:
            ok, err, log = False, "Mode tidak dikenal", ""

    finished_at = datetime.now()
    duration_sec = (finished_at - started_at).total_seconds()

    # Hitung jumlah baris hasil per versi (file terbaru)
    counts = {}
    for ver in ["v2", "v3", "v4", "v5"]:
        path = find_report(ver)
        n = 0
        if path and os.path.exists(path):
            try:
                n = len(pd.read_csv(path))
            except Exception:
                n = 0
        counts[ver] = n

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
    }

    if ok:
        st.success(f"✅ {label} selesai.")
    else:
        st.error(f"❌ {err}")

    with st.expander("📋 Log", expanded=not ok):
        st.text(log if log and log.strip() else "(kosong)")

    # Jangan st.rerun() langsung agar status & log sempat terbaca;
    # user bisa refresh manual atau klik tab hasil.
    # Jika ingin auto-refresh tabel saja, uncomment:
    # st.rerun()

# =====================================================================
# HASIL PER STRATEGI
# =====================================================================
st.subheader("📊 Hasil Screener per Strategi")

t2, t3, t4, t5 = st.tabs([
    "V2 Breakout",
    "V3 Retest Fibo",
    "V4 Order Block",
    "V5 CHOCH",
])

with t2:
    show_report("v2", "V2 Breakout")
with t3:
    show_report("v3", "V3 Retest Fibo")
with t4:
    show_report("v4", "V4 Order Block")
with t5:
    show_report("v5", "V5 CHOCH")

st.markdown("---")
st.caption("IDX Master Screener AI • Bukan rekomendasi investasi")