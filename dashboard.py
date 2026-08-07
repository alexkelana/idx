"""
IDX Master Screener AI — Streamlit Dashboard
- Panel rezim IHSG
- Jalankan AI adaptive / V2 / V3 / V4 / V5
- Tab hasil per strategi (CSV terpisah)
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

# ----- Sidebar -----
st.sidebar.header("⚙️ Modal & Risiko")
account_size = st.sidebar.number_input(
    "Total Modal (Rp)", min_value=1_000_000, value=50_000_000, step=1_000_000, format="%d"
)
risk_pct = st.sidebar.number_input(
    "Risiko per posisi (%)", min_value=0.1, max_value=10.0, value=1.0, step=0.1
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
    master_screener_ai.run_orchestrator(account_size=float(account_size), risk_pct=float(risk_pct))


# ----- Panel rezim IHSG -----
st.subheader("📡 Kondisi Pasar IHSG")
if st.button("🔍 Cek Rezim Sekarang", width="stretch") or "ihsg_regime" not in st.session_state:
    with st.spinner("Mengambil data IHSG..."):
        try:
            import master_screener_ai
            st.session_state["ihsg_regime"] = master_screener_ai.analyze_ihsg_regime()
        except Exception as e:
            st.session_state["ihsg_regime"] = {
                "regime": "UNKNOWN", "reason": [str(e)], "strategies": [], "last_close": None
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

# ----- Jalankan screener -----
if run_button:
    params = {"account_size": float(account_size), "risk_per_trade_pct": float(risk_pct)}
    label = mode
    with st.spinner(f"Menjalankan {label}..."):
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

    if ok:
        st.success(f"✅ {label} selesai.")
        st.rerun()
    else:
        st.error(f"❌ {err}")
    with st.expander("📋 Log", expanded=not ok):
        st.text(log or "(kosong)")

# ----- Hasil per versi -----
st.subheader("📊 Hasil Screener per Strategi")


def find_report(version: str):
    files = glob.glob(f"idx_report_{version}_*.csv")
    if not files:
        return None
    return max(files, key=os.path.getmtime)


def show_report(version: str, title: str):
    path = find_report(version)
    if not path:
        st.info(f"Belum ada data **{title}**. Jalankan screener {version.upper()} atau AI Adaptive.")
        return
    mtime = datetime.fromtimestamp(os.path.getmtime(path))
    st.caption(f"`{os.path.basename(path)}` • {mtime.strftime('%Y-%m-%d %H:%M')}")
    try:
        df = pd.read_csv(path)
        if df.empty:
            st.warning("File kosong.")
            return
        if "Ticker" in df.columns:
            df = df.drop_duplicates(subset=["Ticker"], keep="first").set_index("Ticker")
        st.dataframe(df, width="stretch", height=480)
        st.caption(f"Total: {len(df)} baris")
    except Exception as e:
        st.error(f"Gagal baca: {e}")


t2, t3, t4, t5 = st.tabs(["V2 Breakout", "V3 Retest Fibo", "V4 Order Block", "V5 CHOCH"])
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