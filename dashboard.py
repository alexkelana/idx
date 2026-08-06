"""
IDX Master Screener AI — Streamlit Dashboard
- Jalankan semua (AI adaptive) atau per versi (V2/V3/V4/V5)
- Tab hasil: Klasik vs SMC
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
    "Analisa IHSG + jalankan strategi Breakout, Retest Fibo, Order Block, atau CHOCH."
)

# =====================================================================
# SIDEBAR
# =====================================================================
st.sidebar.header("⚙️ Pengaturan Modal & Risiko")
account_size = st.sidebar.number_input(
    "Total Modal Trading (Rp)",
    min_value=1_000_000,
    value=50_000_000,
    step=1_000_000,
    format="%d",
)
risk_pct = st.sidebar.number_input(
    "Toleransi Risiko per Posisi (%)",
    min_value=0.1,
    max_value=10.0,
    value=1.0,
    step=0.1,
)

st.sidebar.markdown("---")
st.sidebar.header("🚀 Jalankan Screener")

mode = st.sidebar.radio(
    "Pilih mode",
    [
        "🤖 AI Adaptive (otomatis pilih strategi)",
        "V2 — Breakout",
        "V3 — Retest Fibo",
        "V4 — Order Block (SMC)",
        "V5 — CHOCH (SMC)",
    ],
    index=0,
)

run_button = st.sidebar.button("▶️ JALANKAN", use_container_width=True, type="primary")


def capture_run(fn, *args, **kwargs):
    """Jalankan fungsi sambil tangkap print ke string."""
    old_stdout = sys.stdout
    buf = io.StringIO()
    sys.stdout = buf
    success = False
    error_msg = None
    try:
        fn(*args, **kwargs)
        success = True
    except Exception as e:
        error_msg = str(e)
    finally:
        sys.stdout = old_stdout
    return success, error_msg, buf.getvalue()


def run_v2(params: dict):
    mod = importlib.import_module("idx_breakout_screener_v2")
    if hasattr(mod, "PARAMS") and isinstance(mod.PARAMS, dict):
        mod.PARAMS.update(params)
    mod.run_screener(params=getattr(mod, "PARAMS", params))


def run_v3(params: dict):
    mod = importlib.import_module("idx_breakout_screener_v3")
    if hasattr(mod, "PARAMS") and isinstance(mod.PARAMS, dict):
        mod.PARAMS.update(params)
    mod.run_screener(params=getattr(mod, "PARAMS", params))


def run_v4(params: dict):
    mod = importlib.import_module("idx_breakout_screener_v4_smc")
    mod.run_screener_v4(user_params=params)


def run_v5(params: dict):
    mod = importlib.import_module("idx_breakout_screener_v5_smc")
    mod.run_screener_v5(user_params=params)


def run_ai_adaptive(account_size: float, risk_pct: float):
    import master_screener_ai
    master_screener_ai.run_orchestrator(
        account_size=float(account_size),
        risk_pct=float(risk_pct),
    )


# =====================================================================
# EKSEKUSI
# =====================================================================
if run_button:
    user_params = {
        "account_size": float(account_size),
        "risk_per_trade_pct": float(risk_pct),
    }

    label = mode.split("—")[0].strip() if "—" in mode else mode
    with st.spinner(f"Menjalankan {label}... Mohon tunggu (1–3 menit)."):
        if mode.startswith("🤖"):
            success, err, log = capture_run(run_ai_adaptive, account_size, risk_pct)
        elif mode.startswith("V2"):
            success, err, log = capture_run(run_v2, user_params)
        elif mode.startswith("V3"):
            success, err, log = capture_run(run_v3, user_params)
        elif mode.startswith("V4"):
            success, err, log = capture_run(run_v4, user_params)
        elif mode.startswith("V5"):
            success, err, log = capture_run(run_v5, user_params)
        else:
            success, err, log = False, "Mode tidak dikenal", ""

    if success:
        st.success(f"✅ {label} berhasil dijalankan!")
        st.rerun()
    else:
        st.error(f"❌ Gagal: {err}")

    with st.expander("📋 Log Terminal", expanded=not success):
        st.text(log if log.strip() else "(Tidak ada output)")

st.markdown("---")

# =====================================================================
# HASIL
# =====================================================================
st.subheader("📊 Hasil Screener")


def find_latest_file(group: str):
    """Cari file report terbaru untuk 'klasik' atau 'smc'."""
    patterns = [
        f"idx_master_report_{group}_*.csv",
        f"idx_master_report_{group.upper()}_*.csv",
        f"*_{group}_*.csv",
    ]
    # Fallback nama lama
    if group == "klasik":
        patterns += ["idx_master_report_20*.csv"]
    if group == "smc":
        patterns += ["idx_master_report_SMC_*.csv", "*smc*.csv"]

    files = []
    search_dirs = [
        ".",
        os.path.dirname(os.path.abspath(__file__)) if "__file__" in dir() else ".",
    ]
    for d in search_dirs:
        if not os.path.isdir(d):
            continue
        for pat in patterns:
            files.extend(glob.glob(os.path.join(d, pat)))

    # Hindari ambil file smc saat cari klasik
    if group == "klasik":
        files = [f for f in files if "smc" not in os.path.basename(f).lower()]
    if group == "smc":
        files = [f for f in files if "klasik" not in os.path.basename(f).lower()]

    if not files:
        return None
    return max(files, key=os.path.getmtime)


klasik_file = find_latest_file("klasik")
smc_file = find_latest_file("smc")

tab1, tab2, tab3 = st.tabs([
    "🔵 Klasik (V2 / V3)",
    "🟣 SMC (V4 / V5)",
    "ℹ️ Info Mode",
])

with tab1:
    if klasik_file and os.path.exists(klasik_file):
        mtime = datetime.fromtimestamp(os.path.getmtime(klasik_file))
        st.caption(f"`{os.path.basename(klasik_file)}` • {mtime.strftime('%Y-%m-%d %H:%M')}")
        try:
            df = pd.read_csv(klasik_file)
            if not df.empty:
                st.dataframe(df, use_container_width=True, height=480)
                st.caption(f"Total: {len(df)} baris")
            else:
                st.warning("File Klasik kosong.")
        except Exception as e:
            st.error(f"Gagal baca file: {e}")
    else:
        st.info("Belum ada data Klasik. Jalankan V2, V3, atau AI Adaptive.")

with tab2:
    if smc_file and os.path.exists(smc_file):
        mtime = datetime.fromtimestamp(os.path.getmtime(smc_file))
        st.caption(f"`{os.path.basename(smc_file)}` • {mtime.strftime('%Y-%m-%d %H:%M')}")
        try:
            df = pd.read_csv(smc_file)
            if not df.empty:
                st.dataframe(df, use_container_width=True, height=480)
                st.caption(f"Total: {len(df)} baris")
            else:
                st.warning("File SMC kosong.")
        except Exception as e:
            st.error(f"Gagal baca file: {e}")
    else:
        st.info("Belum ada data SMC. Jalankan V4, V5, atau AI Adaptive.")

with tab3:
    st.markdown("""
**Mode AI Adaptive**  
Menganalisa rezim IHSG lalu otomatis menjalankan strategi yang sesuai.

| Mode | Strategi | File hasil |
|------|----------|------------|
| **V2** | Breakout | `idx_master_report_klasik_*.csv` |
| **V3** | Retest Fibo | `idx_master_report_klasik_*.csv` |
| **V4** | Order Block SMC | `idx_master_report_smc_*.csv` |
| **V5** | CHOCH SMC | `idx_master_report_smc_*.csv` |

Gunakan modal & risiko di sidebar. Proses biasanya 1–3 menit per strategi.
""")

st.markdown("---")
st.caption("IDX Master Screener AI • Bukan rekomendasi investasi")