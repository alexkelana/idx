"""
IDX Master Screener AI — Streamlit Dashboard
Membaca file terpisah: Klasik (V2/V3) dan SMC (V4/V5)
"""

import streamlit as st
import pandas as pd
import os
import glob
from datetime import datetime
import master_screener_ai
import io
import sys

st.set_page_config(
    page_title="IDX Master Screener Dashboard",
    page_icon="📈",
    layout="wide"
)

st.title("📈 IDX Master Screener AI Dashboard")
st.markdown(
    "Dashboard menganalisa IHSG dan menjalankan strategi yang relevan "
    "(Breakout, Retest Fibo, Order Block, atau CHOCH)."
)

# --- SIDEBAR ---
st.sidebar.header("⚙️ Pengaturan Modal & Risiko")
account_size = st.sidebar.number_input(
    "Total Modal Trading (Rp)",
    min_value=1_000_000,
    value=50_000_000,
    step=1_000_000,
    format="%d"
)
risk_pct = st.sidebar.number_input(
    "Toleransi Risiko per Posisi (%)",
    min_value=0.1,
    max_value=10.0,
    value=1.0,
    step=0.1
)

st.sidebar.markdown("---")
run_button = st.sidebar.button("🚀 JALANKAN SCREENER AI", use_container_width=True)


def find_latest_file(group: str) -> str | None:
    """Cari file terbaru untuk grup 'klasik' atau 'smc'."""
    patterns = [
        f"idx_master_report_{group}_*.csv",
        f"*{group}*.csv",
    ]
    search_dirs = [".", os.path.dirname(os.path.abspath(__file__)) if "__file__" in dir() else "."]

    files = []
    for d in search_dirs:
        if not os.path.isdir(d):
            continue
        for pat in patterns:
            files.extend(glob.glob(os.path.join(d, pat)))

    if not files:
        return None
    return max(files, key=os.path.getmtime)


# --- JALANKAN SCREENER ---
if run_button:
    with st.spinner("Memindai pasar dan menjalankan strategi... (1–3 menit)"):
        old_stdout = sys.stdout
        new_stdout = io.StringIO()
        sys.stdout = new_stdout
        success = False
        error_msg = None

        try:
            master_screener_ai.run_orchestrator(
                account_size=float(account_size),
                risk_pct=float(risk_pct)
            )
            success = True
        except Exception as e:
            error_msg = str(e)
        finally:
            sys.stdout = old_stdout
            log_output = new_stdout.getvalue()

        if success:
            st.success("✅ Screener berhasil dijalankan!")
            st.rerun()
        else:
            st.error(f"❌ Terjadi kesalahan: {error_msg}")

        with st.expander("📋 Lihat Log Terminal", expanded=not success):
            st.text(log_output if log_output.strip() else "(Tidak ada output log)")

st.markdown("---")
st.subheader("📊 Hasil Screener Hari Ini")

# --- BACA FILE TERPISAH ---
klasik_file = find_latest_file("klasik")
smc_file = find_latest_file("smc")

tab1, tab2 = st.tabs(["🔵 Klasik (Breakout / Fibo)", "🟣 Smart Money (OB / CHOCH)"])

with tab1:
    if klasik_file and os.path.exists(klasik_file):
        mtime = datetime.fromtimestamp(os.path.getmtime(klasik_file))
        st.caption(f"File: `{os.path.basename(klasik_file)}` • {mtime.strftime('%Y-%m-%d %H:%M')}")
        try:
            df = pd.read_csv(klasik_file)
            if not df.empty:
                st.dataframe(df.style.format(precision=2), use_container_width=True, height=500)
                st.caption(f"Total: {len(df)} setup")
            else:
                st.warning("File Klasik kosong.")
        except Exception as e:
            st.error(f"Gagal membaca file Klasik: {e}")
    else:
        st.info("Belum ada data Klasik (V2/V3). Jalankan Screener AI terlebih dahulu.")

with tab2:
    if smc_file and os.path.exists(smc_file):
        mtime = datetime.fromtimestamp(os.path.getmtime(smc_file))
        st.caption(f"File: `{os.path.basename(smc_file)}` • {mtime.strftime('%Y-%m-%d %H:%M')}")
        try:
            df = pd.read_csv(smc_file)
            if not df.empty:
                st.dataframe(df.style.format(precision=2), use_container_width=True, height=500)
                st.caption(f"Total: {len(df)} setup")
            else:
                st.warning("File SMC kosong.")
        except Exception as e:
            st.error(f"Gagal membaca file SMC: {e}")
    else:
        st.info("Belum ada data SMC (V4/V5). Jalankan Screener AI terlebih dahulu.")

st.markdown("---")
st.caption("IDX Master Screener AI • Data Yahoo Finance • Bukan rekomendasi investasi")