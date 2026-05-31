"""Run History page — browse and compare past backtest reports."""
from __future__ import annotations

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import streamlit as st
import pandas as pd

from src.config import load_config
from dashboard.run_history import scan_reports, read_report

# ── Page config ───────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Run History",
    page_icon="📜",
    layout="wide",
)

# ── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.title("📊 Forecast Research")
    st.caption("Daily commodity ranking system")
    st.divider()
    refresh = st.button("🔄 Refresh list")
    st.divider()
    st.caption("⚠️ Research tooling — not investment advice.")

# ── Cached loaders ────────────────────────────────────────────────────────────

@st.cache_data
def _load_cfg():
    return load_config()

@st.cache_data(ttl=60)
def _scan(_reports_dir: str) -> list[dict]:
    return scan_reports(Path(_reports_dir))

# ── Main ──────────────────────────────────────────────────────────────────────

st.title("📜 Run History")

cfg = _load_cfg()
reports_dir = cfg.paths.outputs_reports

if refresh:
    st.cache_data.clear()

reports = _scan(str(reports_dir))

if not reports:
    st.info(
        f"No reports found in `{reports_dir}`. "
        "Run a backtest on the **Run Experiment** page and click **Save run report**."
    )
    st.stop()

# ── Report index table ────────────────────────────────────────────────────────

st.markdown(f"### {len(reports)} reports found in `outputs/reports/`")

table_rows = [
    {
        "Filename": r["filename"],
        "Date": r["date"] or "—",
        "Phase": r["phase"] or "—",
        "Horizon": r["horizon"] or "—",
        "Title": r["title"][:60],
        "Size (KB)": r["size_kb"],
    }
    for r in reports
]
df_table = pd.DataFrame(table_rows)
st.dataframe(df_table, use_container_width=True, height=220)

# ── Single report viewer ──────────────────────────────────────────────────────

st.divider()
st.markdown("### View report")

filenames = [r["filename"] for r in reports]
selected_file = st.selectbox("Select report", filenames)

if selected_file:
    selected_meta = next(r for r in reports if r["filename"] == selected_file)
    content = read_report(selected_meta["path"])
    with st.expander(f"📄 {selected_file}", expanded=True):
        st.markdown(content)

# ── Side-by-side comparison ───────────────────────────────────────────────────

st.divider()
st.markdown("### Compare two reports side by side")

col_a, col_b = st.columns(2)
with col_a:
    file_a = st.selectbox("Report A", filenames, key="cmp_a")
with col_b:
    file_b = st.selectbox("Report B", filenames,
                          index=min(1, len(filenames) - 1), key="cmp_b")

if file_a and file_b:
    meta_a = next(r for r in reports if r["filename"] == file_a)
    meta_b = next(r for r in reports if r["filename"] == file_b)
    c1, c2 = st.columns(2)
    with c1:
        st.markdown(f"**{file_a}**")
        st.markdown(read_report(meta_a["path"]))
    with c2:
        st.markdown(f"**{file_b}**")
        st.markdown(read_report(meta_b["path"]))
