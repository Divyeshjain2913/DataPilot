"""
DataPilot - Smart Data Analyzer
High-Precision Enterprise Analytics Platform (Fluxo / FinTech Edition)
"""

from __future__ import annotations
import os
import streamlit as st
import pandas as pd
import numpy as np

# Internal modules
from utils.helpers import (
    format_bytes,
    format_compact_number,
    format_percentage,
    generate_sparkline_svg,
    generate_minibars_svg,
    dataframe_to_csv_bytes,
    dataframe_to_excel_bytes,
    export_quality_report_json,
    generate_executive_markdown_report,
    inject_custom_css,
    render_html
)
from modules.data_loader import (
    load_file,
    get_sample_datasets_info,
    load_sample_dataset
)
from modules.data_profiler import profile_dataset
from modules.data_cleaner import (
    inspect_cleaning_candidates,
    apply_cleaning_pipeline
)
from modules.data_analyzer import (
    compute_numerical_summary,
    compute_categorical_summary,
    compute_correlation_matrix,
    generate_automated_insights
)
from modules.visualization import (
    create_chart,
    build_archetype_dashboard,
    ENTERPRISE_PALETTE
)

# -----------------------------------------------------------------------------
# PAGE CONFIGURATION
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="DataPilot | Enterprise Analytics Platform",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Inject High-Precision FinTech Dark CSS
inject_custom_css()

# -----------------------------------------------------------------------------
# SESSION STATE MANAGEMENT
# -----------------------------------------------------------------------------
if "active_page" not in st.session_state:
    st.session_state["active_page"] = "Dashboard"

if "df" not in st.session_state:
    st.session_state["df"] = None

if "raw_df" not in st.session_state:
    st.session_state["raw_df"] = None

if "dataset_meta" not in st.session_state:
    st.session_state["dataset_meta"] = None

if "profile" not in st.session_state:
    st.session_state["profile"] = None

if "insights" not in st.session_state:
    st.session_state["insights"] = None

if "cleaning_diff" not in st.session_state:
    st.session_state["cleaning_diff"] = None


def set_active_dataset(df: pd.DataFrame, meta: dict):
    st.session_state["raw_df"] = df.copy()
    st.session_state["df"] = df.copy()
    st.session_state["dataset_meta"] = meta
    st.session_state["profile"] = profile_dataset(df)
    st.session_state["insights"] = generate_automated_insights(df, st.session_state["profile"])
    st.session_state["cleaning_diff"] = None


# -----------------------------------------------------------------------------
# GLOBAL SIDEBAR (Grouped Navigation, Zero Emojis, No Ligature Bugs)
# -----------------------------------------------------------------------------
with st.sidebar:
    render_html("""
    <div class="brand-container">
        <div class="brand-icon-box">
            <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round">
                <polygon points="12 2 2 7 12 12 22 7 12 2"></polygon>
                <polyline points="2 17 12 22 22 17"></polyline>
                <polyline points="2 12 12 17 22 12"></polyline>
            </svg>
        </div>
        <div>
            <h1 class="brand-title">DataPilot</h1>
            <p class="brand-subtitle">Enterprise Data Platform</p>
        </div>
    </div>
    """)

    nav_sections = [
        {
            "title": "WORKSPACE",
            "pages": [
                ("Dashboard", "Dashboard"),
                ("Upload Dataset", "Upload Dataset"),
                ("Data Preview", "Data Preview"),
            ]
        },
        {
            "title": "DATA ANALYSIS",
            "pages": [
                ("Data Cleaning", "Data Cleaning"),
                ("Data Quality", "Data Quality"),
                ("Exploratory Analysis", "Exploratory Analysis"),
                ("Visualizations", "Visualizations"),
                ("AI Insights", "AI Insights"),
            ]
        },
        {
            "title": "REPORTING",
            "pages": [
                ("Export Reports", "Export Reports"),
            ]
        },
        {
            "title": "SETTINGS",
            "pages": [
                ("Settings", "Settings"),
            ]
        }
    ]

    for sec in nav_sections:
        render_html(f'<div class="nav-section-title">{sec["title"]}</div>')
        for label, p_key in sec["pages"]:
            is_active = (st.session_state["active_page"] == p_key)
            if st.button(
                label,
                key=f"nav_{p_key}",
                use_container_width=True,
                type="primary" if is_active else "secondary"
            ):
                if not is_active:
                    st.session_state["active_page"] = p_key
                    st.rerun()

    # Dataset Status Card near bottom
    if st.session_state["df"] is not None:
        meta = st.session_state["dataset_meta"] or {}
        fname = meta.get("filename", "Active Dataset")
        rows = len(st.session_state["df"])
        cols = len(st.session_state["df"].columns)
        score = st.session_state["profile"].get("quality_score", 0) if st.session_state["profile"] else 0

        render_html(f"""
        <div class="dataset-status-box">
            <div style="display: flex; align-items: center; justify-content: space-between;">
                <span style="font-size: 10px; font-weight: 700; text-transform: uppercase; letter-spacing: 0.06em; color: var(--text-muted);">ACTIVE DATASET</span>
                <span class="status-pill live"><span class="status-dot"></span>Live</span>
            </div>
            <div class="dataset-status-title" title="{fname}">{fname}</div>
            <div class="dataset-status-meta">{rows:,} rows • {cols} cols • {score:.0f}% Q-Score</div>
        </div>
        """)
        if st.button("Unload Dataset", key="btn_unload_ds", use_container_width=True):
            st.session_state["df"] = None
            st.session_state["raw_df"] = None
            st.session_state["dataset_meta"] = None
            st.session_state["profile"] = None
            st.session_state["insights"] = None
            st.session_state["cleaning_diff"] = None
            st.session_state["active_page"] = "Dashboard"
            st.rerun()
    else:
        render_html("""
        <div class="dataset-status-box" style="text-align: center; padding: 12px 10px;">
            <span style="font-size: 11px; font-weight: 700; text-transform: uppercase; color: var(--text-muted); display: block;">NO DATASET ACTIVE</span>
            <span style="font-size: 11px; color: var(--text-muted); margin-top: 2px; display: block;">Awaiting file ingestion</span>
        </div>
        """)

    # Bottom Engine Box (Matching reference bottom-left orange/accent card)
    render_html("""
    <div class="sidebar-engine-card">
        <div class="sidebar-engine-title">
            <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round">
                <circle cx="12" cy="12" r="10"></circle>
                <polygon points="12 8 8 12 12 16 12 8"></polygon>
            </svg>
            Enterprise Engine
        </div>
        <p class="sidebar-engine-desc">2,048 MB capacity pipeline with in-memory execution and zero data persistence.</p>
    </div>
    """)


# -----------------------------------------------------------------------------
# GLOBAL TOP HEADER BAR (Matching reference header layout)
# -----------------------------------------------------------------------------
has_data = st.session_state["df"] is not None
active_page = st.session_state["active_page"]
dataset_name = st.session_state["dataset_meta"].get("filename", "") if has_data else ""

top_nav_c1, top_nav_c2 = st.columns([5, 1])

with top_nav_c1:
    if has_data:
        render_html(f"""
        <div class="top-navbar-container">
            <div class="top-navbar-left">
                <span class="top-navbar-crumb-root">Workspace</span>
                <span class="top-navbar-crumb-sep">/</span>
                <span class="top-navbar-crumb-current">{active_page}</span>
                <span class="top-navbar-crumb-file">{dataset_name}</span>
            </div>
            <div class="top-navbar-right">
                <span class="status-pill live"><span class="status-dot"></span>Active Engine</span>
                <span class="status-pill primary"><span class="status-dot"></span>In-Memory</span>
                <span class="status-pill neutral">2GB Limit</span>
            </div>
        </div>
        """)
    else:
        render_html(f"""
        <div class="top-navbar-container">
            <div class="top-navbar-left">
                <span class="top-navbar-crumb-root">Workspace</span>
                <span class="top-navbar-crumb-sep">/</span>
                <span class="top-navbar-crumb-current">{active_page}</span>
            </div>
            <div class="top-navbar-right">
                <span class="status-pill empty"><span class="status-dot"></span>No Dataset</span>
                <span class="status-pill neutral">Local Mode</span>
            </div>
        </div>
        """)

with top_nav_c2:
    if has_data:
        if st.button("Switch Dataset", key="top_switch_btn", use_container_width=True):
            st.session_state["active_page"] = "Upload Dataset"
            st.rerun()
    else:
        if st.button("Upload Dataset", key="top_upload_btn", type="primary", use_container_width=True):
            st.session_state["active_page"] = "Upload Dataset"
            st.rerun()


# =============================================================================
# PAGE 1: DASHBOARD (MATCHING FLUXO / FINTECH REFERENCE ARCHITECTURE)
# =============================================================================
if active_page == "Dashboard":
    if not has_data:
        render_html("""
        <div style="margin-bottom: 22px;">
            <h2 style="font-size: 24px; font-weight: 700; color: #FFFFFF; letter-spacing: -0.02em; margin: 0 0 4px 0;">Workspace Overview</h2>
            <p style="font-size: 13px; color: var(--text-secondary); margin: 0;">Automated data profiling, cleaning pipeline, and interactive exploratory intelligence.</p>
        </div>
        """)

        # Clean Onboarding Hero
        render_html("""
        <div class="ent-box" style="margin-bottom: 22px; padding: 22px 24px;">
            <div style="display: flex; align-items: center; justify-content: space-between;">
                <div>
                    <h3 style="font-size: 16px; font-weight: 700; color: #FFFFFF; margin: 0 0 6px 0;">Ready to Analyze Your Dataset</h3>
                    <p style="font-size: 13px; color: var(--text-secondary); margin: 0; max-width: 620px;">
                        Upload any CSV, Excel, or TSV dataset up to 2GB, or load a pre-configured enterprise benchmark below to explore the analytical capabilities.
                    </p>
                </div>
            </div>
        </div>
        """)

        render_html('<div style="font-size: 14px; font-weight: 600; color: #FFFFFF; margin-bottom: 12px; letter-spacing: -0.01em;">Enterprise Benchmark Datasets</div>')

        samples = get_sample_datasets_info()
        s_cols = st.columns(3)

        for idx, (skey, sinfo) in enumerate(samples.items()):
            with s_cols[idx]:
                render_html(f"""
                <div class="sample-box">
                    <div>
                        <div style="display: flex; align-items: center; justify-content: space-between; margin-bottom: 8px;">
                            <span class="sample-box-badge">{sinfo['badge']}</span>
                            <span class="sample-box-meta">{sinfo['records']}</span>
                        </div>
                        <h4 class="sample-box-title">{sinfo['name']}</h4>
                        <p class="sample-box-desc">{sinfo['description']}</p>
                    </div>
                </div>
                """)
                if st.button("Load Dataset", key=f"btn_demo_{skey}", use_container_width=True):
                    with st.spinner("Ingesting dataset..."):
                        d_df, d_meta = load_sample_dataset(skey)
                        set_active_dataset(d_df, d_meta)
                        st.rerun()

    else:
        df = st.session_state["df"]
        profile = st.session_state["profile"]
        archetype = profile.get("archetype", "general")
        dash_data = build_archetype_dashboard(df, archetype=archetype, theme="dark")
        kpis = dash_data.get("kpis", [])

        # TOP GREETING & WORKSPACE BAR (Reference: "Good morning, Alex" style)
        render_html("""
        <div style="display: flex; align-items: center; justify-content: space-between; margin-bottom: 18px;">
            <div>
                <h2 style="font-size: 24px; font-weight: 700; color: #FFFFFF; letter-spacing: -0.02em; margin: 0 0 2px 0;">Executive Performance Overview</h2>
                <p style="font-size: 12.5px; color: var(--text-secondary); margin: 0;">Real-time analytical telemetry and feature distributions.</p>
            </div>
            <div style="display: flex; gap: 8px;">
                <span class="status-pill primary">Telemetry Active</span>
                <span class="status-pill live">All Systems Nominal</span>
            </div>
        </div>
        """)

        # TOP 4 KPI CARDS WITH EMBEDDED MINI SPARKLINES (Reference Layout)
        kpi_cols = st.columns(len(kpis))
        for idx, kpi in enumerate(kpis):
            with kpi_cols[idx]:
                spark_vals = kpi.get("spark", [10, 15, 20, 25])
                color = kpi.get("color", "#38BDF8")
                is_bar = kpi.get("is_bar", False)
                chart_svg = generate_minibars_svg(spark_vals, color=color, height=38, width=170) if is_bar else generate_sparkline_svg(spark_vals, color=color, height=38, width=170)

                render_html(f"""
                <div class="ent-kpi-card">
                    <div>
                        <div class="ent-kpi-top">
                            <span class="ent-kpi-label">{kpi['label']}</span>
                            <span class="status-pill {'live' if idx != 2 else 'primary'}">{kpi.get('sub', '')}</span>
                        </div>
                        <div class="ent-kpi-value">{kpi['value']}</div>
                    </div>
                    <div class="ent-kpi-chart">
                        {chart_svg}
                    </div>
                </div>
                """)

        st.markdown("<div style='height: 6px;'></div>", unsafe_allow_html=True)

        # MIDDLE ANALYTICS GRID: 65% / 35% SPLIT (Reference Layout)
        charts = dash_data.get("charts", [])
        main_chart = next((c for c in charts if c.get("type") == "main"), (charts[0] if charts else None))
        sec_chart = next((c for c in charts if c.get("type") == "secondary"), (charts[1] if len(charts) > 1 else None))

        mid_c1, mid_c2 = st.columns([65, 35])

        with mid_c1:
            if main_chart:
                st.plotly_chart(main_chart["fig"], use_container_width=True)

        with mid_c2:
            if sec_chart:
                st.plotly_chart(sec_chart["fig"], use_container_width=True)

        # BOTTOM GRID: DATA PROFILE STREAM (55%) + QUALITY PILLARS / GOALS (45%) (Reference Layout)
        bot_c1, bot_c2 = st.columns([55, 45])

        with bot_c1:
            table_rows = []
            for col_info in profile.get("columns", [])[:5]:
                non_null_rate = col_info["non_null_pct"]
                status_badge = "live" if non_null_rate >= 95 else ("warning" if non_null_rate >= 80 else "danger")
                status_text = "Optimal" if non_null_rate >= 95 else "Degraded"
                table_rows.append(f"""
                <tr>
                    <td style="font-weight: 600;">{col_info['name']}</td>
                    <td style="color: var(--text-secondary);">{col_info['inferred_type'].capitalize()}</td>
                    <td>{non_null_rate:.1f}%</td>
                    <td style="font-family: var(--font-mono);">{col_info['unique_count']:,}</td>
                    <td><span class="status-pill {status_badge}">{status_text}</span></td>
                </tr>
                """)

            full_table_html = f"""
            <div class="ent-box">
                <div class="ent-box-header">
                    <h4 class="ent-box-title">
                        <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="#38BDF8" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round">
                            <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"></path>
                            <polyline points="14 2 14 8 20 8"></polyline>
                        </svg>
                        Active Feature Telemetry
                    </h4>
                    <span class="status-pill neutral">Top Columns</span>
                </div>
                <table class="ent-table">
                    <thead>
                        <tr>
                            <th>Feature</th>
                            <th>Type</th>
                            <th>Completeness</th>
                            <th>Unique Values</th>
                            <th>Status</th>
                        </tr>
                    </thead>
                    <tbody>
                        {''.join(table_rows)}
                    </tbody>
                </table>
            </div>
            """
            render_html(full_table_html)

        with bot_c2:
            q_score = profile.get("quality_score", 0)
            q_grade = profile.get("quality_grade", "N/A")
            comp_rate = profile.get("completeness_pct", 100)
            uniq_rate = profile.get("uniqueness_pct", 100)
            valid_rate = profile.get("validity_pct", 100)
            cons_rate = profile.get("consistency_pct", 100)

            full_scorecard_html = f"""
            <div class="ent-box">
                <div class="ent-box-header">
                    <h4 class="ent-box-title">
                        <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="#10B981" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round">
                            <path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"></path>
                        </svg>
                        Quality Scorecard: {q_score:.1f}%
                    </h4>
                    <span class="status-pill {'live' if q_score >= 85 else 'warning'}">Grade {q_grade}</span>
                </div>
                <div class="goal-item">
                    <div class="goal-header">
                        <span class="goal-label">Completeness Health</span>
                        <span class="goal-val">{comp_rate:.1f}%</span>
                    </div>
                    <div class="goal-track"><div class="goal-bar" style="width: {comp_rate}%; background-color: #38BDF8;"></div></div>
                </div>
                <div class="goal-item">
                    <div class="goal-header">
                        <span class="goal-label">Record Uniqueness</span>
                        <span class="goal-val">{uniq_rate:.1f}%</span>
                    </div>
                    <div class="goal-track"><div class="goal-bar" style="width: {uniq_rate}%; background-color: #818CF8;"></div></div>
                </div>
                <div class="goal-item">
                    <div class="goal-header">
                        <span class="goal-label">Distribution Validity</span>
                        <span class="goal-val">{valid_rate:.1f}%</span>
                    </div>
                    <div class="goal-track"><div class="goal-bar" style="width: {valid_rate}%; background-color: #00E5FF;"></div></div>
                </div>
                <div class="goal-item" style="margin-bottom: 0;">
                    <div class="goal-header">
                        <span class="goal-label">Schema Consistency</span>
                        <span class="goal-val">{cons_rate:.1f}%</span>
                    </div>
                    <div class="goal-track"><div class="goal-bar" style="width: {cons_rate}%; background-color: #10B981;"></div></div>
                </div>
            </div>
            """
            render_html(full_scorecard_html)


# =============================================================================
# PAGE 2: UPLOAD DATASET
# =============================================================================
elif active_page == "Upload Dataset":
    render_html("""
    <div style="margin-bottom: 20px;">
        <h2 style="font-size: 24px; font-weight: 700; color: #FFFFFF; letter-spacing: -0.02em; margin: 0 0 4px 0;">Dataset Ingestion Pipeline</h2>
        <p style="font-size: 13px; color: var(--text-secondary); margin: 0;">Ingest high-volume CSV, Excel, or TSV data streams up to 2GB with multi-encoding detection.</p>
    </div>
    """)

    up_c1, up_c2 = st.columns([3, 2])

    with up_c1:
        render_html("""
        <div class="ent-box">
            <div class="ent-box-header">
                <h3 class="ent-box-title">Upload Local Data Stream</h3>
                <span class="status-pill neutral">Max 2,048 MB</span>
            </div>
            <p class="ent-box-desc" style="margin-bottom: 14px;">Drag and drop or browse to ingest enterprise datasets.</p>
        </div>
        """)

        uploaded = st.file_uploader(
            "Upload file",
            type=["csv", "xlsx", "xls", "tsv", "txt"],
            label_visibility="collapsed"
        )

        if uploaded is not None:
            with st.spinner("Ingesting and profiling dataset stream..."):
                try:
                    loaded_df, loaded_meta = load_file(uploaded)
                    set_active_dataset(loaded_df, loaded_meta)
                    st.success(f"Ingested {loaded_meta['filename']} ({loaded_meta['rows']:,} rows, {loaded_meta['columns']} columns) in {loaded_meta['load_time_sec']}s")
                except Exception as e:
                    st.error(f"Ingestion failed: {e}")

    with up_c2:
        render_html("""
        <div class="ent-box">
            <div class="ent-box-header">
                <h3 class="ent-box-title">Industry Benchmarks</h3>
                <span class="status-pill neutral">One-Click Load</span>
            </div>
            <p class="ent-box-desc" style="margin-bottom: 12px;">Standard sample datasets for immediate system verification.</p>
        </div>
        """)

        samples = get_sample_datasets_info()
        for skey, sinfo in samples.items():
            s_c1, s_c2 = st.columns([3, 1])
            with s_c1:
                render_html(f"""
                <div style="padding: 6px 0;">
                    <div style="font-size: 13px; font-weight: 600; color: #FFFFFF;">{sinfo['name']}</div>
                    <div style="font-size: 11.5px; color: var(--text-muted);">{sinfo['records']} • {sinfo['badge']}</div>
                </div>
                """)
            with s_c2:
                if st.button("Load", key=f"btn_up_{skey}", use_container_width=True):
                    with st.spinner("Loading demo dataset..."):
                        d_df, d_meta = load_sample_dataset(skey)
                        set_active_dataset(d_df, d_meta)
                        st.session_state["active_page"] = "Data Preview"
                        st.rerun()

    if has_data:
        meta = st.session_state["dataset_meta"] or {}
        st.markdown("<hr style='border: none; border-top: 1px solid var(--border-subtle); margin: 24px 0;'>", unsafe_allow_html=True)
        render_html('<div style="font-size: 14px; font-weight: 600; color: #FFFFFF; margin-bottom: 12px;">Current Ingestion Telemetry</div>')

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Total Records", f"{meta.get('rows', 0):,}")
        m2.metric("Total Features", meta.get("columns", 0))
        m3.metric("Memory Footprint", meta.get("memory_usage_str", "N/A"))
        m4.metric("Ingestion Latency", f"{meta.get('load_time_sec', 0)}s")


# =============================================================================
# PAGE 3: DATA PREVIEW
# =============================================================================
elif active_page == "Data Preview":
    render_html("""
    <div style="margin-bottom: 20px;">
        <h2 style="font-size: 24px; font-weight: 700; color: #FFFFFF; letter-spacing: -0.02em; margin: 0 0 4px 0;">Data Preview & Inspection</h2>
        <p style="font-size: 13px; color: var(--text-secondary); margin: 0;">Interactive spreadsheet table with dynamic search, filtering, and schema profiling.</p>
    </div>
    """)

    if not has_data:
        st.info("No active dataset. Ingest a dataset first to preview records.")
    else:
        df = st.session_state["df"]
        profile = st.session_state["profile"]

        ctrl_c1, ctrl_c2, ctrl_c3 = st.columns([2, 2, 1])
        with ctrl_c1:
            search_term = st.text_input("Filter records", placeholder="Search within table...")
        with ctrl_c2:
            visible_cols = st.multiselect("Columns", options=list(df.columns), default=list(df.columns))
        with ctrl_c3:
            row_limit = st.selectbox("Display Limit", [10, 20, 50, 100], index=1)

        preview_data = df[visible_cols] if visible_cols else df

        if search_term:
            mask = np.column_stack([
                preview_data[col].astype(str).str.contains(search_term, case=False, na=False)
                for col in preview_data.columns
            ]).any(axis=1)
            preview_data = preview_data[mask]
            st.caption(f"{len(preview_data):,} matching record(s) found.")

        st.dataframe(preview_data.head(row_limit), use_container_width=True, height=440)

        render_html('<div style="font-size: 14px; font-weight: 600; color: #FFFFFF; margin: 20px 0 10px 0;">Inferred Column Taxonomy</div>')
        pills = []
        for c in profile.get("columns", []):
            itype = c["inferred_type"]
            pill_type = "primary" if itype == "numeric" else ("live" if itype == "datetime" else "neutral")
            pills.append(
                f'<span class="status-pill {pill_type}" style="margin: 3px;" title="Native: {c["native_dtype"]} | Missing: {c["null_pct"]}%">'
                f'{c["name"]} ({itype})'
                f'</span>'
            )
        render_html(f'<div style="line-height: 2;">{"".join(pills)}</div>')


# =============================================================================
# PAGE 4: DATA CLEANING
# =============================================================================
elif active_page == "Data Cleaning":
    render_html("""
    <div style="margin-bottom: 20px;">
        <h2 style="font-size: 24px; font-weight: 700; color: #FFFFFF; letter-spacing: -0.02em; margin: 0 0 4px 0;">Data Cleaning Engine</h2>
        <p style="font-size: 13px; color: var(--text-secondary); margin: 0;">Configure precision cleaning rules with dry-run simulation diffs and instant rollback.</p>
    </div>
    """)

    if not has_data:
        st.info("No active dataset. Ingest a dataset first to execute data cleaning.")
    else:
        df = st.session_state["df"]
        raw_df = st.session_state["raw_df"]
        issues = inspect_cleaning_candidates(df)

        if st.session_state["cleaning_diff"]:
            diff = st.session_state["cleaning_diff"]
            st.success(
                f"Cleaning executed: {diff['rows_before']:,} → {diff['rows_after']:,} rows | "
                f"{diff['nulls_before']:,} → {diff['nulls_after']:,} missing cells | "
                f"{diff['dups_removed']:,} duplicates dropped."
            )

        clean_options: dict = {}

        # 1. Missing Values
        mv = issues.get("missing_values", {})
        mv_class = "has-danger" if mv["status"] == "severe" else ("has-warning" if mv["status"] == "warning" else "is-clean")
        render_html(f"""
        <div class="ent-box" style="margin-bottom: 12px;">
            <div style="display: flex; align-items: center; justify-content: space-between; margin-bottom: 6px;">
                <span style="font-size: 13.5px; font-weight: 600; color: #FFFFFF;">Missing Values</span>
                <span class="status-pill {'danger' if mv['status'] == 'severe' else ('warning' if mv['status'] == 'warning' else 'live')}">
                    {mv['status'].upper()} • {mv['count']:,} CELLS
                </span>
            </div>
            <div style="font-size: 12.5px; color: var(--text-secondary); line-height: 1.45;">{mv['details']}</div>
        </div>
        """)
        if mv["count"] > 0:
            c1, c2 = st.columns(2)
            with c1:
                clean_options["missing_strategy"] = st.selectbox(
                    "Missing Values Strategy",
                    options=[
                        ("none", "Leave Unchanged"),
                        ("impute_smart", "Impute (Numeric: Median, Categorical: Mode)"),
                        ("drop_rows", "Drop Incomplete Rows"),
                        ("constant", "Fill with Constant (0 / 'Unknown')"),
                        ("ffill", "Forward & Backward Fill")
                    ],
                    format_func=lambda x: x[1],
                    key="sel_mv_strat"
                )[0]
            with c2:
                if clean_options["missing_strategy"] == "impute_smart":
                    clean_options["numeric_impute_method"] = st.radio("Numeric Imputation Metric", ["median", "mean"], horizontal=True, key="rad_num_imp")

        # 2. Duplicate Records
        dp = issues.get("duplicates", {})
        render_html(f"""
        <div class="ent-box" style="margin-bottom: 12px;">
            <div style="display: flex; align-items: center; justify-content: space-between; margin-bottom: 6px;">
                <span style="font-size: 13.5px; font-weight: 600; color: #FFFFFF;">Duplicate Records</span>
                <span class="status-pill {'danger' if dp['status'] == 'severe' else ('warning' if dp['status'] == 'warning' else 'live')}">
                    {dp['status'].upper()} • {dp['count']:,} ROWS
                </span>
            </div>
            <div style="font-size: 12.5px; color: var(--text-secondary); line-height: 1.45;">{dp['details']}</div>
        </div>
        """)
        if dp["count"] > 0:
            c1, c2 = st.columns(2)
            with c1:
                clean_options["remove_duplicates"] = st.checkbox("Deduplicate dataset", value=True, key="chk_rem_dup")
            with c2:
                clean_options["duplicate_action"] = st.selectbox("Deduplication Rule", ["first", "last", "all"], format_func=lambda x: f"Keep {x.capitalize()}", key="sel_dup_act")

        # 3. Type Standardization
        it = issues.get("incorrect_types", {})
        render_html(f"""
        <div class="ent-box" style="margin-bottom: 12px;">
            <div style="display: flex; align-items: center; justify-content: space-between; margin-bottom: 6px;">
                <span style="font-size: 13.5px; font-weight: 600; color: #FFFFFF;">Data Types & Formatting</span>
                <span class="status-pill {'warning' if it['status'] == 'warning' else 'live'}">
                    {it['status'].upper()} • {it['count']} COLUMNS
                </span>
            </div>
            <div style="font-size: 12.5px; color: var(--text-secondary); line-height: 1.45;">{it['details']}</div>
        </div>
        """)
        if it["count"] > 0:
            clean_options["convert_numeric"] = st.checkbox("Standardize eligible string columns to numeric", value=True, key="chk_conv_num")
            clean_options["numeric_columns"] = list(it.get("columns", {}).keys())

        # 4. Whitespace Trimming
        ws = issues.get("extra_whitespace", {})
        render_html(f"""
        <div class="ent-box" style="margin-bottom: 12px;">
            <div style="display: flex; align-items: center; justify-content: space-between; margin-bottom: 6px;">
                <span style="font-size: 13.5px; font-weight: 600; color: #FFFFFF;">Extra Whitespace</span>
                <span class="status-pill {'warning' if ws['status'] == 'warning' else 'live'}">
                    {ws['status'].upper()} • {ws['count']:,} CELLS
                </span>
            </div>
            <div style="font-size: 12.5px; color: var(--text-secondary); line-height: 1.45;">{ws['details']}</div>
        </div>
        """)
        if ws["count"] > 0:
            clean_options["trim_whitespace"] = st.checkbox("Trim leading/trailing and collapse redundant spaces", value=True, key="chk_trim_ws")

        # 5. Outliers
        ot = issues.get("outliers", {})
        render_html(f"""
        <div class="ent-box" style="margin-bottom: 12px;">
            <div style="display: flex; align-items: center; justify-content: space-between; margin-bottom: 6px;">
                <span style="font-size: 13.5px; font-weight: 600; color: #FFFFFF;">Outlier Remediation</span>
                <span class="status-pill {'danger' if ot['status'] == 'severe' else ('warning' if ot['status'] == 'warning' else 'live')}">
                    {ot['status'].upper()} • {ot['count']:,} POINTS
                </span>
            </div>
            <div style="font-size: 12.5px; color: var(--text-secondary); line-height: 1.45;">{ot['details']}</div>
        </div>
        """)
        if ot["count"] > 0:
            c1, c2 = st.columns(2)
            with c1:
                clean_options["outlier_treatment"] = st.selectbox(
                    "Treatment",
                    ["none", "cap", "drop"],
                    format_func=lambda x: "Leave Unchanged" if x == "none" else ("Winsorize (Cap at IQR)" if x == "cap" else "Drop Outlier Rows"),
                    index=1,
                    key="sel_outlier_treat"
                )
            with c2:
                clean_options["iqr_multiplier"] = st.slider("IQR Multiplier", min_value=1.0, max_value=3.0, value=1.5, step=0.1, key="sld_iqr")

        st.markdown("<hr style='border: none; border-top: 1px solid var(--border-subtle); margin: 24px 0;'>", unsafe_allow_html=True)

        b1, b2, b3 = st.columns([2, 2, 1])
        with b1:
            if st.button("Apply Cleaning Transformations", type="primary", use_container_width=True):
                with st.spinner("Executing transformations..."):
                    cleaned_df, diff_res = apply_cleaning_pipeline(df, clean_options)
                    st.session_state["df"] = cleaned_df
                    st.session_state["profile"] = profile_dataset(cleaned_df)
                    st.session_state["insights"] = generate_automated_insights(cleaned_df, st.session_state["profile"])
                    st.session_state["cleaning_diff"] = diff_res
                    st.rerun()

        with b2:
            if st.button("Simulate Diff (Dry Run)", use_container_width=True):
                sim_df, sim_diff = apply_cleaning_pipeline(df, clean_options)
                render_html(f"""
                <div class="ent-box" style="margin-top: 10px;">
                    <span style="font-size: 11px; text-transform: uppercase; color: var(--text-muted); font-weight: 600;">Simulation Outcome</span>
                    <div style="font-size: 13px; color: var(--text-main); margin-top: 6px;">
                        • Records: {sim_diff['rows_before']:,} → <b>{sim_diff['rows_after']:,}</b> ({sim_diff['rows_removed']} dropped)<br>
                        • Missing: {sim_diff['nulls_before']:,} → <b>{sim_diff['nulls_after']:,}</b> ({sim_diff['nulls_resolved']} imputed)<br>
                        • Duplicates: {sim_diff['dups_before']:,} → <b>{sim_diff['dups_after']:,}</b> ({sim_diff['dups_removed']} dropped)
                    </div>
                </div>
                """)

        with b3:
            if st.button("Reset", use_container_width=True):
                st.session_state["df"] = raw_df.copy()
                st.session_state["profile"] = profile_dataset(raw_df)
                st.session_state["insights"] = generate_automated_insights(raw_df, st.session_state["profile"])
                st.session_state["cleaning_diff"] = None
                st.rerun()


# =============================================================================
# PAGE 5: DATA QUALITY
# =============================================================================
elif active_page == "Data Quality":
    render_html("""
    <div style="margin-bottom: 20px;">
        <h2 style="font-size: 24px; font-weight: 700; color: #FFFFFF; letter-spacing: -0.02em; margin: 0 0 4px 0;">Data Quality & Integrity Audit</h2>
        <p style="font-size: 13px; color: var(--text-secondary); margin: 0;">Multi-dimensional assessment evaluating Completeness, Uniqueness, Validity, and Consistency.</p>
    </div>
    """)

    if not has_data:
        st.info("No active dataset. Ingest a dataset first to view Data Quality Report.")
    else:
        profile = st.session_state["profile"]
        q_score = profile.get("quality_score", 0)
        q_grade = profile.get("quality_grade", "N/A")

        render_html(f"""
        <div class="ent-box" style="display: flex; align-items: center; justify-content: space-between; padding: 22px 26px;">
            <div>
                <span style="font-size: 11px; font-weight: 700; text-transform: uppercase; color: var(--text-muted); letter-spacing: 0.06em;">Composite Data Quality Index</span>
                <div style="font-size: 42px; font-weight: 700; color: #FFFFFF; line-height: 1.1; margin-top: 4px; letter-spacing: -0.03em;">
                    {q_score:.1f}%
                </div>
                <div style="font-size: 12.5px; color: var(--text-secondary); margin-top: 4px;">
                    Evaluated across {profile['total_rows']:,} records and {profile['total_columns']} features
                </div>
            </div>
            <div>
                <span class="status-pill {'live' if q_score >= 85 else 'warning'}" style="font-size: 15px; padding: 8px 16px; border-radius: 6px;">
                    Grade {q_grade}
                </span>
            </div>
        </div>
        """)

        p1, p2, p3, p4 = st.columns(4)
        with p1:
            comp = profile.get("completeness_pct", 100)
            st.metric("Completeness", f"{comp:.1f}%", f"{profile['total_missing_cells']:,} null cells")
            st.progress(comp / 100)
        with p2:
            uniq = profile.get("uniqueness_pct", 100)
            st.metric("Uniqueness", f"{uniq:.1f}%", f"{profile['duplicate_rows']:,} dups")
            st.progress(uniq / 100)
        with p3:
            valid = profile.get("validity_pct", 100)
            st.metric("Validity (Outliers)", f"{valid:.1f}%", f"{profile['total_outliers']:,} outliers")
            st.progress(valid / 100)
        with p4:
            cons = profile.get("consistency_pct", 100)
            st.metric("Consistency", f"{cons:.1f}%", f"{profile['total_whitespace_cells']} issues")
            st.progress(cons / 100)

        render_html('<div style="font-size: 14px; font-weight: 600; color: #FFFFFF; margin: 24px 0 10px 0;">Feature Quality Breakdown</div>')
        col_audit = []
        for c in profile.get("columns", []):
            col_audit.append({
                "Feature": c["name"],
                "Semantic Type": c["inferred_type"].capitalize(),
                "Non-Null %": f"{c['non_null_pct']:.1f}%",
                "Missing": c["null_count"],
                "Unique Values": c["unique_count"],
                "Outliers (IQR)": c["outlier_count"] if c["inferred_type"] == "numeric" else "-",
                "Whitespace Issues": c["whitespace_count"],
                "Quality Score": f"{c['quality_score']:.0f}%"
            })
        st.dataframe(pd.DataFrame(col_audit), use_container_width=True, height=360)


# =============================================================================
# PAGE 6: EXPLORATORY ANALYSIS (EDA)
# =============================================================================
elif active_page == "Exploratory Analysis":
    render_html("""
    <div style="margin-bottom: 20px;">
        <h2 style="font-size: 24px; font-weight: 700; color: #FFFFFF; letter-spacing: -0.02em; margin: 0 0 4px 0;">Exploratory Data Analysis (EDA)</h2>
        <p style="font-size: 13px; color: var(--text-secondary); margin: 0;">Parametric metrics, distribution variance, and multidimensional correlation matrices.</p>
    </div>
    """)

    if not has_data:
        st.info("No active dataset. Ingest a dataset first to explore data distributions.")
    else:
        df = st.session_state["df"]
        num_summary = compute_numerical_summary(df)
        cat_summary = compute_categorical_summary(df)
        corr_mat, corr_pairs = compute_correlation_matrix(df)

        tab1, tab2, tab3 = st.tabs(["Numerical Metrics", "Categorical Profiles", "Correlation Matrix"])

        with tab1:
            if not num_summary.empty:
                st.dataframe(num_summary, use_container_width=True, height=260)
                num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
                d_c1, d_c2 = st.columns(2)
                with d_c1:
                    sel_num = st.selectbox("Select Feature for Distribution Histogram", num_cols, key="sel_hist_col")
                    bins = st.slider("Histogram Bins", 10, 80, 30, key="sld_bins")
                    fig_h = create_chart(df, "Histogram", x_col=sel_num, bins=bins, title=f"Distribution: {sel_num}")
                    st.plotly_chart(fig_h, use_container_width=True)
                with d_c2:
                    fig_b = create_chart(df, "Box", y_col=sel_num, title=f"Box Plot: {sel_num}")
                    st.plotly_chart(fig_b, use_container_width=True)
            else:
                st.info("No numerical features available.")

        with tab2:
            if not cat_summary.empty:
                st.dataframe(cat_summary, use_container_width=True, height=260)
                cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
                if cat_cols:
                    sel_cat = st.selectbox("Select Categorical Feature to Profile", cat_cols, key="sel_cat_col")
                    c_data = df[sel_cat].value_counts().head(10).reset_index()
                    c_data.columns = [sel_cat, "Frequency"]
                    fig_cat = create_chart(c_data, "Bar", x_col=sel_cat, y_col="Frequency", title=f"Top Values: {sel_cat}")
                    st.plotly_chart(fig_cat, use_container_width=True)
            else:
                st.info("No categorical features available.")

        with tab3:
            if not corr_mat.empty:
                fig_corr = create_chart(df, "Correlation Heatmap", title="Pearson Correlation Matrix")
                st.plotly_chart(fig_corr, use_container_width=True)

                if corr_pairs:
                    render_html('<div style="font-size: 14px; font-weight: 600; color: #FFFFFF; margin: 18px 0 10px 0;">Highest Correlated Feature Pairs</div>')
                    p_df = pd.DataFrame(corr_pairs[:6])
                    p_df.columns = ["Feature A", "Feature B", "r", "Absolute r", "Direction", "Strength"]
                    st.dataframe(p_df, use_container_width=True)
            else:
                st.info("At least 2 numerical columns are required for correlation analysis.")


# =============================================================================
# PAGE 7: VISUALIZATIONS (BUILDER)
# =============================================================================
elif active_page == "Visualizations":
    render_html("""
    <div style="margin-bottom: 20px;">
        <h2 style="font-size: 24px; font-weight: 700; color: #FFFFFF; letter-spacing: -0.02em; margin: 0 0 4px 0;">Interactive Visualization Builder</h2>
        <p style="font-size: 13px; color: var(--text-secondary); margin: 0;">Configure dimensional charts, metric aggregations, and layout geometries.</p>
    </div>
    """)

    if not has_data:
        st.info("No active dataset. Ingest a dataset first to build custom visualizations.")
    else:
        df = st.session_state["df"]
        all_cols = list(df.columns)
        num_cols = df.select_dtypes(include=[np.number]).columns.tolist()

        c_build_c1, c_build_c2 = st.columns([35, 65])

        with c_build_c1:
            render_html("""
            <div class="ent-box">
                <div class="ent-box-header">
                    <h3 class="ent-box-title">Chart Parameters</h3>
                </div>
            </div>
            """)

            chart_type = st.selectbox(
                "Chart Type",
                ["Bar", "Line", "Area", "Pie", "Scatter", "Histogram", "Box", "Correlation Heatmap"],
                index=0
            )

            x_axis = None
            y_axis = None
            color_dim = None
            agg_method = "none"

            if chart_type != "Correlation Heatmap":
                x_axis = st.selectbox("X-Axis (Dimension)", all_cols, index=0)

                y_options = ["None"] + all_cols
                default_y_idx = 1 if len(y_options) > 1 else 0
                if num_cols:
                    for i, col in enumerate(y_options):
                        if col in num_cols:
                            default_y_idx = i
                            break
                y_choice = st.selectbox("Y-Axis (Metric)", y_options, index=default_y_idx)
                y_axis = None if y_choice == "None" else y_choice

                color_opts = ["None"] + all_cols
                color_choice = st.selectbox("Group / Color Dimension", color_opts, index=0)
                color_dim = None if color_choice == "None" else color_choice

                if chart_type in ["Bar", "Line", "Area"] and y_axis:
                    agg_method = st.selectbox("Aggregation", ["none", "sum", "mean", "count", "min", "max"], format_func=lambda x: x.capitalize())

            title_input = st.text_input("Chart Title", value=f"{chart_type} Chart")

        with c_build_c2:
            try:
                fig = create_chart(
                    df=df,
                    chart_type=chart_type,
                    x_col=x_axis,
                    y_col=y_axis,
                    color_col=color_dim,
                    agg_func=agg_method,
                    title=title_input
                )
                st.plotly_chart(fig, use_container_width=True)
            except Exception as e:
                st.error(f"Render error: {e}")


# =============================================================================
# PAGE 8: AI INSIGHTS
# =============================================================================
elif active_page == "AI Insights":
    render_html("""
    <div style="margin-bottom: 20px;">
        <h2 style="font-size: 24px; font-weight: 700; color: #FFFFFF; letter-spacing: -0.02em; margin: 0 0 4px 0;">Automated Statistical Insights</h2>
        <p style="font-size: 13px; color: var(--text-secondary); margin: 0;">Algorithmic observations calculated from dataset distributions, volume drivers, and correlations.</p>
    </div>
    """)

    if not has_data:
        st.info("No active dataset. Ingest a dataset first to generate analytical observations.")
    else:
        insights = st.session_state["insights"] or []
        if not insights:
            st.info("No anomalous variance detected across features. Distributions align with expected baselines.")
        else:
            for idx, ins in enumerate(insights):
                render_html(f"""
                <div class="ent-box" style="margin-bottom: 14px;">
                    <div style="display: flex; align-items: center; justify-content: space-between; margin-bottom: 6px;">
                        <span class="status-pill primary">{ins.get('type', 'Insight').upper()}</span>
                        <span style="font-size: 11.5px; font-family: var(--font-mono); color: var(--accent-blue);">{ins.get('metric', '')}</span>
                    </div>
                    <h4 style="font-size: 14px; font-weight: 600; color: #FFFFFF; margin: 0 0 6px 0;">{ins.get('title', '')}</h4>
                    <p style="font-size: 12.5px; color: var(--text-secondary); line-height: 1.5; margin-bottom: 10px;">{ins.get('narrative', '')}</p>
                    <div style="background-color: var(--bg-elevated); border-left: 2px solid var(--accent-blue); padding: 8px 12px; border-radius: 4px; font-size: 12px; color: #FFFFFF;">
                        <b>Operational Note:</b> {ins.get('recommendation', '')}
                    </div>
                </div>
                """)


# =============================================================================
# PAGE 9: EXPORT REPORTS
# =============================================================================
elif active_page == "Export Reports":
    render_html("""
    <div style="margin-bottom: 20px;">
        <h2 style="font-size: 24px; font-weight: 700; color: #FFFFFF; letter-spacing: -0.02em; margin: 0 0 4px 0;">Export & Report Center</h2>
        <p style="font-size: 13px; color: var(--text-secondary); margin: 0;">Export cleaned dataset formats, quality audit logs, and executive summary reports.</p>
    </div>
    """)

    if not has_data:
        st.info("No active dataset. Ingest or clean a dataset first to download reports.")
    else:
        df = st.session_state["df"]
        profile = st.session_state["profile"]
        insights = st.session_state["insights"]
        meta = st.session_state["dataset_meta"] or {}
        base_name = os.path.splitext(meta.get("filename", "dataset.csv"))[0]

        e_c1, e_c2 = st.columns(2)

        with e_c1:
            render_html("""
            <div class="ent-box">
                <div class="ent-box-header">
                    <h4 class="ent-box-title">Cleaned CSV Dataset</h4>
                    <span class="status-pill live">UTF-8 BOM</span>
                </div>
                <p class="ent-box-desc">Production-ready CSV file formatted with universal Excel encoding.</p>
            </div>
            """)
            csv_data = dataframe_to_csv_bytes(df)
            st.download_button(
                "Download Cleaned CSV",
                data=csv_data,
                file_name=f"{base_name}_cleaned.csv",
                mime="text/csv",
                use_container_width=True
            )

        with e_c2:
            render_html("""
            <div class="ent-box">
                <div class="ent-box-header">
                    <h4 class="ent-box-title">Excel Spreadsheet (.xlsx)</h4>
                    <span class="status-pill neutral">OpenPyXL</span>
                </div>
                <p class="ent-box-desc">Standard Excel workbook format with schema headers.</p>
            </div>
            """)
            try:
                excel_data = dataframe_to_excel_bytes(df)
                st.download_button(
                    "Download Excel (.xlsx)",
                    data=excel_data,
                    file_name=f"{base_name}.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    use_container_width=True
                )
            except Exception as e:
                st.warning(f"Excel export unavailable: {e}")

        st.markdown("<div style='height: 12px;'></div>", unsafe_allow_html=True)

        e_c3, e_c4 = st.columns(2)

        with e_c3:
            render_html("""
            <div class="ent-box">
                <div class="ent-box-header">
                    <h4 class="ent-box-title">Data Quality Audit (JSON)</h4>
                    <span class="status-pill neutral">JSON Schema</span>
                </div>
                <p class="ent-box-desc">Full machine-readable quality breakdown with column scores.</p>
            </div>
            """)
            json_data = export_quality_report_json(profile)
            st.download_button(
                "Download Quality JSON",
                data=json_data,
                file_name=f"{base_name}_quality_audit.json",
                mime="application/json",
                use_container_width=True
            )

        with e_c4:
            render_html("""
            <div class="ent-box">
                <div class="ent-box-header">
                    <h4 class="ent-box-title">Executive Summary Report</h4>
                    <span class="status-pill primary">Markdown</span>
                </div>
                <p class="ent-box-desc">Comprehensive narrative with KPIs, column profiles, and recommendations.</p>
            </div>
            """)
            md_report = generate_executive_markdown_report(meta.get("filename", "dataset"), df, profile, insights)
            st.download_button(
                "Download Executive Report (MD)",
                data=md_report,
                file_name=f"{base_name}_executive_report.md",
                mime="text/markdown",
                use_container_width=True
            )


# =============================================================================
# PAGE 10: SETTINGS
# =============================================================================
elif active_page == "Settings":
    render_html("""
    <div style="margin-bottom: 20px;">
        <h2 style="font-size: 24px; font-weight: 700; color: #FFFFFF; letter-spacing: -0.02em; margin: 0 0 4px 0;">Platform Settings</h2>
        <p style="font-size: 13px; color: var(--text-secondary); margin: 0;">Platform configuration, ingestion engine limits, and architecture specifications.</p>
    </div>
    """)

    s1, s2 = st.columns(2)

    with s1:
        render_html("""
        <div class="ent-box">
            <div class="ent-box-header">
                <h3 class="ent-box-title">Platform Specifications</h3>
                <span class="status-pill live">Active</span>
            </div>
            <div style="font-size: 12.5px; color: var(--text-secondary); line-height: 1.6;">
                • <b>Theme:</b> Enterprise FinTech Dark (#080A0F)<br>
                • <b>Typography:</b> Inter / JetBrains Mono<br>
                • <b>UI Standards:</b> Production SaaS BI (Clean vector curves, no emojis)<br>
                • <b>Engine Version:</b> DataPilot v2.4.0 PRO
            </div>
        </div>
        """)

    with s2:
        render_html("""
        <div class="ent-box">
            <div class="ent-box-header">
                <h3 class="ent-box-title">Ingestion Engine Parameters</h3>
                <span class="status-pill neutral">System Limits</span>
            </div>
            <div style="font-size: 12.5px; color: var(--text-secondary); line-height: 1.6;">
                • <b>Max Upload Limit:</b> 2,048 MB (2 GB)<br>
                • <b>Encoding Support:</b> UTF-8, Latin1, CP1252, ISO-8859-1<br>
                • <b>Delimiters:</b> Auto-detect (Comma, Semicolon, Tab)<br>
                • <b>Data Privacy:</b> 100% In-Memory Processing (Zero Persistence)
            </div>
        </div>
        """)
