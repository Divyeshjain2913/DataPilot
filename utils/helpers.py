"""
DataPilot - Helper Utilities (High-Precision Enterprise Edition)
Number formatting, byte conversion, file exports, SVG sparkline generators, and CSS injection.
"""

from __future__ import annotations
import io
import json
import textwrap
from typing import Any, Dict, List, Optional
import pandas as pd
import streamlit as st


def format_bytes(size_in_bytes: float | int) -> str:
    """Format bytes into clean human-readable string (KB, MB, GB)."""
    if size_in_bytes is None or size_in_bytes < 0:
        return "0 B"
    units = ["B", "KB", "MB", "GB", "TB"]
    size = float(size_in_bytes)
    unit_index = 0
    while size >= 1024.0 and unit_index < len(units) - 1:
        size /= 1024.0
        unit_index += 1
    return f"{size:.1f} {units[unit_index]}"


def format_compact_number(val: float | int, prefix: str = "", suffix: str = "") -> str:
    """Format large numbers into clean enterprise compact strings (e.g. 1.2K, 3.4M)."""
    if val is None or pd.isna(val):
        return "N/A"
    try:
        num = float(val)
    except (ValueError, TypeError):
        return str(val)

    abs_num = abs(num)
    sign = "-" if num < 0 else ""

    if abs_num >= 1_000_000_000:
        formatted = f"{abs_num / 1_000_000_000:.1f}B"
    elif abs_num >= 1_000_000:
        formatted = f"{abs_num / 1_000_000:.1f}M"
    elif abs_num >= 1_000:
        formatted = f"{abs_num / 1_000:.1f}K"
    elif abs_num.is_integer():
        formatted = f"{int(abs_num):,}"
    else:
        formatted = f"{abs_num:,.2f}"

    return f"{sign}{prefix}{formatted}{suffix}"


def format_percentage(val: float | int, decimals: int = 1) -> str:
    """Format float into percentage string."""
    if val is None or pd.isna(val):
        return "0.0%"
    return f"{float(val):.{decimals}f}%"


def generate_sparkline_svg(values: List[float], color: str = "#38BDF8", height: int = 40, width: int = 160) -> str:
    """
    Generate an ultra-lightweight inline SVG sparkline with smooth area gradient
    matching the micro-charts in high-end FinTech dashboards.
    """
    if not values or len(values) < 2:
        values = [10.0, 15.0, 13.0, 22.0, 18.0, 25.0, 21.0, 30.0]

    min_val = min(values)
    max_val = max(values)
    val_range = max(max_val - min_val, 1e-5)

    points = []
    step_x = width / (len(values) - 1)
    pad_y = 4
    usable_h = height - (pad_y * 2)

    for i, v in enumerate(values):
        x = i * step_x
        y = pad_y + (1.0 - (v - min_val) / val_range) * usable_h
        points.append((round(x, 1), round(y, 1)))

    # Path d
    path_d = f"M {points[0][0]} {points[0][1]}"
    for p in points[1:]:
        path_d += f" L {p[0]} {p[1]}"

    # Area fill path
    area_d = f"{path_d} L {width} {height} L 0 {height} Z"

    grad_id = f"grad_{abs(hash(tuple(values[:4]))) % 10000}"

    svg = f"""<svg width="100%" height="{height}" viewBox="0 0 {width} {height}" fill="none" xmlns="http://www.w3.org/2000/svg" style="display:block; overflow:visible;">
        <defs>
            <linearGradient id="{grad_id}" x1="0" y1="0" x2="0" y2="1">
                <stop offset="0%" stop-color="{color}" stop-opacity="0.25"/>
                <stop offset="100%" stop-color="{color}" stop-opacity="0.0"/>
            </linearGradient>
        </defs>
        <path d="{area_d}" fill="url(#{grad_id})"/>
        <path d="{path_d}" stroke="{color}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
    </svg>"""
    return svg


def generate_minibars_svg(values: List[float], color: str = "#38BDF8", height: int = 40, width: int = 160) -> str:
    """Generate a micro-bar chart SVG matching the 4th KPI card in the reference image."""
    if not values or len(values) < 2:
        values = [12.0, 18.0, 24.0, 15.0, 32.0, 28.0, 38.0, 42.0]

    max_val = max(max(values), 1e-5)
    n = len(values)
    bar_w = max(4.0, (width / n) - 3)
    bars_svg = []

    for i, v in enumerate(values):
        x = i * (bar_w + 3)
        h = max(3.0, (v / max_val) * (height - 4))
        y = height - h
        # Highlight last 2 bars with brighter accent
        bar_color = color if i < n - 2 else "#00E5FF"
        bars_svg.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_w:.1f}" height="{h:.1f}" rx="2" fill="{bar_color}" opacity="{0.5 + 0.5 * (i / n)}"/>')

    return f"""<svg width="100%" height="{height}" viewBox="0 0 {width} {height}" fill="none" xmlns="http://www.w3.org/2000/svg" style="display:block;">
        {''.join(bars_svg)}
    </svg>"""


def render_html(html_str: str) -> None:
    """Render HTML safely using native st.html (zero markdown escaping or indentation leaks)."""
    if hasattr(st, "html"):
        st.html(html_str)
    else:
        lines = [line.strip() for line in html_str.strip().split("\n") if line.strip()]
        clean_html = "".join(lines)
        st.markdown(clean_html, unsafe_allow_html=True)


def dataframe_to_csv_bytes(df: pd.DataFrame) -> bytes:
    """Convert dataframe to CSV bytes with UTF-8 BOM."""
    return df.to_csv(index=False, encoding="utf-8-sig").encode("utf-8-sig")


def dataframe_to_excel_bytes(df: pd.DataFrame) -> bytes:
    """Convert dataframe to Excel bytes (.xlsx)."""
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name="DataPilot_Cleaned")
    return output.getvalue()


def export_quality_report_json(profile_data: Dict[str, Any]) -> str:
    """Export profile and data quality metrics to clean JSON."""
    clean_dict: Dict[str, Any] = {}
    for k, v in profile_data.items():
        if isinstance(v, (int, float, str, bool, list, dict)) or v is None:
            clean_dict[k] = v
        elif isinstance(v, pd.DataFrame):
            clean_dict[k] = v.to_dict(orient="records")
        elif isinstance(v, pd.Series):
            clean_dict[k] = v.to_dict()
        else:
            clean_dict[k] = str(v)
    return json.dumps(clean_dict, indent=2, default=str)


def generate_executive_markdown_report(
    dataset_name: str,
    df: pd.DataFrame,
    profile: Dict[str, Any],
    insights: list[Dict[str, Any]]
) -> str:
    """Generate executive analysis report in Markdown."""
    lines = [
        f"# DataPilot Executive Analysis Report",
        f"**Dataset:** {dataset_name}",
        f"**Timestamp:** {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S UTC')}",
        "",
        "---",
        "",
        "## 1. Executive Summary & KPIs",
        f"- **Total Records:** {len(df):,}",
        f"- **Total Features:** {len(df.columns)}",
        f"- **Data Quality Score:** {profile.get('quality_score', 0):.1f}% ({profile.get('quality_grade', 'N/A')})",
        f"- **Completeness Rate:** {profile.get('completeness_pct', 0):.1f}%",
        f"- **Duplicate Rows:** {profile.get('duplicate_rows', 0):,} ({profile.get('duplicate_pct', 0):.1f}%)",
        f"- **Memory Footprint:** {profile.get('memory_usage_str', 'N/A')}",
        "",
        "---",
        "",
        "## 2. Statistical Insights",
    ]

    if insights:
        for idx, ins in enumerate(insights, 1):
            lines.append(f"### {idx}. {ins.get('title', 'Insight')}")
            lines.append(f"**Classification:** {ins.get('type', 'General')} | **Metric:** `{ins.get('metric', 'N/A')}`")
            lines.append(f"{ins.get('narrative', '')}")
            if ins.get('recommendation'):
                lines.append(f"> **Operational Note:** {ins.get('recommendation')}")
            lines.append("")
    else:
        lines.append("No anomalous variance detected.\n")

    lines.extend([
        "---",
        "",
        "## 3. Column-by-Column Data Profiling",
        "| Feature Name | Semantic Type | Non-Null % | Unique Count | Outliers (IQR) | Quality Score |",
        "|---|---|---|---|---|---|",
    ])

    col_profiles = profile.get("columns", [])
    for col in col_profiles:
        name = col.get("name", "")
        ctype = col.get("inferred_type", "")
        non_null = f"{col.get('non_null_pct', 0):.1f}%"
        unique = f"{col.get('unique_count', 0):,}"
        outliers = f"{col.get('outlier_count', 0)}" if col.get("inferred_type") == "numeric" else "-"
        qscore = f"{col.get('quality_score', 0):.0f}%"
        lines.append(f"| `{name}` | {ctype} | {non_null} | {unique} | {outliers} | {qscore} |")

    lines.extend([
        "",
        "---",
        "",
        "## 4. Methodology Notes",
        "- **Outlier Detection:** Tukey's Interquartile Range (IQR) criterion with 1.5 multiplier.",
        "- **Quality Composite:** 35% Completeness + 25% Uniqueness + 20% Validity + 20% Consistency.",
        ""
    ])

    return "\n".join(lines)


def inject_custom_css() -> None:
    """Inject strict dark enterprise CSS into Streamlit page."""
    try:
        with open("assets/style.css", "r", encoding="utf-8") as f:
            css_content = f.read()
    except Exception:
        css_content = ""

    if hasattr(st, "html"):
        st.html(f"<style>{css_content}</style>")
    else:
        st.markdown(f"<style>{css_content}</style>", unsafe_allow_html=True)
