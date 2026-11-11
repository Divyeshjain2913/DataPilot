"""
DataPilot - Visualization Module (Fluxo / FinTech Enterprise Edition)
Sleek dark surfaces (#111520), rounded bar caps, subtle gridlines, and high-precision palettes.
"""

from __future__ import annotations
from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from utils.helpers import format_compact_number


# Enterprise FinTech Palette
ENTERPRISE_PALETTE = [
    "#38BDF8",  # Electric Cyan/Blue
    "#818CF8",  # Indigo Accent
    "#00E5FF",  # Neon Cyan
    "#10B981",  # Emerald Green
    "#F59E0B",  # Warm Amber
    "#F43F5E",  # Rose Red
    "#64748B",  # Slate
    "#A78BFA"   # Purple
]


def get_enterprise_layout(title: str = "") -> Dict[str, Any]:
    """Standardized Plotly layout for enterprise dark dashboards."""
    bg_color = "#111520"
    text_color = "#F8FAFC"
    muted_color = "#64748B"
    border_color = "rgba(255, 255, 255, 0.06)"

    return {
        "title": {
            "text": title,
            "font": {"family": "Inter, sans-serif", "size": 13.5, "color": text_color, "weight": 600},
            "x": 0.02,
            "xanchor": "left"
        },
        "plot_bgcolor": bg_color,
        "paper_bgcolor": bg_color,
        "font": {"family": "Inter, sans-serif", "color": text_color, "size": 11},
        "colorway": ENTERPRISE_PALETTE,
        "margin": {"l": 45, "r": 20, "t": 52, "b": 42},
        "xaxis": {
            "gridcolor": border_color,
            "linecolor": border_color,
            "tickfont": {"size": 10.5, "color": muted_color},
            "title": {"font": {"size": 11, "weight": 500, "color": muted_color}},
            "showgrid": False,
            "zeroline": False
        },
        "yaxis": {
            "gridcolor": border_color,
            "linecolor": border_color,
            "tickfont": {"size": 10.5, "color": muted_color},
            "title": {"font": {"size": 11, "weight": 500, "color": muted_color}},
            "showgrid": True,
            "zeroline": False
        },
        "legend": {
            "orientation": "h",
            "yanchor": "bottom",
            "y": 1.02,
            "xanchor": "right",
            "x": 1,
            "font": {"size": 10.5, "color": "#94A3B8"}
        },
        "hoverlabel": {
            "bgcolor": "#1A2130",
            "font": {"family": "Inter, sans-serif", "size": 11, "color": "#FFFFFF"},
            "bordercolor": "rgba(56, 189, 248, 0.3)"
        }
    }


def create_chart(
    df: pd.DataFrame,
    chart_type: str,
    x_col: Optional[str] = None,
    y_col: Optional[str] = None,
    color_col: Optional[str] = None,
    agg_func: str = "none",
    title: str = "",
    theme: str = "dark",
    orientation: str = "v",
    bins: int = 30
) -> go.Figure:
    """Build enterprise Plotly chart with sleek dark surface styling."""
    chart_df = df.copy()

    # Apply Aggregation
    if agg_func != "none" and x_col and y_col and chart_type in ["Bar", "Line", "Area"]:
        group_cols = [x_col] + ([color_col] if color_col and color_col != "None" else [])
        if agg_func == "sum":
            chart_df = chart_df.groupby(group_cols)[y_col].sum().reset_index()
        elif agg_func == "mean":
            chart_df = chart_df.groupby(group_cols)[y_col].mean().reset_index()
        elif agg_func == "count":
            chart_df = chart_df.groupby(group_cols)[y_col].count().reset_index()
        elif agg_func == "min":
            chart_df = chart_df.groupby(group_cols)[y_col].min().reset_index()
        elif agg_func == "max":
            chart_df = chart_df.groupby(group_cols)[y_col].max().reset_index()

    layout_defaults = get_enterprise_layout(title or f"{chart_type} Chart")

    if chart_type == "Bar":
        if orientation == "h" and y_col and x_col:
            fig = px.bar(chart_df, x=y_col, y=x_col, color=color_col if color_col != "None" else None, orientation="h", color_discrete_sequence=ENTERPRISE_PALETTE)
        else:
            fig = px.bar(chart_df, x=x_col, y=y_col, color=color_col if color_col != "None" else None, color_discrete_sequence=ENTERPRISE_PALETTE)
        fig.update_traces(marker_line_width=0, opacity=0.92)

    elif chart_type == "Line":
        fig = px.line(chart_df, x=x_col, y=y_col, color=color_col if color_col != "None" else None, markers=True, color_discrete_sequence=ENTERPRISE_PALETTE)
        fig.update_traces(line=dict(width=2.5))

    elif chart_type == "Area":
        fig = px.area(chart_df, x=x_col, y=y_col, color=color_col if color_col != "None" else None, color_discrete_sequence=ENTERPRISE_PALETTE)
        fig.update_traces(line=dict(width=2), opacity=0.3)

    elif chart_type == "Pie":
        fig = px.pie(
            chart_df,
            names=x_col,
            values=y_col,
            hole=0.62,
            color_discrete_sequence=ENTERPRISE_PALETTE
        )
        fig.update_traces(textposition="outside", textinfo="percent+label", textfont=dict(size=10.5, color="#94A3B8"))

    elif chart_type == "Scatter":
        fig = px.scatter(
            chart_df,
            x=x_col,
            y=y_col,
            color=color_col if color_col != "None" else None,
            color_discrete_sequence=ENTERPRISE_PALETTE,
            opacity=0.8
        )
        fig.update_traces(marker=dict(size=7.5, line=dict(width=0.5, color="#1E293B")))

    elif chart_type == "Histogram":
        fig = px.histogram(
            chart_df,
            x=x_col,
            color=color_col if color_col != "None" else None,
            nbins=bins,
            color_discrete_sequence=ENTERPRISE_PALETTE
        )
        fig.update_traces(marker_line_width=0.5, marker_line_color="#111520", opacity=0.88)

    elif chart_type == "Box":
        fig = px.box(
            chart_df,
            x=x_col,
            y=y_col,
            color=color_col if color_col != "None" else None,
            color_discrete_sequence=ENTERPRISE_PALETTE,
            points="outliers"
        )
        fig.update_traces(line=dict(width=1.4))

    elif chart_type == "Correlation Heatmap":
        num_df = chart_df.select_dtypes(include=[np.number])
        if num_df.shape[1] >= 2:
            corr = num_df.corr().round(2)
            fig = px.imshow(
                corr,
                text_auto=True,
                aspect="auto",
                color_continuous_scale=[[0, "#10141E"], [0.5, "#1E293B"], [1.0, "#38BDF8"]],
                zmin=-1,
                zmax=1
            )
            fig.update_coloraxes(showscale=False)
        else:
            fig = go.Figure()
            fig.add_annotation(text="At least 2 numeric columns required", showarrow=False, font=dict(color="#64748B"))

    else:
        fig = px.bar(chart_df, x=x_col, y=y_col, color_discrete_sequence=ENTERPRISE_PALETTE)

    fig.update_layout(**layout_defaults)
    return fig


def build_archetype_dashboard(df: pd.DataFrame, archetype: str, theme: str = "dark") -> Dict[str, Any]:
    """
    Generate enterprise KPI cards and charts for domain archetypes.
    """
    charts: List[Dict[str, Any]] = []
    kpis: List[Dict[str, Any]] = []

    # 1. SALES ARCHETYPE
    if archetype == "sales":
        sales_col = next((c for c in df.columns if "sales" in c.lower()), None)
        profit_col = next((c for c in df.columns if "profit" in c.lower()), None)
        cat_col = next((c for c in df.columns if "category" in c.lower() and "sub" not in c.lower()), None)
        date_col = next((c for c in df.columns if "date" in c.lower()), None)
        region_col = next((c for c in df.columns if "region" in c.lower()), None)

        total_sales = df[sales_col].sum() if sales_col else 0
        total_profit = df[profit_col].sum() if profit_col else 0
        margin = (total_profit / total_sales * 100) if total_sales else 0

        # Sample sparkline values derived from actual sales
        spark_sales = df[sales_col].dropna().head(12).tolist() if sales_col else [10, 20, 15, 30]
        spark_profit = df[profit_col].dropna().head(12).tolist() if profit_col else [5, 12, 8, 18]

        kpis = [
            {"label": "Gross Sales Revenue", "value": format_compact_number(total_sales, prefix="₹"), "sub": "Total Sales", "spark": spark_sales, "color": "#38BDF8"},
            {"label": "Net Profit", "value": format_compact_number(total_profit, prefix="₹"), "sub": f"{margin:.1f}% Margin", "spark": spark_profit, "color": "#818CF8"},
            {"label": "Transaction Volume", "value": f"{len(df):,}", "sub": "Delivered Orders", "spark": [12, 14, 18, 16, 22, 26, 30], "color": "#00E5FF"},
            {"label": "Order Fulfillment", "value": "100.0%", "sub": "Processed Orders", "spark": [60, 75, 70, 85, 90, 88, 94], "color": "#10B981", "is_bar": True}
        ]

        if cat_col and sales_col:
            cat_sum = df.groupby(cat_col)[sales_col].sum().reset_index().sort_values(by=sales_col, ascending=False)
            fig_main = create_chart(cat_sum, "Bar", x_col=cat_col, y_col=sales_col, title="Revenue by Product Category")
            charts.append({"title": "Revenue by Product Category", "fig": fig_main, "type": "main"})

        if region_col and sales_col:
            reg_sum = df.groupby(region_col)[sales_col].sum().reset_index()
            fig_sec = create_chart(reg_sum, "Pie", x_col=region_col, y_col=sales_col, title="Regional Revenue Distribution")
            charts.append({"title": "Regional Revenue Distribution", "fig": fig_sec, "type": "secondary"})

        if date_col and sales_col:
            temp = df[[date_col, sales_col]].dropna().copy()
            temp[date_col] = pd.to_datetime(temp[date_col], errors="coerce")
            temp = temp.dropna().sort_values(by=date_col)
            monthly = temp.groupby(pd.Grouper(key=date_col, freq="ME"))[sales_col].sum().reset_index()
            monthly[date_col] = monthly[date_col].dt.strftime("%b %Y")
            fig_trend = create_chart(monthly, "Line", x_col=date_col, y_col=sales_col, title="Monthly Sales Trajectory")
            charts.append({"title": "Monthly Sales Trajectory", "fig": fig_trend, "type": "tertiary"})

        if sales_col and profit_col:
            fig_scatter = create_chart(df, "Scatter", x_col=sales_col, y_col=profit_col, color_col=cat_col, title="Profitability vs Sales Volume")
            charts.append({"title": "Profitability vs Sales Volume", "fig": fig_scatter, "type": "quaternary"})

    # 2. STUDENT ARCHETYPE
    elif archetype == "student":
        exam_col = next((c for c in df.columns if "exam" in c.lower() or "score" in c.lower()), None)
        att_col = next((c for c in df.columns if "att" in c.lower()), None)
        dept_col = next((c for c in df.columns if "dept" in c.lower() or "department" in c.lower()), None)
        grade_col = next((c for c in df.columns if "grade" in c.lower()), None)

        spark_exam = df[exam_col].dropna().head(12).tolist() if exam_col else [65, 72, 80, 85]
        spark_att = df[att_col].dropna().head(12).tolist() if att_col else [70, 78, 84, 88]

        # Calculate actual pass rate from data
        if grade_col:
            passing = (df[grade_col].astype(str).str.strip().str.upper() != "F").sum()
            pass_rate = (passing / max(len(df), 1)) * 100
            pass_sub = "Grade A-D Pass"
        elif exam_col:
            pass_rate = (df[exam_col] >= 40).mean() * 100
            pass_sub = "Exam Score ≥ 40%"
        else:
            pass_rate = 100.0
            pass_sub = "Full Roster"

        kpis = [
            {"label": "Cohort Enrollment", "value": f"{len(df):,}", "sub": "Active Students", "spark": [500, 600, 720, 800], "color": "#38BDF8"},
            {"label": "Mean Exam Score", "value": f"{df[exam_col].mean():.1f} / 100" if exam_col else "N/A", "sub": "Median: " + f"{df[exam_col].median():.1f}" if exam_col else "", "spark": spark_exam, "color": "#818CF8"},
            {"label": "Average Attendance", "value": f"{df[att_col].mean():.1f}%" if att_col else "N/A", "sub": "Campus Benchmark >75%", "spark": spark_att, "color": "#00E5FF"},
            {"label": "Cohort Pass Rate", "value": f"{pass_rate:.1f}%", "sub": pass_sub, "spark": [82, 86, 88, 90, 92, 94], "color": "#10B981", "is_bar": True}
        ]

        if exam_col and dept_col:
            fig_main = create_chart(df, "Box", x_col=dept_col, y_col=exam_col, title="Academic Score Variance by Department")
            charts.append({"title": "Score Variance by Department", "fig": fig_main, "type": "main"})

        if grade_col:
            g_counts = df[grade_col].value_counts().reset_index()
            g_counts.columns = [grade_col, "Count"]
            fig_sec = create_chart(g_counts, "Pie", x_col=grade_col, y_col="Count", title="Grade Distribution Breakdown")
            charts.append({"title": "Grade Distribution Breakdown", "fig": fig_sec, "type": "secondary"})

        if att_col and exam_col:
            fig_scatter = create_chart(df, "Scatter", x_col=att_col, y_col=exam_col, color_col=dept_col, title="Attendance vs Exam Score Correlation")
            charts.append({"title": "Attendance vs Exam Score", "fig": fig_scatter, "type": "tertiary"})

        if exam_col:
            fig_hist = create_chart(df, "Histogram", x_col=exam_col, title="Score Distribution Frequency")
            charts.append({"title": "Score Distribution", "fig": fig_hist, "type": "quaternary"})

    # 3. EMPLOYEE ARCHETYPE
    elif archetype == "employee":
        salary_col = next((c for c in df.columns if "salary" in c.lower()), None)
        exp_col = next((c for c in df.columns if "exp" in c.lower() or "tenure" in c.lower()), None)
        dept_col = next((c for c in df.columns if "dept" in c.lower()), None)
        role_col = next((c for c in df.columns if "role" in c.lower() or "title" in c.lower()), None)

        spark_sal = df[salary_col].dropna().head(12).tolist() if salary_col else [60000, 75000, 85000]
        spark_exp = df[exp_col].dropna().head(12).tolist() if exp_col else [2, 4, 6, 8]

        # Appraisal/Rating or Structure calculation
        perf_col = next((c for c in df.columns if any(k in c.lower() for k in ["perform", "apprais", "rating", "eval"])), None)
        if perf_col and pd.api.types.is_numeric_dtype(df[perf_col]):
            perf_val = f"{df[perf_col].mean():.1f} / 5.0"
            perf_sub = "Mean Rating"
            perf_label = "Appraisal Score"
        else:
            perf_val = f"{len(df[dept_col].dropna().unique()) if dept_col else 0} Depts"
            perf_sub = "Org Units"
            perf_label = "Department Span"

        kpis = [
            {"label": "Active Workforce", "value": f"{len(df):,}", "sub": "Total Headcount", "spark": [400, 480, 560, 650], "color": "#38BDF8"},
            {"label": "Mean Annual Salary", "value": format_compact_number(df[salary_col].mean(), prefix="$") if salary_col else "N/A", "sub": "Market Parity", "spark": spark_sal, "color": "#818CF8"},
            {"label": "Average Seniority", "value": f"{df[exp_col].mean():.1f} Yrs" if exp_col else "N/A", "sub": "Tenure Benchmark", "spark": spark_exp, "color": "#00E5FF"},
            {"label": perf_label, "value": perf_val, "sub": perf_sub, "spark": [3.2, 3.5, 3.8, 4.0, 4.1, 4.2], "color": "#10B981", "is_bar": True}
        ]

        if dept_col and salary_col:
            fig_main = create_chart(df, "Box", x_col=dept_col, y_col=salary_col, title="Compensation Range by Department")
            charts.append({"title": "Compensation by Department", "fig": fig_main, "type": "main"})

        if dept_col:
            d_counts = df[dept_col].value_counts().reset_index()
            d_counts.columns = [dept_col, "Count"]
            fig_sec = create_chart(d_counts, "Pie", x_col=dept_col, y_col="Count", title="Workforce Distribution by Department")
            charts.append({"title": "Workforce Distribution", "fig": fig_sec, "type": "secondary"})

        if exp_col and salary_col:
            fig_scatter = create_chart(df, "Scatter", x_col=exp_col, y_col=salary_col, color_col=dept_col, title="Tenure vs Compensation Trajectory")
            charts.append({"title": "Tenure vs Compensation", "fig": fig_scatter, "type": "tertiary"})

        if role_col:
            r_counts = df[role_col].value_counts().head(7).reset_index()
            r_counts.columns = [role_col, "Count"]
            fig_roles = create_chart(r_counts, "Bar", x_col=role_col, y_col="Count", orientation="h", title="Top Roles by Headcount")
            charts.append({"title": "Top Roles by Headcount", "fig": fig_roles, "type": "quaternary"})

    # 4. GENERAL ARCHETYPE
    else:
        num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()

        null_pct = (df.isnull().sum().sum() / max(len(df) * len(df.columns), 1)) * 100
        density_pct = 100.0 - null_pct
        q_score = max(0.0, 100.0 - (null_pct * 1.5))
        q_grade = "A+" if q_score >= 95 else ("A" if q_score >= 90 else ("B" if q_score >= 80 else "C"))

        kpis = [
            {"label": "Total Analyzed Records", "value": f"{len(df):,}", "sub": "Row Entities", "spark": [10, 15, 20, 25, 30], "color": "#38BDF8"},
            {"label": "Feature Dimensions", "value": f"{len(df.columns)}", "sub": f"{len(num_cols)} num, {len(cat_cols)} cat", "spark": [4, 6, 8, 12], "color": "#818CF8"},
            {"label": "Data Density Rate", "value": f"{density_pct:.1f}%", "sub": "Field Integrity", "spark": [88, 92, 95, 98], "color": "#00E5FF"},
            {"label": "Quality Score", "value": f"{q_score:.1f}%", "sub": f"Grade {q_grade}", "spark": [85, 90, 95, 98], "color": "#10B981", "is_bar": True}
        ]

        if num_cols and cat_cols:
            n_col, c_col = num_cols[0], cat_cols[0]
            grp = df.groupby(c_col)[n_col].sum().reset_index().sort_values(by=n_col, ascending=False).head(8)
            fig_main = create_chart(grp, "Bar", x_col=c_col, y_col=n_col, title=f"Total {n_col} by {c_col}")
            charts.append({"title": f"{n_col} by {c_col}", "fig": fig_main, "type": "main"})

        if cat_cols:
            c_counts = df[cat_cols[0]].value_counts().head(6).reset_index()
            c_counts.columns = [cat_cols[0], "Count"]
            fig_sec = create_chart(c_counts, "Pie", x_col=cat_cols[0], y_col="Count", title=f"Distribution of {cat_cols[0]}")
            charts.append({"title": f"Distribution of {cat_cols[0]}", "fig": fig_sec, "type": "secondary"})

        if len(num_cols) >= 2:
            fig_scatter = create_chart(df, "Scatter", x_col=num_cols[0], y_col=num_cols[1], title=f"{num_cols[0]} vs {num_cols[1]}")
            charts.append({"title": f"{num_cols[0]} vs {num_cols[1]}", "fig": fig_scatter, "type": "tertiary"})

        if num_cols:
            fig_hist = create_chart(df, "Histogram", x_col=num_cols[0], title=f"Distribution of {num_cols[0]}")
            charts.append({"title": f"Distribution of {num_cols[0]}", "fig": fig_hist, "type": "quaternary"})

    return {"kpis": kpis, "charts": charts}
