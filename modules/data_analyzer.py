"""
DataPilot - Data Analyzer Module
Computes deep statistical summaries, correlation matrices, multi-group aggregations,
and generates automated, statistically-grounded analytical insights.
"""

from __future__ import annotations
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from utils.helpers import format_compact_number, format_percentage


def compute_numerical_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Compute detailed descriptive statistics for all numerical columns."""
    num_df = df.select_dtypes(include=[np.number])
    if num_df.empty:
        return pd.DataFrame()

    stats = []
    for col in num_df.columns:
        s = num_df[col].dropna()
        if len(s) == 0:
            continue
        q25 = s.quantile(0.25)
        q75 = s.quantile(0.75)
        iqr = q75 - q25
        stats.append({
            "Column": col,
            "Count": int(len(s)),
            "Mean": round(float(s.mean()), 2),
            "Std Dev": round(float(s.std()), 2) if len(s) > 1 else 0.0,
            "Min": round(float(s.min()), 2),
            "25%": round(float(q25), 2),
            "Median (50%)": round(float(s.median()), 2),
            "75%": round(float(q75), 2),
            "Max": round(float(s.max()), 2),
            "IQR": round(float(iqr), 2),
            "Skewness": round(float(s.skew()), 2) if len(s) > 2 else 0.0
        })

    return pd.DataFrame(stats)


def compute_categorical_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Compute frequency and diversity metrics for categorical and text columns."""
    cat_df = df.select_dtypes(include=["object", "category"])
    if cat_df.empty:
        return pd.DataFrame()

    stats = []
    for col in cat_df.columns:
        s = cat_df[col].dropna().astype(str)
        if len(s) == 0:
            continue
        val_counts = s.value_counts()
        top_val = val_counts.index[0]
        top_freq = int(val_counts.iloc[0])
        top_pct = round((top_freq / len(s)) * 100, 1)

        stats.append({
            "Column": col,
            "Non-Null Count": len(s),
            "Unique Values": s.nunique(),
            "Top Value": top_val,
            "Top Frequency": top_freq,
            "Top Share (%)": f"{top_pct}%"
        })

    return pd.DataFrame(stats)


def compute_correlation_matrix(df: pd.DataFrame, method: str = "pearson") -> Tuple[pd.DataFrame, List[Dict[str, Any]]]:
    """
    Compute correlation matrix for numeric columns and extract top correlated pairs.
    """
    num_df = df.select_dtypes(include=[np.number])
    if num_df.shape[1] < 2:
        return pd.DataFrame(), []

    corr = num_df.corr(method=method).round(3)

    # Extract pairs
    pairs = []
    cols = list(corr.columns)
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            col1 = cols[i]
            col2 = cols[j]
            val = corr.loc[col1, col2]
            if pd.notna(val):
                pairs.append({
                    "col1": col1,
                    "col2": col2,
                    "corr": float(val),
                    "abs_corr": abs(float(val)),
                    "direction": "Positive" if val > 0 else "Negative",
                    "strength": "Strong" if abs(val) >= 0.7 else ("Moderate" if abs(val) >= 0.4 else "Weak")
                })

    pairs.sort(key=lambda x: x["abs_corr"], reverse=True)
    return corr, pairs


def generate_automated_insights(df: pd.DataFrame, profile: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
    """
    Generate rigorous analytical and statistical observations based on the uploaded dataset.
    Never relies on external hallucinating LLM; uses actual computed numbers.
    """
    insights: List[Dict[str, Any]] = []
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()

    # 1. Primary Aggregate Drivers
    if num_cols and cat_cols:
        # Find numeric column with highest variation or importance
        target_num = None
        for cand in ["sales", "profit", "salary", "revenue", "exam_score", "marks", "attendance"]:
            matches = [c for c in num_cols if cand in c.lower()]
            if matches:
                target_num = matches[0]
                break
        if not target_num and num_cols:
            target_num = num_cols[0]

        target_cat = None
        for cand in ["category", "department", "region", "segment", "grade"]:
            matches = [c for c in cat_cols if cand in c.lower()]
            if matches:
                target_cat = matches[0]
                break
        if not target_cat and cat_cols:
            # Pick categorical with 2 to 15 unique values
            candidates = [c for c in cat_cols if 2 <= df[c].nunique() <= 15]
            if candidates:
                target_cat = candidates[0]

        if target_num and target_cat:
            grp = df.groupby(target_cat)[target_num].sum().sort_values(ascending=False)
            if not grp.empty:
                top_group = grp.index[0]
                top_val = grp.iloc[0]
                total_sum = grp.sum()
                pct_share = (top_val / total_sum * 100) if total_sum != 0 else 0

                insights.append({
                    "type": "Driver",
                    "badge_color": "primary",
                    "title": f"Top Volume Driver: {top_group} in {target_cat}",
                    "metric": f"{pct_share:.1f}% Share",
                    "narrative": (
                        f"Across all records, **{top_group}** accounts for the highest aggregated **{target_num}**, "
                        f"contributing **{format_compact_number(top_val)}** out of total **{format_compact_number(total_sum)}** "
                        f"({pct_share:.1f}% of cumulative total)."
                    ),
                    "recommendation": f"Focus resource allocation and strategic monitoring on the {top_group} cluster."
                })

    # 2. Dominant Category Distribution
    for cat in cat_cols[:2]:
        s = df[cat].dropna()
        if len(s) > 0 and 2 <= s.nunique() <= 20:
            val_counts = s.value_counts()
            top_name = val_counts.index[0]
            top_count = int(val_counts.iloc[0])
            top_share = (top_count / len(s)) * 100
            if top_share > 35.0:
                insights.append({
                    "type": "Distribution",
                    "badge_color": "info",
                    "title": f"High Concentration in {cat}: '{top_name}'",
                    "metric": f"{top_share:.1f}% Density",
                    "narrative": (
                        f"The **{cat}** attribute exhibits high concentration. The single value **'{top_name}'** "
                        f"represents **{top_count:,} out of {len(s):,} entries** ({top_share:.1f}% of all non-null records)."
                    ),
                    "recommendation": f"Verify whether '{top_name}' should be further segmented into sub-categories."
                })
                break

    # 3. Correlation Findings
    if len(num_cols) >= 2:
        _, pairs = compute_correlation_matrix(df)
        if pairs:
            top_pair = pairs[0]
            if top_pair["abs_corr"] >= 0.40:
                insights.append({
                    "type": "Correlation",
                    "badge_color": "success" if top_pair["corr"] > 0 else "warning",
                    "title": f"{top_pair['strength']} {top_pair['direction']} Correlation: {top_pair['col1']} & {top_pair['col2']}",
                    "metric": f"r = {top_pair['corr']:+.2f}",
                    "narrative": (
                        f"A statistically notable {top_pair['direction'].lower()} linear correlation "
                        f"(Pearson r = **{top_pair['corr']:+.2f}**) was measured between **{top_pair['col1']}** and **{top_pair['col2']}**. "
                        f"Changes in {top_pair['col1']} strongly mirror movements in {top_pair['col2']}."
                    ),
                    "recommendation": f"Consider using {top_pair['col1']} as a leading predictive feature for {top_pair['col2']}."
                })

    # 4. Outlier & Skewness Anomalies
    for num in num_cols:
        s = df[num].dropna()
        if len(s) >= 20:
            skew = s.skew()
            q25, q75 = s.quantile(0.25), s.quantile(0.75)
            iqr = q75 - q25
            if iqr > 0:
                outliers = s[(s < (q25 - 1.5 * iqr)) | (s > (q75 + 1.5 * iqr))]
                if len(outliers) > 0 and (len(outliers) / len(s)) > 0.015:
                    insights.append({
                        "type": "Anomaly",
                        "badge_color": "danger",
                        "title": f"Heavy Tail & Outliers in '{num}'",
                        "metric": f"{len(outliers)} Extreme Points",
                        "narrative": (
                            f"**{num}** exhibits a skewness coefficient of **{skew:+.2f}** with **{len(outliers):,} statistical outlier(s)** "
                            f"beyond 1.5x IQR bounds. Min value is {format_compact_number(s.min())} while Max reaches {format_compact_number(s.max())}."
                        ),
                        "recommendation": "Use median over mean for summary metrics, or apply Winsorization in Data Cleaning."
                    })
                    break

    # 5. Time Trend Analysis
    date_cols = [c for c in df.columns if "date" in c.lower() or "time" in c.lower() or pd.api.types.is_datetime64_any_dtype(df[c])]
    if date_cols and num_cols:
        dcol = date_cols[0]
        ncol = num_cols[0]
        try:
            temp_df = df[[dcol, ncol]].dropna().copy()
            temp_df[dcol] = pd.to_datetime(temp_df[dcol], errors="coerce")
            temp_df = temp_df.dropna().sort_values(by=dcol)
            if len(temp_df) > 20:
                midpoint = len(temp_df) // 2
                h1_mean = temp_df.iloc[:midpoint][ncol].mean()
                h2_mean = temp_df.iloc[midpoint:][ncol].mean()
                if h1_mean != 0:
                    pct_change = ((h2_mean - h1_mean) / abs(h1_mean)) * 100
                    direction = "Growth" if pct_change > 0 else "Decline"
                    insights.append({
                        "type": "Trend",
                        "badge_color": "success" if pct_change > 0 else "warning",
                        "title": f"Chronological Trend: {direction} in {ncol}",
                        "metric": f"{pct_change:+.1f}% Shift",
                        "narrative": (
                            f"Splitting records over chronological order of **{dcol}**, the average **{ncol}** shifted from "
                            f"**{format_compact_number(h1_mean)}** in the first half to **{format_compact_number(h2_mean)}** in the second half "
                            f"({pct_change:+.1f}% net change)."
                        ),
                        "recommendation": f"Monitor the temporal trajectory of {ncol} to assess seasonal or sustained momentum."
                    })
        except Exception:
            pass

    # 6. Data Quality Hygiene Note
    if profile:
        score = profile.get("quality_score", 100)
        missing_cnt = profile.get("total_missing_cells", 0)
        dups_cnt = profile.get("duplicate_rows", 0)
        if score < 85:
            insights.append({
                "type": "Quality",
                "badge_color": "warning",
                "title": f"Data Integrity Notice: Quality Score {score:.0f}%",
                "metric": f"Grade: {profile.get('quality_grade', 'N/A')}",
                "narrative": (
                    f"Dataset scored **{score:.1f}% ({profile.get('quality_grade', 'N/A')})**. Found **{missing_cnt:,} missing cell(s)** "
                    f"and **{dups_cnt:,} duplicate record(s)**. Applying automated cleaning will boost analytical confidence."
                ),
                "recommendation": "Navigate to the Data Cleaning tab to run imputation and deduplication."
            })

    return insights

