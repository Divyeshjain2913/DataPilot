"""
DataPilot - Data Profiler Module
Provides deep dataset profiling, column classification, missingness, duplicate analysis,
outlier detection (IQR & Z-score), and composite Data Quality Scoring (0-100%).
"""

from __future__ import annotations
import re
from typing import Any, Dict, List, Tuple
import numpy as np
import pandas as pd
from utils.helpers import format_bytes


def classify_column(series: pd.Series, col_name: str) -> str:
    """Classify a pandas Series into semantic types: numeric, categorical, datetime, text, boolean, or id."""
    col_lower = col_name.lower()

    # 1. Check ID-like columns
    if any(k in col_lower for k in ["id", "uuid", "guid", "code", "index", "key"]) and not pd.api.types.is_numeric_dtype(series):
        if series.nunique(dropna=True) / max(len(series.dropna()), 1) > 0.8:
            return "id"

    # 2. Check Boolean
    if pd.api.types.is_bool_dtype(series):
        return "boolean"
    non_null = series.dropna()
    if len(non_null) > 0:
        unique_vals = set(non_null.unique())
        if unique_vals.issubset({True, False, 1, 0, "1", "0", "true", "false", "yes", "no", "Yes", "No", "Y", "N"}):
            return "boolean"

    # 3. Check Datetime
    if pd.api.types.is_datetime64_any_dtype(series):
        return "datetime"
    if series.dtype == "object":
        sample = non_null.head(50).astype(str)
        # Check if sample looks like ISO dates or common date strings
        date_pattern = re.compile(r"^\d{4}[-/]\d{1,2}[-/]\d{1,2}|\d{1,2}[-/]\d{1,2}[-/]\d{2,4}")
        if len(sample) > 0 and (sample.str.contains(date_pattern).mean() > 0.75):
            try:
                pd.to_datetime(sample, errors="raise")
                return "datetime"
            except Exception:
                pass

    # 4. Check Numeric
    if pd.api.types.is_numeric_dtype(series):
        return "numeric"

    # 5. Check Categorical vs Text
    if series.dtype == "object" or isinstance(series.dtype, pd.CategoricalDtype):
        unique_count = series.nunique(dropna=True)
        total_valid = len(non_null)
        if total_valid == 0:
            return "categorical"
        unique_ratio = unique_count / total_valid
        if unique_count <= 60 or unique_ratio < 0.2:
            return "categorical"
        return "text"

    return "text"


def detect_outliers_iqr(series: pd.Series, multiplier: float = 1.5) -> Tuple[int, float, float]:
    """Detect outliers in a numeric series using Tukey's IQR rule."""
    clean_series = series.dropna()
    if len(clean_series) < 5:
        return 0, 0.0, 0.0

    q25 = clean_series.quantile(0.25)
    q75 = clean_series.quantile(0.75)
    iqr = q75 - q25

    if iqr == 0:
        return 0, float(q25), float(q75)

    lower_bound = float(q25 - multiplier * iqr)
    upper_bound = float(q75 + multiplier * iqr)

    outliers = clean_series[(clean_series < lower_bound) | (clean_series > upper_bound)]
    return len(outliers), lower_bound, upper_bound


def detect_extra_whitespace(series: pd.Series) -> int:
    """Count how many non-null string cells contain leading, trailing, or double whitespace."""
    if not (series.dtype == "object" or isinstance(series.dtype, pd.StringDtype)):
        return 0
    non_null = series.dropna().astype(str)
    has_ws = non_null.str.match(r"^\s+|\s+$|\s{2,}")
    return int(has_ws.sum())


def detect_inconsistent_casing(series: pd.Series) -> int:
    """Detect categories that exist in multiple casing variations (e.g. 'Tech' and 'tech')."""
    if series.dtype != "object" and not isinstance(series.dtype, pd.CategoricalDtype):
        return 0
    non_null = series.dropna().astype(str)
    val_counts = non_null.value_counts()
    lowered = val_counts.index.str.strip().str.lower()
    duplicates_lower = lowered[lowered.duplicated(keep=False)]
    return len(duplicates_lower)


def infer_dataset_archetype(df: pd.DataFrame) -> str:
    """Infer whether dataset is Sales, Student/Education, Employee/HR, or General."""
    cols_lower = [c.lower() for c in df.columns]

    sales_keywords = ["sales", "profit", "revenue", "order", "discount", "customer", "product", "shipping", "sub_category"]
    student_keywords = ["student", "marks", "grade", "attendance", "exam", "study", "placement", "course", "subject"]
    employee_keywords = ["employee", "salary", "department", "experience", "hire", "tenure", "attrition", "rating", "performance"]

    sales_matches = sum(any(k in c for k in sales_keywords) for c in cols_lower)
    student_matches = sum(any(k in c for k in student_keywords) for c in cols_lower)
    emp_matches = sum(any(k in c for k in employee_keywords) for c in cols_lower)

    scores = {"sales": sales_matches, "student": student_matches, "employee": emp_matches}
    best_match = max(scores, key=scores.get)

    if scores[best_match] >= 2:
        return best_match
    return "general"


def profile_dataset(df: pd.DataFrame) -> Dict[str, Any]:
    """Perform a comprehensive profile and data quality assessment of the DataFrame."""
    total_rows = len(df)
    total_cols = len(df.columns)
    total_cells = max(total_rows * total_cols, 1)

    null_counts = df.isnull().sum()
    total_nulls = int(null_counts.sum())
    missing_pct = round((total_nulls / total_cells) * 100, 2)

    duplicate_rows = int(df.duplicated().sum())
    duplicate_pct = round((duplicate_rows / max(total_rows, 1)) * 100, 2)

    memory_bytes = int(df.memory_usage(deep=True).sum())
    memory_str = format_bytes(memory_bytes)

    empty_cols = [c for c in df.columns if null_counts[c] == total_rows and total_rows > 0]

    col_profiles: List[Dict[str, Any]] = []
    type_counts = {"numeric": 0, "categorical": 0, "datetime": 0, "text": 0, "boolean": 0, "id": 0}
    total_outliers = 0
    total_whitespace_cells = 0
    total_casing_issues = 0

    for col in df.columns:
        series = df[col]
        inferred = classify_column(series, col)
        type_counts[inferred] = type_counts.get(inferred, 0) + 1

        n_null = int(null_counts[col])
        n_valid = total_rows - n_null
        non_null_pct = round((n_valid / max(total_rows, 1)) * 100, 1)
        null_col_pct = round((n_null / max(total_rows, 1)) * 100, 1)
        n_unique = int(series.nunique(dropna=True))
        unique_pct = round((n_unique / max(total_rows, 1)) * 100, 1)

        # Non-null sample values
        samples = series.dropna().head(3).tolist()

        # Stats & Outliers
        stats: Dict[str, Any] = {}
        outlier_count = 0
        outlier_bounds = (0.0, 0.0)

        if inferred == "numeric":
            valid_numeric = pd.to_numeric(series, errors="coerce").dropna()
            if len(valid_numeric) > 0:
                stats["min"] = float(valid_numeric.min())
                stats["max"] = float(valid_numeric.max())
                stats["mean"] = float(valid_numeric.mean())
                stats["median"] = float(valid_numeric.median())
                stats["std"] = float(valid_numeric.std()) if len(valid_numeric) > 1 else 0.0
                stats["skew"] = float(valid_numeric.skew()) if len(valid_numeric) > 2 else 0.0

                outlier_count, lower_b, upper_b = detect_outliers_iqr(valid_numeric)
                outlier_bounds = (round(lower_b, 2), round(upper_b, 2))
                total_outliers += outlier_count

        elif inferred in ["categorical", "text"]:
            val_counts = series.dropna().value_counts()
            if len(val_counts) > 0:
                top_val = val_counts.index[0]
                top_freq = int(val_counts.iloc[0])
                stats["mode"] = str(top_val)
                stats["mode_freq"] = top_freq
                stats["mode_pct"] = round((top_freq / max(n_valid, 1)) * 100, 1)

        # Text cleanliness
        ws_count = detect_extra_whitespace(series)
        total_whitespace_cells += ws_count
        case_issues = detect_inconsistent_casing(series)
        total_casing_issues += case_issues

        # Column Quality Score (0 - 100)
        # Completeness penalty:
        comp_score = non_null_pct  # 0 to 100
        # Outlier penalty for numeric
        outlier_penalty = min((outlier_count / max(n_valid, 1)) * 100, 25) if inferred == "numeric" else 0
        ws_penalty = min((ws_count / max(n_valid, 1)) * 50, 20)
        col_quality = max(0.0, min(100.0, comp_score - outlier_penalty - ws_penalty))

        col_profiles.append({
            "name": col,
            "native_dtype": str(series.dtype),
            "inferred_type": inferred,
            "non_null_count": n_valid,
            "non_null_pct": non_null_pct,
            "null_count": n_null,
            "null_pct": null_col_pct,
            "unique_count": n_unique,
            "unique_pct": unique_pct,
            "sample_values": samples,
            "outlier_count": outlier_count,
            "outlier_bounds": outlier_bounds,
            "whitespace_count": ws_count,
            "casing_issues": case_issues,
            "quality_score": round(col_quality, 1),
            "stats": stats
        })

    # Overall Data Quality Score Calculation
    # 1. Completeness (weight 35%)
    completeness_score = max(0.0, 100.0 - missing_pct)

    # 2. Uniqueness (weight 25%)
    uniqueness_score = max(0.0, 100.0 - (duplicate_pct * 2.0))

    # 3. Validity (weight 20%)
    numeric_cells = sum(c["non_null_count"] for c in col_profiles if c["inferred_type"] == "numeric")
    if numeric_cells > 0:
        outlier_ratio = total_outliers / numeric_cells
        validity_score = max(0.0, 100.0 - (outlier_ratio * 100.0 * 1.5))
    else:
        validity_score = 100.0

    # 4. Consistency (weight 20%)
    consistency_penalties = 0.0
    if empty_cols:
        consistency_penalties += min(len(empty_cols) * 10.0, 30.0)
    if total_whitespace_cells > 0:
        consistency_penalties += min((total_whitespace_cells / max(total_cells, 1)) * 100 * 2.0, 25.0)
    if total_casing_issues > 0:
        consistency_penalties += min(total_casing_issues * 3.0, 20.0)
    consistency_score = max(0.0, 100.0 - consistency_penalties)

    overall_score = round(
        0.35 * completeness_score +
        0.25 * uniqueness_score +
        0.20 * validity_score +
        0.20 * consistency_score,
        1
    )

    # Assign Letter Grade
    if overall_score >= 93: grade = "A+"
    elif overall_score >= 85: grade = "A"
    elif overall_score >= 75: grade = "B"
    elif overall_score >= 65: grade = "C"
    elif overall_score >= 50: grade = "D"
    else: grade = "F"

    archetype = infer_dataset_archetype(df)

    return {
        "total_rows": total_rows,
        "total_columns": total_cols,
        "total_cells": total_cells,
        "total_missing_cells": total_nulls,
        "missing_pct": missing_pct,
        "duplicate_rows": duplicate_rows,
        "duplicate_pct": duplicate_pct,
        "empty_columns": empty_cols,
        "memory_usage_bytes": memory_bytes,
        "memory_usage_str": memory_str,
        "type_counts": type_counts,
        "columns": col_profiles,
        "quality_score": overall_score,
        "quality_grade": grade,
        "completeness_pct": round(completeness_score, 1),
        "uniqueness_pct": round(uniqueness_score, 1),
        "validity_pct": round(validity_score, 1),
        "consistency_pct": round(consistency_score, 1),
        "total_outliers": total_outliers,
        "total_whitespace_cells": total_whitespace_cells,
        "total_casing_issues": total_casing_issues,
        "archetype": archetype
    }

