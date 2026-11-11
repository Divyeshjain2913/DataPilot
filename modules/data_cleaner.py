"""
DataPilot - Data Cleaner Module
Interactive, non-destructive data cleaning engine supporting 8 modular operations,
before-and-after diff inspection, and configurable transformations.
"""

from __future__ import annotations
from typing import Any, Dict, List, Tuple, Optional
import numpy as np
import pandas as pd
from utils.helpers import format_bytes


def inspect_cleaning_candidates(df: pd.DataFrame) -> Dict[str, Dict[str, Any]]:
    """
    Inspect the DataFrame across all 8 cleaning dimensions and return
    issue counts, severity indicators, and recommended solutions.
    """
    total_rows = len(df)
    results: Dict[str, Dict[str, Any]] = {}

    # 1. Missing Values
    null_counts = df.isnull().sum()
    cols_with_nulls = {col: int(cnt) for col, cnt in null_counts.items() if cnt > 0}
    total_null_cells = int(null_counts.sum())
    null_rows = int(df.isnull().any(axis=1).sum())

    if total_null_cells == 0:
        results["missing_values"] = {
            "status": "clean",
            "count": 0,
            "affected_rows": 0,
            "details": "No missing values detected. All fields are complete.",
            "recommended": "Leave unchanged",
            "columns": {}
        }
    else:
        sev = "severe" if (null_rows / max(total_rows, 1)) > 0.15 else "warning"
        results["missing_values"] = {
            "status": sev,
            "count": total_null_cells,
            "affected_rows": null_rows,
            "details": f"{total_null_cells:,} missing cells found across {len(cols_with_nulls)} column(s). Affects {null_rows:,} rows ({(null_rows / max(total_rows, 1)) * 100:.1f}%).",
            "recommended": "Impute numeric with Median, categorical with Mode, or drop sparse rows.",
            "columns": cols_with_nulls
        }

    # 2. Duplicate Records
    dup_count = int(df.duplicated().sum())
    if dup_count == 0:
        results["duplicates"] = {
            "status": "clean",
            "count": 0,
            "affected_rows": 0,
            "details": "No duplicate rows found. Every record is unique.",
            "recommended": "None required"
        }
    else:
        sev = "severe" if (dup_count / max(total_rows, 1)) > 0.05 else "warning"
        results["duplicates"] = {
            "status": sev,
            "count": dup_count,
            "affected_rows": dup_count,
            "details": f"{dup_count:,} duplicate rows identified ({(dup_count / max(total_rows, 1)) * 100:.1f}% of dataset).",
            "recommended": "Remove duplicate rows, keeping the first occurrence.",
        }

    # 3. Incorrect Data Types (Numeric stored as object or mixed)
    convertible_cols: Dict[str, str] = {}
    for col in df.select_dtypes(include=["object"]).columns:
        s = df[col].dropna()
        if len(s) == 0:
            continue
        # Check if numeric when stripped of currency/commas
        cleaned_s = s.astype(str).str.replace(r"[\$,₹€£,\s]", "", regex=True)
        numeric_test = pd.to_numeric(cleaned_s, errors="coerce")
        if numeric_test.notnull().mean() > 0.85:
            convertible_cols[col] = "numeric"
        else:
            # Check if date with preliminary pattern check to avoid parsing arbitrary text strings
            sample_str = s.head(30).astype(str)
            if sample_str.str.contains(r"^\d{4}[-/]\d{1,2}[-/]\d{1,2}|\d{1,2}[-/]\d{1,2}[-/]\d{2,4}", regex=True).mean() > 0.6:
                date_test = pd.to_datetime(s, errors="coerce")
                if date_test.notnull().mean() > 0.85:
                    convertible_cols[col] = "datetime"

    if len(convertible_cols) == 0:
        results["incorrect_types"] = {
            "status": "clean",
            "count": 0,
            "affected_rows": 0,
            "details": "All column data types align with their underlying values.",
            "recommended": "None required",
            "columns": {}
        }
    else:
        results["incorrect_types"] = {
            "status": "warning",
            "count": len(convertible_cols),
            "affected_rows": 0,
            "details": f"{len(convertible_cols)} column(s) are stored as text but contain numbers or dates: {', '.join(convertible_cols.keys())}.",
            "recommended": "Convert to native numeric or datetime types for better analysis.",
            "columns": convertible_cols
        }

    # 4. Extra Whitespace
    ws_cols: Dict[str, int] = {}
    for col in df.select_dtypes(include=["object", "string"]).columns:
        s = df[col].dropna().astype(str)
        cnt = int(s.str.match(r"^\s+|\s+$|\s{2,}").sum())
        if cnt > 0:
            ws_cols[col] = cnt

    total_ws = sum(ws_cols.values())
    if total_ws == 0:
        results["extra_whitespace"] = {
            "status": "clean",
            "count": 0,
            "affected_rows": 0,
            "details": "Text columns are clean with no leading or trailing whitespace.",
            "recommended": "None required",
            "columns": {}
        }
    else:
        results["extra_whitespace"] = {
            "status": "warning",
            "count": total_ws,
            "affected_rows": total_ws,
            "details": f"{total_ws:,} cell(s) across {len(ws_cols)} column(s) contain unwanted whitespace.",
            "recommended": "Strip leading/trailing spaces and collapse multiple spaces.",
            "columns": ws_cols
        }

    # 5. Inconsistent Text (Casing variations)
    casing_cols: Dict[str, int] = {}
    for col in df.select_dtypes(include=["object", "category"]).columns:
        s = df[col].dropna().astype(str)
        counts = s.value_counts()
        lowered = counts.index.str.strip().str.lower()
        num_dups = int(lowered.duplicated().sum())
        if num_dups > 0:
            casing_cols[col] = num_dups

    if len(casing_cols) == 0:
        results["inconsistent_text"] = {
            "status": "clean",
            "count": 0,
            "affected_rows": 0,
            "details": "Text capitalization is consistent across categorical values.",
            "recommended": "None required",
            "columns": {}
        }
    else:
        results["inconsistent_text"] = {
            "status": "warning",
            "count": sum(casing_cols.values()),
            "affected_rows": sum(casing_cols.values()),
            "details": f"{len(casing_cols)} column(s) contain mixed casing (e.g. 'Tech' vs 'tech'): {', '.join(casing_cols.keys())}.",
            "recommended": "Standardize to Title Case or lowercase.",
            "columns": casing_cols
        }

    # 6. Invalid Dates
    date_cols: Dict[str, int] = {}
    for col in df.columns:
        col_lower = col.lower()
        if "date" in col_lower or "time" in col_lower or pd.api.types.is_datetime64_any_dtype(df[col]):
            s = df[col].dropna()
            parsed = pd.to_datetime(s, errors="coerce")
            unparseable = int(parsed.isnull().sum())
            if unparseable > 0:
                date_cols[col] = unparseable

    if len(date_cols) == 0:
        results["invalid_dates"] = {
            "status": "clean",
            "count": 0,
            "affected_rows": 0,
            "details": "Date values are valid or properly formatted.",
            "recommended": "None required",
            "columns": {}
        }
    else:
        results["invalid_dates"] = {
            "status": "warning",
            "count": sum(date_cols.values()),
            "affected_rows": sum(date_cols.values()),
            "details": f"{sum(date_cols.values())} unparseable or corrupted date strings found in {', '.join(date_cols.keys())}.",
            "recommended": "Coerce to valid standard ISO datetime format (YYYY-MM-DD).",
            "columns": date_cols
        }

    # 7. Outliers
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    outlier_cols: Dict[str, int] = {}
    for col in numeric_cols:
        s = df[col].dropna()
        if len(s) >= 5:
            q25, q75 = s.quantile(0.25), s.quantile(0.75)
            iqr = q75 - q25
            if iqr > 0:
                n_out = int(((s < (q25 - 1.5 * iqr)) | (s > (q75 + 1.5 * iqr))).sum())
                if n_out > 0:
                    outlier_cols[col] = n_out

    total_outliers = sum(outlier_cols.values())
    if total_outliers == 0:
        results["outliers"] = {
            "status": "clean",
            "count": 0,
            "affected_rows": 0,
            "details": "No statistical outliers detected under 1.5x IQR rule.",
            "recommended": "None required",
            "columns": {}
        }
    else:
        sev = "warning" if (total_outliers / max(total_rows, 1)) < 0.1 else "severe"
        results["outliers"] = {
            "status": sev,
            "count": total_outliers,
            "affected_rows": total_outliers,
            "details": f"{total_outliers:,} statistical outlier points detected across {len(outlier_cols)} numeric column(s).",
            "recommended": "Cap (Winsorize) at 1.5x IQR boundary or inspect for data entry errors.",
            "columns": outlier_cols
        }

    # 8. Empty Columns (100% missing or >90% missing)
    empty_cols = [c for c in df.columns if null_counts[c] == total_rows and total_rows > 0]
    sparse_cols = [c for c in df.columns if (null_counts[c] / max(total_rows, 1)) >= 0.90 and c not in empty_cols]

    if len(empty_cols) == 0 and len(sparse_cols) == 0:
        results["empty_columns"] = {
            "status": "clean",
            "count": 0,
            "affected_rows": 0,
            "details": "No completely empty or ultra-sparse (>90% empty) columns found.",
            "recommended": "None required",
            "columns": []
        }
    else:
        all_empty = empty_cols + sparse_cols
        results["empty_columns"] = {
            "status": "severe" if len(empty_cols) > 0 else "warning",
            "count": len(all_empty),
            "affected_rows": total_rows,
            "details": f"{len(empty_cols)} completely empty column(s) and {len(sparse_cols)} sparse column(s) (>90% null).",
            "recommended": "Remove empty/sparse columns to simplify dataset structure.",
            "columns": all_empty
        }

    return results


def apply_cleaning_pipeline(
    df: pd.DataFrame,
    options: Dict[str, Any]
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Apply configured cleaning operations non-destructively and return (cleaned_df, report_diff).
    """
    cleaned = df.copy()
    logs: List[str] = []

    rows_before = len(cleaned)
    nulls_before = int(cleaned.isnull().sum().sum())
    dups_before = int(cleaned.duplicated().sum())
    memory_before = int(cleaned.memory_usage(deep=True).sum())

    # 1. Empty Columns Removal
    if options.get("remove_empty_cols", False):
        empty_to_drop = [c for c in cleaned.columns if cleaned[c].isnull().sum() == len(cleaned)]
        if options.get("remove_sparse_cols", False):
            sparse = [c for c in cleaned.columns if (cleaned[c].isnull().sum() / max(len(cleaned), 1)) >= 0.90]
            empty_to_drop = list(set(empty_to_drop + sparse))

        if empty_to_drop:
            cleaned = cleaned.drop(columns=empty_to_drop)
            logs.append(f"Dropped {len(empty_to_drop)} empty/sparse column(s): {', '.join(empty_to_drop)}")

    # 2. Duplicate Records Handling
    if options.get("remove_duplicates", False):
        dup_action = options.get("duplicate_action", "first")
        initial_count = len(cleaned)
        if dup_action == "all":
            cleaned = cleaned.drop_duplicates(keep=False)
        else:
            cleaned = cleaned.drop_duplicates(keep=dup_action)
        removed_dups = initial_count - len(cleaned)
        if removed_dups > 0:
            logs.append(f"Removed {removed_dups:,} duplicate row(s) (kept {dup_action}).")

    # 3. Extra Whitespace Trimming
    if options.get("trim_whitespace", False):
        text_cols = cleaned.select_dtypes(include=["object", "string"]).columns
        trimmed_count = 0
        for col in text_cols:
            non_null = cleaned[col].dropna()
            if len(non_null) > 0:
                cleaned[col] = cleaned[col].apply(
                    lambda x: " ".join(str(x).strip().split()) if pd.notna(x) else x
                )
                trimmed_count += 1
        if trimmed_count > 0:
            logs.append(f"Trimmed leading/trailing and redundant whitespace across {trimmed_count} text column(s).")

    # 4. Inconsistent Text Standardization
    casing_mode = options.get("text_casing", "none")
    if casing_mode in ["title", "lower", "upper"]:
        casing_cols = options.get("casing_columns", [])
        if not casing_cols:
            casing_cols = list(cleaned.select_dtypes(include=["object"]).columns)

        modified = 0
        for col in casing_cols:
            if col in cleaned.columns:
                if casing_mode == "title":
                    cleaned[col] = cleaned[col].apply(lambda x: str(x).strip().title() if pd.notna(x) else x)
                elif casing_mode == "lower":
                    cleaned[col] = cleaned[col].apply(lambda x: str(x).strip().lower() if pd.notna(x) else x)
                elif casing_mode == "upper":
                    cleaned[col] = cleaned[col].apply(lambda x: str(x).strip().upper() if pd.notna(x) else x)
                modified += 1
        if modified > 0:
            logs.append(f"Standardized text casing to {casing_mode.capitalize()} Case in {modified} column(s).")

    # 5. Type Conversions & Invalid Dates
    if options.get("convert_dates", False):
        date_cols = options.get("date_columns", [])
        for col in date_cols:
            if col in cleaned.columns:
                try:
                    cleaned[col] = pd.to_datetime(cleaned[col], errors="coerce")
                    logs.append(f"Converted '{col}' to datetime format.")
                except Exception as e:
                    logs.append(f"Could not convert '{col}' to date: {e}")

    if options.get("convert_numeric", False):
        num_cols = options.get("numeric_columns", [])
        for col in num_cols:
            if col in cleaned.columns:
                try:
                    # Clean currency symbols and commas first
                    cleaned[col] = cleaned[col].astype(str).str.replace(r"[\$,₹€£,\s]", "", regex=True)
                    cleaned[col] = pd.to_numeric(cleaned[col], errors="coerce")
                    logs.append(f"Converted '{col}' to numeric.")
                except Exception as e:
                    logs.append(f"Could not convert '{col}' to numeric: {e}")

    # 6. Missing Values Imputation / Dropping
    missing_strategy = options.get("missing_strategy", "none")
    if missing_strategy != "none":
        num_cols = cleaned.select_dtypes(include=[np.number]).columns
        cat_cols = cleaned.select_dtypes(include=["object", "category"]).columns

        if missing_strategy == "drop_rows":
            drop_target = options.get("missing_drop_columns", "all")
            initial_len = len(cleaned)
            if drop_target == "all":
                cleaned = cleaned.dropna()
            else:
                target_cols = [c for c in options.get("missing_cols_list", []) if c in cleaned.columns]
                cleaned = cleaned.dropna(subset=target_cols)
            logs.append(f"Dropped {initial_len - len(cleaned):,} rows with missing values.")

        elif missing_strategy == "impute_smart":
            # Mean or Median for numeric, Mode for categorical
            num_method = options.get("numeric_impute_method", "median")
            imputed_cells = 0
            for col in num_cols:
                n_miss = cleaned[col].isnull().sum()
                if n_miss > 0:
                    fill_val = cleaned[col].median() if num_method == "median" else cleaned[col].mean()
                    if pd.notna(fill_val):
                        cleaned[col] = cleaned[col].fillna(fill_val)
                        imputed_cells += int(n_miss)

            for col in cat_cols:
                n_miss = cleaned[col].isnull().sum()
                if n_miss > 0:
                    mode_series = cleaned[col].mode()
                    if not mode_series.empty:
                        fill_val = mode_series.iloc[0]
                        cleaned[col] = cleaned[col].fillna(fill_val)
                        imputed_cells += int(n_miss)

            logs.append(f"Imputed {imputed_cells:,} missing cells (numeric with {num_method}, text with mode).")

        elif missing_strategy == "constant":
            const_num = options.get("constant_numeric_val", 0)
            const_text = options.get("constant_text_val", "Unknown")
            for col in num_cols:
                cleaned[col] = cleaned[col].fillna(const_num)
            for col in cat_cols:
                cleaned[col] = cleaned[col].fillna(const_text)
            logs.append(f"Filled missing numeric values with '{const_num}' and text with '{const_text}'.")

        elif missing_strategy == "ffill":
            cleaned = cleaned.ffill().bfill()
            logs.append("Applied forward and backward fill across missing records.")

    # 7. Outliers Handling
    outlier_treatment = options.get("outlier_treatment", "none")
    if outlier_treatment in ["cap", "drop"]:
        target_outlier_cols = options.get("outlier_columns", [])
        if not target_outlier_cols:
            target_outlier_cols = list(cleaned.select_dtypes(include=[np.number]).columns)

        multiplier = float(options.get("iqr_multiplier", 1.5))
        treated_count = 0

        if outlier_treatment == "cap":
            for col in target_outlier_cols:
                if col in cleaned.columns and pd.api.types.is_numeric_dtype(cleaned[col]):
                    s = cleaned[col].dropna()
                    if len(s) >= 5:
                        q25, q75 = s.quantile(0.25), s.quantile(0.75)
                        iqr = q75 - q25
                        if iqr > 0:
                            lb, ub = q25 - multiplier * iqr, q75 + multiplier * iqr
                            n_out = ((cleaned[col] < lb) | (cleaned[col] > ub)).sum()
                            if n_out > 0:
                                cleaned[col] = cleaned[col].clip(lower=lb, upper=ub)
                                treated_count += int(n_out)
            if treated_count > 0:
                logs.append(f"Winsorized (capped) {treated_count:,} outlier values across {len(target_outlier_cols)} columns using {multiplier}x IQR.")

        elif outlier_treatment == "drop":
            init_rows = len(cleaned)
            for col in target_outlier_cols:
                if col in cleaned.columns and pd.api.types.is_numeric_dtype(cleaned[col]):
                    s = cleaned[col].dropna()
                    if len(s) >= 5:
                        q25, q75 = s.quantile(0.25), s.quantile(0.75)
                        iqr = q75 - q25
                        if iqr > 0:
                            lb, ub = q25 - multiplier * iqr, q75 + multiplier * iqr
                            cleaned = cleaned[(cleaned[col].isnull()) | ((cleaned[col] >= lb) & (cleaned[col] <= ub))]
            dropped_rows = init_rows - len(cleaned)
            if dropped_rows > 0:
                logs.append(f"Dropped {dropped_rows:,} rows containing extreme statistical outliers.")

    rows_after = len(cleaned)
    nulls_after = int(cleaned.isnull().sum().sum())
    dups_after = int(cleaned.duplicated().sum())
    memory_after = int(cleaned.memory_usage(deep=True).sum())

    diff_report = {
        "rows_before": rows_before,
        "rows_after": rows_after,
        "rows_removed": rows_before - rows_after,
        "nulls_before": nulls_before,
        "nulls_after": nulls_after,
        "nulls_resolved": nulls_before - nulls_after,
        "dups_before": dups_before,
        "dups_after": dups_after,
        "dups_removed": dups_before - dups_after,
        "memory_before_str": format_bytes(memory_before),
        "memory_after_str": format_bytes(memory_after),
        "logs": logs
    }

    return cleaned, diff_report
