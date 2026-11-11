"""
Comprehensive test script for DataPilot modules.
"""
import sys
import os
import pandas as pd
import numpy as np

def run_tests():
    print("Testing DataPilot Modules...")

    # 1. Test Data Loader
    from modules.data_loader import load_file, get_sample_datasets_info, load_sample_dataset
    samples = get_sample_datasets_info()
    assert len(samples) == 3, f"Expected 3 sample datasets, got {len(samples)}"
    print("[PASS] Sample datasets info retrieved successfully.")

    for skey in ["sales", "student", "employee"]:
        df, meta = load_sample_dataset(skey)
        assert len(df) > 0, f"Dataset {skey} is empty"
        assert meta["rows"] == len(df), f"Metadata row count mismatch for {skey}"
        print(f"[PASS] Loaded {skey} sample dataset: {df.shape}")

    # 2. Test Data Profiler
    from modules.data_profiler import profile_dataset
    profile = profile_dataset(df)
    assert "quality_score" in profile, "Quality score missing in profile"
    assert "quality_grade" in profile, "Quality grade missing in profile"
    assert len(profile["columns"]) == len(df.columns), "Column count mismatch in profile"
    print(f"[PASS] Data Profiler: Quality score = {profile['quality_score']}% ({profile['quality_grade']})")

    # 3. Test Data Cleaner
    from modules.data_cleaner import inspect_cleaning_candidates, apply_cleaning_pipeline
    issues = inspect_cleaning_candidates(df)
    assert "missing_values" in issues, "Missing values inspection missing"
    assert "duplicates" in issues, "Duplicates inspection missing"
    assert "outliers" in issues, "Outliers inspection missing"
    print("[PASS] Data Cleaner: Cleaning candidates inspected successfully.")

    clean_opts = {
        "remove_duplicates": True,
        "duplicate_action": "first",
        "trim_whitespace": True,
        "text_casing": "title",
        "missing_strategy": "impute_smart",
        "numeric_impute_method": "median",
        "outlier_treatment": "cap",
        "iqr_multiplier": 1.5,
        "remove_empty_cols": True
    }
    cleaned_df, diff = apply_cleaning_pipeline(df, clean_opts)
    assert len(cleaned_df) <= len(df), "Cleaned dataframe length should be <= original"
    print(f"[PASS] Data Cleaner: Cleaning pipeline executed. Diff logs: {len(diff['logs'])} actions.")

    # 4. Test Data Analyzer
    from modules.data_analyzer import (
        compute_numerical_summary,
        compute_categorical_summary,
        compute_correlation_matrix,
        generate_automated_insights
    )
    num_summary = compute_numerical_summary(df)
    cat_summary = compute_categorical_summary(df)
    corr_mat, corr_pairs = compute_correlation_matrix(df)
    insights = generate_automated_insights(df, profile)

    print(f"[PASS] Data Analyzer: Num stats cols = {len(num_summary)}, Cat stats cols = {len(cat_summary)}")
    print(f"[PASS] Data Analyzer: Generated {len(insights)} analytical insights.")
    for ins in insights:
        print(f"       - [{ins['type']}] {ins['title']}: {ins['metric']}")

    # 5. Test Visualization
    from modules.visualization import create_chart, build_archetype_dashboard
    dash = build_archetype_dashboard(df, archetype=profile["archetype"], theme="light")
    assert len(dash["kpis"]) > 0, "No KPIs returned for dashboard"
    assert len(dash["charts"]) > 0, "No charts returned for dashboard"
    print(f"[PASS] Visualization: Generated archetype dashboard with {len(dash['kpis'])} KPIs and {len(dash['charts'])} charts.")

    # 6. Test Exports
    from utils.helpers import (
        dataframe_to_csv_bytes,
        dataframe_to_excel_bytes,
        export_quality_report_json,
        generate_executive_markdown_report
    )
    csv_bytes = dataframe_to_csv_bytes(cleaned_df)
    assert len(csv_bytes) > 0, "CSV bytes generation failed"
    excel_bytes = dataframe_to_excel_bytes(cleaned_df)
    assert len(excel_bytes) > 0, "Excel bytes generation failed"
    json_str = export_quality_report_json(profile)
    assert len(json_str) > 0, "JSON export failed"
    md_rep = generate_executive_markdown_report("employee_sample.csv", cleaned_df, profile, insights)
    assert len(md_rep) > 0, "Executive markdown report generation failed"
    print("[PASS] Utils: All export generators (CSV, Excel, JSON, Markdown) verified.")

    print("\nALL DATAPILOT TESTS PASSED PERFECTLY!")

if __name__ == "__main__":
    run_tests()
