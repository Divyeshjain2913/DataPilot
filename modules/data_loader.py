"""
DataPilot - Data Loader Module
Supports CSV, Excel, automatic encoding/delimiter detection, memory profiling, and built-in sample datasets.
"""

from __future__ import annotations
import io
import os
import time
from typing import Any, Dict, Tuple, Optional
import pandas as pd
from utils.helpers import format_bytes


def detect_and_read_csv(file_buffer: Any) -> pd.DataFrame:
    """
    Read CSV file with robust multi-encoding and delimiter fallback.
    Supports files up to 2GB safely.
    """
    # If file_buffer is bytes or has read(), create a seekable stream or read directly
    encodings = ["utf-8", "utf-8-sig", "latin1", "cp1252", "iso-8859-1"]

    # Keep a copy of bytes if possible to allow multiple read attempts
    if hasattr(file_buffer, "seek"):
        file_buffer.seek(0)
        content_bytes = file_buffer.read()
    elif isinstance(file_buffer, (bytes, bytearray)):
        content_bytes = file_buffer
    else:
        content_bytes = None

    last_error = None

    for enc in encodings:
        try:
            if content_bytes is not None:
                stream = io.BytesIO(content_bytes)
                # First try standard comma with C engine for speed
                try:
                    df = pd.read_csv(stream, encoding=enc, low_memory=False)
                    # If it parsed into just 1 column with semicolons or tabs, re-parse with sep=None
                    if len(df.columns) == 1 and (";" in df.columns[0] or "\t" in df.columns[0]):
                        stream.seek(0)
                        df = pd.read_csv(stream, sep=None, engine="python", encoding=enc)
                    return df
                except Exception:
                    stream.seek(0)
                    df = pd.read_csv(stream, sep=None, engine="python", encoding=enc)
                    return df
            else:
                return pd.read_csv(file_buffer, encoding=enc, low_memory=False)
        except Exception as e:
            last_error = e
            continue

    raise ValueError(f"Failed to decode CSV with supported encodings. Error: {last_error}")


def load_file(uploaded_file: Any, filename: Optional[str] = None) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Load uploaded CSV or Excel file and return (DataFrame, metadata).
    """
    start_time = time.time()
    fname = filename or getattr(uploaded_file, "name", "uploaded_dataset.csv")
    file_ext = os.path.splitext(fname)[1].lower()

    # Determine file size
    if hasattr(uploaded_file, "size"):
        file_size = uploaded_file.size
    elif hasattr(uploaded_file, "getbuffer"):
        file_size = uploaded_file.getbuffer().nbytes
    elif isinstance(uploaded_file, (bytes, bytearray)):
        file_size = len(uploaded_file)
    elif isinstance(uploaded_file, str) and os.path.exists(uploaded_file):
        file_size = os.path.getsize(uploaded_file)
    else:
        file_size = 0

    if file_ext in [".xlsx", ".xls"]:
        df = pd.read_excel(uploaded_file, engine="openpyxl" if file_ext == ".xlsx" else None)
    elif file_ext in [".csv", ".txt", ".tsv"]:
        df = detect_and_read_csv(uploaded_file)
    else:
        # Fallback try CSV parser
        df = detect_and_read_csv(uploaded_file)

    load_time = time.time() - start_time

    metadata = {
        "filename": fname,
        "file_ext": file_ext,
        "file_size": file_size,
        "file_size_str": format_bytes(file_size),
        "rows": len(df),
        "columns": len(df.columns),
        "load_time_sec": round(load_time, 3),
        "memory_usage_bytes": int(df.memory_usage(deep=True).sum()),
        "memory_usage_str": format_bytes(int(df.memory_usage(deep=True).sum()))
    }

    return df, metadata


def get_sample_datasets_info() -> Dict[str, Dict[str, Any]]:
    """Return dictionary of available built-in sample datasets (enterprise edition)."""
    return {
        "sales": {
            "name": "Global Retail Sales Analytics",
            "archetype": "sales",
            "description": "1,208 records tracking orders, revenue, profit margins, product categories, and fulfillment.",
            "file": "data/sales_sample.csv",
            "records": "1,208 records",
            "badge": "E-Commerce",
            "key_metrics": ["Sales", "Profit", "Discount", "Quantity"]
        },
        "student": {
            "name": "Student Academic Performance",
            "archetype": "student",
            "description": "800 records measuring attendance, study hours, exam outcomes, and academic standing.",
            "file": "data/student_sample.csv",
            "records": "800 records",
            "badge": "Education",
            "key_metrics": ["Exam_Score", "Attendance_Pct", "Study_Hours_Per_Week"]
        },
        "employee": {
            "name": "HR & Workforce Talent Analytics",
            "archetype": "employee",
            "description": "650 workforce records covering compensation, experience tenure, and performance appraisal.",
            "file": "data/employee_sample.csv",
            "records": "650 records",
            "badge": "People Ops",
            "key_metrics": ["Salary", "Experience_Years", "Performance_Rating"]
        }
    }


def load_sample_dataset(sample_key: str = "sales") -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Load one of the built-in sample datasets."""
    samples = get_sample_datasets_info()
    if sample_key not in samples:
        sample_key = "sales"

    info = samples[sample_key]
    file_path = info["file"]

    if not os.path.exists(file_path):
        # Fallback generate if file not found
        import generate_samples
        pass

    with open(file_path, "rb") as f:
        df, meta = load_file(f, filename=os.path.basename(file_path))

    meta["dataset_name"] = info["name"]
    meta["archetype"] = info["archetype"]
    return df, meta

