"""
Unit test for Step 5: Enhanced Executive Dossier & Report Export
Verifies HTML dossier (print-ready) and Markdown report generation with
cluster personas, Mamdani FIS anomaly telemetry, DAG audit, and studio history.
"""
import sys
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

import pandas as pd
import numpy as np

from utils.type_detector import detect_column_types
from utils.cleaner import detect_and_clean
from utils.ml_engine import run_ml_analysis
from utils.summarizer import generate_executive_summary, generate_html_dossier, generate_markdown_dossier

def test_step5_dossier():
    print("--- 1. Generating mock pipeline context ---")
    np.random.seed(42)
    n = 100
    df = pd.DataFrame({
        'Sales': np.random.exponential(scale=1000, size=n),
        'Profit': np.random.normal(loc=200, scale=50, size=n),
        'Quantity': np.random.randint(1, 20, size=n),
        'Region': np.random.choice(['North', 'South', 'East', 'West'], size=n)
    })
    # Inject outliers
    df.loc[0, 'Sales'] = 25000.0
    df.loc[1, 'Sales'] = 30000.0
    df.loc[1, 'Profit'] = -5000.0

    col_types = detect_column_types(df)
    clean_df, cleaning_report = detect_and_clean(df, col_types)
    ml_results = run_ml_analysis(clean_df, col_types, n_clusters=3, contamination=0.05)
    exec_brief = generate_executive_summary(clean_df, col_types, cleaning_report, ml_results)

    studio_history = [
        "IMPUTE_MISSING on Sales (strategy=median)",
        "FILTER_ROWS on Profit >= 0",
        "DEDUPLICATE_ROWS on all features"
    ]

    print("--- 2. Testing generate_html_dossier ---")
    html_output = generate_html_dossier(
        clean_df, col_types, cleaning_report, ml_results, exec_brief,
        dataset_name="Sales_Q4_Test",
        studio_history=studio_history
    )
    assert isinstance(html_output, str), "HTML output must be a string"
    assert "<!DOCTYPE html>" in html_output, "HTML must contain doctype"
    assert "@media print" in html_output, "HTML must contain print media stylesheet"
    assert "window.print()" in html_output, "HTML must contain print button action"
    assert "Sales_Q4_Test" in html_output, "HTML must include dataset name"
    assert "Behavioral Personas & Cluster Segmentation" in html_output, "HTML must include clusters"
    assert "Anomaly Telemetry" in html_output, "HTML must include anomaly section"
    assert "5-Stage Automated Cleaning DAG Audit" in html_output, "HTML must include DAG audit"
    assert "Custom Data Studio Transformations" in html_output, "HTML must include studio history"
    print("✓ generate_html_dossier verified successfully")

    print("--- 3. Testing generate_markdown_dossier (Zero-Tabulate Dependency) ---")
    md_output = generate_markdown_dossier(
        clean_df, col_types, cleaning_report, ml_results, exec_brief,
        dataset_name="Sales_Q4_Test",
        studio_history=studio_history
    )
    assert isinstance(md_output, str), "Markdown output must be a string"
    assert "# ⚡ InsightAnalyst AI — Executive Audit Dossier" in md_output, "Markdown must contain title"
    assert "Sales_Q4_Test" in md_output, "Markdown must include dataset name"
    assert "| **Total Rows Evaluated** |" in md_output, "Markdown must contain KPI table"
    assert "## 👥 Behavioral Cluster Personas" in md_output, "Markdown must include cluster section"
    assert "## ⚠️ High-Risk Anomalies & Fuzzy Severity" in md_output, "Markdown must include anomaly section"
    assert "## 🛠️ Custom Data Studio Transformations" in md_output, "Markdown must include studio section"
    assert "## 🛡️ Pipeline & 5-Stage Cleaning DAG Audit" in md_output, "Markdown must include DAG table"
    print("✓ generate_markdown_dossier verified successfully")

    print("\n=== STEP 5 DOSSIER TESTS ALL PASSED ===")

if __name__ == "__main__":
    test_step5_dossier()
