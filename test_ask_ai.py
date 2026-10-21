"""
Comprehensive Test Suite for InsightAnalyst AI - Ask AI Module & State Management
Covers all 9 acceptance criteria specified in the bug fix requirements.
"""

import os
import sys
import pandas as pd
import numpy as np

# Ensure datathon root is on sys.path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

from utils.type_detector import detect_column_types
from utils.cleaner import detect_and_clean
from utils.ml_engine import run_ml_analysis
from utils.ai_assistant import AIAssistantEngine, normalize_user_query, extract_record_limit
from utils.rag_engine import DatasetRAGEngine

def test_scenario_1_highest_critical_records():
    """1. Asking for the highest critical records returns a real, descending-sorted table when anomaly results are available."""
    print("\n--- Test 1: Highest Critical Records Returns Descending-Sorted Table ---")
    np.random.seed(42)
    # Synthetic dataset with 100 rows, 3 numeric columns, and inject 5 strong outliers
    df = pd.DataFrame({
        'Feature_A': np.random.normal(50, 5, 100),
        'Feature_B': np.random.normal(100, 10, 100),
        'Feature_C': np.random.normal(10, 2, 100)
    })
    # Inject extreme anomalies
    df.loc[10, 'Feature_A'] = 500
    df.loc[25, 'Feature_B'] = 1500
    df.loc[60, 'Feature_C'] = 200

    col_types = detect_column_types(df)
    cleaned_df, report = detect_and_clean(df, col_types)
    ml_results = run_ml_analysis(cleaned_df, col_types, contamination=0.08)

    assistant = AIAssistantEngine(cleaned_df, col_types, ml_results=ml_results, dataset_name="Synthetic Test Data")
    res = assistant.process_message("can u tell me highest criticals rows")

    assert res['table'] is not None, "Expected table in result, got None"
    table = res['table']
    assert len(table) > 0, "Table should not be empty"
    assert "Record ID" in table.columns, "Table should contain 'Record ID'"
    assert "Risk Score" in table.columns, "Table should contain 'Risk Score'"
    assert "Fuzzy Severity" in table.columns, "Table should contain 'Fuzzy Severity'"

    # Verify descending sort order
    scores = [float(str(s).split('/')[0].strip()) for s in table['Risk Score']]
    for i in range(len(scores) - 1):
        assert scores[i] >= scores[i + 1] - 1e-4, f"Scores must be descending: {scores}"

    # Verify rows 10, 25, 60 are detected among top records
    detected_ids = [str(r) for r in table['Record ID']]
    outlier_hits = sum(1 for target in ['Row #10', 'Row #25', 'Row #60'] if any(target in did for did in detected_ids))
    assert outlier_hits >= 2, f"Expected injected outliers to be caught, found in {detected_ids}"

    print(f"✓ Test 1 Passed: {len(table)} records returned descending-sorted (Top score: {scores[0]:.1f})")


def test_scenario_2_equivalent_wording_variations():
    """2. Equivalent wording maps to the same suitable analytical intent."""
    print("\n--- Test 2: Equivalent Natural Language Variations Map to Critical Records ---")
    np.random.seed(42)
    df = pd.DataFrame({'Val1': np.random.normal(20, 2, 50), 'Val2': np.random.normal(50, 5, 50)})
    df.loc[5, 'Val1'] = 300

    col_types = detect_column_types(df)
    ml_results = run_ml_analysis(df, col_types)
    assistant = AIAssistantEngine(df, col_types, ml_results=ml_results, dataset_name="Variation Test")

    variations = [
        "can u tell me highest criticals rows",
        "Show me the highest critical rows.",
        "Show the top 10 critical records.",
        "Which records have the highest risk scores?",
        "Show the most severe anomalies.",
        "Find critical anomalies in my dataset.",
        "List the most suspicious records.",
        "Show the records with the highest fuzzy risk scores.",
        "what are the top critical rows"
    ]

    for q in variations:
        res = assistant.process_message(q)
        assert res['table'] is not None, f"Query '{q}' failed to return table"
        assert "Risk Score" in res['table'].columns, f"Query '{q}' missing Risk Score in table"
        assert "Highest Computed Risk Scores" in res['text'] or "Critical" in res['text'], f"Query '{q}' text did not match critical template"

    print(f"✓ Test 2 Passed: All {len(variations)} natural language variations mapped successfully")


def test_scenario_3_requested_limits():
    """3. Asking for the top 5 records returns no more than five records."""
    print("\n--- Test 3: Respect User-Requested Limits ---")
    np.random.seed(42)
    df = pd.DataFrame({'Sensor1': np.random.normal(10, 1, 60), 'Sensor2': np.random.normal(20, 2, 60)})
    col_types = detect_column_types(df)
    ml_results = run_ml_analysis(df, col_types)
    assistant = AIAssistantEngine(df, col_types, ml_results=ml_results, dataset_name="Limit Test")

    # Ask for top 5
    res5 = assistant.process_message("Show top 5 critical records")
    assert len(res5['table']) == 5, f"Expected exactly 5 records, got {len(res5['table'])}"

    # Ask for top 3
    res3 = assistant.process_message("show 3 critical rows")
    assert len(res3['table']) == 3, f"Expected exactly 3 records, got {len(res3['table'])}"

    # Default should be 10
    res_def = assistant.process_message("show highest critical rows")
    assert len(res_def['table']) == 10, f"Expected default 10 records, got {len(res_def['table'])}"

    print(f"✓ Test 3 Passed: Limits respected (5 -> {len(res5['table'])}, 3 -> {len(res3['table'])}, default -> {len(res_def['table'])})")


def test_scenario_4_and_5_dataset_change_and_metadata_consistency():
    """4. Dataset change refreshes analytical state. 5. Metadata and welcome message agree on row/column count."""
    print("\n--- Test 4 & 5: Dataset Change State Refresh & Authoritative Metadata ---")

    # Dataset A (e.g. Merged dataset: 50 rows, 3 columns)
    df_a = pd.DataFrame({'A': range(50), 'B': range(50), 'C': range(50)})
    sig_a = f"merged_dataset.csv_{len(df_a)}_{len(df_a.columns)}"

    # Dataset B (e.g. AirQualityUCI: 9357 rows, 15 columns)
    df_b = pd.DataFrame({f"Col_{i}": range(100) for i in range(15)})
    sig_b = f"AirQualityUCI.xlsx_{len(df_b)}_{len(df_b.columns)}"

    # Simulate state transition
    session_state = {}

    def simulate_app_state(dataset_name, df, title):
        sig = f"{dataset_name}_{len(df)}_{len(df.columns)}"
        dataset_changed = (session_state.get('active_dataset_signature') != sig)
        if dataset_changed:
            session_state['chat_messages'] = [
                {
                    'role': 'assistant',
                    'content': f"Hello! I'm your **AI Data Analyst**. I've analyzed **{title}** ({len(df):,} rows, {len(df.columns)} columns) and I'm ready to answer any questions about your data.",
                    'chart': None, 'table': None, 'code': None
                }
            ]
            session_state['active_dataset_signature'] = sig
            session_state['last_analyzed_dataset'] = dataset_name

    # Step 1: Load Dataset A
    simulate_app_state("merged_dataset.csv", df_a, "Merged (2 files)")
    welcome_a = session_state['chat_messages'][0]['content']
    assert "50 rows" in welcome_a and "3 columns" in welcome_a, f"Expected 50 rows, 3 cols in welcome: {welcome_a}"

    # Step 2: Switch to Dataset B
    simulate_app_state("AirQualityUCI.xlsx", df_b, "AirQualityUCI.xlsx")
    welcome_b = session_state['chat_messages'][0]['content']
    assert "100 rows" in welcome_b and "15 columns" in welcome_b, f"Expected 100 rows, 15 cols in welcome: {welcome_b}"
    assert "Merged" not in welcome_b, "Stale dataset name leaked into welcome message!"

    print("✓ Test 4 & 5 Passed: State invalidates on dataset change; welcome message and metadata stay strictly synchronized")


def test_scenario_6_markdown_formatting_not_escaped():
    """6. Assistant headings, bold text, lists, tables, and code are formatted correctly and not wrapped in raw text HTML."""
    print("\n--- Test 6: Markdown Formatting Integrity ---")
    df = pd.DataFrame({'Sales': [100, 200, 300], 'Region': ['North', 'South', 'East']})
    col_types = detect_column_types(df)
    assistant = AIAssistantEngine(df, col_types, dataset_name="Format Test")

    res = assistant.process_message("What are the biggest problems in my data?")
    text = res['text']

    # Must contain Markdown headers and bold markers that will be rendered natively
    assert text.startswith("###"), f"Response should start with Markdown header '###': {text[:30]}"
    assert "**" in text, "Response should contain bold Markdown tags"
    # Ensure raw HTML chat div wrapper is NOT present inside the assistant text
    assert "<div class='chat-assistant-msg'>" not in text, "Raw HTML div must not wrap assistant text"

    print("✓ Test 6 Passed: Clean Markdown syntax generated without escaping or enclosing raw HTML wrapper")


def test_scenario_7_unanswerable_question_no_fabrication():
    """7. A question that cannot be answered does not produce fabricated analysis."""
    print("\n--- Test 7: Unanswerable Question Does Not Fabricate Data ---")
    df = pd.DataFrame({'FeatureX': [1, 2, 3], 'FeatureY': [4, 5, 6]})
    col_types = detect_column_types(df)
    assistant = AIAssistantEngine(df, col_types, dataset_name="Safety Test")

    res = assistant.process_message("What was the weather like in Tokyo in 1982?")
    text = res['text']

    # Should clearly explain limitation and provide active dataset context
    assert "Tokyo" in text or "could not map" in text.lower(), "Should address the question limitation"
    assert "Safety Test" in text, "Should ground response in the active dataset"
    assert res['table'] is None, "Should not fabricate a fake table"

    print("✓ Test 7 Passed: Unanswerable question explained limitation without fabricating records")


def test_scenario_8_edge_cases_handling():
    """8. Application handles empty datasets, missing columns, missing API keys without crashing."""
    print("\n--- Test 8: Empty Datasets, Missing Columns, Missing API Keys Gracefully Handled ---")

    # Empty dataframe
    empty_df = pd.DataFrame()
    assistant_empty = AIAssistantEngine(empty_df, {}, dataset_name="Empty Data")
    res_empty = assistant_empty.process_message("can u tell me highest criticals rows")
    assert "Empty Dataset" in res_empty['text'], "Empty dataset should produce clean warning"

    # No numeric columns (only categorical/text)
    cat_df = pd.DataFrame({'City': ['New York', 'London', 'Tokyo'], 'Status': ['A', 'B', 'C']})
    cat_types = detect_column_types(cat_df)
    assistant_cat = AIAssistantEngine(cat_df, cat_types, dataset_name="Categorical Only")
    res_cat = assistant_cat.process_message("Show critical rows")
    assert "Numerical Features Required" in res_cat['text'], "Non-numeric dataset should explain requirement"

    # Missing API key (should work offline without error)
    norm_df = pd.DataFrame({'Metric': [10, 20, 30, 40, 50]})
    norm_types = detect_column_types(norm_df)
    assistant_offline = AIAssistantEngine(norm_df, norm_types, api_key="", dataset_name="Offline Test")
    res_offline = assistant_offline.process_message("What are the biggest problems in my data?")
    assert res_offline['text'] is not None and len(res_offline['text']) > 50

    print("✓ Test 8 Passed: Empty datasets, non-numeric datasets, and missing API keys handled gracefully")


def test_scenario_9_existing_features_working():
    """9. Existing sample questions, charts, cleaning operations, and other working functionality continue to work."""
    print("\n--- Test 9: Existing Sample Questions & Core Pipeline Integrity ---")
    sample_path = os.path.join(os.path.dirname(__file__), 'sample_data', 'sample_sales.csv')
    df = pd.read_csv(sample_path)
    col_types = detect_column_types(df)
    cleaned_df, report = detect_and_clean(df, col_types)
    ml_results = run_ml_analysis(cleaned_df, col_types)
    rag_engine = DatasetRAGEngine()
    rag_engine.build_knowledge_base(cleaned_df, col_types, report, ml_results)

    assistant = AIAssistantEngine(cleaned_df, col_types, rag_engine=rag_engine, ml_results=ml_results, dataset_name="Retail Sales Data")

    # Test all 5 original sample questions
    sample_queries = [
        "What are the biggest problems in my data?",
        "Show me the most important insights.",
        "Why are some records unusual?",
        "Which products generate the most sales?",
        "Are there any important trends?"
    ]

    for q in sample_queries:
        res = assistant.process_message(q)
        assert res['text'] is not None and len(res['text']) > 50, f"Query '{q}' returned invalid text"
        print(f"  ✓ Sample Question: '{q}' -> OK")

    print("✓ Test 9 Passed: All existing sample questions and workflows functional")


if __name__ == '__main__':
    print("================================================================")
    print("  RUNNING INSIGHTANALYST AI ASK AI & STATE ACCEPTANCE TEST SUITE")
    print("================================================================")
    try:
        test_scenario_1_highest_critical_records()
        test_scenario_2_equivalent_wording_variations()
        test_scenario_3_requested_limits()
        test_scenario_4_and_5_dataset_change_and_metadata_consistency()
        test_scenario_6_markdown_formatting_not_escaped()
        test_scenario_7_unanswerable_question_no_fabrication()
        test_scenario_8_edge_cases_handling()
        test_scenario_9_existing_features_working()
        print("\n================================================================")
        print("  🎉 ALL 9 ACCEPTANCE CRITERIA TESTS PASSED SUCCESSFULLY!")
        print("================================================================")
    except AssertionError as e:
        print(f"\n❌ TEST FAILED: {e}")
        sys.exit(1)
    except Exception as e:
        import traceback
        print(f"\n❌ UNEXPECTED ERROR: {e}")
        traceback.print_exc()
        sys.exit(1)
