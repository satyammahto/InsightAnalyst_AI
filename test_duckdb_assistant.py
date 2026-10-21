"""
Unit test to verify DuckDB integration inside AIAssistantEngine
"""
import sys
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

import pandas as pd
from utils.ai_assistant import AIAssistantEngine
from utils.type_detector import detect_column_types

def test_duckdb_assistant_integration():
    print("--- Testing DuckDB Integration in AIAssistantEngine ---")

    df = pd.DataFrame({
        "Category": ["Electronics", "Clothing", "Home", "Electronics", "Clothing", "Home"],
        "Revenue": [1200.0, 450.0, 300.0, 2100.0, 600.0, 850.0],
        "Units": [4, 15, 3, 7, 20, 10]
    })
    col_types = detect_column_types(df)
    assistant = AIAssistantEngine(df, col_types, dataset_name="Retail Test")

    # 1. Raw SQL execution via Ask AI
    raw_res = assistant.process_message("SELECT Category, SUM(Revenue) AS total_rev FROM dataset GROUP BY Category ORDER BY total_rev DESC")
    assert raw_res['table'] is not None, "Expected table in raw SQL response"
    assert "DuckDB" in raw_res['text'] or "SQL" in raw_res['text'], "Expected DuckDB mention in text"
    assert raw_res['code_snippet'] is not None, "Expected SQL snippet in code_snippet"
    assert len(raw_res['table']) == 3, f"Expected 3 rows, got {len(raw_res['table'])}"
    print(f"✓ Raw SQL query execution passed: {len(raw_res['table'])} rows returned")

    # 2. DuckDB SUMMARIZE command via Ask AI
    sum_res = assistant.process_message("SUMMARIZE dataset;")
    assert sum_res['table'] is not None, "Expected table in SUMMARIZE response"
    assert len(sum_res['table']) == 3, "Expected 3 columns summarized"
    print("✓ SUMMARIZE command via Ask AI passed")

    # 3. Deterministic natural-language to DuckDB query
    nl_res = assistant.process_message("top 2 Revenue")
    assert nl_res['table'] is not None, "Expected table for 'top 2 Revenue'"
    assert len(nl_res['table']) == 2, f"Expected 2 rows, got {len(nl_res['table'])}"
    print("✓ NL to DuckDB query 'top 2 Revenue' passed")

    print("\n=== ALL DUCKDB ASSISTANT INTEGRATION TESTS PASSED ===")

if __name__ == "__main__":
    test_duckdb_assistant_integration()
