"""
Unit tests for DuckDBEngine
"""
import sys
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

import pandas as pd
import numpy as np
from utils.duckdb_engine import DuckDBEngine

def run_tests():
    print("--- Starting DuckDB Engine Unit Tests ---")

    # Sample DataFrame with special column names (like AirQualityUCI)
    df = pd.DataFrame({
        "Region": ["North", "South", "East", "West", "North", "South", "East", "West"],
        "Sales": [100.5, 250.0, 75.2, 310.0, 150.0, 220.0, 80.0, 400.0],
        "Units Sold": [10, 25, 8, 30, 15, 20, 9, 38],
        "CO(GT)": [2.6, 2.0, 1.8, 3.1, 2.4, 1.9, 1.5, 3.8]
    })

    engine = DuckDBEngine(df)

    # Test 1: Query Execution
    res = engine.execute_query("SELECT COUNT(*) AS total_count FROM dataset")
    assert res["success"] is True, f"Query failed: {res['error']}"
    assert res["df"].iloc[0]["total_count"] == 8, "Expected 8 rows"
    print(f"✓ Test 1: Query execution passed ({res['execution_time_ms']} ms)")

    # Test 2: Table aliases (df and data)
    res_df_alias = engine.execute_query("SELECT AVG(Sales) AS avg_sales FROM df")
    assert res_df_alias["success"] is True, f"Query on 'df' alias failed: {res_df_alias['error']}"
    res_data_alias = engine.execute_query("SELECT AVG(Sales) AS avg_sales FROM data")
    assert res_data_alias["success"] is True, f"Query on 'data' alias failed: {res_data_alias['error']}"
    print("✓ Test 2: Table aliases ('dataset', 'df', 'data') passed")

    # Test 3: Quoted column with special characters
    res_special = engine.execute_query('SELECT "CO(GT)", Sales FROM dataset WHERE "CO(GT)" > 2.0')
    assert res_special["success"] is True, f"Special char query failed: {res_special['error']}"
    assert len(res_special["df"]) == 4, f"Expected 4 rows, got {len(res_special['df'])}"
    print("✓ Test 3: Quoted special characters in column names passed")

    # Test 4: Security - Block DROP TABLE
    res_drop = engine.execute_query("DROP TABLE dataset")
    assert res_drop["success"] is False, "Security check failed: DROP TABLE was not blocked"
    print(f"✓ Test 4: Security blocked DROP TABLE: '{res_drop['error']}'")

    # Test 5: Security - Block DELETE
    res_delete = engine.execute_query("DELETE FROM dataset WHERE Sales < 100")
    assert res_delete["success"] is False, "Security check failed: DELETE was not blocked"
    print(f"✓ Test 5: Security blocked DELETE: '{res_delete['error']}'")

    # Test 6: Security - Block INSERT and UPDATE
    res_insert = engine.execute_query("INSERT INTO dataset VALUES ('Test', 1, 1, 1)")
    assert res_insert["success"] is False, "Security check failed: INSERT was not blocked"
    res_update = engine.execute_query("UPDATE dataset SET Sales = 0")
    assert res_update["success"] is False, "Security check failed: UPDATE was not blocked"
    print("✓ Test 6: Security blocked INSERT & UPDATE")

    # Test 7: Security - Block chained queries
    res_chain = engine.execute_query("SELECT * FROM dataset; DROP TABLE dataset")
    assert res_chain["success"] is False, "Security check failed: Chained semicolon queries not blocked"
    print(f"✓ Test 7: Security blocked chained query: '{res_chain['error']}'")

    # Test 8: Schema metadata
    summary = engine.get_schema_summary()
    assert len(summary) == 4, f"Expected 4 schema items, got {len(summary)}"
    prompt_ctx = engine.get_schema_prompt_context()
    assert "CREATE TABLE dataset" in prompt_ctx
    print("✓ Test 8: Schema summary & prompt context passed")

    # Test 9: Offline deterministic NL-to-SQL translation
    sql, expl = engine.translate_nl_to_sql_offline("top 3 sales")
    assert sql is not None, "Failed to translate 'top 3 sales'"
    assert "LIMIT 3" in sql
    res_top = engine.execute_query(sql)
    assert res_top["success"] is True and len(res_top["df"]) == 3
    print("✓ Test 9: NL-to-SQL for 'top 3 sales' passed")

    # Test 10: NL-to-SQL Grouped aggregation
    sql_agg, expl_agg = engine.translate_nl_to_sql_offline("average Sales by Region")
    assert sql_agg is not None, "Failed to translate 'average Sales by Region'"
    assert "GROUP BY" in sql_agg
    res_agg = engine.execute_query(sql_agg)
    assert res_agg["success"] is True and len(res_agg["df"]) == 4
    print("✓ Test 10: NL-to-SQL for 'average Sales by Region' passed")

    # Test 11: Auto-visualization
    fig = engine.auto_visualize(res_agg["df"], title="Average Sales by Region")
    assert fig is not None, "Failed to generate auto-visualization for grouped aggregation"
    print("✓ Test 11: Auto-visualization generation passed")

    # Test 12: DuckDB SUMMARIZE command
    res_sum = engine.execute_query("SUMMARIZE dataset;")
    assert res_sum["success"] is True, f"SUMMARIZE failed: {res_sum['error']}"
    assert len(res_sum["df"]) == 4, "SUMMARIZE should return 4 rows (1 per column)"
    print("✓ Test 12: Native DuckDB SUMMARIZE execution passed")

    print("\n=== ALL DUCKDB ENGINE TESTS PASSED! ===")

if __name__ == "__main__":
    run_tests()
