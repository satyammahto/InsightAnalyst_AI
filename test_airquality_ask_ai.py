"""
Verification test against the real user-uploaded AirQualityUCI dataset.
"""
import os
import sys
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

from utils.type_detector import detect_column_types
from utils.cleaner import detect_and_clean
from utils.ml_engine import run_ml_analysis
from utils.ai_assistant import AIAssistantEngine

air_quality_path = os.path.join(os.path.dirname(__file__), 'uploaded_datasets', 'uploaded_AirQualityUCI.xlsx')

if not os.path.exists(air_quality_path):
    print("AirQualityUCI.xlsx not found at:", air_quality_path)
    sys.exit(0)

print(f"Loading {air_quality_path}...")
raw_df = pd.read_excel(air_quality_path)
print(f"Loaded AirQualityUCI: {raw_df.shape[0]:,} rows, {raw_df.shape[1]} columns")

col_types = detect_column_types(raw_df)
cleaned_df, report = detect_and_clean(raw_df, col_types)
print(f"Cleaned shape: {cleaned_df.shape}, Hygiene: {report.get('hygiene_score')}")

ml_results = run_ml_analysis(cleaned_df, col_types)
print(f"ML analysis complete: outliers detected = {ml_results['outliers']['count'] if ml_results.get('outliers') else 0}")

assistant = AIAssistantEngine(
    cleaned_df,
    col_types,
    ml_results=ml_results,
    dataset_name="AirQualityUCI.xlsx",
    cleaning_report=report
)

print("\n--- Test A: User exact query 'can u tell me highest criticals rows' ---")
res = assistant.process_message("can u tell me highest criticals rows")
print("Response text excerpt:\n", res['text'][:250])
assert res['table'] is not None, "Expected table in result"
print("\nTop critical records table:\n", res['table'].head(5))
assert len(res['table']) == 10, f"Expected 10 rows, got {len(res['table'])}"

# Check descending risk scores
scores = [float(str(s).split('/')[0].strip()) for s in res['table']['Risk Score']]
print("Risk Scores:", scores)
for i in range(len(scores) - 1):
    assert scores[i] >= scores[i + 1] - 1e-4, "Scores not descending!"

print("\n--- Test B: Query with explicit limit 'show top 5 critical records' ---")
res5 = assistant.process_message("show top 5 critical records")
assert len(res5['table']) == 5, f"Expected 5 records, got {len(res5['table'])}"
print(f"Successfully returned exactly 5 records with scores: {[str(s) for s in res5['table']['Risk Score']]}")

print("\n--- Test C: Query for column statistics 'what is the average T?' ---")
res_stat = assistant.process_message("what is the average T?")
assert res_stat['table'] is not None, "Expected table for column stats"
print("Stats Table:\n", res_stat['table'])

print("\n✓ All real-world AirQualityUCI tests passed successfully!")
