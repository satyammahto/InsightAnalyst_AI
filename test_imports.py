"""Quick import test to verify all dependencies are installed correctly."""
import sys
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
print(f"Python: {sys.version}")

try:
    import streamlit as st
    print(f"✓ streamlit {st.__version__}")
except ImportError as e:
    print(f"✗ streamlit: {e}")

try:
    import pandas as pd
    print(f"✓ pandas {pd.__version__}")
except ImportError as e:
    print(f"✗ pandas: {e}")

try:
    import numpy as np
    print(f"✓ numpy {np.__version__}")
except ImportError as e:
    print(f"✗ numpy: {e}")

try:
    import sklearn
    print(f"✓ scikit-learn {sklearn.__version__}")
except ImportError as e:
    print(f"✗ scikit-learn: {e}")

try:
    import plotly
    print(f"✓ plotly {plotly.__version__}")
except ImportError as e:
    print(f"✗ plotly: {e}")

try:
    import pyarrow
    print(f"✓ pyarrow {pyarrow.__version__}")
except ImportError as e:
    print(f"✗ pyarrow: {e}")

try:
    import duckdb
    print(f"✓ duckdb {duckdb.__version__}")
except ImportError as e:
    print(f"✗ duckdb: {e}")

try:
    import google.generativeai as genai
    print("✓ google-generativeai")
except ImportError as e:
    print(f"✗ google-generativeai: {e}")

try:
    import openpyxl
    print(f"✓ openpyxl {openpyxl.__version__}")
except ImportError as e:
    print(f"✗ openpyxl: {e}")

# Test app-level imports
print("\n--- Testing app module imports ---")
try:
    sys.path.insert(0, '.')
    from utils.type_detector import detect_column_types
    print("✓ utils.type_detector")
except Exception as e:
    print(f"✗ utils.type_detector: {e}")

try:
    from utils.cleaner import detect_and_clean
    print("✓ utils.cleaner")
except Exception as e:
    print(f"✗ utils.cleaner: {e}")

try:
    from utils.batch_processor import process_dataset_in_batches, stream_file_to_disk, safe_read_csv
    print("✓ utils.batch_processor")
except Exception as e:
    print(f"✗ utils.batch_processor: {e}")

try:
    from utils.rag_engine import DatasetRAGEngine
    print("✓ utils.rag_engine")
except Exception as e:
    print(f"✗ utils.rag_engine: {e}")

try:
    from utils.ai_assistant import AIAssistantEngine
    print("✓ utils.ai_assistant")
except Exception as e:
    print(f"✗ utils.ai_assistant: {e}")

try:
    from utils.fuzzy_engine import calculate_fuzzy_severity, get_fuzzy_membership_curves
    print("✓ utils.fuzzy_engine")
except Exception as e:
    print(f"✗ utils.fuzzy_engine: {e}")

try:
    from utils.ml_engine import run_ml_analysis
    print("✓ utils.ml_engine")
except Exception as e:
    print(f"✗ utils.ml_engine: {e}")

try:
    from utils.dashboard import (
        plot_numeric_distribution,
        plot_categorical_counts,
        plot_date_trend,
        plot_correlation_heatmap,
        plot_cluster_scatter,
        plot_outlier_scatter,
        plot_custom_user_chart
    )
    print("✓ utils.dashboard")
except Exception as e:
    print(f"✗ utils.dashboard: {e}")

try:
    from utils.summarizer import generate_executive_summary, generate_html_dossier
    print("✓ utils.summarizer")
except Exception as e:
    print(f"✗ utils.summarizer: {e}")

try:
    from utils.query_engine import filter_dataset_by_nl
    print("✓ utils.query_engine")
except Exception as e:
    print(f"✗ utils.query_engine: {e}")

# Test loading sample data and running the pipeline
print("\n--- Testing pipeline execution ---")
try:
    import os
    sample_path = os.path.join(os.path.dirname(__file__), 'sample_data', 'sample_sales.csv')
    raw_df = safe_read_csv(sample_path)
    print(f"✓ Loaded sample_sales.csv: {raw_df.shape}")
    
    col_types = detect_column_types(raw_df)
    print(f"✓ Type detection: numeric={len(col_types['numeric'])}, cat={len(col_types['categorical'])}, date={len(col_types['date'])}")
    
    cleaned_df, report = detect_and_clean(raw_df, col_types)
    print(f"✓ Cleaning: {report['initial_shape']} → {report['final_shape']}, hygiene={report['hygiene_score']}")
    
    ml_results = run_ml_analysis(cleaned_df, col_types)
    print(f"✓ ML analysis: status={ml_results['status']}, execution={ml_results['execution_ms']}")
    
    rag = DatasetRAGEngine()
    rag.build_knowledge_base(cleaned_df, col_types, report, ml_results)
    print(f"✓ RAG engine: {len(rag.chunks)} chunks indexed")
    
    brief = generate_executive_summary(cleaned_df, col_types, report, ml_results, None, rag)
    print(f"✓ Executive summary generated")
    
    print("\n=== ALL TESTS PASSED ===")
except Exception as e:
    import traceback
    print(f"✗ Pipeline error: {e}")
    traceback.print_exc()
