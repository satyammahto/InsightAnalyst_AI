"""
Unit tests for CleaningStudio module
"""
import sys
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

import pandas as pd
import numpy as np
from utils.cleaning_studio import CleaningStudio

def run_tests():
    print("--- Starting CleaningStudio Unit Tests ---")

    df = pd.DataFrame({
        "City": [" New York ", "los angeles!", "CHICAGO", "new york", None],
        "Sales": [100.0, 200.0, np.nan, 5000.0, 300.0],
        "Units": [10, 20, 15, 50, 25],
        "Date": ["2024-01-15", "2024-02-20", "2024-03-25", "2024-04-30", "2024-05-05"]
    })

    # Test 1: Imputation
    df_imp, tel_imp = CleaningStudio.impute_missing(df, "Sales", strategy="median")
    assert df_imp["Sales"].isnull().sum() == 0, "Missing Sales should be imputed"
    assert tel_imp["nulls_before"] == 1 and tel_imp["nulls_after"] == 0
    print("✓ Test 1: Imputation (median) passed")

    # Test 2: Outlier Clipping
    df_outl, tel_outl = CleaningStudio.handle_outliers(df_imp, "Sales", method="clip_percentiles", lower_pct=5.0, upper_pct=95.0)
    assert df_outl["Sales"].max() < 5000.0, "Outlier 5000 should be clipped"
    print("✓ Test 2: Outlier clipping passed")

    # Test 3: Text Cleaning
    df_txt, tel_txt = CleaningStudio.clean_text_column(df, "City", casing="title", strip_whitespace=True, remove_special_chars=True)
    assert "Los Angeles" in df_txt["City"].values, f"Expected 'Los Angeles', got {df_txt['City'].values}"
    print("✓ Test 3: Text cleaning (casing, whitespace, special chars) passed")

    # Test 4: Datetime Features
    df_dt, tel_dt = CleaningStudio.extract_datetime_features(df, "Date", ["year", "month", "dayofweek", "is_weekend"])
    assert "Date_year" in df_dt.columns and "Date_dayofweek" in df_dt.columns
    print("✓ Test 4: Datetime extraction passed")

    # Test 5: Feature Scaling
    df_scale, tel_scale = CleaningStudio.scale_features(df_imp, ["Sales", "Units"], method="standard")
    assert abs(df_scale["Units"].mean()) < 1e-4, "Standard scaled feature mean should be ~0"
    print("✓ Test 5: Feature scaling (StandardScaler) passed")

    # Test 6: Derived Column
    df_der, tel_der = CleaningStudio.derive_column(df, "Revenue_Per_Unit", "Sales", "divide", "Units")
    assert "Revenue_Per_Unit" in df_der.columns
    assert df_der.loc[0, "Revenue_Per_Unit"] == 10.0
    print("✓ Test 6: Derived column formula passed")

    # Test 7: Drop Column
    df_drop, tel_drop = CleaningStudio.drop_columns(df, ["City"])
    assert "City" not in df_drop.columns
    print("✓ Test 7: Drop columns passed")

    print("\n=== ALL CLEANING STUDIO TESTS PASSED ===")

if __name__ == "__main__":
    run_tests()
