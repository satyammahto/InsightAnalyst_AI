"""
Interactive Data Cleaning Studio for InsightAnalyst AI
------------------------------------------------------
Enables user-configurable, live data transformations with:
- Missing value imputation (mean, median, mode, constant, ffill, bfill, drop)
- Outlier treatment (percentile clipping, IQR bounds, 3-sigma drop)
- Categorical & text normalization (casing, whitespace, regex sanitization, find/replace)
- Datetime feature extraction (year, month, day, hour, dayofweek, is_weekend)
- Numeric feature scaling (StandardScaler, MinMaxScaler, RobustScaler, Log1p)
- Column operations (drop, rename, math formula derivation)
- Before vs. After comparison telemetry
- Undo stack and full pipeline reset
"""

import re
from typing import Dict, Any, List, Optional, Tuple
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler


class CleaningStudio:
    """
    Stateful execution engine for custom interactive dataset transformations.
    """

    @staticmethod
    def impute_missing(
        df: pd.DataFrame,
        column: str,
        strategy: str = "median",
        constant_value: Any = None
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Imputes missing values in a specified column.
        Strategies: 'median', 'mean', 'mode', 'constant', 'ffill', 'bfill', 'drop_rows'
        """
        df_out = df.copy()
        if column not in df_out.columns:
            raise ValueError(f"Column '{column}' not found in dataset.")

        null_count_before = int(df_out[column].isnull().sum())
        rows_before = len(df_out)

        if strategy == "drop_rows":
            df_out = df_out.dropna(subset=[column]).reset_index(drop=True)
            details = f"Dropped {rows_before - len(df_out):,} rows with missing '{column}'."
        elif strategy == "mean":
            val = df_out[column].mean()
            df_out[column] = df_out[column].fillna(val)
            details = f"Filled {null_count_before:,} missing values with mean ({val:.2f})."
        elif strategy == "median":
            val = df_out[column].median()
            df_out[column] = df_out[column].fillna(val)
            details = f"Filled {null_count_before:,} missing values with median ({val:.2f})."
        elif strategy == "mode":
            mode_series = df_out[column].mode()
            val = mode_series.iloc[0] if not mode_series.empty else "Unknown"
            df_out[column] = df_out[column].fillna(val)
            details = f"Filled {null_count_before:,} missing values with mode ('{val}')."
        elif strategy == "constant":
            val = constant_value if constant_value is not None else 0
            df_out[column] = df_out[column].fillna(val)
            details = f"Filled {null_count_before:,} missing values with constant value '{val}'."
        elif strategy == "ffill":
            df_out[column] = df_out[column].ffill().bfill()
            details = f"Filled missing values using forward fill."
        elif strategy == "bfill":
            df_out[column] = df_out[column].bfill().ffill()
            details = f"Filled missing values using backward fill."
        else:
            raise ValueError(f"Unsupported imputation strategy '{strategy}'.")

        null_count_after = int(df_out[column].isnull().sum())
        telemetry = {
            "operation": f"Impute '{column}' ({strategy})",
            "column": column,
            "nulls_before": null_count_before,
            "nulls_after": null_count_after,
            "rows_before": rows_before,
            "rows_after": len(df_out),
            "details": details
        }
        return df_out, telemetry

    @staticmethod
    def handle_outliers(
        df: pd.DataFrame,
        column: str,
        method: str = "clip_percentiles",
        lower_pct: float = 1.0,
        upper_pct: float = 99.0
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Handles outliers in a numeric column.
        Methods:
        - 'clip_percentiles': Clips values at specified lower/upper percentiles
        - 'clip_iqr': Clips values at [Q1 - 1.5*IQR, Q3 + 1.5*IQR]
        - 'drop_iqr': Drops rows exceeding 1.5*IQR bounds
        - 'drop_3sigma': Drops rows where |z| > 3
        """
        df_out = df.copy()
        if column not in df_out.columns:
            raise ValueError(f"Column '{column}' not found.")
        if not pd.api.types.is_numeric_dtype(df_out[column]):
            raise ValueError(f"Column '{column}' must be numeric.")

        series = df_out[column].dropna()
        rows_before = len(df_out)
        outliers_modified = 0

        if method == "clip_percentiles":
            low_val = np.percentile(series, lower_pct)
            high_val = np.percentile(series, upper_pct)
            mask_outliers = (df_out[column] < low_val) | (df_out[column] > high_val)
            outliers_modified = int(mask_outliers.sum())
            df_out[column] = df_out[column].clip(lower=low_val, upper=high_val)
            details = f"Clipped {outliers_modified:,} values to range [{low_val:.2f}, {high_val:.2f}] (P{lower_pct:.0f}-P{upper_pct:.0f})."

        elif method == "clip_iqr":
            q25 = np.percentile(series, 25)
            q75 = np.percentile(series, 75)
            iqr = q75 - q25
            low_bound = q25 - 1.5 * iqr
            high_bound = q75 + 1.5 * iqr
            mask_outliers = (df_out[column] < low_bound) | (df_out[column] > high_bound)
            outliers_modified = int(mask_outliers.sum())
            df_out[column] = df_out[column].clip(lower=low_bound, upper=high_bound)
            details = f"Clipped {outliers_modified:,} values outside IQR bounds [{low_bound:.2f}, {high_bound:.2f}]."

        elif method == "drop_iqr":
            q25 = np.percentile(series, 25)
            q75 = np.percentile(series, 75)
            iqr = q75 - q25
            low_bound = q25 - 1.5 * iqr
            high_bound = q75 + 1.5 * iqr
            valid_mask = (df_out[column] >= low_bound) & (df_out[column] <= high_bound)
            outliers_modified = int((~valid_mask).sum())
            df_out = df_out[valid_mask].reset_index(drop=True)
            details = f"Dropped {outliers_modified:,} outlier rows outside IQR bounds."

        elif method == "drop_3sigma":
            mean_val = series.mean()
            std_val = series.std()
            if std_val > 0:
                z_scores = np.abs((df_out[column] - mean_val) / std_val)
                valid_mask = z_scores <= 3.0
                outliers_modified = int((~valid_mask).sum())
                df_out = df_out[valid_mask].reset_index(drop=True)
                details = f"Dropped {outliers_modified:,} rows with z-score > 3.0 (mean={mean_val:.2f}, std={std_val:.2f})."
            else:
                details = "Column standard deviation is 0. No rows dropped."
        else:
            raise ValueError(f"Unsupported outlier method '{method}'.")

        telemetry = {
            "operation": f"Outlier treatment on '{column}' ({method})",
            "column": column,
            "modified_count": outliers_modified,
            "rows_before": rows_before,
            "rows_after": len(df_out),
            "details": details
        }
        return df_out, telemetry

    @staticmethod
    def clean_text_column(
        df: pd.DataFrame,
        column: str,
        casing: str = "title",
        strip_whitespace: bool = True,
        remove_special_chars: bool = False,
        replace_text: Optional[Tuple[str, str]] = None
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Cleans and sanitizes a text column.
        """
        df_out = df.copy()
        if column not in df_out.columns:
            raise ValueError(f"Column '{column}' not found.")

        # Ensure string type
        series = df_out[column].astype(str)

        if strip_whitespace:
            series = series.str.strip().replace(r'\s+', ' ', regex=True)

        if remove_special_chars:
            series = series.replace(r'[^a-zA-Z0-9\s]', '', regex=True)

        if casing == "lower":
            series = series.str.lower()
        elif casing == "upper":
            series = series.str.upper()
        elif casing == "title":
            series = series.str.title()

        if replace_text and len(replace_text) == 2:
            old_str, new_str = replace_text
            if old_str:
                series = series.str.replace(old_str, new_str, regex=False)

        df_out[column] = series
        telemetry = {
            "operation": f"Text cleaning on '{column}'",
            "column": column,
            "casing": casing,
            "details": f"Applied {casing} casing, whitespace compaction, and sanitation."
        }
        return df_out, telemetry

    @staticmethod
    def extract_datetime_features(
        df: pd.DataFrame,
        column: str,
        features: List[str]
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Extracts sub-features from a datetime column.
        Supported features: 'year', 'month', 'day', 'dayofweek', 'hour', 'is_weekend'
        """
        df_out = df.copy()
        if column not in df_out.columns:
            raise ValueError(f"Column '{column}' not found.")

        dt_series = pd.to_datetime(df_out[column], errors='coerce')
        created_cols = []

        for f in features:
            f_clean = f.lower().strip()
            new_col_name = f"{column}_{f_clean}"
            if f_clean == "year":
                df_out[new_col_name] = dt_series.dt.year
                created_cols.append(new_col_name)
            elif f_clean == "month":
                df_out[new_col_name] = dt_series.dt.month
                created_cols.append(new_col_name)
            elif f_clean == "day":
                df_out[new_col_name] = dt_series.dt.day
                created_cols.append(new_col_name)
            elif f_clean == "hour":
                df_out[new_col_name] = dt_series.dt.hour
                created_cols.append(new_col_name)
            elif f_clean == "dayofweek":
                df_out[new_col_name] = dt_series.dt.day_name()
                created_cols.append(new_col_name)
            elif f_clean == "is_weekend":
                df_out[new_col_name] = dt_series.dt.dayofweek.isin([5, 6]).astype(int)
                created_cols.append(new_col_name)

        telemetry = {
            "operation": f"Extract datetime features from '{column}'",
            "column": column,
            "created_columns": created_cols,
            "details": f"Extracted {len(created_cols)} temporal columns: {', '.join(created_cols)}."
        }
        return df_out, telemetry

    @staticmethod
    def scale_features(
        df: pd.DataFrame,
        columns: List[str],
        method: str = "standard"
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Scales one or more numeric features.
        Methods: 'standard' (Z-score), 'minmax' [0, 1], 'robust' (IQR), 'log1p' (Log(1+x))
        """
        df_out = df.copy()
        valid_cols = [c for c in columns if c in df_out.columns and pd.api.types.is_numeric_dtype(df_out[c])]
        if not valid_cols:
            raise ValueError("No valid numeric columns selected for scaling.")

        if method == "standard":
            scaler = StandardScaler()
            df_out[valid_cols] = scaler.fit_transform(df_out[valid_cols].fillna(0))
            details = f"Applied StandardScaler (mean=0, variance=1) across {len(valid_cols)} columns."
        elif method == "minmax":
            scaler = MinMaxScaler()
            df_out[valid_cols] = scaler.fit_transform(df_out[valid_cols].fillna(0))
            details = f"Applied MinMaxScaler [0, 1] across {len(valid_cols)} columns."
        elif method == "robust":
            scaler = RobustScaler()
            df_out[valid_cols] = scaler.fit_transform(df_out[valid_cols].fillna(0))
            details = f"Applied RobustScaler (median and IQR) across {len(valid_cols)} columns."
        elif method == "log1p":
            for c in valid_cols:
                # Ensure non-negative before log
                min_v = df_out[c].min()
                if min_v < 0:
                    df_out[c] = np.log1p(df_out[c] - min_v)
                else:
                    df_out[c] = np.log1p(df_out[c].clip(lower=0))
            details = f"Applied Log1p transformation across {len(valid_cols)} columns."
        else:
            raise ValueError(f"Unsupported scaling method '{method}'.")

        telemetry = {
            "operation": f"Scale features ({method})",
            "columns": valid_cols,
            "details": details
        }
        return df_out, telemetry

    @staticmethod
    def derive_column(
        df: pd.DataFrame,
        new_col_name: str,
        col_a: str,
        operation: str,
        col_b_or_scalar: Any
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Creates a new derived column using arithmetic or combination.
        Operations: 'add', 'subtract', 'multiply', 'divide', 'ratio_pct', 'concat'
        """
        df_out = df.copy()
        if not new_col_name or not new_col_name.strip():
            raise ValueError("New column name cannot be empty.")
        if col_a not in df_out.columns:
            raise ValueError(f"Column '{col_a}' not found.")

        is_col_b = isinstance(col_b_or_scalar, str) and col_b_or_scalar in df_out.columns
        b_val = df_out[col_b_or_scalar] if is_col_b else float(col_b_or_scalar)

        if operation == "add":
            df_out[new_col_name] = df_out[col_a] + b_val
        elif operation == "subtract":
            df_out[new_col_name] = df_out[col_a] - b_val
        elif operation == "multiply":
            df_out[new_col_name] = df_out[col_a] * b_val
        elif operation == "divide":
            # Safe division
            divisor = b_val.replace(0, np.nan) if is_col_b else (np.nan if b_val == 0 else b_val)
            df_out[new_col_name] = (df_out[col_a] / divisor).fillna(0)
        elif operation == "ratio_pct":
            divisor = b_val.replace(0, np.nan) if is_col_b else (np.nan if b_val == 0 else b_val)
            df_out[new_col_name] = ((df_out[col_a] / divisor) * 100.0).fillna(0)
        elif operation == "concat":
            df_out[new_col_name] = df_out[col_a].astype(str) + "_" + (b_val.astype(str) if is_col_b else str(b_val))
        else:
            raise ValueError(f"Unsupported formula operation '{operation}'.")

        telemetry = {
            "operation": f"Derived column '{new_col_name}'",
            "new_column": new_col_name,
            "details": f"Created '{new_col_name}' = {col_a} {operation} {col_b_or_scalar}."
        }
        return df_out, telemetry

    @staticmethod
    def drop_columns(df: pd.DataFrame, columns: List[str]) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Drops selected columns from the dataset."""
        df_out = df.copy()
        to_drop = [c for c in columns if c in df_out.columns]
        if not to_drop:
            return df_out, {"operation": "Drop columns", "details": "No matching columns dropped."}

        df_out = df_out.drop(columns=to_drop)
        telemetry = {
            "operation": f"Drop {len(to_drop)} columns",
            "dropped_columns": to_drop,
            "details": f"Removed columns: {', '.join(to_drop)}."
        }
        return df_out, telemetry
