"""
Classical Mamdani Fuzzy Inference Engine for Anomaly Severity Assessment
InsightAnalyst AI - Enterprise Data Engine

This module implements a classical Mamdani-style Fuzzy Inference System (FIS)
that sits on top of Isolation Forest anomaly detection to determine HOW SEVERE
an anomaly is based on three linguistic inputs:
1. Anomaly Intensity (from normalized inverse Isolation Forest score)
2. Feature Deviation (from robust z-score / MAD across numerical columns)
3. Business Impact (from value/volume magnitude and percentile rank)

Architecture:
  Crisp Inputs (0-100)
       ↓
  Fuzzification (Triangular / Trapezoidal Membership Functions)
       ↓
  Fuzzy Rules Evaluation (Mamdani Min-Conjunction, Max-Disjunction)
       ↓
  Rule Implication (Min Clipping) & Aggregation (Max T-Conorm)
       ↓
  Centroid Defuzzification
       ↓
  Crisp Severity Score (0-100) → Low / Medium / High / Critical
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, List, Tuple, Optional

# -------------------------------------------------------------
# 1. Classical Membership Functions (Pure NumPy, 100% Robust)
# -------------------------------------------------------------
def trimf(x: np.ndarray | float, abc: List[float]) -> np.ndarray | float:
    """Triangular membership function: abc = [left_foot, peak, right_foot]."""
    a, b, c = float(abc[0]), float(abc[1]), float(abc[2])
    if isinstance(x, (int, float, np.number)):
        if x <= a or x >= c:
            return 0.0
        if a < x <= b:
            return (x - a) / (b - a) if b != a else 1.0
        return (c - x) / (c - b) if c != b else 1.0

    x_arr = np.asarray(x, dtype=float)
    y = np.zeros_like(x_arr)
    # Left slope
    if b != a:
        idx1 = np.logical_and(a < x_arr, x_arr <= b)
        y[idx1] = (x_arr[idx1] - a) / (b - a)
    else:
        y[x_arr == b] = 1.0
    # Right slope
    if c != b:
        idx2 = np.logical_and(b < x_arr, x_arr < c)
        y[idx2] = (c - x_arr[idx2]) / (c - b)
    return y


def trapmf(x: np.ndarray | float, abcd: List[float]) -> np.ndarray | float:
    """Trapezoidal membership function: abcd = [left_foot, left_shoulder, right_shoulder, right_foot]."""
    a, b, c, d = float(abcd[0]), float(abcd[1]), float(abcd[2]), float(abcd[3])
    if isinstance(x, (int, float, np.number)):
        if x <= a or x >= d:
            return 0.0
        if a < x < b:
            return (x - a) / (b - a) if b != a else 1.0
        if b <= x <= c:
            return 1.0
        return (d - x) / (d - c) if d != c else 1.0

    x_arr = np.asarray(x, dtype=float)
    y = np.zeros_like(x_arr)
    if b != a:
        idx1 = np.logical_and(a < x_arr, x_arr < b)
        y[idx1] = (x_arr[idx1] - a) / (b - a)
    idx2 = np.logical_and(b <= x_arr, x_arr <= c)
    y[idx2] = 1.0
    if d != c:
        idx3 = np.logical_and(c < x_arr, x_arr < d)
        y[idx3] = (d - x_arr[idx3]) / (d - c)
    return y


# -------------------------------------------------------------
# 2. Fuzzy Universe & Linguistic Terms Definition
# -------------------------------------------------------------
# Universe of Discourse: 0 to 100 for all variables
UNIVERSE = np.linspace(0.0, 100.0, 201)

# Input Membership Function Parameters (0 - 100)
INPUT_MF_PARAMS = {
    'low': [0.0, 0.0, 20.0, 45.0],      # trapmf: high at 0, tapers off by 45
    'medium': [20.0, 50.0, 80.0],       # trimf: centered at 50, overlaps with low and high
    'high': [55.0, 80.0, 100.0, 100.0]  # trapmf: rises at 55, flat top 80-100
}

# Output Severity Membership Function Parameters (0 - 100)
OUTPUT_MF_PARAMS = {
    'low': [0.0, 0.0, 15.0, 35.0],        # trapmf
    'medium': [20.0, 42.5, 62.0],         # trimf
    'high': [48.0, 68.0, 85.0],           # trimf
    'critical': [72.0, 88.0, 100.0, 100.0] # trapmf
}


class MamdaniFuzzyInferenceSystem:
    """
    Classical Mamdani Fuzzy Inference System for Anomaly Severity.
    Evaluates 14 explicit fuzzy inference rules using Min conjunction and Max aggregation,
    followed by Centroid defuzzification.
    """
    def __init__(self):
        self.universe = UNIVERSE
        self.input_params = INPUT_MF_PARAMS
        self.output_params = OUTPUT_MF_PARAMS

    def fuzzify_input(self, val: float) -> Dict[str, float]:
        """Calculates membership grades [Low, Medium, High] for an input in [0, 100]."""
        v = float(np.clip(val, 0.0, 100.0))
        return {
            'low': float(trapmf(v, self.input_params['low'])),
            'medium': float(trimf(v, self.input_params['medium'])),
            'high': float(trapmf(v, self.input_params['high']))
        }

    def evaluate(self, intensity: float, deviation: float, impact: float) -> Tuple[float, str, Dict[str, float]]:
        """
        Executes Mamdani inference for given crisp inputs:
        Returns:
          - defuzzified_score: float in [0, 100]
          - severity_label: 'Low' | 'Medium' | 'High' | 'Critical'
          - rule_activations: dict of firing strengths for each severity class
        """
        # Step 1: Fuzzification
        mu_I = self.fuzzify_input(intensity)
        mu_D = self.fuzzify_input(deviation)
        mu_B = self.fuzzify_input(impact)

        # Step 2: Fuzzy Rules Evaluation (Mamdani Min-Conjunction, Max-Disjunction)
        # RULE 1: IF intensity is LOW AND deviation is LOW AND impact is LOW THEN severity is LOW
        r1 = min(mu_I['low'], mu_D['low'], mu_B['low'])

        # RULE 2: IF intensity is MEDIUM OR deviation is MEDIUM THEN severity is MEDIUM
        r2 = max(mu_I['medium'], mu_D['medium'])

        # RULE 3: IF intensity is HIGH AND deviation is HIGH THEN severity is HIGH
        r3 = min(mu_I['high'], mu_D['high'])

        # RULE 4: IF intensity is HIGH AND business impact is HIGH THEN severity is CRITICAL
        r4 = min(mu_I['high'], mu_B['high'])

        # RULE 5: IF feature deviation is HIGH AND business impact is HIGH THEN severity is CRITICAL
        r5 = min(mu_D['high'], mu_B['high'])

        # RULE 6: IF intensity is HIGH AND deviation is MEDIUM THEN severity is HIGH
        r6 = min(mu_I['high'], mu_D['medium'])

        # RULE 7: IF intensity is MEDIUM AND business impact is HIGH THEN severity is HIGH
        r7 = min(mu_I['medium'], mu_B['high'])

        # RULE 8: IF intensity is LOW AND deviation is HIGH THEN severity is MEDIUM
        r8 = min(mu_I['low'], mu_D['high'])

        # RULE 9: IF intensity is HIGH AND deviation is HIGH AND business impact is HIGH THEN severity is CRITICAL
        r9 = min(mu_I['high'], mu_D['high'], mu_B['high'])

        # Additional complementary rules for full 3D boundary coverage:
        # RULE 10: IF intensity is HIGH AND deviation is LOW AND impact is LOW THEN severity is MEDIUM
        r10 = min(mu_I['high'], mu_D['low'], mu_B['low'])

        # RULE 11: IF intensity is LOW AND deviation is MEDIUM AND impact is LOW THEN severity is LOW
        r11 = min(mu_I['low'], mu_D['medium'], mu_B['low'])

        # RULE 12: IF intensity is MEDIUM AND deviation is HIGH AND impact is LOW THEN severity is MEDIUM
        r12 = min(mu_I['medium'], mu_D['high'], mu_B['low'])

        # RULE 13: IF intensity is MEDIUM AND deviation is MEDIUM AND impact is MEDIUM THEN severity is MEDIUM
        r13 = min(mu_I['medium'], mu_D['medium'], mu_B['medium'])

        # RULE 14: IF intensity is LOW AND business impact is HIGH THEN severity is MEDIUM
        r14 = min(mu_I['low'], mu_B['high'])

        # Aggregate Rule Firing Strengths per Consequent Class (Max S-Norm)
        alpha_low = max(r1, r11)
        alpha_medium = max(r2, r8, r10, r12, r13, r14)
        alpha_high = max(r3, r6, r7)
        alpha_critical = max(r4, r5, r9)

        # Step 3: Consequent Implication (Mamdani Min Clipping)
        mf_low = trapmf(self.universe, self.output_params['low'])
        mf_medium = trimf(self.universe, self.output_params['medium'])
        mf_high = trimf(self.universe, self.output_params['high'])
        mf_critical = trapmf(self.universe, self.output_params['critical'])

        clipped_low = np.minimum(alpha_low, mf_low)
        clipped_medium = np.minimum(alpha_medium, mf_medium)
        clipped_high = np.minimum(alpha_high, mf_high)
        clipped_critical = np.minimum(alpha_critical, mf_critical)

        # Step 4: Rule Aggregation (Max T-Conorm)
        aggregated = np.maximum.reduce([clipped_low, clipped_medium, clipped_high, clipped_critical])

        # Step 5: Defuzzification (Centroid / Center of Gravity Method)
        sum_weights = np.sum(aggregated)
        if sum_weights > 1e-6:
            centroid_score = float(np.sum(self.universe * aggregated) / sum_weights)
        else:
            # Fallback if no rule fired significantly
            centroid_score = float(np.clip(0.4 * intensity + 0.35 * deviation + 0.25 * impact, 0.0, 100.0))

        centroid_score = float(np.clip(centroid_score, 0.0, 100.0))
        severity_label = classify_severity(centroid_score)

        activations = {
            'low': float(alpha_low),
            'medium': float(alpha_medium),
            'high': float(alpha_high),
            'critical': float(alpha_critical)
        }

        return centroid_score, severity_label, activations


# -------------------------------------------------------------
# 3. Singleton Engine & Scikit-Fuzzy Bridge
# -------------------------------------------------------------
_FUZZY_ENGINE_INSTANCE: Optional[MamdaniFuzzyInferenceSystem] = None

def create_fuzzy_system() -> MamdaniFuzzyInferenceSystem:
    """Initializes and returns the singleton Mamdani Fuzzy Inference System."""
    global _FUZZY_ENGINE_INSTANCE
    if _FUZZY_ENGINE_INSTANCE is None:
        _FUZZY_ENGINE_INSTANCE = MamdaniFuzzyInferenceSystem()
    return _FUZZY_ENGINE_INSTANCE


# -------------------------------------------------------------
# 4. Input Feature Extraction & Normalization
# -------------------------------------------------------------
def normalize_anomaly_score(
    decision_scores: np.ndarray,
    is_anomaly_mask: np.ndarray
) -> np.ndarray:
    """
    Transforms raw Isolation Forest decision_function scores into normalized
    Anomaly Intensity values in [0, 100] where:
      0 = very normal
      100 = extremely anomalous

    Remember: in Isolation Forest, lower/negative scores indicate higher anomalies.
    """
    if len(decision_scores) == 0:
        return np.array([], dtype=float)

    scores = np.asarray(decision_scores, dtype=float)
    intensity = np.zeros_like(scores, dtype=float)

    # Find minimum and threshold
    min_score = np.min(scores)
    max_score = np.max(scores)

    # For anomalous records (pred == -1), score is below contamination cutoff
    anom_scores = scores[is_anomaly_mask] if np.any(is_anomaly_mask) else scores
    anom_cutoff = np.max(anom_scores) if len(anom_scores) > 0 else 0.0

    # Normalization logic:
    # Points below anom_cutoff receive higher intensity from 50 to 100
    # Points above anom_cutoff receive lower intensity from 0 to 50
    for i, s in enumerate(scores):
        if is_anomaly_mask[i]:
            # Scale within anomalous range: anom_cutoff -> 55, min_score -> 100
            denom = abs(min_score - anom_cutoff) if abs(min_score - anom_cutoff) > 1e-5 else 1.0
            diff = (anom_cutoff - s) / denom
            intensity[i] = 55.0 + 45.0 * np.clip(diff, 0.0, 1.0)
        else:
            # Scale within inlier range: max_score -> 5, anom_cutoff -> 45
            denom = abs(max_score - anom_cutoff) if abs(max_score - anom_cutoff) > 1e-5 else 1.0
            diff = (max_score - s) / denom
            intensity[i] = 5.0 + 40.0 * np.clip(diff, 0.0, 1.0)

    return np.clip(intensity, 0.0, 100.0)


def calculate_feature_deviation(
    df: pd.DataFrame,
    numeric_cols: List[str]
) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
    """
    Measures how far each record's numerical values deviate from normal dataset values.
    Uses robust z-score: z = 0.6745 * |x - median| / MAD.
    Aggregates the top 3 absolute deviations for each record and normalizes to [0, 100].
    Handles zero MAD, constant columns, missing values, and single columns without crashing.
    """
    n_rows = len(df)
    if n_rows == 0 or not numeric_cols:
        return np.zeros(n_rows, dtype=float), [{} for _ in range(n_rows)]

    valid_cols = [c for c in numeric_cols if c in df.columns]
    if not valid_cols:
        return np.zeros(n_rows, dtype=float), [{} for _ in range(n_rows)]

    z_matrix = np.zeros((n_rows, len(valid_cols)), dtype=float)
    col_medians = {}

    for j, c in enumerate(valid_cols):
        series = pd.to_numeric(df[c], errors='coerce').fillna(df[c].median() if df[c].notnull().any() else 0.0)
        med = float(series.median())
        col_medians[c] = med

        # Median Absolute Deviation
        abs_diff = np.abs(series - med)
        mad = float(abs_diff.median())

        if mad > 1e-6:
            z_vals = 0.6745 * abs_diff / mad
        else:
            # Fallback to standard deviation if MAD is zero
            std_val = float(series.std())
            if std_val > 1e-6:
                z_vals = abs_diff / std_val
            else:
                # Column is constant: zero deviation
                z_vals = np.zeros(n_rows, dtype=float)

        z_matrix[:, j] = z_vals

    aggregated_deviations = np.zeros(n_rows, dtype=float)
    top_deviations_list = []

    k_top = min(3, len(valid_cols))

    for i in range(n_rows):
        row_z = z_matrix[i, :]
        sorted_indices = np.argsort(row_z)[::-1]
        
        top_k_indices = sorted_indices[:k_top]
        top_mean_z = float(np.mean(row_z[top_k_indices])) if len(top_k_indices) > 0 else 0.0

        # Normalization: robust z-score of 4.5 maps to 100/100 (extreme deviation)
        # z=1.5 maps to ~33/100, z=3.0 maps to ~66/100
        score = np.clip((top_mean_z / 4.5) * 100.0, 0.0, 100.0)
        aggregated_deviations[i] = score

        # Store top feature deviation details for human explanation
        top_details = []
        for idx in top_k_indices:
            col_name = valid_cols[idx]
            val = float(df[col_name].iloc[i]) if pd.notnull(df[col_name].iloc[i]) else 0.0
            top_details.append({
                'column': col_name,
                'this_val': val,
                'normal_median': col_medians[col_name],
                'robust_z': float(row_z[idx])
            })
        top_deviations_list.append(top_details)

    return aggregated_deviations, top_deviations_list


def calculate_business_impact(
    df: pd.DataFrame,
    numeric_cols: List[str]
) -> Tuple[np.ndarray, str]:
    """
    Creates a generic business-impact signal normalized to [0, 100].
    Dynamically identifies business-value columns using keyword matching
    (sales, revenue, amount, price, profit, cost, value, etc.).
    Falls back gracefully to the column with highest variance or magnitude.
    """
    n_rows = len(df)
    if n_rows == 0 or not numeric_cols:
        return np.full(n_rows, 50.0, dtype=float), "Neutral Fallback"

    valid_cols = [c for c in numeric_cols if c in df.columns]
    if not valid_cols:
        return np.full(n_rows, 50.0, dtype=float), "Neutral Fallback"

    # Keywords for business value
    keywords = ['sales', 'revenue', 'amount', 'price', 'profit', 'cost', 'value', 'transaction', 'total', 'spend', 'charge', 'balance', 'fee', 'salary', 'income']

    chosen_col = None
    for kw in keywords:
        for c in valid_cols:
            if kw in c.lower():
                chosen_col = c
                break
        if chosen_col:
            break

    if not chosen_col:
        # Fallback to column with largest dynamic range
        chosen_col = valid_cols[0]

    series = pd.to_numeric(df[chosen_col], errors='coerce').fillna(df[chosen_col].median() if df[chosen_col].notnull().any() else 0.0)

    # Compute percentile rank (0 to 100)
    if len(series) > 1 and series.nunique() > 1:
        ranks = series.rank(pct=True).values * 100.0
    else:
        ranks = np.full(n_rows, 50.0, dtype=float)

    return np.clip(ranks, 0.0, 100.0), chosen_col


# -------------------------------------------------------------
# 5. End-to-End Severity Calculation & Classification
# -------------------------------------------------------------
def classify_severity(score: float) -> str:
    """
    Converts a crisp defuzzified severity score (0-100) into human-readable label:
    - 0-30:   Low (Green)
    - 30-55:  Medium (Yellow)
    - 55-80:  High (Orange)
    - 80-100: Critical (Red)
    """
    s = float(score)
    if s < 30.0:
        return "Low"
    elif s < 55.0:
        return "Medium"
    elif s < 80.0:
        return "High"
    else:
        return "Critical"


def explain_severity(
    intensity: float,
    deviation: float,
    impact: float,
    severity_score: float,
    severity_label: str,
    top_deviations: Optional[List[Dict[str, Any]]] = None
) -> str:
    """
    Generates a natural, human-readable business explanation for why a record was
    classified with a given anomaly severity.
    """
    if severity_label == "Critical":
        core = "Highly unusual record with major feature deviations and high business impact."
    elif severity_label == "High":
        core = "Significant anomaly with strong deviations from typical records."
    elif severity_label == "Medium":
        core = "Moderate anomaly with some unusual feature values worth checking."
    else:
        core = "Minor anomaly with limited deviation from normal baseline patterns."

    drivers_str = ""
    if top_deviations and len(top_deviations) > 0:
        lead_feat = top_deviations[0]
        col = lead_feat.get('column', 'Metric')
        val = lead_feat.get('this_val', 0.0)
        med = lead_feat.get('normal_median', 0.0)
        drivers_str = f" Primary driver: {col} is {val:,.1f} (normal baseline is {med:,.1f})."

    return f"{core}{drivers_str}"


def calculate_fuzzy_severity(
    outlier_df: pd.DataFrame,
    numeric_cols: List[str],
    raw_decision_scores: np.ndarray,
    is_anomaly_mask: np.ndarray
) -> pd.DataFrame:
    """
    Integrates the complete Mamdani Fuzzy Inference System with Isolation Forest results.
    Computes:
      - Anomaly_Intensity (0-100)
      - Feature_Deviation (0-100)
      - Business_Impact (0-100)
      - Severity_Score (0-100)
      - Severity_Label (Normal / Low / Medium / High / Critical)
      - Severity_Explanation
      - Top_Deviations
    Keeps normal records marked as Normal with Severity_Score = 0.
    """
    df = outlier_df.copy()
    n_rows = len(df)
    
    if n_rows == 0:
        return df

    # Step 1: Compute Inputs
    try:
        intensities = normalize_anomaly_score(raw_decision_scores, is_anomaly_mask)
    except Exception:
        intensities = np.full(n_rows, 50.0)

    try:
        deviations, top_dev_list = calculate_feature_deviation(df, numeric_cols)
    except Exception:
        deviations = np.full(n_rows, 50.0)
        top_dev_list = [[] for _ in range(n_rows)]

    try:
        impacts, impact_col = calculate_business_impact(df, numeric_cols)
    except Exception:
        impacts = np.full(n_rows, 50.0)
        impact_col = "Fallback"

    # Step 2: Initialize Fuzzy Inference System
    fis = create_fuzzy_system()

    severity_scores = np.zeros(n_rows, dtype=float)
    severity_labels = ["Normal" for _ in range(n_rows)]
    explanations = ["Normal record following baseline patterns." for _ in range(n_rows)]

    # Step 3: Run Mamdani inference only on anomalous records (as required for performance)
    for i in range(n_rows):
        if is_anomaly_mask[i]:
            score, label, _ = fis.evaluate(
                intensity=intensities[i],
                deviation=deviations[i],
                impact=impacts[i]
            )
            severity_scores[i] = round(score, 1)
            severity_labels[i] = label
            explanations[i] = explain_severity(
                intensities[i], deviations[i], impacts[i],
                score, label, top_dev_list[i] if i < len(top_dev_list) else None
            )

    df['Anomaly_Intensity'] = np.round(intensities, 1)
    df['Feature_Deviation'] = np.round(deviations, 1)
    df['Business_Impact'] = np.round(impacts, 1)
    df['Severity_Score'] = severity_scores
    df['Severity_Label'] = severity_labels
    df['Severity_Explanation'] = explanations
    df['Top_Deviations'] = top_dev_list

    return df


# -------------------------------------------------------------
# 6. Visualization Data Helpers for Dashboard & Viva Demo
# -------------------------------------------------------------
def get_fuzzy_membership_curves() -> Dict[str, Any]:
    """
    Returns coordinate data for Plotly visualization of input and output
    membership functions, useful for explaining fuzzy sets during college vivas.
    """
    u = UNIVERSE

    # Input membership curves
    input_curves = {
        'universe': u,
        'low': trapmf(u, INPUT_MF_PARAMS['low']),
        'medium': trimf(u, INPUT_MF_PARAMS['medium']),
        'high': trapmf(u, INPUT_MF_PARAMS['high'])
    }

    # Output severity curves
    output_curves = {
        'universe': u,
        'low': trapmf(u, OUTPUT_MF_PARAMS['low']),
        'medium': trimf(u, OUTPUT_MF_PARAMS['medium']),
        'high': trimf(u, OUTPUT_MF_PARAMS['high']),
        'critical': trapmf(u, OUTPUT_MF_PARAMS['critical'])
    }

    return {
        'input_curves': input_curves,
        'output_curves': output_curves
    }
