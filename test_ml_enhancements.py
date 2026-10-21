"""
Unit tests for Machine Learning & Clustering Enhancements (Step 4)
"""
import sys
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

import pandas as pd
import numpy as np
from utils.type_detector import detect_column_types
from utils.ml_engine import run_ml_analysis
from utils.dashboard import plot_elbow_silhouette, plot_cluster_radar, plot_outliers_by_cluster

def run_tests():
    print("--- Starting ML & Clustering Enhancements Unit Tests ---")
    np.random.seed(42)

    # Synthetic multi-cluster dataset
    cluster1 = np.random.normal(loc=[10, 50, 100], scale=[2, 5, 10], size=(40, 3))
    cluster2 = np.random.normal(loc=[30, 20, 200], scale=[3, 4, 15], size=(40, 3))
    cluster3 = np.random.normal(loc=[50, 80, 50], scale=[4, 6, 8], size=(40, 3))
    data = np.vstack([cluster1, cluster2, cluster3])

    df = pd.DataFrame(data, columns=["Sensor_A", "Sensor_B", "Sensor_C"])
    # Inject outliers
    df.loc[0, "Sensor_A"] = 500
    df.loc[41, "Sensor_B"] = 400

    col_types = detect_column_types(df)
    results = run_ml_analysis(df, col_types, n_clusters=3, contamination=0.08)

    assert results["status"] == "success", f"ML analysis failed: {results}"
    assert results["clustering"] is not None, "Clustering results are None"

    # Test 1: Optimal K Diagnostics
    clust = results["clustering"]
    assert "k_diagnostics" in clust, "Missing k_diagnostics in clustering results"
    k_diag = clust["k_diagnostics"]
    assert len(k_diag["k_values"]) >= 3, "Expected at least 3 k candidate values"
    assert len(k_diag["inertias"]) == len(k_diag["k_values"])
    assert len(k_diag["silhouettes"]) == len(k_diag["k_values"])
    assert 2 <= k_diag["recommended_k"] <= 8
    print(f"✓ Test 1: k_diagnostics verified (Recommended k={k_diag['recommended_k']})")

    # Test 2: Cluster Persona Summary Table
    assert "cluster_summary_df" in clust, "Missing cluster_summary_df in clustering results"
    summary_df = clust["cluster_summary_df"]
    assert len(summary_df) == 3, f"Expected 3 cluster rows, got {len(summary_df)}"
    assert "Persona Archetype" in summary_df.columns
    assert "Defining Strengths" in summary_df.columns
    print(f"✓ Test 2: cluster_summary_df verified ({len(summary_df)} clusters summarized)")

    # Test 3: Radar Profiles
    assert "radar_profiles" in clust, "Missing radar_profiles in clustering results"
    radar = clust["radar_profiles"]
    assert len(radar) == 3, f"Expected 3 radar profiles, got {len(radar)}"
    print("✓ Test 3: radar_profiles verified")

    # Test 4: Outliers By Cluster
    outliers = results["outliers"]
    assert outliers is not None, "Outliers results are None"
    assert "outliers_by_cluster_df" in outliers
    out_cl_df = outliers["outliers_by_cluster_df"]
    assert out_cl_df is not None and len(out_cl_df) > 0, "Outliers by cluster table is empty"
    assert "Critical" in out_cl_df.columns and "Anomalies" in out_cl_df.columns
    print(f"✓ Test 4: outliers_by_cluster_df verified ({len(out_cl_df)} cluster cohorts)")

    # Test 5: Dashboard Visualizations
    fig_diag = plot_elbow_silhouette(k_diag)
    assert fig_diag is not None, "plot_elbow_silhouette failed"

    fig_rad = plot_cluster_radar(radar)
    assert fig_rad is not None, "plot_cluster_radar failed"

    fig_out = plot_outliers_by_cluster(out_cl_df)
    assert fig_out is not None, "plot_outliers_by_cluster failed"
    print("✓ Test 5: All 3 Plotly chart figures successfully generated")

    print("\n=== ALL ML & CLUSTERING ENHANCEMENTS TESTS PASSED! ===")

if __name__ == "__main__":
    run_tests()
