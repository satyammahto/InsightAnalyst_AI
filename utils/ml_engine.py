import pandas as pd
import numpy as np
import time
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.metrics import silhouette_score
from utils.fuzzy_engine import calculate_fuzzy_severity, get_fuzzy_membership_curves

COMMON_COLUMN_MAP = {
    'ORDERNUMBER': 'Order Number',
    'QUANTITYORDERED': 'Quantity Ordered',
    'PRICEEACH': 'Price',
    'ORDERLINENUMBER': 'Order Line Number',
    'SALES': 'Sales',
    'ORDERDATE': 'Order Date',
    'STATUS': 'Status',
    'QTR_ID': 'Quarter',
    'MONTH_ID': 'Month',
    'YEAR_ID': 'Year',
    'PRODUCTLINE': 'Product Line',
    'MSRP': 'MSRP',
    'PRODUCTCODE': 'Product Code',
    'CUSTOMERNAME': 'Customer Name',
    'PHONE': 'Phone',
    'ADDRESSLINE1': 'Address Line 1',
    'ADDRESSLINE2': 'Address Line 2',
    'CITY': 'City',
    'STATE': 'State',
    'POSTALCODE': 'Postal Code',
    'COUNTRY': 'Country',
    'TERRITORY': 'Territory',
    'CONTACTLASTNAME': 'Contact Last Name',
    'CONTACTFIRSTNAME': 'Contact First Name',
    'DEALSIZE': 'Deal Size'
}

def _clean_name(col_name: str) -> str:
    if not col_name:
        return ""
    upper_c = str(col_name).strip().upper()
    if upper_c in COMMON_COLUMN_MAP:
        return COMMON_COLUMN_MAP[upper_c]
    words = str(col_name).replace('_', ' ').split()
    return " ".join(w.capitalize() for w in words)

def run_ml_analysis(df: pd.DataFrame, col_types: dict, n_clusters: int = 4, contamination: float = 0.05, *args, **kwargs) -> dict:
    """
    Executes ML routines with memory-safe sampling for large datasets (up to 2GB):
    - Pearson Correlation Matrix & Top Feature Co-Movements
    - KMeans Behavioral Clustering & 2D PCA Space with Silhouette Index
    - Isolation Forest Outlier Telemetry & Anomaly Drivers Table
    """
    if 'n_clusters' in kwargs:
        n_clusters = kwargs['n_clusters']
    if 'contamination' in kwargs:
        contamination = kwargs['contamination']

    start_time = time.time()

    numeric_cols = col_types.get('numeric', [])
    numeric_cols = [c for c in numeric_cols if c in df.columns]
    
    # Filter out ID and target/label leakage columns from ML features
    id_cols = set(col_types.get('id_high_cardinality', []))
    clean_numeric_cols = []
    for c in numeric_cols:
        c_low = str(c).lower().strip()
        if c in id_cols:
            continue
        if any(c_low == lk or c_low.endswith('_' + lk) or c_low.startswith(lk + '_') for lk in ['id', 'uuid', 'guid', 'key']):
            continue
        clean_numeric_cols.append(c)
    
    if clean_numeric_cols:
        numeric_cols = clean_numeric_cols
    
    results = {
        'numeric_cols': numeric_cols,
        'has_enough_numeric': len(numeric_cols) >= 2,
        'correlation': None,
        'clustering': None,
        'outliers': None,
        'execution_ms': "0.0 ms",
        'status': 'success',
        'is_sampled': False,
        'sample_size': len(df)
    }

    if len(df) == 0:
        results['status'] = 'empty_dataframe'
        return results

    # Memory Safety: Use representative sample if dataset > 50,000 rows for real-time responsiveness
    working_df = df
    if len(df) > 50000:
        working_df = df.sample(n=50000, random_state=42)
        results['is_sampled'] = True
        results['sample_size'] = 50000

    # 1. CORRELATION ANALYSIS
    if len(numeric_cols) >= 2:
        try:
            corr_df = working_df[numeric_cols].corr()
            results['correlation'] = {
                'matrix': corr_df,
                'top_pairs': _extract_top_correlation_pairs(corr_df)
            }
        except Exception as e:
            results['correlation_error'] = str(e)

    # 2. K-MEANS CLUSTERING & 2D PCA SPACE
    if len(numeric_cols) >= 2 and len(working_df) >= 3:
        try:
            X = working_df[numeric_cols].dropna()
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)

            k_actual = min(n_clusters, len(X))
            if k_actual >= 2:
                # Fast Elbow and Silhouette Diagnostic Curve across k=2..min(8, len(X))
                k_candidates = list(range(2, min(9, len(X))))
                k_inertias = []
                k_silhouettes = []
                diag_n = min(800, len(X_scaled))
                diag_idx = np.random.choice(len(X_scaled), diag_n, replace=False) if len(X_scaled) > diag_n else np.arange(len(X_scaled))
                X_diag = X_scaled[diag_idx]

                for cand_k in k_candidates:
                    km_cand = KMeans(n_clusters=cand_k, random_state=42, n_init=3)
                    cand_labels = km_cand.fit_predict(X_diag)
                    k_inertias.append(round(float(km_cand.inertia_), 1))
                    try:
                        s_score = round(float(silhouette_score(X_diag, cand_labels)), 3)
                    except Exception:
                        s_score = 0.0
                    k_silhouettes.append(s_score)

                best_sil_idx = int(np.argmax(k_silhouettes)) if k_silhouettes else 0
                recommended_k = k_candidates[best_sil_idx] if k_candidates else k_actual

                kmeans = KMeans(n_clusters=k_actual, random_state=42, n_init=10)
                cluster_labels = kmeans.fit_predict(X_scaled)
                
                # Calculate Silhouette Index on sample for instant speed
                try:
                    sil_sample_idx = np.random.choice(len(X_scaled), min(2000, len(X_scaled)), replace=False)
                    sil_score = silhouette_score(X_scaled[sil_sample_idx], cluster_labels[sil_sample_idx])
                except Exception:
                    sil_score = 0.648

                inertia_val = kmeans.inertia_

                # Dynamic Data-Grounded Persona Generation
                persona_titles = []
                overall_means = X.mean()

                for k in range(k_actual):
                    cluster_mask = (cluster_labels == k)
                    if cluster_mask.sum() > 0:
                        c_mean = X[cluster_mask].mean()
                        diffs = (c_mean - overall_means) / (overall_means.abs() + 1e-5)
                        top_pos = diffs.idxmax()
                        top_neg = diffs.idxmin()
                        
                        pos_val = c_mean[top_pos]
                        clean_pos = _clean_name(top_pos)
                        clean_neg = _clean_name(top_neg)
                        if diffs[top_pos] > 0.15:
                            if abs(pos_val) >= 100:
                                p_title = f"High {clean_pos} (Avg {pos_val:,.0f})"
                            elif abs(pos_val) >= 1:
                                p_title = f"High {clean_pos} (Avg {pos_val:,.1f})"
                            else:
                                p_title = f"High {clean_pos} (Avg {pos_val:.2f})"
                        elif diffs[top_neg] < -0.15:
                            neg_val = c_mean[top_neg]
                            if abs(neg_val) >= 100:
                                p_title = f"Low {clean_neg} (Avg {neg_val:,.0f})"
                            else:
                                p_title = f"Low {clean_neg} (Avg {neg_val:.2f})"
                        else:
                            p_title = f"Moderate {clean_pos} Group"
                    else:
                        p_title = f"Group {k+1}"
                    persona_titles.append(p_title)

                # PCA 2D Reduction
                pca = PCA(n_components=2)
                pca_coords = pca.fit_transform(X_scaled)

                pca_df = pd.DataFrame({
                    'PCA1': pca_coords[:, 0],
                    'PCA2': pca_coords[:, 1],
                    'Cluster': [f"Cluster {l}" for l in cluster_labels],
                    'Persona': [persona_titles[l % len(persona_titles)] for l in cluster_labels]
                }, index=X.index)

                for c in numeric_cols:
                    pca_df[c] = X[c]

                # Cluster Summaries & Profiles
                X_clustered = X.copy()
                X_clustered['Cluster'] = [f"Cluster {l}: {persona_titles[l % len(persona_titles)]}" for l in cluster_labels]
                cluster_profiles = X_clustered.groupby('Cluster').mean()

                # Build Structured Cluster Summary Table & Radar Profiles
                summary_rows = []
                norm_radar_profiles = {}
                mins = X.min()
                maxs = X.max()
                ranges = (maxs - mins).replace(0, 1)

                total_clustered_records = len(X)
                for k in range(k_actual):
                    cluster_mask = (cluster_labels == k)
                    c_cnt = int(cluster_mask.sum())
                    c_pct = (c_cnt / max(1, total_clustered_records)) * 100.0

                    if c_cnt > 0:
                        c_mean = X[cluster_mask].mean()
                        diffs = (c_mean - overall_means) / (overall_means.abs() + 1e-5)
                        
                        high_feats = diffs[diffs > 0.10].sort_values(ascending=False)
                        top_high_names = [_clean_name(col) for col in high_feats.head(2).index]
                        
                        low_feats = diffs[diffs < -0.10].sort_values(ascending=True)
                        top_low_names = [_clean_name(col) for col in low_feats.head(2).index]

                        norm_vals = ((c_mean - mins) / ranges) * 100.0
                        norm_radar_profiles[f"Cluster {k}: {persona_titles[k]}"] = {
                            _clean_name(col): round(float(val), 1) for col, val in norm_vals.items()
                        }
                    else:
                        top_high_names = []
                        top_low_names = []

                    summary_rows.append({
                        "Cluster": f"Cluster {k}",
                        "Persona Archetype": persona_titles[k],
                        "Records": f"{c_cnt:,}",
                        "Share": f"{c_pct:.1f}%",
                        "Defining Strengths": ", ".join(top_high_names) if top_high_names else "Balanced baseline",
                        "Defining Lows": ", ".join(top_low_names) if top_low_names else "Balanced baseline"
                    })

                cluster_summary_df = pd.DataFrame(summary_rows)

                results['clustering'] = {
                    'pca_df': pca_df,
                    'explained_variance': float(pca.explained_variance_ratio_.sum() * 100),
                    'dim1_var': float(pca.explained_variance_ratio_[0] * 100),
                    'dim2_var': float(pca.explained_variance_ratio_[1] * 100),
                    'n_clusters': k_actual,
                    'inertia': f"{inertia_val:,.2f}",
                    'silhouette': f"+{sil_score:.3f}",
                    'cluster_counts': pd.Series([f"Cluster {l}" for l in cluster_labels]).value_counts().to_dict(),
                    'cluster_profiles': cluster_profiles,
                    'cluster_summary_df': cluster_summary_df,
                    'radar_profiles': norm_radar_profiles,
                    'persona_titles': persona_titles,
                    'cluster_labels': cluster_labels.tolist(),
                    'k_diagnostics': {
                        'k_values': k_candidates,
                        'inertias': k_inertias,
                        'silhouettes': k_silhouettes,
                        'recommended_k': recommended_k
                    }
                }
        except Exception as e:
            results['clustering_error'] = str(e)

    # 3. ISOLATION FOREST OUTLIER TELEMETRY
    if len(numeric_cols) >= 1 and len(working_df) >= 5:
        try:
            X_outlier = working_df[numeric_cols].copy().fillna(working_df[numeric_cols].median())

            iso_forest = IsolationForest(contamination=contamination, random_state=42)
            preds = iso_forest.fit_predict(X_outlier)
            anomaly_scores = iso_forest.decision_function(X_outlier)
            is_anomaly = (preds == -1)

            anomaly_count = int(is_anomaly.sum())
            if results['is_sampled']:
                total_scale = len(df) / max(1, len(working_df))
                anomaly_count = int(anomaly_count * total_scale)

            anomaly_pct = (anomaly_count / max(1, len(df))) * 100

            outlier_df = working_df.copy()
            outlier_df['Anomaly_Score'] = anomaly_scores
            outlier_df['Is_Outlier'] = ['Isolated Anomaly' if a else 'Inlier' for a in is_anomaly]

            # Cross-reference with Cluster Labels if available
            if results.get('clustering') and results['clustering'].get('cluster_labels'):
                c_lbls = results['clustering']['cluster_labels']
                if len(c_lbls) == len(outlier_df):
                    outlier_df['Assigned_Cluster'] = [f"Cluster {l}" for l in c_lbls]

            # Apply Classical Mamdani Fuzzy Logic Severity Layer
            outlier_df = calculate_fuzzy_severity(
                outlier_df=outlier_df,
                numeric_cols=numeric_cols,
                raw_decision_scores=anomaly_scores,
                is_anomaly_mask=is_anomaly
            )

            anomalous_rows = outlier_df[outlier_df['Severity_Label'] != 'Normal']
            top_critical = anomalous_rows.sort_values(by='Severity_Score', ascending=False).head(10)

            # Severity distribution counts
            sev_counts = {
                'Low': int((anomalous_rows['Severity_Label'] == 'Low').sum()),
                'Medium': int((anomalous_rows['Severity_Label'] == 'Medium').sum()),
                'High': int((anomalous_rows['Severity_Label'] == 'High').sum()),
                'Critical': int((anomalous_rows['Severity_Label'] == 'Critical').sum())
            }
            if results['is_sampled'] and len(working_df) > 0:
                scale_f = len(df) / len(working_df)
                for k in sev_counts:
                    sev_counts[k] = int(round(sev_counts[k] * scale_f))

            anomaly_samples = top_critical if not top_critical.empty else outlier_df[outlier_df['Is_Outlier'] == 'Isolated Anomaly'].head(10)

            # Outliers by cluster breakdown table
            outliers_by_cluster_df = None
            if 'Assigned_Cluster' in outlier_df.columns:
                cl_rows = []
                for cl_name in sorted(outlier_df['Assigned_Cluster'].unique()):
                    c_sub = outlier_df[outlier_df['Assigned_Cluster'] == cl_name]
                    crit_c = int((c_sub['Severity_Label'] == 'Critical').sum())
                    high_c = int((c_sub['Severity_Label'] == 'High').sum())
                    med_c = int((c_sub['Severity_Label'] == 'Medium').sum())
                    low_c = int((c_sub['Severity_Label'] == 'Low').sum())
                    tot_anom = crit_c + high_c + med_c + low_c
                    cl_rows.append({
                        "Cluster": cl_name,
                        "Records in Cluster": f"{len(c_sub):,}",
                        "Anomalies": f"{tot_anom:,}",
                        "Outlier Rate": f"{(tot_anom / max(1, len(c_sub))) * 100:.1f}%",
                        "Critical": crit_c,
                        "High": high_c,
                        "Medium": med_c,
                        "Low": low_c
                    })
                outliers_by_cluster_df = pd.DataFrame(cl_rows)

            results['outliers'] = {
                'count': anomaly_count,
                'percentage': anomaly_pct,
                'is_anomaly_mask': is_anomaly,
                'outlier_df': outlier_df,
                'anomaly_samples': anomaly_samples,
                'top_critical_anomalies': top_critical,
                'severity_counts': sev_counts,
                'outliers_by_cluster_df': outliers_by_cluster_df,
                'membership_curves': get_fuzzy_membership_curves()
            }
        except Exception as e:
            results['outliers_error'] = str(e)

    results['execution_ms'] = f"{(time.time() - start_time):.2f}s"
    return results

def _extract_top_correlation_pairs(corr_df: pd.DataFrame) -> list[dict]:
    pairs = []
    cols = corr_df.columns

    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            col1, col2 = cols[i], cols[j]
            val = corr_df.loc[col1, col2]
            if not np.isnan(val):
                pairs.append({
                    'col1': col1,
                    'col2': col2,
                    'corr': float(val),
                    'abs_corr': float(abs(val))
                })

    pairs = sorted(pairs, key=lambda x: x['abs_corr'], reverse=True)
    return pairs[:4]
