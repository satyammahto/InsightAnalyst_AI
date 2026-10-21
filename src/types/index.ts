export type DataRow = Record<string, any>;

export interface ColumnTelemetry {
  featureName: string;
  schemaType: string;
  nonNullCount: string;
  uniquePct: string;
  missingPct: string;
  sampleValues: string;
}

export interface ColumnClassification {
  numeric: string[];
  categorical: string[];
  date: string[];
  id_high_cardinality: string[];
  text: string[];
  details: Record<string, string>;
  telemetry_table: ColumnTelemetry[];
}

export interface DAGStep {
  step_num: string;
  title: string;
  tag: string;
  desc: string;
  badge: string;
  latency: string;
}

export interface CleaningReport {
  initial_shape: [number, number];
  final_shape: [number, number];
  duplicates_removed: number;
  total_nulls_imputed: number;
  nulls_imputed: Record<string, number>;
  string_tokens_sanitized: number;
  dates_converted: string[];
  hygiene_score: string;
  dag_steps: DAGStep[];
  execution_ms: string;
}

export interface CorrelationPair {
  col1: string;
  col2: string;
  corr: number;
}

export interface CorrelationResult {
  matrix: Record<string, Record<string, number>>;
  columns: string[];
  top_pairs: CorrelationPair[];
}

export interface TopDeviation {
  column: string;
  normal_median: number;
  this_val: number;
  robust_z: number;
}

export type SeverityRiskLevel = 'Low' | 'Medium' | 'High' | 'Critical';

export interface CriticalAnomaly {
  index: number;
  Severity_Label: SeverityRiskLevel;
  Severity_Score: number;
  Anomaly_Intensity: number;
  Feature_Deviation: number;
  Business_Impact: number;
  Severity_Explanation: string;
  Top_Deviations: TopDeviation[];
  rowData: DataRow;
}

export interface OutlierResult {
  count: number;
  percentage: number;
  severity_counts: {
    Low: number;
    Medium: number;
    High: number;
    Critical: number;
  };
  top_critical_anomalies: CriticalAnomaly[];
  all_anomalies: CriticalAnomaly[];
  membership_curves: {
    universe: number[];
    inputs: {
      low: number[];
      medium: number[];
      high: number[];
    };
    outputs: {
      low: number[];
      medium: number[];
      high: number[];
      critical: number[];
    };
  };
  outliers_by_cluster_df: Array<{ cluster: string; count: number }>;
}

export interface KDiagnostics {
  k_values: number[];
  inertias: number[];
  silhouettes: number[];
  recommended_k: number;
}

export interface ClusterResult {
  n_clusters: number;
  silhouette: string;
  explained_variance: number;
  persona_titles: string[];
  cluster_labels: number[];
  pca_df: Array<{
    pc1: number;
    pc2: number;
    cluster: number;
    cluster_name: string;
    index: number;
  }>;
  radar_profiles: {
    features: string[];
    groups: Array<{
      groupName: string;
      color: string;
      values: number[];
    }>;
  };
  cluster_summary_df: Array<Record<string, any>>;
  k_diagnostics: KDiagnostics;
}

export interface MLResults {
  numeric_cols: string[];
  has_enough_numeric: boolean;
  correlation: CorrelationResult | null;
  clustering: ClusterResult | null;
  outliers: OutlierResult | null;
  execution_ms: string;
  status: string;
  is_sampled: boolean;
  sample_size: number;
}

export interface RAGChunk {
  chunk_id: string;
  title: string;
  category: string;
  content: string;
}

export interface ActionableInsight {
  icon: string;
  what: string;
  why: string;
  action: string;
  technical_label: string;
  technical_val: string;
}

export interface ExecutiveBrief {
  what_we_found: string[];
  what_should_you_look_at: string[];
  actionable_insights: ActionableInsight[];
  ingestion_friendly: string;
  driver_friendly: string;
  segmentation_friendly: string;
  outlier_friendly: string;
}

export interface ChatMessage {
  role: 'user' | 'assistant';
  content: string;
  chart?: any;
  table?: DataRow[];
  code?: string;
}

export interface StudioTransformationRecord {
  id: string;
  operation: string;
  details: string;
  timestamp: string;
}
