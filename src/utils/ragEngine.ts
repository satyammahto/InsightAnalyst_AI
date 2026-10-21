import {
  CleaningReport,
  ColumnClassification,
  DataRow,
  MLResults,
  RAGChunk
} from '../types';
import { cleanName } from './cleanName';

export class DatasetRAGEngine {
  chunks: RAGChunk[] = [];

  buildKnowledgeBase(
    df: DataRow[],
    colTypes: ColumnClassification,
    cleaningReport: CleaningReport,
    mlResults: MLResults
  ) {
    this.chunks = [];
    const nRows = df.length;
    const nCols = df.length > 0 ? Object.keys(df[0]).length : 0;

    // CHUNK 1: Schema & Data Ingestion
    this.chunks.push({
      chunk_id: 'SCHEMA_INGESTION',
      title: 'Dataset Volume & Schema Topology',
      category: 'Schema',
      content: `Active dataset has ${nRows.toLocaleString()} rows and ${nCols} columns. Numeric features: ${colTypes.numeric.map(cleanName).join(', ') || 'None'}. Categorical attributes: ${colTypes.categorical.map(cleanName).join(', ') || 'None'}. Dates: ${colTypes.date.map(cleanName).join(', ') || 'None'}. High-cardinality keys: ${colTypes.id_high_cardinality.join(', ') || 'None'}.`
    });

    // CHUNK 2: Data Quality & Defensive Cleaning DAG
    const nNulls = cleaningReport.total_nulls_imputed;
    const nDups = cleaningReport.duplicates_removed;
    this.chunks.push({
      chunk_id: 'HYGIENE_DAG_TELEMETRY',
      title: '5-Stage Defensive Cleaning & Hygiene Telemetry',
      category: 'Data Quality',
      content: `Hygiene Score rated at ${cleaningReport.hygiene_score}. Pipeline completed in ${cleaningReport.execution_ms}. Deduplication purged ${nDups.toLocaleString()} redundant records. Imputation filled ${nNulls.toLocaleString()} empty values across features. String normalizer sanitized ${cleaningReport.string_tokens_sanitized.toLocaleString()} tokens.`
    });

    // CHUNK 3: Correlations & Feature Drivers
    if (mlResults.correlation && mlResults.correlation.top_pairs.length > 0) {
      const top3 = mlResults.correlation.top_pairs.slice(0, 3).map(
        p => `${cleanName(p.col1)} <-> ${cleanName(p.col2)} (r = ${p.corr >= 0 ? '+' : ''}${p.corr})`
      );
      this.chunks.push({
        chunk_id: 'FEATURE_CORRELATIONS',
        title: 'Key Correlation Pairs & Co-Movements',
        category: 'Relationships',
        content: `Leading statistical relationships: ${top3.join('; ')}.`
      });
    }

    // CHUNK 4: Behavioral Clusters & Segmentation
    if (mlResults.clustering) {
      const c = mlResults.clustering;
      this.chunks.push({
        chunk_id: 'KMEANS_CLUSTERING',
        title: 'KMeans Behavioral Group Personas',
        category: 'Clustering',
        content: `Dataset partitions naturally into ${c.n_clusters} clusters with silhouette index ${c.silhouette} and ${c.explained_variance.toFixed(1)}% variance explained via 2D PCA. Detected archetypes: ${c.persona_titles.map((p, idx) => `Group ${idx + 1}: ${p}`).join(', ')}.`
      });
    }

    // CHUNK 5: Anomaly Risk & Mamdani Fuzzy Severity
    if (mlResults.outliers) {
      const o = mlResults.outliers;
      this.chunks.push({
        chunk_id: 'MAMDANI_ANOMALY_SEVERITY',
        title: 'Mamdani Classical Fuzzy Anomaly Assessment',
        category: 'Anomalies',
        content: `Detected ${o.count.toLocaleString()} unusual records (${o.percentage}% of dataset). Mamdani Classical FIS classified severity as: Critical (${o.severity_counts.Critical.toLocaleString()}), High (${o.severity_counts.High.toLocaleString()}), Medium (${o.severity_counts.Medium.toLocaleString()}), Low (${o.severity_counts.Low.toLocaleString()}). Critical records show high anomaly intensity combined with extreme feature deviations.`
      });
    }
  }

  search(query: string, topK: number = 3): RAGChunk[] {
    if (this.chunks.length === 0) return [];
    const qLower = query.toLowerCase();
    const queryWords = qLower.split(/\s+/).filter(w => w.length > 2);

    const scored = this.chunks.map(chunk => {
      let score = 0;
      const text = (chunk.title + ' ' + chunk.content + ' ' + chunk.category).toLowerCase();
      for (const w of queryWords) {
        if (text.includes(w)) score += 2;
      }
      return { chunk, score };
    });

    scored.sort((a, b) => b.score - a.score);
    return scored.slice(0, topK).map(s => s.chunk);
  }

  generateContextString(): string {
    return this.chunks.map(c => `[${c.chunk_id}] ${c.title}:\n${c.content}`).join('\n\n');
  }
}
