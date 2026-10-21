import {
  CleaningReport,
  ColumnClassification,
  DataRow,
  MLResults
} from '../types';
import { cleanName } from './cleanName';
import { DatasetRAGEngine } from './ragEngine';

export interface AIResponse {
  text: string;
  chart?: any;
  table?: DataRow[];
  codeSnippet?: string;
}

export class AIAssistant {
  constructor(
    private df: DataRow[],
    private colTypes: ColumnClassification,
    private ragEngine: DatasetRAGEngine,
    private mlResults: MLResults,
    private cleaningReport: CleaningReport,
    private datasetTitle: string = 'Active Dataset',
    private apiKey?: string
  ) {}

  async processMessage(userMessage: string): Promise<AIResponse> {
    const qNorm = userMessage.toLowerCase().trim();
    const numericCols = this.colTypes.numeric;
    const catCols = this.colTypes.categorical;
    const outInfo = this.mlResults.outliers;

    // 1. QUESTION: Critical Rows / Anomaly Risk
    if (
      qNorm.includes('critical') ||
      qNorm.includes('highest critical') ||
      qNorm.includes('anomal') ||
      qNorm.includes('outlier')
    ) {
      if (outInfo && outInfo.top_critical_anomalies.length > 0) {
        const topAnom = outInfo.top_critical_anomalies.slice(0, 5);
        const tableData = topAnom.map(a => {
          const lead = a.Top_Deviations[0];
          return {
            'Record': `Row #${a.index}`,
            'Severity': a.Severity_Label,
            'Score': `${a.Severity_Score.toFixed(1)}/100`,
            'Intensity': `${a.Anomaly_Intensity.toFixed(1)}`,
            'Deviation': `${a.Feature_Deviation.toFixed(1)}`,
            'Primary Factor': lead ? `${cleanName(lead.column)}: ${lead.this_val.toLocaleString()} (Median: ${lead.normal_median.toLocaleString()})` : 'N/A'
          };
        });

        const chartData = {
          type: 'bar',
          labels: topAnom.map(a => `Row #${a.index}`),
          datasets: [
            {
              label: 'Severity Score (Mamdani FIS)',
              data: topAnom.map(a => a.Severity_Score),
              backgroundColor: topAnom.map(a => a.Severity_Label === 'Critical' ? '#E11D48' : a.Severity_Label === 'High' ? '#EA580C' : '#D97706')
            }
          ]
        };

        const codeSnippet = `SELECT * FROM dataset WHERE severity_label IN ('Critical', 'High') ORDER BY severity_score DESC LIMIT 5;`;

        return {
          text: `Here are the **top ${topAnom.length} highest-risk records** evaluated by the **Mamdani Classical Fuzzy Inference Engine**.\n\n` +
            `• **Most Critical:** Row #${topAnom[0].index} with severity score **${topAnom[0].Severity_Score.toFixed(1)}/100** (${topAnom[0].Severity_Label}).\n` +
            `• **Evaluation Logic:** Evaluated across Anomaly Intensity, Feature Deviations (robust z-score), and Business Impact magnitude.\n` +
            `• **Recommended Action:** Review these records for data entry errors, potential fraud, or high-value operational anomalies.`,
          chart: chartData,
          table: tableData,
          codeSnippet
        };
      }
    }

    // 2. QUESTION: Biggest Problems / Data Quality
    if (
      qNorm.includes('biggest problem') ||
      qNorm.includes('problem') ||
      qNorm.includes('quality') ||
      qNorm.includes('dirty') ||
      qNorm.includes('missing')
    ) {
      const nDups = this.cleaningReport.duplicates_removed;
      const nNulls = this.cleaningReport.total_nulls_imputed;
      const nAnom = outInfo ? outInfo.count : 0;
      const nCrit = outInfo ? outInfo.severity_counts.Critical : 0;

      return {
        text: `### 🩺 Data Quality Assessment for **${this.datasetTitle}**\n\n` +
          `1. **Missing Data Imputation:** **${nNulls.toLocaleString()} missing values** were identified and automatically imputed (median for continuous numbers, mode for categorical labels).\n` +
          `2. **Duplicates Purged:** **${nDups.toLocaleString()} redundant records** were removed during initial ingestion.\n` +
          `3. **Anomalies & Extreme Deviations:** **${nAnom.toLocaleString()} records** deviate significantly from typical values (${nCrit.toLocaleString()} rated Critical severity).\n` +
          `4. **Hygiene Score:** Currently at **${this.cleaningReport.hygiene_score}** after defensive DAG stabilization.\n\n` +
          `**What to do next:** You can further refine outlier clipping or drop specific columns in the **Interactive Cleaning Studio** tab.`,
        codeSnippet: `-- Check remaining nulls or inspect cleaned columns\nSELECT COUNT(*) AS total_rows FROM dataset;`
      };
    }

    // 3. QUESTION: Key Insights / Findings
    if (
      qNorm.includes('key insight') ||
      qNorm.includes('important insight') ||
      qNorm.includes('finding') ||
      qNorm.includes('summary')
    ) {
      const topPairs = this.mlResults.correlation?.top_pairs || [];
      const clust = this.mlResults.clustering;

      let pairText = 'No dominant correlation pair found.';
      if (topPairs.length > 0) {
        const p = topPairs[0];
        pairText = `**${cleanName(p.col1)}** and **${cleanName(p.col2)}** have a ${p.corr > 0 ? 'positive' : 'negative'} relationship (correlation: **${p.corr >= 0 ? '+' : ''}${p.corr.toFixed(2)}**).`;
      }

      return {
        text: `### 💡 Key Strategic Insights for **${this.datasetTitle}**\n\n` +
          `• **Primary Relationship:** ${pairText}\n` +
          `• **Behavioral Segmentation:** The records naturally segment into **${clust ? clust.n_clusters : 4} distinct personas**, explaining **${clust ? clust.explained_variance.toFixed(1) : 75}%** of total dataset variance.\n` +
          `• **Data Hygiene:** The dataset has an overall hygiene rating of **${this.cleaningReport.hygiene_score}** with clean standardized formats.\n\n` +
          `**Actionable Tip:** Switch to the **Charts** tab to explore how your main metrics behave across segments.`,
        codeSnippet: topPairs.length > 0 ? `SELECT [${topPairs[0].col1}], [${topPairs[0].col2}] FROM dataset ORDER BY [${topPairs[0].col1}] DESC LIMIT 10;` : undefined
      };
    }

    // 4. QUESTION: Why Unusual?
    if (qNorm.includes('why') && (qNorm.includes('unusual') || qNorm.includes('flagged') || qNorm.includes('anom'))) {
      if (outInfo && outInfo.top_critical_anomalies.length > 0) {
        const first = outInfo.top_critical_anomalies[0];
        const topDev = first.Top_Deviations[0];
        return {
          text: `### 🔍 Why Records Are Flagged as Unusual\n\n` +
            `Records are flagged using **Isolation Forest** paired with a **Classical Mamdani Fuzzy Inference System**.\n\n` +
            `For example, in **Row #${first.index}**:\n` +
            `• **Anomaly Intensity:** **${first.Anomaly_Intensity.toFixed(1)}/100** (how isolated it is in feature space).\n` +
            `• **Feature Deviation:** **${first.Feature_Deviation.toFixed(1)}/100**.\n` +
            `• **Root Cause:** In **${topDev ? cleanName(topDev.column) : 'the primary feature'}**, the value is **${topDev ? topDev.this_val.toLocaleString() : 'N/A'}**, which deviates **${topDev ? topDev.robust_z : 'several'}x** from the normal baseline median (**${topDev ? topDev.normal_median.toLocaleString() : 'N/A'}**).\n` +
            `• **Business Impact:** High volume contribution yields a final severity score of **${first.Severity_Score.toFixed(1)}/100** (${first.Severity_Label}).`,
          table: [
            {
              'Record': `Row #${first.index}`,
              'Attribute': topDev ? cleanName(topDev.column) : 'Key Metric',
              'Record Value': topDev ? topDev.this_val.toLocaleString() : 'N/A',
              'Normal Median': topDev ? topDev.normal_median.toLocaleString() : 'N/A',
              'Deviation Multiple': topDev ? `${topDev.robust_z}x` : 'N/A'
            }
          ]
        };
      }
    }

    // 5. QUESTION: Top Drivers / Best Categories
    if (
      qNorm.includes('top driver') ||
      qNorm.includes('most sales') ||
      qNorm.includes('revenue') ||
      qNorm.includes('channel') ||
      qNorm.includes('category')
    ) {
      if (catCols.length > 0 && numericCols.length > 0) {
        const catCol = catCols[0];
        const numCol = numericCols[0];
        const groupSums: Record<string, number> = {};

        for (const r of this.df) {
          const key = String(r[catCol] ?? 'Other');
          const val = Number(r[numCol]) || 0;
          groupSums[key] = (groupSums[key] || 0) + val;
        }

        const sortedGroups = Object.entries(groupSums).sort((a, b) => b[1] - a[1]);
        const top5 = sortedGroups.slice(0, 5);

        return {
          text: `### 🏆 Top Drivers by **${cleanName(numCol)}** across **${cleanName(catCol)}**\n\n` +
            `**${top5[0][0]}** leads the dataset with **${top5[0][1].toLocaleString(undefined, { maximumFractionDigits: 1 })}** in total ${cleanName(numCol)}.\n\n` +
            top5.map((g, i) => `${i + 1}. **${g[0]}**: ${g[1].toLocaleString(undefined, { maximumFractionDigits: 1 })}`).join('\n'),
          chart: {
            type: 'bar',
            labels: top5.map(g => g[0]),
            datasets: [
              {
                label: `Total ${cleanName(numCol)}`,
                data: top5.map(g => g[1]),
                backgroundColor: '#0284C7'
              }
            ]
          },
          table: top5.map(g => ({
            [cleanName(catCol)]: g[0],
            [`Total ${cleanName(numCol)}`]: g[1].toLocaleString(undefined, { maximumFractionDigits: 1 })
          })),
          codeSnippet: `SELECT [${catCol}], SUM([${numCol}]) AS total_${numCol.toLowerCase()} FROM dataset GROUP BY [${catCol}] ORDER BY total_${numCol.toLowerCase()} DESC LIMIT 5;`
        };
      }
    }

    // 6. QUESTION: Trends / Dates
    if (qNorm.includes('trend') || qNorm.includes('date') || qNorm.includes('over time')) {
      const dateCols = this.colTypes.date;
      if (dateCols.length > 0 && numericCols.length > 0) {
        const dateCol = dateCols[0];
        const numCol = numericCols[0];

        const dateAgg: Record<string, number> = {};
        for (const r of this.df) {
          const dt = String(r[dateCol] ?? '').slice(0, 10);
          if (dt) {
            dateAgg[dt] = (dateAgg[dt] || 0) + (Number(r[numCol]) || 0);
          }
        }

        const sortedDates = Object.entries(dateAgg).sort((a, b) => a[0].localeCompare(b[0]));

        return {
          text: `### 📈 Timeline Trend for **${cleanName(numCol)}** over **${cleanName(dateCol)}**\n\n` +
            `Tracked across **${sortedDates.length} distinct dates**. Earliest record: **${sortedDates[0][0]}**, latest record: **${sortedDates[sortedDates.length - 1][0]}**.`,
          chart: {
            type: 'line',
            labels: sortedDates.map(d => d[0]),
            datasets: [
              {
                label: cleanName(numCol),
                data: sortedDates.map(d => d[1]),
                borderColor: '#0284C7',
                backgroundColor: 'rgba(2, 132, 199, 0.1)',
                tension: 0.3
              }
            ]
          },
          codeSnippet: `SELECT [${dateCol}], SUM([${numCol}]) AS daily_${numCol.toLowerCase()} FROM dataset GROUP BY [${dateCol}] ORDER BY [${dateCol}] ASC;`
        };
      }
    }

    // 7. Call Server-Side Gemini API if online, or intelligent fallback
    try {
      const response = await fetch('/api/chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          message: userMessage,
          context: this.ragEngine.generateContextString(),
          apiKey: this.apiKey
        })
      });

      if (response.ok) {
        const data = await response.json();
        if (data.text) {
          return {
            text: data.text,
            codeSnippet: `-- Generated by InsightAnalyst AI\nSELECT * FROM dataset LIMIT 10;`
          };
        }
      }
    } catch {
      // Fallback seamlessly to local engine
    }

    // Default intelligent local synthesis
    const matchedChunks = this.ragEngine.search(userMessage, 2);
    let summaryText = `I analyzed your query: **"${userMessage}"** against the dataset knowledge base.\n\n`;
    if (matchedChunks.length > 0) {
      summaryText += `**Relevant Information:**\n` + matchedChunks.map(c => `• **${c.title}:** ${c.content}`).join('\n\n');
    } else {
      summaryText += `The dataset contains **${this.df.length.toLocaleString()} rows** and **${Object.keys(this.df[0] || {}).length} columns** with an overall quality hygiene rating of **${this.cleaningReport.hygiene_score}**.\n\n` +
        `Try asking:\n` +
        `• *"Show me the highest critical rows"*\n` +
        `• *"What are the biggest problems in my data?"*\n` +
        `• *"Which products generate the most sales?"*`;
    }

    return {
      text: summaryText,
      codeSnippet: `SELECT * FROM dataset LIMIT 10;`
    };
  }
}
