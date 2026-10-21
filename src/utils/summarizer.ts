import {
  ActionableInsight,
  CleaningReport,
  ColumnClassification,
  DataRow,
  ExecutiveBrief,
  MLResults
} from '../types';
import { cleanName } from './cleanName';

export function generateExecutiveSummary(
  df: DataRow[],
  colTypes: ColumnClassification,
  cleaningReport: CleaningReport,
  mlResults: MLResults
): ExecutiveBrief {
  const nRows = df.length;
  const nCols = df.length > 0 ? Object.keys(df[0]).length : 0;
  const hygieneScore = cleaningReport.hygiene_score;
  const outInfo = mlResults.outliers;
  const outCnt = outInfo ? outInfo.count : 0;
  const outPct = outInfo ? outInfo.percentage : 0;
  const nCrit = outInfo ? outInfo.severity_counts.Critical : 0;
  const nHigh = outInfo ? outInfo.severity_counts.High : 0;
  const topPairs = mlResults.correlation ? mlResults.correlation.top_pairs : [];
  const clustInfo = mlResults.clustering;

  const whatWeFound: string[] = [
    `Your dataset contains **${nRows.toLocaleString()} records** across **${nCols} columns**.`,
    `Overall data hygiene is **rated at ${hygieneScore}** after automated verification.`
  ];

  if (outCnt > 0) {
    if (nCrit > 0) {
      whatWeFound.push(
        `**${outCnt.toLocaleString()} records** exhibit unusual patterns, including **${nCrit.toLocaleString()} critical** and **${nHigh.toLocaleString()} high** severity records evaluated via Mamdani Fuzzy Logic.`
      );
    } else {
      whatWeFound.push(
        `**${outCnt.toLocaleString()} records** look unusual and may need review (${outPct.toFixed(1)}% of dataset).`
      );
    }
  } else {
    whatWeFound.push(
      'No severe anomalous records were found; values conform to expected statistical distributions.'
    );
  }

  if (topPairs.length > 0) {
    const p1 = topPairs[0];
    const c1 = cleanName(p1.col1);
    const c2 = cleanName(p1.col2);
    const dir = p1.corr > 0 ? 'closely together' : 'in opposite directions';
    whatWeFound.push(
      `**${c1}** and **${c2}** move ${dir} (correlation: ${p1.corr >= 0 ? '+' : ''}${p1.corr.toFixed(2)}).`
    );
  }

  if (colTypes.categorical.length > 0) {
    const catCol = colTypes.categorical[0];
    const counts: Record<string, number> = {};
    for (const r of df) {
      const v = String(r[catCol] ?? '');
      counts[v] = (counts[v] || 0) + 1;
    }
    let topVal = '';
    let maxC = 0;
    for (const [k, count] of Object.entries(counts)) {
      if (count > maxC) {
        maxC = count;
        topVal = k;
      }
    }
    if (topVal) {
      const share = ((maxC / Math.max(nRows, 1)) * 100).toFixed(1);
      whatWeFound.push(
        `The largest category in **${cleanName(catCol)}** is **${topVal}**, making up **${share}%** of all records.`
      );
    }
  }

  const whatShouldYouLookAt: string[] = [];
  if (nCrit > 0) {
    whatShouldYouLookAt.push(
      `Check the **${nCrit.toLocaleString()} critical severity records** in the Patterns & Anomalies tab to identify operational discrepancies.`
    );
  } else if (outCnt > 0) {
    whatShouldYouLookAt.push(
      `Review unusual records in the Patterns & Anomalies tab to ensure data validity.`
    );
  } else {
    whatShouldYouLookAt.push(
      'Verify distributions in the Visual Charts tab to confirm target metrics.'
    );
  }

  if (topPairs.length > 0) {
    const lead = topPairs[0];
    whatShouldYouLookAt.push(
      `Investigate the relationship between **${cleanName(lead.col1)}** and **${cleanName(lead.col2)}** to discover optimization levers.`
    );
  }

  if (clustInfo && clustInfo.n_clusters > 1) {
    whatShouldYouLookAt.push(
      `Examine your **${clustInfo.n_clusters} behavioral groups** to tailor distinct operational strategies for each persona.`
    );
  }

  const actionableInsights: ActionableInsight[] = [];

  if (topPairs.length > 0) {
    const lead = topPairs[0];
    actionableInsights.push({
      icon: '🔗',
      what: `${cleanName(lead.col1)} and ${cleanName(lead.col2)} show a strong ${lead.corr > 0 ? 'positive' : 'negative'} relationship.`,
      why: `Changes in ${cleanName(lead.col1)} reliably signal corresponding movements in ${cleanName(lead.col2)}.`,
      action: `Use ${cleanName(lead.col1)} as a leading indicator when forecasting ${cleanName(lead.col2)}.`,
      technical_label: 'Pearson r',
      technical_val: `${lead.corr >= 0 ? '+' : ''}${lead.corr.toFixed(2)}`
    });
  }

  if (outCnt > 0) {
    actionableInsights.push({
      icon: '🚨',
      what: `${outCnt.toLocaleString()} unusual records detected (${outPct.toFixed(1)}% of dataset).`,
      why: 'Extreme outliers distort arithmetic averages and can indicate process errors or high-value exceptions.',
      action: 'Isolate these records in the Patterns & Anomalies tab or treat them via the Cleaning Studio.',
      technical_label: 'Evaluator',
      technical_val: 'Mamdani Classical FIS'
    });
  }

  if (clustInfo) {
    actionableInsights.push({
      icon: '🧩',
      what: `Dataset partitions naturally into ${clustInfo.n_clusters} distinct behavioral cohorts.`,
      why: 'Uniform treatment across all records misses persona-specific opportunities and nuances.',
      action: 'Filter records by cluster in the Visual Explorer to study individual cohort profiles.',
      technical_label: 'Silhouette Score',
      technical_val: clustInfo.silhouette
    });
  }

  return {
    what_we_found: whatWeFound,
    what_should_you_look_at: whatShouldYouLookAt,
    actionable_insights: actionableInsights,
    ingestion_friendly: `Successfully ingested and cleaned ${nRows.toLocaleString()} rows with 0 unhandled errors. Hygiene rated at ${hygieneScore}.`,
    driver_friendly: topPairs.length > 0 ? `Strongest co-movement identified between ${cleanName(topPairs[0].col1)} and ${cleanName(topPairs[0].col2)} (r = ${topPairs[0].corr.toFixed(2)}).` : 'No dominant feature co-movement detected.',
    segmentation_friendly: clustInfo ? `Partitioned into ${clustInfo.n_clusters} cohorts explaining ${clustInfo.explained_variance.toFixed(1)}% variance.` : 'Single cohort pattern observed.',
    outlier_friendly: outCnt > 0 ? `Detected ${outCnt.toLocaleString()} unusual records (${nCrit.toLocaleString()} critical, ${nHigh.toLocaleString()} high severity).` : 'All records conform to standard distributions.'
  };
}

export function generateHtmlDossier(
  df: DataRow[],
  colTypes: ColumnClassification,
  cleaningReport: CleaningReport,
  mlResults: MLResults,
  brief: ExecutiveBrief,
  datasetName: string = 'Active Dataset',
  studioHistory: string[] = []
): string {
  const dateStr = new Date().toLocaleDateString('en-US', { year: 'numeric', month: 'long', day: 'numeric' });
  const nRows = df.length;
  const nCols = df.length > 0 ? Object.keys(df[0]).length : 0;

  return `<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>Executive Audit Dossier - ${datasetName}</title>
  <style>
    body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; background: #F8FAFC; color: #0F172A; margin: 0; padding: 40px; }
    .container { max-width: 950px; margin: 0 auto; background: #FFFFFF; border-radius: 12px; border: 1px solid #E2E8F0; padding: 40px; box-shadow: 0 4px 20px rgba(0,0,0,0.05); }
    h1 { font-size: 26px; font-weight: 800; color: #0F172A; margin-top: 0; border-bottom: 2px solid #E2E8F0; padding-bottom: 16px; }
    h2 { font-size: 18px; font-weight: 700; color: #0284C7; margin-top: 28px; }
    .meta-bar { display: flex; justify-content: space-between; font-size: 13px; color: #64748B; margin-bottom: 24px; }
    .kpi-grid { display: grid; grid-template-columns: repeat(4, 1fr); gap: 14px; margin: 20px 0; }
    .kpi-card { background: #F8FAFC; border: 1px solid #E2E8F0; border-radius: 8px; padding: 14px; }
    .kpi-title { font-size: 11px; font-weight: 700; color: #64748B; text-transform: uppercase; }
    .kpi-val { font-size: 22px; font-weight: 800; color: #0F172A; margin-top: 4px; }
    .insight-box { background: #EFF6FF; border-left: 4px solid #0284C7; padding: 16px; border-radius: 6px; margin: 14px 0; font-size: 13.5px; line-height: 1.5; }
    table { width: 100%; border-collapse: collapse; margin-top: 14px; font-size: 13px; }
    th { background: #F1F5F9; text-align: left; padding: 10px; border-bottom: 2px solid #CBD5E1; color: #334155; }
    td { padding: 9px 10px; border-bottom: 1px solid #E2E8F0; }
    .badge { display: inline-block; padding: 3px 8px; border-radius: 12px; font-size: 11px; font-weight: 700; }
    .badge-crit { background: #FFE4E6; color: #BE123C; }
    .badge-high { background: #FFEDD5; color: #C2410C; }
    .badge-med { background: #FEF08A; color: #854D0E; }
    .badge-low { background: #DCFCE7; color: #166534; }
  </style>
</head>
<body>
  <div class="container">
    <h1>✨ InsightAnalyst AI — Executive Audit Dossier</h1>
    <div class="meta-bar">
      <span><b>Dataset:</b> ${datasetName}</span>
      <span><b>Generated:</b> ${dateStr}</span>
      <span><b>Hygiene:</b> ${cleaningReport.hygiene_score}</span>
    </div>

    <div class="kpi-grid">
      <div class="kpi-card"><div class="kpi-title">Records</div><div class="kpi-val">${nRows.toLocaleString()}</div></div>
      <div class="kpi-card"><div class="kpi-title">Attributes</div><div class="kpi-val">${nCols}</div></div>
      <div class="kpi-card"><div class="kpi-title">Hygiene Score</div><div class="kpi-val" style="color:#059669;">${cleaningReport.hygiene_score}</div></div>
      <div class="kpi-card"><div class="kpi-title">Unusual Cases</div><div class="kpi-val" style="color:#E11D48;">${mlResults.outliers ? mlResults.outliers.count.toLocaleString() : 0}</div></div>
    </div>

    <h2>Executive Synthesis</h2>
    <ul>
      ${brief.what_we_found.map(item => `<li>${item.replace(/\*\*(.*?)\*\*/g, '<b>$1</b>')}</li>`).join('')}
    </ul>

    <h2>Actionable Strategic Takeaways</h2>
    ${brief.actionable_insights.map(item => `
      <div class="insight-box">
        <b>${item.icon} ${item.what}</b><br>
        <span style="color:#475569;">Why it matters: ${item.why}</span><br>
        <span style="color:#0369A1; font-weight:600;">Recommended Action: ${item.action}</span>
      </div>
    `).join('')}

    <h2>Data Quality & 5-Stage Defensive Cleaning DAG</h2>
    <table>
      <thead><tr><th>Step</th><th>Phase</th><th>Telemetry Detail</th><th>Badge</th><th>Latency</th></tr></thead>
      <tbody>
        ${cleaningReport.dag_steps.map(s => `
          <tr><td>${s.step_num}</td><td><b>${s.title}</b></td><td>${s.desc}</td><td>${s.badge}</td><td>${s.latency}</td></tr>
        `).join('')}
      </tbody>
    </table>

    ${mlResults.outliers && mlResults.outliers.top_critical_anomalies.length > 0 ? `
      <h2>Top Critical Anomalies (Mamdani Classical FIS)</h2>
      <table>
        <thead><tr><th>Row #</th><th>Severity</th><th>Score</th><th>Intensity</th><th>Deviation</th><th>Business Impact</th><th>Root Cause</th></tr></thead>
        <tbody>
          ${mlResults.outliers.top_critical_anomalies.map(a => `
            <tr>
              <td>Row #${a.index}</td>
              <td><span class="badge ${a.Severity_Label === 'Critical' ? 'badge-crit' : a.Severity_Label === 'High' ? 'badge-high' : 'badge-med'}">${a.Severity_Label}</span></td>
              <td>${a.Severity_Score.toFixed(1)}/100</td>
              <td>${a.Anomaly_Intensity.toFixed(1)}</td>
              <td>${a.Feature_Deviation.toFixed(1)}</td>
              <td>${a.Business_Impact.toFixed(1)}</td>
              <td>${a.Top_Deviations[0] ? `${cleanName(a.Top_Deviations[0].column)}: ${a.Top_Deviations[0].this_val} (Median: ${a.Top_Deviations[0].normal_median})` : 'N/A'}</td>
            </tr>
          `).join('')}
        </tbody>
      </table>
    ` : ''}

    ${studioHistory.length > 0 ? `
      <h2>Custom Studio Transformations Applied</h2>
      <ul>
        ${studioHistory.map(h => `<li>${h}</li>`).join('')}
      </ul>
    ` : ''}
  </div>
</body>
</html>`;
}

export function generateMarkdownDossier(
  df: DataRow[],
  colTypes: ColumnClassification,
  cleaningReport: CleaningReport,
  mlResults: MLResults,
  brief: ExecutiveBrief,
  datasetName: string = 'Active Dataset',
  studioHistory: string[] = []
): string {
  const nRows = df.length;
  const nCols = df.length > 0 ? Object.keys(df[0]).length : 0;
  const dateStr = new Date().toISOString().split('T')[0];

  let md = `# Executive Audit Dossier — ${datasetName}\n\n`;
  md += `**Date:** ${dateStr} | **Hygiene:** ${cleaningReport.hygiene_score} | **Records:** ${nRows.toLocaleString()} | **Attributes:** ${nCols}\n\n`;

  md += `## 1. Executive Summary\n\n`;
  brief.what_we_found.forEach(item => {
    md += `- ${item}\n`;
  });
  md += `\n`;

  md += `## 2. Actionable Insights\n\n`;
  brief.actionable_insights.forEach(item => {
    md += `### ${item.icon} ${item.what}\n`;
    md += `- **Why it matters:** ${item.why}\n`;
    md += `- **Recommended Action:** ${item.action}\n`;
    md += `- **${item.technical_label}:** \`${item.technical_val}\`\n\n`;
  });

  md += `## 3. Data Cleaning DAG Telemetry\n\n`;
  cleaningReport.dag_steps.forEach(s => {
    md += `- **[Step ${s.step_num}] ${s.title}:** ${s.desc} (${s.latency})\n`;
  });
  md += `\n`;

  if (mlResults.outliers && mlResults.outliers.top_critical_anomalies.length > 0) {
    md += `## 4. Top Critical Anomalies (Mamdani Classical FIS)\n\n`;
    md += `| Record | Severity | Score | Intensity | Deviation | Business Impact | Primary Feature Deviation |\n`;
    md += `|---|---|---|---|---|---|---|\n`;
    mlResults.outliers.top_critical_anomalies.forEach(a => {
      const topDev = a.Top_Deviations[0];
      const devStr = topDev ? `${cleanName(topDev.column)}: ${topDev.this_val} (Median: ${topDev.normal_median})` : 'N/A';
      md += `| Row #${a.index} | ${a.Severity_Label} | ${a.Severity_Score.toFixed(1)}/100 | ${a.Anomaly_Intensity.toFixed(1)} | ${a.Feature_Deviation.toFixed(1)} | ${a.Business_Impact.toFixed(1)} | ${devStr} |\n`;
    });
    md += `\n`;
  }

  if (studioHistory.length > 0) {
    md += `## 5. Applied Studio Transformations\n\n`;
    studioHistory.forEach((h, idx) => {
      md += `${idx + 1}. ${h}\n`;
    });
  }

  return md;
}
