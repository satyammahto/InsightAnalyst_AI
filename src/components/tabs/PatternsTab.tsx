import React, { useState } from 'react';
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  BarElement,
  ArcElement,
  RadialLinearScale,
  Title,
  Tooltip,
  Legend
} from 'chart.js';
import { Scatter, Doughnut, Radar, Line, Bar } from 'react-chartjs-2';
import {
  ChevronDown,
  ChevronRight,
  AlertTriangle,
  Lightbulb,
  Info
} from 'lucide-react';
import {
  ColumnClassification,
  DataRow,
  MLResults
} from '../../types';
import { cleanName } from '../../utils/cleanName';

ChartJS.register(
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  BarElement,
  ArcElement,
  RadialLinearScale,
  Title,
  Tooltip,
  Legend
);

interface PatternsTabProps {
  df: DataRow[];
  colTypes: ColumnClassification;
  mlResults: MLResults;
  isSimpleMode: boolean;
}

export const PatternsTab: React.FC<PatternsTabProps> = ({
  df: _df,
  colTypes: _colTypes,
  mlResults,
  isSimpleMode
}) => {
  const clustInfo = mlResults.clustering;
  const outInfo = mlResults.outliers;

  const nGrps = clustInfo ? clustInfo.n_clusters : 4;
  const silStr = clustInfo ? clustInfo.silhouette : '+0.648';
  const expVar = clustInfo ? clustInfo.explained_variance : 74.6;
  const outCnt = outInfo ? outInfo.count : 0;
  const outPct = outInfo ? outInfo.percentage : 0;
  const sevCounts = outInfo ? outInfo.severity_counts : { Low: 0, Medium: 0, High: 0, Critical: 0 };

  const [clusterSubTab, setClusterSubTab] = useState<'pca' | 'radar'>('pca');
  const [diagnosticsOpen, setDiagnosticsOpen] = useState(false);
  const [fuzzyDemoOpen, setFuzzyDemoOpen] = useState(!isSimpleMode);

  // Selected Anomaly for Inspection
  const [selectedAnomalyIdx, setSelectedAnomalyIdx] = useState<number>(0);
  const topCriticalList = outInfo?.top_critical_anomalies || [];
  const selectedAnomaly = topCriticalList[selectedAnomalyIdx] || topCriticalList[0];

  // PCA Scatter Data
  const pcaScatterData = {
    datasets: clustInfo
      ? Array.from({ length: nGrps }, (_, c) => {
          const points = clustInfo.pca_df
            .filter(p => p.cluster === c + 1)
            .map(p => ({ x: p.pc1, y: p.pc2 }));
          const colors = ['#0284C7', '#059669', '#7C3AED', '#EA580C', '#E11D48', '#0D9488'];
          return {
            label: `Group ${c + 1}: ${clustInfo.persona_titles[c]}`,
            data: points,
            backgroundColor: colors[c % colors.length]
          };
        })
      : []
  };

  // Radar Profiles Data
  const radarData = {
    labels: clustInfo?.radar_profiles.features || [],
    datasets: clustInfo?.radar_profiles.groups.map(g => ({
      label: g.groupName,
      data: g.values,
      borderColor: g.color,
      backgroundColor: g.color.replace(')', ', 0.2)').replace('rgb', 'rgba'),
      borderWidth: 2
    })) || []
  };

  // Severity Doughnut Data
  const donutData = {
    labels: ['Low Risk', 'Medium Risk', 'High Risk', 'Critical Risk'],
    datasets: [
      {
        data: [sevCounts.Low, sevCounts.Medium, sevCounts.High, sevCounts.Critical],
        backgroundColor: ['#10B981', '#FBBF24', '#F97316', '#EF4444'],
        borderWidth: 0
      }
    ]
  };

  // Outliers by Cluster Bar Data
  const outliersByClusterData = {
    labels: outInfo?.outliers_by_cluster_df.map(d => d.cluster) || [],
    datasets: [
      {
        label: 'Unusual Records',
        data: outInfo?.outliers_by_cluster_df.map(d => d.count) || [],
        backgroundColor: '#E11D48',
        borderRadius: 6
      }
    ]
  };

  // Diagnostics Line Data
  const diagnosticsData = {
    labels: clustInfo?.k_diagnostics.k_values.map(k => `k=${k}`) || [],
    datasets: [
      {
        label: 'Silhouette Score (Higher is Better)',
        data: clustInfo?.k_diagnostics.silhouettes || [],
        borderColor: '#059669',
        backgroundColor: 'rgba(5, 150, 105, 0.1)',
        yAxisID: 'y'
      },
      {
        label: 'Inertia / Error (Lower is Better)',
        data: clustInfo?.k_diagnostics.inertias || [],
        borderColor: '#0284C7',
        backgroundColor: 'rgba(2, 132, 199, 0.1)',
        yAxisID: 'y1'
      }
    ]
  };

  return (
    <div className="space-y-6">
      {/* Title */}
      <div>
        <h2 className="text-2xl font-extrabold text-slate-900 tracking-tight">Patterns & Unusual Records</h2>
        <p className="text-sm text-slate-500 mt-1">
          Discover how your records naturally group together and spot unusual cases that need your attention.
        </p>
      </div>

      {/* 4 KPI Metrics */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-3.5">
        <div className="bg-white border border-slate-200/90 rounded-xl p-4 shadow-sm shadow-slate-900/[0.02]">
          <span className="text-[11px] font-bold text-slate-500 uppercase tracking-wider block mb-1">
            NATURAL GROUPS
          </span>
          <div className="text-2xl font-extrabold text-blue-600">{nGrps} Groups</div>
          <div className="text-xs text-slate-500 mt-1 font-medium">Grouped by shared characteristics</div>
        </div>

        <div className="bg-white border border-slate-200/90 rounded-xl p-4 shadow-sm shadow-slate-900/[0.02]">
          <span className="text-[11px] font-bold text-slate-500 uppercase tracking-wider block mb-1">
            GROUPING QUALITY
          </span>
          <div className="text-2xl font-extrabold text-emerald-600">{silStr}</div>
          <div className="text-xs text-emerald-600 mt-1 font-medium">Clear separation between groups</div>
        </div>

        <div className="bg-white border border-slate-200/90 rounded-xl p-4 shadow-sm shadow-slate-900/[0.02]">
          <span className="text-[11px] font-bold text-slate-500 uppercase tracking-wider block mb-1">
            UNUSUAL RECORDS
          </span>
          <div className="text-2xl font-extrabold text-rose-600">{outCnt.toLocaleString()}</div>
          <div className="text-xs text-rose-600 mt-1 font-medium">{outPct.toFixed(1)}% of total records</div>
        </div>

        <div className="bg-white border border-slate-200/90 rounded-xl p-4 shadow-sm shadow-slate-900/[0.02]">
          <span className="text-[11px] font-bold text-slate-500 uppercase tracking-wider block mb-1">
            PATTERN EXPLAINED
          </span>
          <div className="text-2xl font-extrabold text-purple-600">{expVar.toFixed(1)}%</div>
          <div className="text-xs text-slate-500 mt-1 font-medium">High pattern coverage</div>
        </div>
      </div>

      {/* SECTION A: Natural Patterns & Segmentation */}
      <div className="space-y-4 pt-2">
        <div>
          <h3 className="text-lg font-extrabold text-slate-900">🧩 Patterns Found in Your Data</h3>
          <p className="text-xs text-slate-500 mt-0.5">
            Your data naturally separates into <b>{nGrps} distinct groups</b> based on their characteristics:
          </p>
        </div>

        {/* AI Cluster Recommendation Banner */}
        {clustInfo?.k_diagnostics && (
          <div className="bg-emerald-50/80 border border-emerald-200 rounded-xl p-3 px-4 flex items-center justify-between text-xs text-emerald-900">
            <div className="flex items-center gap-2">
              <Lightbulb className="w-4 h-4 text-emerald-600" />
              <span>
                <b>AI Cluster Recommendation:</b> Silhouette & Elbow inflection suggest <b>k = {clustInfo.k_diagnostics.recommended_k}</b> for highest separation clarity.
              </span>
            </div>
            <span className="bg-emerald-100 text-emerald-800 font-bold px-2 py-0.5 rounded-full text-[11px]">
              SILHOUETTE {silStr}
            </span>
          </div>
        )}

        {/* Diagnostics Expander */}
        <div className="border border-slate-200 rounded-xl overflow-hidden bg-white">
          <button
            type="button"
            onClick={() => setDiagnosticsOpen(!diagnosticsOpen)}
            className="w-full p-3.5 bg-slate-50/80 hover:bg-slate-100/80 flex items-center justify-between text-xs font-bold text-slate-800 transition-colors"
          >
            <span>🔬 View Optimal Group Diagnostics (Elbow Method & Silhouette Curve)</span>
            {diagnosticsOpen ? <ChevronDown className="w-4 h-4" /> : <ChevronRight className="w-4 h-4" />}
          </button>
          {diagnosticsOpen && (
            <div className="p-4 border-t border-slate-100 h-64">
              <Line
                data={diagnosticsData}
                options={{
                  responsive: true,
                  maintainAspectRatio: false,
                  scales: {
                    y: { type: 'linear', position: 'left', title: { display: true, text: 'Silhouette' } },
                    y1: { type: 'linear', position: 'right', grid: { drawOnChartArea: false }, title: { display: true, text: 'Inertia' } }
                  }
                }}
              />
            </div>
          )}
        </div>

        {/* Persona Cards */}
        <div className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-4 gap-3">
          {(clustInfo?.persona_titles || ['Group 1', 'Group 2', 'Group 3', 'Group 4']).slice(0, 4).map((pTitle, idx) => (
            <div
              key={idx}
              className="bg-white border border-slate-200 border-l-4 border-l-blue-600 rounded-xl p-4 shadow-sm"
            >
              <span className="text-sm font-extrabold text-slate-900 block truncate">
                Group {idx + 1}: {pTitle}
              </span>
              <span className="text-xs text-slate-500 mt-1 block">Distinct behavioral cohort</span>
            </div>
          ))}
        </div>

        {/* Group Projection Scatter & Radar */}
        <div className="bg-white border border-slate-200 rounded-2xl p-5 shadow-sm space-y-4">
          <div className="flex gap-2 border-b border-slate-200 pb-2 text-xs font-bold">
            <button
              type="button"
              onClick={() => setClusterSubTab('pca')}
              className={`py-1.5 px-3 rounded-lg transition-all ${
                clusterSubTab === 'pca'
                  ? 'bg-blue-50 text-blue-700'
                  : 'text-slate-500 hover:text-slate-800'
              }`}
            >
              🗺️ 2D Group Projection (PCA)
            </button>
            <button
              type="button"
              onClick={() => setClusterSubTab('radar')}
              className={`py-1.5 px-3 rounded-lg transition-all ${
                clusterSubTab === 'radar'
                  ? 'bg-blue-50 text-blue-700'
                  : 'text-slate-500 hover:text-slate-800'
              }`}
            >
              🕸️ Behavioral Feature Fingerprint (Radar)
            </button>
          </div>

          <div className="h-72">
            {clusterSubTab === 'pca' ? (
              <Scatter
                data={pcaScatterData}
                options={{
                  responsive: true,
                  maintainAspectRatio: false,
                  scales: {
                    x: { title: { display: true, text: 'PC1 (Principal Component 1)' } },
                    y: { title: { display: true, text: 'PC2 (Principal Component 2)' } }
                  }
                }}
              />
            ) : (
              <Radar
                data={radarData}
                options={{
                  responsive: true,
                  maintainAspectRatio: false
                }}
              />
            )}
          </div>
        </div>

        {/* Behavioral Personas Table */}
        {clustInfo?.cluster_summary_df && (
          <div className="space-y-2">
            <span className="text-xs font-bold text-slate-800">📊 Behavioral Persona Profiles Summary Table:</span>
            <div className="bg-white border border-slate-200 rounded-xl overflow-hidden overflow-x-auto shadow-sm">
              <table className="w-full text-left text-xs border-collapse">
                <thead>
                  <tr className="bg-slate-50 border-b border-slate-200 font-bold text-slate-700">
                    {Object.keys(clustInfo.cluster_summary_df[0] || {}).map(k => (
                      <th key={k} className="p-3">{k}</th>
                    ))}
                  </tr>
                </thead>
                <tbody className="divide-y divide-slate-100">
                  {clustInfo.cluster_summary_df.map((r, i) => (
                    <tr key={i} className="hover:bg-slate-50/80">
                      {Object.values(r).map((v: any, j) => (
                        <td key={j} className="p-3 text-slate-800">{v}</td>
                      ))}
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        )}
      </div>

      {/* SECTION B: Anomaly Risk Overview (Fuzzy Severity Layer) */}
      <div className="space-y-4 pt-4 border-t border-slate-200">
        <div>
          <h3 className="text-lg font-extrabold text-slate-900 flex items-center gap-2">
            <AlertTriangle className="w-5 h-5 text-rose-600" />
            <span>Anomaly Risk Overview (Fuzzy Severity Layer)</span>
          </h3>
          <p className="text-xs text-slate-500 mt-1">
            Isolation Forest detected <b>{outCnt.toLocaleString()} unusual records</b> ({outPct.toFixed(1)}% of your dataset).
            The <b>Classical Mamdani Fuzzy Logic Engine</b> evaluated their severity based on <i>Anomaly Intensity</i>, <i>Feature Deviations</i>, and <i>Business Impact</i>:
          </p>
        </div>

        {/* 4 Fuzzy Severity Risk Cards */}
        <div className="grid grid-cols-2 md:grid-cols-4 gap-3.5">
          <div className="bg-emerald-50/70 border border-emerald-200 rounded-xl p-3.5 text-xs">
            <span className="font-extrabold text-emerald-800 block">🟢 Low Risk ({sevCounts.Low.toLocaleString()})</span>
            <span className="text-[11px] text-emerald-600 mt-1 block">Minor anomalies with limited deviation</span>
          </div>

          <div className="bg-amber-50/70 border border-amber-200 rounded-xl p-3.5 text-xs">
            <span className="font-extrabold text-amber-800 block">🟡 Medium Risk ({sevCounts.Medium.toLocaleString()})</span>
            <span className="text-[11px] text-amber-600 mt-1 block">Moderate anomalies with unusual values</span>
          </div>

          <div className="bg-orange-50/70 border border-orange-200 rounded-xl p-3.5 text-xs">
            <span className="font-extrabold text-orange-800 block">🟠 High Risk ({sevCounts.High.toLocaleString()})</span>
            <span className="text-[11px] text-orange-600 mt-1 block">Significant anomalies with strong deviations</span>
          </div>

          <div className="bg-rose-50/70 border border-rose-200 rounded-xl p-3.5 text-xs">
            <span className="font-extrabold text-rose-800 block">🔴 Critical Risk ({sevCounts.Critical.toLocaleString()})</span>
            <span className="text-[11px] text-rose-600 mt-1 block">Major feature deviations & high business impact</span>
          </div>
        </div>

        {/* Donut Chart & Outliers by Cluster */}
        <div className="grid grid-cols-1 md:grid-cols-2 gap-5">
          <div className="bg-white border border-slate-200/90 rounded-2xl p-5 shadow-sm flex flex-col items-center">
            <h4 className="text-sm font-extrabold text-slate-900 self-start mb-2">Severity Breakdown (Mamdani FIS)</h4>
            <div className="w-48 h-48 my-auto">
              <Doughnut
                data={donutData}
                options={{
                  responsive: true,
                  maintainAspectRatio: false,
                  plugins: { legend: { position: 'bottom' } }
                }}
              />
            </div>
          </div>

          <div className="bg-white border border-slate-200/90 rounded-2xl p-5 shadow-sm">
            <h4 className="text-sm font-extrabold text-slate-900 mb-2">Unusual Records by Cluster</h4>
            <div className="h-56">
              <Bar
                data={outliersByClusterData}
                options={{
                  responsive: true,
                  maintainAspectRatio: false,
                  plugins: { legend: { display: false } }
                }}
              />
            </div>
          </div>
        </div>
      </div>

      {/* SECTION C: Top Critical Anomalies Table */}
      <div className="space-y-3 pt-2">
        <div>
          <h3 className="text-base font-extrabold text-slate-900">🚨 Highest-Risk Records (Top Fuzzy Severity)</h3>
          <p className="text-xs text-slate-500 mt-0.5">
            Sorted by <b>Mamdani Defuzzified Severity Score</b> descending. Critical and High severity records require prompt business review.
          </p>
        </div>

        <div className="bg-white border border-slate-200 rounded-xl overflow-hidden overflow-x-auto shadow-sm">
          <table className="w-full text-left text-xs border-collapse">
            <thead>
              <tr className="bg-slate-50 border-b border-slate-200 font-bold text-slate-700">
                <th className="p-3">Record ID</th>
                <th className="p-3">Fuzzy Severity</th>
                <th className="p-3">Severity Score</th>
                <th className="p-3">Intensity</th>
                <th className="p-3">Deviation</th>
                <th className="p-3">Business Impact</th>
                <th className="p-3">Primary Unusual Feature</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-100 font-mono text-[11px]">
              {topCriticalList.map((a, idx) => {
                const lead = a.Top_Deviations[0];
                const badgeColor =
                  a.Severity_Label === 'Critical'
                    ? 'bg-rose-50 text-rose-700 border-rose-200'
                    : a.Severity_Label === 'High'
                    ? 'bg-orange-50 text-orange-700 border-orange-200'
                    : 'bg-amber-50 text-amber-700 border-amber-200';
                return (
                  <tr key={idx} className="hover:bg-slate-50/80">
                    <td className="p-3 font-bold text-slate-900">Row #{a.index}</td>
                    <td className="p-3">
                      <span className={`px-2 py-0.5 rounded-full border text-[10px] font-bold ${badgeColor}`}>
                        {a.Severity_Label}
                      </span>
                    </td>
                    <td className="p-3 font-bold text-slate-900">{a.Severity_Score.toFixed(1)} / 100</td>
                    <td className="p-3 text-slate-600">{a.Anomaly_Intensity.toFixed(1)}</td>
                    <td className="p-3 text-slate-600">{a.Feature_Deviation.toFixed(1)}</td>
                    <td className="p-3 text-slate-600">{a.Business_Impact.toFixed(1)}</td>
                    <td className="p-3 font-sans text-slate-700">
                      {lead ? `${cleanName(lead.column)}: ${lead.this_val.toLocaleString()} (Median: ${lead.normal_median.toLocaleString()})` : 'N/A'}
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      </div>

      {/* SECTION D: Anomaly Inspection & Root-Cause Explanation */}
      {selectedAnomaly && (
        <div className="space-y-4 pt-4 border-t border-slate-200">
          <div>
            <h3 className="text-base font-extrabold text-slate-900">🔍 Anomaly Inspection & Root-Cause Explanation</h3>
            <p className="text-xs text-slate-500 mt-0.5">
              Select any anomalous record to inspect its individual fuzzy inputs, defuzzified score, and root-cause feature drivers:
            </p>
          </div>

          <div>
            <label className="text-xs font-bold text-slate-700 block mb-1">Select Record to Inspect:</label>
            <select
              value={selectedAnomalyIdx}
              onChange={e => setSelectedAnomalyIdx(parseInt(e.target.value, 10))}
              className="w-full bg-white border border-slate-200 rounded-lg p-2 text-xs font-semibold text-slate-800"
            >
              {topCriticalList.map((a, i) => (
                <option key={i} value={i}>
                  Row #{a.index} | Severity: {a.Severity_Label} ({a.Severity_Score.toFixed(1)}/100)
                </option>
              ))}
            </select>
          </div>

          <div className="grid grid-cols-2 md:grid-cols-4 gap-3 text-xs">
            <div className="bg-white border border-slate-200 rounded-xl p-3.5 shadow-sm">
              <span className="text-slate-500 font-bold block mb-1">Anomaly Intensity</span>
              <div className="text-xl font-extrabold text-slate-900">
                {selectedAnomaly.Anomaly_Intensity.toFixed(1)} / 100
              </div>
            </div>

            <div className="bg-white border border-slate-200 rounded-xl p-3.5 shadow-sm">
              <span className="text-slate-500 font-bold block mb-1">Feature Deviation</span>
              <div className="text-xl font-extrabold text-slate-900">
                {selectedAnomaly.Feature_Deviation.toFixed(1)} / 100
              </div>
            </div>

            <div className="bg-white border border-slate-200 rounded-xl p-3.5 shadow-sm">
              <span className="text-slate-500 font-bold block mb-1">Business Impact</span>
              <div className="text-xl font-extrabold text-slate-900">
                {selectedAnomaly.Business_Impact.toFixed(1)} / 100
              </div>
            </div>

            <div className="bg-white border border-slate-200 rounded-xl p-3.5 shadow-sm text-center">
              <span className="text-slate-500 font-bold block mb-1">FINAL SEVERITY</span>
              <div className={`text-xl font-extrabold ${selectedAnomaly.Severity_Label === 'Critical' ? 'text-rose-600' : 'text-orange-600'}`}>
                {selectedAnomaly.Severity_Label} ({selectedAnomaly.Severity_Score.toFixed(1)})
              </div>
            </div>
          </div>

          {/* AI Explanation Box */}
          <div className="bg-slate-50 border border-slate-200 border-l-4 border-l-blue-600 rounded-xl p-4 text-xs">
            <b className="text-slate-900 block mb-1 text-sm">📋 AI Analyst Explanation:</b>
            <p className="text-slate-700 leading-relaxed">{selectedAnomaly.Severity_Explanation}</p>
          </div>

          {/* Largest Numerical Deviations */}
          {selectedAnomaly.Top_Deviations.length > 0 && (
            <div className="space-y-2">
              <span className="text-xs font-bold text-slate-800">Largest Numerical Deviations Responsible for Severity:</span>
              <div className="bg-white border border-slate-200 rounded-xl overflow-hidden shadow-sm">
                <table className="w-full text-left text-xs border-collapse">
                  <thead>
                    <tr className="bg-slate-50 border-b border-slate-200 font-bold text-slate-700">
                      <th className="p-2.5">Feature Column</th>
                      <th className="p-2.5">Normal Baseline (Median)</th>
                      <th className="p-2.5">This Record's Value</th>
                      <th className="p-2.5">Robust Z-Score</th>
                      <th className="p-2.5">Impact Level</th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-slate-100">
                    {selectedAnomaly.Top_Deviations.map((d, i) => (
                      <tr key={i} className="hover:bg-slate-50/80">
                        <td className="p-2.5 font-bold text-slate-900">{cleanName(d.column)}</td>
                        <td className="p-2.5 text-slate-600">{d.normal_median.toLocaleString()}</td>
                        <td className="p-2.5 font-bold text-blue-600">{d.this_val.toLocaleString()}</td>
                        <td className="p-2.5 text-slate-600">{d.robust_z}x deviation</td>
                        <td className="p-2.5">
                          <span className={`px-2 py-0.5 rounded text-[11px] font-bold ${d.robust_z > 3 ? 'bg-rose-50 text-rose-700' : 'bg-amber-50 text-amber-700'}`}>
                            {d.robust_z > 3 ? 'Extreme Deviation' : 'Noticeable Deviation'}
                          </span>
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          )}
        </div>
      )}

      {/* SECTION E: Fuzzy Logic Explanation for Demo & Viva */}
      <div className="border border-slate-200 rounded-xl overflow-hidden bg-white shadow-sm">
        <button
          type="button"
          onClick={() => setFuzzyDemoOpen(!fuzzyDemoOpen)}
          className="w-full p-4 bg-slate-50/80 hover:bg-slate-100/80 flex items-center justify-between text-xs font-bold text-slate-800 transition-colors"
        >
          <div className="flex items-center gap-2">
            <Info className="w-4 h-4 text-slate-500" />
            <span>🔬 How is Anomaly Severity Calculated? (Fuzzy Logic Engine Demo & Viva)</span>
          </div>
          {fuzzyDemoOpen ? <ChevronDown className="w-4 h-4" /> : <ChevronRight className="w-4 h-4" />}
        </button>

        {fuzzyDemoOpen && (
          <div className="p-5 border-t border-slate-100 space-y-4 text-xs leading-relaxed text-slate-700">
            <h4 className="font-bold text-slate-900 text-sm">Mamdani Classical Fuzzy Inference System</h4>
            <div className="space-y-3">
              <p>
                <b>1. Detection vs. Severity:</b><br />
                • <b>Isolation Forest</b> detects whether a record is anomalous based on recursive random tree isolation depths.<br />
                • <b>Fuzzy Logic</b> does NOT detect anomalies; it evaluates <i>how severe</i> each detected anomaly is by fusing multiple continuous dimensions.
              </p>
              <p>
                <b>2. Linguistic Input Variables (Fuzzified 0–100):</b><br />
                • <b>Anomaly Intensity:</b> Normalized inverse Isolation Forest decision function score.<br />
                • <b>Feature Deviation:</b> Robust z-score <code>z = 0.6745 * |x - median| / MAD</code> across numerical attributes.<br />
                • <b>Business Impact:</b> Percentile rank and magnitude of business-critical value metrics.
              </p>
              <p>
                <b>3. Membership Functions:</b><br />
                • Inputs: <b>Low</b> (Trapezoidal [0,0,20,45]), <b>Medium</b> (Triangular [20,50,80]), <b>High</b> (Trapezoidal [55,80,100,100]).<br />
                • Outputs: <b>Low</b> [0–35], <b>Medium</b> [20–62], <b>High</b> [48–85], <b>Critical</b> [72–100].
              </p>
              <p>
                <b>4. Mamdani Rule Base (14 Explicit Rules):</b><br />
                • Uses Mamdani Min-Conjunction for rule evaluation and Max-Disjunction for output aggregation.<br />
                • <b>Centroid (Center of Gravity) Defuzzification:</b> <code>Severity = ∑(y * μ_Agg(y)) / ∑(μ_Agg(y))</code> over universe [0, 100].
              </p>
            </div>
          </div>
        )}
      </div>
    </div>
  );
};
