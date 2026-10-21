import React from 'react';
import { FileText, Download } from 'lucide-react';
import {
  CleaningReport,
  ColumnClassification,
  DataRow,
  ExecutiveBrief,
  MLResults
} from '../../types';
import { cleanName } from '../../utils/cleanName';
import { generateHtmlDossier, generateMarkdownDossier } from '../../utils/summarizer';

interface InsightsTabProps {
  df: DataRow[];
  colTypes: ColumnClassification;
  cleaningReport: CleaningReport;
  mlResults: MLResults;
  executiveBrief: ExecutiveBrief;
  datasetName: string;
  studioHistory: string[];
}

export const InsightsTab: React.FC<InsightsTabProps> = ({
  df,
  colTypes,
  cleaningReport,
  mlResults,
  executiveBrief,
  datasetName,
  studioHistory
}) => {
  const topPairs = mlResults.correlation ? mlResults.correlation.top_pairs : [];

  const downloadHtml = () => {
    const html = generateHtmlDossier(
      df,
      colTypes,
      cleaningReport,
      mlResults,
      executiveBrief,
      datasetName,
      studioHistory
    );
    const blob = new Blob([html], { type: 'text/html' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `Executive_Dossier_${datasetName}.html`;
    a.click();
    URL.revokeObjectURL(url);
  };

  const downloadMd = () => {
    const md = generateMarkdownDossier(
      df,
      colTypes,
      cleaningReport,
      mlResults,
      executiveBrief,
      datasetName,
      studioHistory
    );
    const blob = new Blob([md], { type: 'text/markdown' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `Executive_Report_${datasetName}.md`;
    a.click();
    URL.revokeObjectURL(url);
  };

  return (
    <div className="space-y-6">
      {/* Title */}
      <div>
        <h2 className="text-2xl font-extrabold text-slate-900 tracking-tight">Key Insights & Actionable Takeaways</h2>
        <p className="text-sm text-slate-500 mt-1">
          Every insight tells you <b>What happened</b>, <b>Why it matters</b>, and <b>What you can do</b>.
        </p>
      </div>

      {/* Actionable Insight Cards Framework */}
      <div className="space-y-4">
        {executiveBrief.actionable_insights.map((item, idx) => (
          <div
            key={idx}
            className="bg-white border border-slate-200/90 border-l-4 border-l-sky-600 rounded-xl p-5 shadow-sm shadow-slate-900/[0.02] space-y-2.5"
          >
            <div className="flex items-center gap-2.5">
              <span className="text-xl">{item.icon}</span>
              <h4 className="text-sm font-extrabold text-slate-900">{item.what}</h4>
            </div>

            <div className="bg-slate-50 border border-slate-100 rounded-lg p-3 text-xs text-slate-700 leading-relaxed">
              <b className="text-slate-900">Why it matters:</b> {item.why}
            </div>

            <div className="flex items-center gap-2 text-xs font-bold text-sky-700">
              <span>🎯 <b>What can you do:</b> {item.action}</span>
            </div>

            <div className="text-[11px] text-slate-400 font-medium">
              {item.technical_label}: <code className="text-sky-600 font-bold">{item.technical_val}</code>
            </div>
          </div>
        ))}
      </div>

      {/* Top Relationships Breakdown */}
      {topPairs.length > 0 && (
        <div className="space-y-3 pt-2">
          <div>
            <h3 className="text-base font-extrabold text-slate-900">🔗 Key Relationships Between Columns</h3>
            <p className="text-xs text-slate-500 mt-0.5">When one of these numbers moves, the other tends to follow:</p>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-3 gap-3.5">
            {topPairs.slice(0, 3).map((p, idx) => {
              const isPositive = p.corr > 0;
              return (
                <div
                  key={idx}
                  className="bg-white border border-slate-200/90 rounded-xl p-4 shadow-sm shadow-slate-900/[0.02]"
                >
                  <div className={`text-xs font-bold ${isPositive ? 'text-sky-600' : 'text-rose-600'}`}>
                    {isPositive ? 'Move Together ↗' : 'Move Oppositely ↘'}
                  </div>
                  <div className="text-sm font-extrabold text-slate-900 my-1.5 truncate">
                    {cleanName(p.col1)} ↔ {cleanName(p.col2)}
                  </div>
                  <div className="text-xs text-slate-500">
                    Relationship strength: <b className="text-slate-800">{p.corr >= 0 ? '+' : ''}{p.corr.toFixed(2)}</b>
                  </div>
                </div>
              );
            })}
          </div>
        </div>
      )}

      {/* Export Executive Intelligence Dossier */}
      <div className="bg-white border border-slate-200/90 border-l-4 border-l-blue-600 rounded-xl p-5 shadow-sm space-y-3">
        <h4 className="text-base font-extrabold text-slate-900">📥 Export Executive Intelligence Dossier</h4>
        <p className="text-xs text-slate-600 leading-relaxed">
          Download a comprehensive executive report containing automated cleaning DAG logs, behavioral cluster personas,
          Mamdani fuzzy anomaly telemetry, and actionable strategic takeaways.
        </p>

        <div className="grid grid-cols-1 sm:grid-cols-2 gap-3 pt-1">
          <button
            type="button"
            onClick={downloadHtml}
            className="py-2 px-4 bg-blue-50 hover:bg-blue-100 text-blue-700 border border-blue-200 rounded-xl text-xs font-bold flex items-center justify-center gap-2 transition-all cursor-pointer"
          >
            <FileText className="w-4 h-4 text-blue-600" />
            <span>📄 Download Standalone HTML Dossier</span>
          </button>

          <button
            type="button"
            onClick={downloadMd}
            className="py-2 px-4 bg-emerald-50 hover:bg-emerald-100 text-emerald-700 border border-emerald-200 rounded-xl text-xs font-bold flex items-center justify-center gap-2 transition-all cursor-pointer"
          >
            <Download className="w-4 h-4 text-emerald-600" />
            <span>📝 Download Markdown Report (.MD)</span>
          </button>
        </div>
      </div>
    </div>
  );
};
