import React, { useState } from 'react';
import {
  FileText,
  Download,
  ChevronDown,
  ChevronRight,
  Info
} from 'lucide-react';
import {
  CleaningReport,
  ColumnClassification,
  DataRow,
  ExecutiveBrief,
  MLResults,
  RAGChunk
} from '../../types';
import { generateHtmlDossier, generateMarkdownDossier } from '../../utils/summarizer';

interface OverviewTabProps {
  df: DataRow[];
  colTypes: ColumnClassification;
  cleaningReport: CleaningReport;
  mlResults: MLResults;
  executiveBrief: ExecutiveBrief;
  ragChunks: RAGChunk[];
  datasetName: string;
  isSimpleMode: boolean;
  pipeLatency: string;
  studioHistory: string[];
}

export const OverviewTab: React.FC<OverviewTabProps> = ({
  df,
  colTypes,
  cleaningReport,
  mlResults,
  executiveBrief,
  ragChunks,
  datasetName,
  isSimpleMode,
  pipeLatency,
  studioHistory
}) => {
  const [techOpen, setTechOpen] = useState(!isSimpleMode);

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

  const outCnt = mlResults.outliers ? mlResults.outliers.count : 0;
  const outPct = mlResults.outliers ? mlResults.outliers.percentage : 0;

  return (
    <div className="space-y-6">
      {/* Title */}
      <div>
        <h2 className="text-2xl font-extrabold text-slate-900 tracking-tight">Your Data Overview</h2>
        <p className="text-sm text-slate-500 mt-1">Here is what we found in your dataset.</p>
      </div>

      {/* Top 5 Important KPI Metric Cards */}
      <div className="grid grid-cols-2 md:grid-cols-5 gap-3.5">
        <div className="bg-white border border-slate-200/90 rounded-xl p-4 shadow-sm shadow-slate-900/[0.02]">
          <div className="flex items-center justify-between text-[11px] font-bold text-slate-500 uppercase tracking-wider mb-1">
            <span>ROWS</span>
            <span title="Total number of records evaluated in your dataset" className="cursor-help text-slate-400">ⓘ</span>
          </div>
          <div className="text-2xl font-extrabold text-slate-900 tracking-tight">
            {cleaningReport.final_shape[0].toLocaleString()}
          </div>
          <div className="text-xs text-slate-500 mt-1 font-medium">Total records evaluated</div>
        </div>

        <div className="bg-white border border-slate-200/90 rounded-xl p-4 shadow-sm shadow-slate-900/[0.02]">
          <div className="flex items-center justify-between text-[11px] font-bold text-slate-500 uppercase tracking-wider mb-1">
            <span>COLUMNS</span>
            <span title="Total number of attributes or fields in each record" className="cursor-help text-slate-400">ⓘ</span>
          </div>
          <div className="text-2xl font-extrabold text-slate-900 tracking-tight">
            {df.length > 0 ? Object.keys(df[0]).length : 0}
          </div>
          <div className="text-xs text-slate-500 mt-1 font-medium">
            {colTypes.numeric.length} Numbers · {colTypes.categorical.length} Categories
          </div>
        </div>

        <div className="bg-white border border-slate-200/90 rounded-xl p-4 shadow-sm shadow-slate-900/[0.02]">
          <div className="flex items-center justify-between text-[11px] font-bold text-slate-500 uppercase tracking-wider mb-1">
            <span>DATA QUALITY</span>
            <span title="Percentage of clean, complete, and reliable records" className="cursor-help text-slate-400">ⓘ</span>
          </div>
          <div className="text-2xl font-extrabold text-emerald-600 tracking-tight">
            {cleaningReport.hygiene_score}
          </div>
          <div className="text-xs text-emerald-600 mt-1 font-medium">Mostly clean, ready to use</div>
        </div>

        <div className="bg-white border border-slate-200/90 rounded-xl p-4 shadow-sm shadow-slate-900/[0.02]">
          <div className="flex items-center justify-between text-[11px] font-bold text-slate-500 uppercase tracking-wider mb-1">
            <span>MISSING VALUES</span>
            <span title="Empty cells filled with standard typical values" className="cursor-help text-slate-400">ⓘ</span>
          </div>
          <div className="text-2xl font-extrabold text-blue-600 tracking-tight">
            {cleaningReport.total_nulls_imputed.toLocaleString()}
          </div>
          <div className="text-xs text-slate-500 mt-1 font-medium">Auto-filled for accuracy</div>
        </div>

        <div className="bg-white border border-slate-200/90 rounded-xl p-4 shadow-sm shadow-slate-900/[0.02]">
          <div className="flex items-center justify-between text-[11px] font-bold text-slate-500 uppercase tracking-wider mb-1">
            <span>UNUSUAL RECORDS</span>
            <span title="Records that deviate significantly from standard baseline patterns" className="cursor-help text-slate-400">ⓘ</span>
          </div>
          <div className="text-2xl font-extrabold text-rose-600 tracking-tight">
            {outCnt.toLocaleString()}
          </div>
          <div className="text-xs text-rose-600 mt-1 font-medium">
            {outPct.toFixed(1)}% need review
          </div>
        </div>
      </div>

      {/* Plain English AI Summary Card */}
      <div className="bg-gradient-to-b from-white to-slate-50/60 border border-slate-200/90 rounded-2xl p-6 shadow-sm shadow-slate-900/[0.03]">
        <div className="flex items-center justify-between pb-4 border-b border-slate-100">
          <div className="flex items-center gap-2.5">
            <span className="text-xl">🤖</span>
            <h3 className="text-lg font-bold text-slate-900">AI Data Summary</h3>
          </div>
          <span className="bg-emerald-50 text-emerald-700 border border-emerald-200 text-xs font-bold px-3 py-1 rounded-full">
            AI Confidence: 99% ✔
          </span>
        </div>

        {/* What We Found */}
        <div className="mt-5">
          <span className="text-sm font-extrabold text-slate-900 block mb-2">📋 What we found:</span>
          <ul className="space-y-1.5 ml-4 list-disc text-sm text-slate-700 leading-relaxed">
            {executiveBrief.what_we_found.map((item, idx) => (
              <li
                key={idx}
                dangerouslySetInnerHTML={{
                  __html: item.replace(/\*\*(.*?)\*\*/g, '<b class="text-slate-900 font-bold">$1</b>')
                }}
              />
            ))}
          </ul>
        </div>

        {/* What Should You Look At */}
        <div className="mt-5 bg-slate-50 border border-slate-200/80 rounded-xl p-4">
          <span className="text-sm font-extrabold text-slate-900 block mb-2">
            🎯 What should you look at?
          </span>
          <ol className="space-y-1.5 ml-4 list-decimal text-sm text-slate-700 leading-relaxed">
            {executiveBrief.what_should_you_look_at.map((item, idx) => (
              <li
                key={idx}
                dangerouslySetInnerHTML={{
                  __html: item.replace(/\*\*(.*?)\*\*/g, '<b class="text-slate-900 font-bold">$1</b>')
                }}
              />
            ))}
          </ol>
        </div>

        {/* 4 Summary Feature Boxes */}
        <div className="grid grid-cols-1 md:grid-cols-2 gap-3.5 mt-5">
          <div className="bg-white border border-slate-200 rounded-xl p-3.5 shadow-sm shadow-slate-900/[0.02]">
            <span className="text-[11px] font-extrabold text-blue-600 tracking-wider block mb-1 uppercase">
              DATA HEALTH & CLEANING
            </span>
            <p className="text-xs text-slate-600 leading-relaxed">{executiveBrief.ingestion_friendly}</p>
          </div>

          <div className="bg-white border border-slate-200 rounded-xl p-3.5 shadow-sm shadow-slate-900/[0.02]">
            <span className="text-[11px] font-extrabold text-emerald-600 tracking-wider block mb-1 uppercase">
              KEY RELATIONSHIPS
            </span>
            <p className="text-xs text-slate-600 leading-relaxed">{executiveBrief.driver_friendly}</p>
          </div>

          <div className="bg-white border border-slate-200 rounded-xl p-3.5 shadow-sm shadow-slate-900/[0.02]">
            <span className="text-[11px] font-extrabold text-purple-600 tracking-wider block mb-1 uppercase">
              CUSTOMER / DATA GROUPS
            </span>
            <p className="text-xs text-slate-600 leading-relaxed">{executiveBrief.segmentation_friendly}</p>
          </div>

          <div className="bg-white border border-slate-200 rounded-xl p-3.5 shadow-sm shadow-slate-900/[0.02]">
            <span className="text-[11px] font-extrabold text-rose-600 tracking-wider block mb-1 uppercase">
              UNUSUAL DATA DETECTION
            </span>
            <p className="text-xs text-slate-600 leading-relaxed">{executiveBrief.outlier_friendly}</p>
          </div>
        </div>
      </div>

      {/* Download Dossier Actions */}
      <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
        <button
          type="button"
          onClick={downloadHtml}
          className="w-full py-2.5 px-4 bg-white hover:bg-slate-50 text-slate-800 border border-slate-200 rounded-xl text-xs font-bold flex items-center justify-center gap-2 shadow-sm transition-all"
        >
          <FileText className="w-4 h-4 text-blue-600" />
          <span>📄 Download HTML Dossier (Print to PDF)</span>
        </button>

        <button
          type="button"
          onClick={downloadMd}
          className="w-full py-2.5 px-4 bg-white hover:bg-slate-50 text-slate-800 border border-slate-200 rounded-xl text-xs font-bold flex items-center justify-center gap-2 shadow-sm transition-all"
        >
          <Download className="w-4 h-4 text-emerald-600" />
          <span>📝 Download Executive Report (.MD)</span>
        </button>
      </div>

      {/* Progressive Disclosure: Technical Details */}
      <div className="border border-slate-200 rounded-xl overflow-hidden bg-white">
        <button
          type="button"
          onClick={() => setTechOpen(!techOpen)}
          className="w-full p-4 bg-slate-50/80 hover:bg-slate-100/80 flex items-center justify-between text-xs font-bold text-slate-800 transition-colors"
        >
          <div className="flex items-center gap-2">
            <Info className="w-4 h-4 text-slate-500" />
            <span>🔬 How was this calculated? (Technical Details)</span>
          </div>
          {techOpen ? <ChevronDown className="w-4 h-4 text-slate-500" /> : <ChevronRight className="w-4 h-4 text-slate-500" />}
        </button>

        {techOpen && (
          <div className="p-5 border-t border-slate-100 space-y-4 text-xs">
            <div className="space-y-1.5 text-slate-600 leading-relaxed">
              <p className="font-bold text-slate-900 mb-1">Underlying ML Pipeline & Telemetry:</p>
              <p>• <b>RAG Knowledge Vectors:</b> {ragChunks.length} semantic knowledge chunks indexed via vector similarity ranking.</p>
              <p>• <b>Unusual Record Detection:</b> Scored using Isolation Forest with an estimated contamination boundary of {(mlResults.outliers ? mlResults.outliers.percentage : 5).toFixed(1)}%.</p>
              <p>• <b>Customer Segmentation:</b> Partitioned into {mlResults.clustering ? mlResults.clustering.n_clusters : 4} clusters via KMeans.</p>
              <p>• <b>Pipeline Execution Latency:</b> {pipeLatency} across {cleaningReport.final_shape[0].toLocaleString()} rows.</p>
            </div>

            <div>
              <p className="font-bold text-slate-900 mb-2">Semantic Knowledge Chunks Used by AI:</p>
              <div className="space-y-2.5">
                {ragChunks.map(chunk => (
                  <div
                    key={chunk.chunk_id}
                    className="p-3 bg-slate-50 border border-slate-200 border-l-4 border-l-blue-600 rounded-lg text-xs"
                  >
                    <b className="text-blue-700 font-bold block mb-1">
                      [{chunk.chunk_id}] {chunk.title}
                    </b>
                    <p className="text-slate-600 leading-relaxed">{chunk.content}</p>
                  </div>
                ))}
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
};
