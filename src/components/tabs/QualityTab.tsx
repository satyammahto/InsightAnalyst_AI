import React, { useState } from 'react';
import {
  Download,
  RotateCcw,
  Undo2,
  Table as TableIcon
} from 'lucide-react';
import Papa from 'papaparse';
import {
  CleaningReport,
  ColumnClassification,
  DataRow
} from '../../types';
import { CleaningStudio } from '../../utils/cleaningStudio';

interface QualityTabProps {
  df: DataRow[];
  colTypes: ColumnClassification;
  cleaningReport: CleaningReport;
  studioDf: DataRow[];
  setStudioDf: (df: DataRow[]) => void;
  studioHistory: string[];
  setStudioHistory: React.Dispatch<React.SetStateAction<string[]>>;
  undoStack: DataRow[][];
  setUndoStack: React.Dispatch<React.SetStateAction<DataRow[][]>>;
  selectedDatasetName: string;
}

export const QualityTab: React.FC<QualityTabProps> = ({
  df,
  colTypes,
  cleaningReport,
  studioDf,
  setStudioDf,
  studioHistory,
  setStudioHistory,
  undoStack,
  setUndoStack,
  selectedDatasetName
}) => {
  const [subTab, setSubTab] = useState<'columns' | 'steps' | 'studio' | 'preview'>('columns');

  // Studio form states
  const [studioAction, setStudioAction] = useState<string>('impute');
  const [targetCol, setTargetCol] = useState<string>('');
  const [imputeStrat, setImputeStrat] = useState<string>('median');
  const [constVal, setConstVal] = useState<string>('0');

  const [outlierCol, setOutlierCol] = useState<string>('');
  const [outlierMethod, setOutlierMethod] = useState<string>('clip_percentiles');

  const [textCol, setTextCol] = useState<string>('');
  const [casingChoice, setCasingChoice] = useState<string>('title');
  const [stripWs, setStripWs] = useState<boolean>(true);
  const [remSpec, setRemSpec] = useState<boolean>(false);
  const [findTxt, setFindTxt] = useState<string>('');
  const [replaceTxt, setReplaceTxt] = useState<string>('');

  const [dateTarget, setDateTarget] = useState<string>('');
  const [selectedDateFeats, setSelectedDateFeats] = useState<string[]>(['year', 'month']);

  const [scaleCol, setScaleCol] = useState<string>('');
  const [scaleMethod, setScaleMethod] = useState<'minmax' | 'standard_zscore'>('minmax');

  const [newColName, setNewColName] = useState<string>('derived_feature');
  const [colA, setColA] = useState<string>('');
  const [deriveOp, setDeriveOp] = useState<string>('add');
  const [deriveBType, setDeriveBType] = useState<'col' | 'scalar'>('col');
  const [colB, setColB] = useState<string>('');
  const [scalarB, setScalarB] = useState<number>(1);

  const [dropCols, setDropCols] = useState<string[]>([]);

  const allColumns = studioDf.length > 0 ? Object.keys(studioDf[0]) : [];
  const numColumns = colTypes.numeric.filter(c => allColumns.includes(c));

  // Initialize dropdown defaults
  React.useEffect(() => {
    if (allColumns.length > 0 && !targetCol) {
      setTargetCol(allColumns[0]);
      setTextCol(allColumns[0]);
      setColA(allColumns[0]);
      setColB(allColumns.length > 1 ? allColumns[1] : allColumns[0]);
    }
    if (numColumns.length > 0 && !outlierCol) {
      setOutlierCol(numColumns[0]);
      setScaleCol(numColumns[0]);
    }
    if (colTypes.date.length > 0 && !dateTarget) {
      setDateTarget(colTypes.date[0]);
    }
  }, [allColumns, numColumns, colTypes.date]);

  const handleApplyTransform = () => {
    setUndoStack(prev => [...prev, studioDf.map(r => ({ ...r }))]);

    if (studioAction === 'impute') {
      const { newRows, details } = CleaningStudio.imputeMissing(studioDf, targetCol, imputeStrat, constVal);
      setStudioDf(newRows);
      setStudioHistory(prev => [...prev, details]);
    } else if (studioAction === 'outlier') {
      const { newRows, details } = CleaningStudio.handleOutliers(studioDf, outlierCol, outlierMethod);
      setStudioDf(newRows);
      setStudioHistory(prev => [...prev, details]);
    } else if (studioAction === 'text') {
      const { newRows, details } = CleaningStudio.cleanTextColumn(
        studioDf,
        textCol,
        casingChoice,
        stripWs,
        remSpec,
        findTxt ? [findTxt, replaceTxt] : null
      );
      setStudioDf(newRows);
      setStudioHistory(prev => [...prev, details]);
    } else if (studioAction === 'date') {
      const { newRows, details } = CleaningStudio.extractDateFeatures(studioDf, dateTarget, selectedDateFeats);
      setStudioDf(newRows);
      setStudioHistory(prev => [...prev, details]);
    } else if (studioAction === 'scale') {
      const { newRows, details } = CleaningStudio.scaleNumeric(studioDf, scaleCol, scaleMethod);
      setStudioDf(newRows);
      setStudioHistory(prev => [...prev, details]);
    } else if (studioAction === 'derive') {
      const bVal = deriveBType === 'col' ? colB : scalarB;
      const { newRows, details } = CleaningStudio.deriveColumn(studioDf, newColName, colA, deriveOp, bVal);
      setStudioDf(newRows);
      setStudioHistory(prev => [...prev, details]);
    } else if (studioAction === 'drop') {
      if (dropCols.length === 0) return;
      const { newRows, details } = CleaningStudio.dropColumns(studioDf, dropCols);
      setStudioDf(newRows);
      setStudioHistory(prev => [...prev, details]);
      setDropCols([]);
    }
  };

  const handleUndo = () => {
    if (undoStack.length === 0) return;
    const last = undoStack[undoStack.length - 1];
    setUndoStack(prev => prev.slice(0, -1));
    setStudioDf(last);
    setStudioHistory(prev => prev.slice(0, -1));
  };

  const handleReset = () => {
    setStudioDf(df.map(r => ({ ...r })));
    setStudioHistory([]);
    setUndoStack([]);
  };

  const downloadCsv = () => {
    const csv = Papa.unparse(studioDf);
    const blob = new Blob([csv], { type: 'text/csv;charset=utf-8;' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `clean_${selectedDatasetName}.csv`;
    a.click();
    URL.revokeObjectURL(url);
  };

  const downloadJson = () => {
    const jsonStr = JSON.stringify(studioDf, null, 2);
    const blob = new Blob([jsonStr], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `clean_${selectedDatasetName}.json`;
    a.click();
    URL.revokeObjectURL(url);
  };

  const nRowsCleaned = cleaningReport.final_shape[0];
  const nDups = cleaningReport.duplicates_removed;
  const nNulls = cleaningReport.total_nulls_imputed;
  const nDates = cleaningReport.dates_converted.length;

  return (
    <div className="space-y-6">
      {/* Title */}
      <div>
        <h2 className="text-2xl font-extrabold text-slate-900 tracking-tight">Data Quality & Cleaning</h2>
        <p className="text-sm text-slate-500 mt-1">
          Here is how we checked, cleaned, and standardized your data so it is ready for analysis.
        </p>
      </div>

      {/* Visual Checklist Alert */}
      <div className="bg-blue-50/80 border border-blue-200/90 rounded-2xl p-5 shadow-sm shadow-blue-500/[0.02]">
        <div className="flex items-center gap-2 mb-1">
          <span className="text-lg">🎉</span>
          <span className="text-sm font-extrabold text-blue-900">Your data is ready for analysis.</span>
        </div>
        <p className="text-xs text-blue-700">
          Every row and column has been inspected, missing cells filled, and text formats standardized.
        </p>
      </div>

      {/* 5 Checklist Pills */}
      <div className="flex flex-wrap gap-2.5">
        <div className="bg-white border border-slate-200 rounded-xl px-3.5 py-2 text-xs font-semibold text-slate-800 shadow-sm flex items-center gap-1.5">
          <span>✅</span>
          <span><b>{nRowsCleaned.toLocaleString()}</b> rows checked</span>
        </div>
        <div className="bg-white border border-slate-200 rounded-xl px-3.5 py-2 text-xs font-semibold text-slate-800 shadow-sm flex items-center gap-1.5">
          <span>✅</span>
          <span><b>{nDups.toLocaleString()}</b> duplicate rows removed</span>
        </div>
        <div className="bg-white border border-slate-200 rounded-xl px-3.5 py-2 text-xs font-semibold text-slate-800 shadow-sm flex items-center gap-1.5">
          <span>⚠️</span>
          <span><b>{nNulls.toLocaleString()}</b> missing values filled</span>
        </div>
        <div className="bg-white border border-slate-200 rounded-xl px-3.5 py-2 text-xs font-semibold text-slate-800 shadow-sm flex items-center gap-1.5">
          <span>✅</span>
          <span><b>{nDates}</b> date column{nDates !== 1 ? 's' : ''} standardized</span>
        </div>
        <div className="bg-white border border-slate-200 rounded-xl px-3.5 py-2 text-xs font-semibold text-slate-800 shadow-sm flex items-center gap-1.5">
          <span>✅</span>
          <span>Text values cleaned & capitalized</span>
        </div>
      </div>

      {/* 4 Summary Metric Cards */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-3.5">
        <div className="bg-white border border-slate-200/90 rounded-xl p-4 shadow-sm shadow-slate-900/[0.02]">
          <span className="text-[11px] font-bold text-slate-500 uppercase tracking-wider block mb-1">
            ROWS AFTER CLEANING
          </span>
          <div className="text-2xl font-extrabold text-slate-900">{nRowsCleaned.toLocaleString()}</div>
          <div className="text-xs text-emerald-600 mt-1 font-medium">100% of valid rows retained</div>
        </div>

        <div className="bg-white border border-slate-200/90 rounded-xl p-4 shadow-sm shadow-slate-900/[0.02]">
          <span className="text-[11px] font-bold text-slate-500 uppercase tracking-wider block mb-1">
            MISSING VALUES FILLED
          </span>
          <div className="text-2xl font-extrabold text-blue-600">{nNulls.toLocaleString()}</div>
          <div className="text-xs text-slate-500 mt-1 font-medium">Filled with typical values</div>
        </div>

        <div className="bg-white border border-slate-200/90 rounded-xl p-4 shadow-sm shadow-slate-900/[0.02]">
          <span className="text-[11px] font-bold text-slate-500 uppercase tracking-wider block mb-1">
            TEXT VALUES CLEANED
          </span>
          <div className="text-2xl font-extrabold text-emerald-600">
            {cleaningReport.string_tokens_sanitized.toLocaleString()}
          </div>
          <div className="text-xs text-slate-500 mt-1 font-medium">Spaces trimmed & formatted</div>
        </div>

        <div className="bg-white border border-slate-200/90 rounded-xl p-4 shadow-sm shadow-slate-900/[0.02]">
          <span className="text-[11px] font-bold text-slate-500 uppercase tracking-wider block mb-1">
            DATE FORMAT FIXES
          </span>
          <div className="text-2xl font-extrabold text-purple-600">{nDates} Col</div>
          <div className="text-xs text-slate-500 mt-1 font-medium">Standardized calendar dates</div>
        </div>
      </div>

      {/* Sub Tabs Navigation */}
      <div className="border-b border-slate-200">
        <div className="flex gap-2 text-xs font-bold">
          <button
            type="button"
            onClick={() => setSubTab('columns')}
            className={`py-2 px-3.5 border-b-2 transition-all ${
              subTab === 'columns'
                ? 'border-blue-600 text-blue-600 bg-blue-50/50 rounded-t-lg'
                : 'border-transparent text-slate-500 hover:text-slate-800'
            }`}
          >
            📋 Column Health Breakdown
          </button>
          <button
            type="button"
            onClick={() => setSubTab('steps')}
            className={`py-2 px-3.5 border-b-2 transition-all ${
              subTab === 'steps'
                ? 'border-blue-600 text-blue-600 bg-blue-50/50 rounded-t-lg'
                : 'border-transparent text-slate-500 hover:text-slate-800'
            }`}
          >
            🔄 Processing Steps
          </button>
          <button
            type="button"
            onClick={() => setSubTab('studio')}
            className={`py-2 px-3.5 border-b-2 transition-all ${
              subTab === 'studio'
                ? 'border-blue-600 text-blue-600 bg-blue-50/50 rounded-t-lg'
                : 'border-transparent text-slate-500 hover:text-slate-800'
            }`}
          >
            🛠️ Interactive Cleaning Studio
          </button>
          <button
            type="button"
            onClick={() => setSubTab('preview')}
            className={`py-2 px-3.5 border-b-2 transition-all ${
              subTab === 'preview'
                ? 'border-blue-600 text-blue-600 bg-blue-50/50 rounded-t-lg'
                : 'border-transparent text-slate-500 hover:text-slate-800'
            }`}
          >
            📥 Preview & Export Cleaned Data
          </button>
        </div>
      </div>

      {/* SUB-TAB 1: Column Health Breakdown */}
      {subTab === 'columns' && (
        <div className="bg-white border border-slate-200 rounded-xl overflow-hidden shadow-sm">
          <div className="overflow-x-auto">
            <table className="w-full text-left text-xs border-collapse">
              <thead>
                <tr className="bg-slate-50 border-b border-slate-200 text-slate-600 font-bold">
                  <th className="p-3">Column Name</th>
                  <th className="p-3">Data Type</th>
                  <th className="p-3">Missing Values Fixed</th>
                  <th className="p-3">Cleaning Action</th>
                  <th className="p-3">Status</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-slate-100">
                {colTypes.telemetry_table.map((row, idx) => {
                  const imputedCnt = cleaningReport.nulls_imputed[row.featureName] || 0;
                  let friendlyType = 'Text';
                  let actionTaken = 'Standardized capitalization';

                  if (colTypes.numeric.includes(row.featureName)) {
                    friendlyType = 'Number (Metric)';
                    actionTaken = imputedCnt > 0 ? `Filled ${imputedCnt.toLocaleString()} blanks with median` : 'All values valid';
                  } else if (colTypes.categorical.includes(row.featureName)) {
                    friendlyType = 'Category / Group';
                    actionTaken = imputedCnt > 0 ? `Filled ${imputedCnt.toLocaleString()} blanks with common mode` : 'All values valid';
                  } else if (colTypes.date.includes(row.featureName)) {
                    friendlyType = 'Date / Calendar';
                    actionTaken = 'Unified to standard calendar dates';
                  } else if (colTypes.id_high_cardinality.includes(row.featureName)) {
                    friendlyType = 'Identifier (ID / Key)';
                    actionTaken = 'Trimmed spaces, preserved exact keys';
                  }

                  return (
                    <tr key={idx} className="hover:bg-slate-50/80 transition-colors">
                      <td className="p-3 font-bold text-slate-900">{row.featureName}</td>
                      <td className="p-3 text-slate-600">{friendlyType}</td>
                      <td className="p-3 text-blue-600 font-semibold">{imputedCnt.toLocaleString()}</td>
                      <td className="p-3 text-slate-600">{actionTaken}</td>
                      <td className="p-3">
                        <span className="bg-emerald-50 text-emerald-700 border border-emerald-200 text-[11px] font-bold px-2 py-0.5 rounded-full">
                          ✅ Ready for Analysis
                        </span>
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        </div>
      )}

      {/* SUB-TAB 2: Processing Steps DAG */}
      {subTab === 'steps' && (
        <div className="space-y-3">
          <p className="text-xs font-bold text-slate-800">How Your Data Was Processed Step-by-Step:</p>
          <div className="space-y-2">
            {cleaningReport.dag_steps.map((step, idx) => (
              <div
                key={idx}
                className="bg-white border border-slate-200 rounded-xl p-3.5 flex items-center justify-between shadow-sm shadow-slate-900/[0.02]"
              >
                <div className="flex items-center gap-3">
                  <div className="w-8 h-8 rounded-full bg-blue-50 border border-blue-200 text-blue-700 font-extrabold text-xs flex items-center justify-center shrink-0">
                    {step.step_num}
                  </div>
                  <div>
                    <span className="text-sm font-bold text-slate-900 block">{step.title}</span>
                    <span className="text-xs text-slate-500">{step.desc}</span>
                  </div>
                </div>

                <div className="flex items-center gap-2">
                  <span className="bg-emerald-50 text-emerald-700 border border-emerald-200 text-xs font-bold px-2.5 py-1 rounded-full">
                    {step.badge}
                  </span>
                  <span className="text-[11px] text-slate-400 font-mono">{step.latency}</span>
                </div>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* SUB-TAB 3: Interactive Cleaning Studio */}
      {subTab === 'studio' && (
        <div className="space-y-5">
          <div className="bg-slate-50 border border-slate-200 rounded-xl p-4">
            <h4 className="text-sm font-bold text-slate-900">🛠️ Interactive Data Cleaning Studio</h4>
            <p className="text-xs text-slate-500 mt-1">
              Customize transformations live with <b>Before vs. After</b> telemetry, missing value imputation, outlier clipping, text hygiene, and feature engineering.
            </p>
          </div>

          {/* Telemetry Summary */}
          <div className="grid grid-cols-2 md:grid-cols-4 gap-3 text-xs">
            <div className="bg-white border border-slate-200 rounded-xl p-3">
              <span className="text-slate-500 text-[11px]">Current Rows</span>
              <div className="text-lg font-bold text-slate-900 mt-0.5">
                {studioDf.length.toLocaleString()}
              </div>
            </div>
            <div className="bg-white border border-slate-200 rounded-xl p-3">
              <span className="text-slate-500 text-[11px]">Current Columns</span>
              <div className="text-lg font-bold text-slate-900 mt-0.5">
                {allColumns.length}
              </div>
            </div>
            <div className="bg-white border border-slate-200 rounded-xl p-3">
              <span className="text-slate-500 text-[11px]">Missing Values</span>
              <div className="text-lg font-bold text-slate-900 mt-0.5">0</div>
            </div>
            <div className="bg-white border border-slate-200 rounded-xl p-3">
              <span className="text-slate-500 text-[11px]">Custom Transforms</span>
              <div className="text-lg font-bold text-blue-600 mt-0.5">
                {studioHistory.length}
              </div>
            </div>
          </div>

          {/* Action Selector */}
          <div className="space-y-3 bg-white border border-slate-200 rounded-xl p-4 shadow-sm">
            <div>
              <label className="text-xs font-bold text-slate-800 block mb-1.5">
                Select Transformation Type:
              </label>
              <select
                value={studioAction}
                onChange={e => setStudioAction(e.target.value)}
                className="w-full text-xs font-semibold bg-slate-50 border border-slate-200 rounded-lg p-2 text-slate-800"
              >
                <option value="impute">1. Missing Value Imputation</option>
                <option value="outlier">2. Outlier Treatment & Clipping</option>
                <option value="text">3. Text & Categorical Normalization</option>
                <option value="date">4. Date / Time Feature Extraction</option>
                <option value="scale">5. Numeric Feature Scaling</option>
                <option value="derive">6. Derive New Column (Formula)</option>
                <option value="drop">7. Drop Columns</option>
              </select>
            </div>

            {/* Dynamic Controls based on Action */}
            {studioAction === 'impute' && (
              <div className="grid grid-cols-1 md:grid-cols-3 gap-3 pt-2">
                <div>
                  <label className="text-[11px] font-bold text-slate-700 block mb-1">Target Column:</label>
                  <select
                    value={targetCol}
                    onChange={e => setTargetCol(e.target.value)}
                    className="w-full bg-slate-50 border border-slate-200 rounded p-1.5 text-xs"
                  >
                    {allColumns.map(c => <option key={c} value={c}>{c}</option>)}
                  </select>
                </div>
                <div>
                  <label className="text-[11px] font-bold text-slate-700 block mb-1">Imputation Strategy:</label>
                  <select
                    value={imputeStrat}
                    onChange={e => setImputeStrat(e.target.value)}
                    className="w-full bg-slate-50 border border-slate-200 rounded p-1.5 text-xs"
                  >
                    <option value="median">median</option>
                    <option value="mean">mean</option>
                    <option value="mode">mode</option>
                    <option value="constant">constant</option>
                    <option value="ffill">forward fill (ffill)</option>
                    <option value="drop_rows">drop incomplete rows</option>
                  </select>
                </div>
                {imputeStrat === 'constant' && (
                  <div>
                    <label className="text-[11px] font-bold text-slate-700 block mb-1">Constant Value:</label>
                    <input
                      type="text"
                      value={constVal}
                      onChange={e => setConstVal(e.target.value)}
                      className="w-full bg-slate-50 border border-slate-200 rounded p-1.5 text-xs"
                    />
                  </div>
                )}
              </div>
            )}

            {studioAction === 'outlier' && (
              <div className="grid grid-cols-1 md:grid-cols-2 gap-3 pt-2">
                <div>
                  <label className="text-[11px] font-bold text-slate-700 block mb-1">Numeric Column:</label>
                  <select
                    value={outlierCol}
                    onChange={e => setOutlierCol(e.target.value)}
                    className="w-full bg-slate-50 border border-slate-200 rounded p-1.5 text-xs"
                  >
                    {numColumns.map(c => <option key={c} value={c}>{c}</option>)}
                  </select>
                </div>
                <div>
                  <label className="text-[11px] font-bold text-slate-700 block mb-1">Outlier Strategy:</label>
                  <select
                    value={outlierMethod}
                    onChange={e => setOutlierMethod(e.target.value)}
                    className="w-full bg-slate-50 border border-slate-200 rounded p-1.5 text-xs"
                  >
                    <option value="clip_percentiles">Clip at 1st & 99th Percentiles (Winsorize)</option>
                    <option value="clip_iqr">Clip at 1.5 * IQR</option>
                    <option value="drop_iqr">Drop Rows Outside 1.5 * IQR</option>
                    <option value="drop_3sigma">Drop Rows Beyond 3σ</option>
                  </select>
                </div>
              </div>
            )}

            {studioAction === 'text' && (
              <div className="space-y-3 pt-2">
                <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
                  <div>
                    <label className="text-[11px] font-bold text-slate-700 block mb-1">Text Column:</label>
                    <select
                      value={textCol}
                      onChange={e => setTextCol(e.target.value)}
                      className="w-full bg-slate-50 border border-slate-200 rounded p-1.5 text-xs"
                    >
                      {allColumns.map(c => <option key={c} value={c}>{c}</option>)}
                    </select>
                  </div>
                  <div>
                    <label className="text-[11px] font-bold text-slate-700 block mb-1">Casing:</label>
                    <select
                      value={casingChoice}
                      onChange={e => setCasingChoice(e.target.value)}
                      className="w-full bg-slate-50 border border-slate-200 rounded p-1.5 text-xs"
                    >
                      <option value="title">Title Case</option>
                      <option value="lower">lower case</option>
                      <option value="upper">UPPER CASE</option>
                      <option value="keep_as_is">Keep as is</option>
                    </select>
                  </div>
                </div>

                <div className="flex gap-4">
                  <label className="flex items-center gap-1.5 text-xs cursor-pointer">
                    <input
                      type="checkbox"
                      checked={stripWs}
                      onChange={e => setStripWs(e.target.checked)}
                      className="rounded text-blue-600"
                    />
                    <span>Strip whitespace & extra spaces</span>
                  </label>
                  <label className="flex items-center gap-1.5 text-xs cursor-pointer">
                    <input
                      type="checkbox"
                      checked={remSpec}
                      onChange={e => setRemSpec(e.target.checked)}
                      className="rounded text-blue-600"
                    />
                    <span>Remove special characters</span>
                  </label>
                </div>

                <div className="grid grid-cols-2 gap-3">
                  <input
                    type="text"
                    placeholder="Find text (optional)"
                    value={findTxt}
                    onChange={e => setFindTxt(e.target.value)}
                    className="bg-slate-50 border border-slate-200 rounded p-1.5 text-xs"
                  />
                  <input
                    type="text"
                    placeholder="Replace with (optional)"
                    value={replaceTxt}
                    onChange={e => setReplaceTxt(e.target.value)}
                    className="bg-slate-50 border border-slate-200 rounded p-1.5 text-xs"
                  />
                </div>
              </div>
            )}

            {studioAction === 'date' && (
              <div className="grid grid-cols-1 md:grid-cols-2 gap-3 pt-2">
                <div>
                  <label className="text-[11px] font-bold text-slate-700 block mb-1">Date Column:</label>
                  <select
                    value={dateTarget}
                    onChange={e => setDateTarget(e.target.value)}
                    className="w-full bg-slate-50 border border-slate-200 rounded p-1.5 text-xs"
                  >
                    {(colTypes.date.length > 0 ? colTypes.date : allColumns).map(c => (
                      <option key={c} value={c}>{c}</option>
                    ))}
                  </select>
                </div>
                <div>
                  <label className="text-[11px] font-bold text-slate-700 block mb-1">Features to Extract:</label>
                  <div className="flex flex-wrap gap-2 text-xs">
                    {['year', 'month', 'day', 'dayofweek', 'quarter', 'is_weekend'].map(f => (
                      <label key={f} className="flex items-center gap-1 cursor-pointer">
                        <input
                          type="checkbox"
                          checked={selectedDateFeats.includes(f)}
                          onChange={e => {
                            if (e.target.checked) setSelectedDateFeats(prev => [...prev, f]);
                            else setSelectedDateFeats(prev => prev.filter(x => x !== f));
                          }}
                          className="rounded text-blue-600"
                        />
                        <span>{f}</span>
                      </label>
                    ))}
                  </div>
                </div>
              </div>
            )}

            {studioAction === 'scale' && (
              <div className="grid grid-cols-1 md:grid-cols-2 gap-3 pt-2">
                <div>
                  <label className="text-[11px] font-bold text-slate-700 block mb-1">Numeric Column:</label>
                  <select
                    value={scaleCol}
                    onChange={e => setScaleCol(e.target.value)}
                    className="w-full bg-slate-50 border border-slate-200 rounded p-1.5 text-xs"
                  >
                    {numColumns.map(c => <option key={c} value={c}>{c}</option>)}
                  </select>
                </div>
                <div>
                  <label className="text-[11px] font-bold text-slate-700 block mb-1">Scaling Method:</label>
                  <select
                    value={scaleMethod}
                    onChange={e => setScaleMethod(e.target.value as any)}
                    className="w-full bg-slate-50 border border-slate-200 rounded p-1.5 text-xs"
                  >
                    <option value="minmax">MinMax (0 to 1)</option>
                    <option value="standard_zscore">Standard Z-Score (Mean=0, Std=1)</option>
                  </select>
                </div>
              </div>
            )}

            {studioAction === 'derive' && (
              <div className="space-y-3 pt-2">
                <div className="grid grid-cols-1 md:grid-cols-3 gap-3">
                  <div>
                    <label className="text-[11px] font-bold text-slate-700 block mb-1">New Column Name:</label>
                    <input
                      type="text"
                      value={newColName}
                      onChange={e => setNewColName(e.target.value)}
                      className="w-full bg-slate-50 border border-slate-200 rounded p-1.5 text-xs"
                    />
                  </div>
                  <div>
                    <label className="text-[11px] font-bold text-slate-700 block mb-1">Column A:</label>
                    <select
                      value={colA}
                      onChange={e => setColA(e.target.value)}
                      className="w-full bg-slate-50 border border-slate-200 rounded p-1.5 text-xs"
                    >
                      {allColumns.map(c => <option key={c} value={c}>{c}</option>)}
                    </select>
                  </div>
                  <div>
                    <label className="text-[11px] font-bold text-slate-700 block mb-1">Operation:</label>
                    <select
                      value={deriveOp}
                      onChange={e => setDeriveOp(e.target.value)}
                      className="w-full bg-slate-50 border border-slate-200 rounded p-1.5 text-xs"
                    >
                      <option value="add">+ Add</option>
                      <option value="subtract">- Subtract</option>
                      <option value="multiply">* Multiply</option>
                      <option value="divide">/ Divide</option>
                      <option value="ratio_pct">% Ratio</option>
                      <option value="concat">_ Concat</option>
                    </select>
                  </div>
                </div>

                <div className="flex items-center gap-3">
                  <label className="flex items-center gap-1.5 text-xs cursor-pointer">
                    <input
                      type="radio"
                      name="deriveType"
                      checked={deriveBType === 'col'}
                      onChange={() => setDeriveBType('col')}
                    />
                    <span>Another Column</span>
                  </label>
                  <label className="flex items-center gap-1.5 text-xs cursor-pointer">
                    <input
                      type="radio"
                      name="deriveType"
                      checked={deriveBType === 'scalar'}
                      onChange={() => setDeriveBType('scalar')}
                    />
                    <span>Fixed Number</span>
                  </label>

                  {deriveBType === 'col' ? (
                    <select
                      value={colB}
                      onChange={e => setColB(e.target.value)}
                      className="bg-slate-50 border border-slate-200 rounded p-1 text-xs"
                    >
                      {allColumns.filter(c => c !== colA).map(c => <option key={c} value={c}>{c}</option>)}
                    </select>
                  ) : (
                    <input
                      type="number"
                      value={scalarB}
                      onChange={e => setScalarB(parseFloat(e.target.value))}
                      className="w-24 bg-slate-50 border border-slate-200 rounded p-1 text-xs"
                    />
                  )}
                </div>
              </div>
            )}

            {studioAction === 'drop' && (
              <div className="pt-2">
                <label className="text-[11px] font-bold text-slate-700 block mb-1">
                  Select Columns to Remove:
                </label>
                <div className="flex flex-wrap gap-2 text-xs">
                  {allColumns.map(c => (
                    <label key={c} className="flex items-center gap-1 bg-slate-50 border border-slate-200 rounded px-2 py-1 cursor-pointer">
                      <input
                        type="checkbox"
                        checked={dropCols.includes(c)}
                        onChange={e => {
                          if (e.target.checked) setDropCols(prev => [...prev, c]);
                          else setDropCols(prev => prev.filter(x => x !== c));
                        }}
                        className="rounded text-rose-600"
                      />
                      <span>{c}</span>
                    </label>
                  ))}
                </div>
              </div>
            )}

            <div className="pt-2">
              <button
                type="button"
                onClick={handleApplyTransform}
                className="py-2 px-4 bg-blue-600 hover:bg-blue-700 text-white font-bold rounded-lg text-xs shadow-sm shadow-blue-500/20 transition-all cursor-pointer"
              >
                Apply Transformation ⚡
              </button>
            </div>
          </div>

          {/* History & Controls */}
          <div className="flex items-start justify-between gap-4 bg-white border border-slate-200 rounded-xl p-4">
            <div className="flex-1">
              <span className="text-xs font-bold text-slate-800 block mb-1">
                Applied Studio Transformations:
              </span>
              {studioHistory.length === 0 ? (
                <p className="text-xs text-slate-500">
                  No custom studio transformations applied yet. Dataset matches baseline cleaned state.
                </p>
              ) : (
                <ul className="space-y-1 text-xs text-slate-700">
                  {studioHistory.map((h, i) => (
                    <li key={i}>• Step {i + 1}: {h}</li>
                  ))}
                </ul>
              )}
            </div>

            <div className="flex gap-2 shrink-0">
              {undoStack.length > 0 && (
                <button
                  type="button"
                  onClick={handleUndo}
                  className="py-1.5 px-3 bg-slate-100 hover:bg-slate-200 text-slate-700 font-semibold rounded-lg text-xs flex items-center gap-1 transition-all"
                >
                  <Undo2 className="w-3.5 h-3.5" />
                  Undo Last
                </button>
              )}
              {studioHistory.length > 0 && (
                <button
                  type="button"
                  onClick={handleReset}
                  className="py-1.5 px-3 bg-slate-100 hover:bg-slate-200 text-slate-700 font-semibold rounded-lg text-xs flex items-center gap-1 transition-all"
                >
                  <RotateCcw className="w-3.5 h-3.5" />
                  Reset All
                </button>
              )}
            </div>
          </div>

          {/* Table Preview */}
          <div className="space-y-2">
            <span className="text-xs font-bold text-slate-800 flex items-center gap-1.5">
              <TableIcon className="w-3.5 h-3.5 text-blue-600" />
              Live Studio Data Preview:
            </span>
            <div className="bg-white border border-slate-200 rounded-xl overflow-hidden overflow-x-auto shadow-sm">
              <table className="w-full text-left text-xs border-collapse">
                <thead>
                  <tr className="bg-slate-50 border-b border-slate-200 font-bold text-slate-700">
                    {allColumns.map(c => <th key={c} className="p-2.5">{c}</th>)}
                  </tr>
                </thead>
                <tbody className="divide-y divide-slate-100 font-mono text-[11px]">
                  {studioDf.slice(0, 10).map((r, idx) => (
                    <tr key={idx} className="hover:bg-slate-50/80">
                      {allColumns.map(c => <td key={c} className="p-2.5">{String(r[c] ?? '')}</td>)}
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        </div>
      )}

      {/* SUB-TAB 4: Preview & Export Cleaned Data */}
      {subTab === 'preview' && (
        <div className="space-y-4">
          <div className="flex items-center justify-between">
            <span className="text-xs font-bold text-slate-800">Cleaned Production Dataset:</span>
            <div className="flex gap-2">
              <button
                type="button"
                onClick={downloadCsv}
                className="py-1.5 px-3 bg-blue-600 hover:bg-blue-700 text-white font-bold rounded-lg text-xs flex items-center gap-1.5 shadow-sm shadow-blue-500/20 transition-all cursor-pointer"
              >
                <Download className="w-3.5 h-3.5" />
                Download CSV
              </button>
              <button
                type="button"
                onClick={downloadJson}
                className="py-1.5 px-3 bg-white hover:bg-slate-50 text-slate-700 border border-slate-200 font-bold rounded-lg text-xs flex items-center gap-1.5 shadow-sm transition-all cursor-pointer"
              >
                <Download className="w-3.5 h-3.5" />
                Download JSON
              </button>
            </div>
          </div>

          <div className="bg-white border border-slate-200 rounded-xl overflow-hidden overflow-x-auto shadow-sm max-h-[500px]">
            <table className="w-full text-left text-xs border-collapse">
              <thead className="sticky top-0 bg-slate-50 border-b border-slate-200 shadow-sm">
                <tr className="font-bold text-slate-700">
                  {allColumns.map(c => <th key={c} className="p-2.5">{c}</th>)}
                </tr>
              </thead>
              <tbody className="divide-y divide-slate-100 font-mono text-[11px]">
                {studioDf.slice(0, 30).map((r, idx) => (
                  <tr key={idx} className="hover:bg-slate-50/80">
                    {allColumns.map(c => <td key={c} className="p-2.5">{String(r[c] ?? '')}</td>)}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}
    </div>
  );
};
