import React, { useRef, useState } from 'react';
import {
  Sparkles,
  Database,
  Upload,
  Settings,
  Key,
  Trash2,
  ChevronDown,
  ChevronRight,
  FileSpreadsheet,
  Layers
} from 'lucide-react';
import Papa from 'papaparse';
import * as XLSX from 'xlsx';
import { PRACTICE_DATASETS, PracticeDataset } from '../sample_data/datasets';
import { DataRow } from '../types';

interface SidebarProps {
  isSimpleMode: boolean;
  setIsSimpleMode: (val: boolean) => void;
  dataSource: 'practice' | 'upload';
  setDataSource: (source: 'practice' | 'upload') => void;
  selectedPracticeId: string;
  setSelectedPracticeId: (id: string) => void;
  onCustomDataUploaded: (name: string, rows: DataRow[]) => void;
  customDatasetName: string | null;
  onRemoveCustomDataset: () => void;
  imputePolicy: string;
  setImputePolicy: (val: string) => void;
  nClusters: number;
  setNClusters: (val: number) => void;
  contamination: number;
  setContamination: (val: number) => void;
  geminiApiKey: string;
  setGeminiApiKey: (val: string) => void;
  activeDatasetTitle: string;
  rowCount: number;
  colCount: number;
}

export const Sidebar: React.FC<SidebarProps> = ({
  isSimpleMode,
  setIsSimpleMode,
  dataSource,
  setDataSource,
  selectedPracticeId,
  setSelectedPracticeId,
  onCustomDataUploaded,
  customDatasetName,
  onRemoveCustomDataset,
  imputePolicy,
  setImputePolicy,
  nClusters,
  setNClusters,
  contamination,
  setContamination,
  geminiApiKey,
  setGeminiApiKey,
  activeDatasetTitle,
  rowCount,
  colCount
}) => {
  const [settingsOpen, setSettingsOpen] = useState(false);
  const [apiKeyOpen, setApiKeyOpen] = useState(false);
  const [multiFileStrategy, setMultiFileStrategy] = useState<'merge' | 'pick'>('merge');
  const [uploadedFilesList, setUploadedFilesList] = useState<Array<{ name: string; rows: DataRow[] }>>([]);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const handleFileUpload = (e: React.ChangeEvent<HTMLInputElement>) => {
    const files = e.target.files;
    if (!files || files.length === 0) return;

    const parsedFiles: Array<{ name: string; rows: DataRow[] }> = [];
    let filesProcessed = 0;

    Array.from(files).forEach(file => {
      const ext = file.name.split('.').pop()?.toLowerCase();
      if (ext === 'xlsx' || ext === 'xls') {
        const reader = new FileReader();
        reader.onload = evt => {
          try {
            const data = new Uint8Array(evt.target?.result as ArrayBuffer);
            const workbook = XLSX.read(data, { type: 'array' });
            const sheetName = workbook.SheetNames[0];
            const sheet = workbook.Sheets[sheetName];
            const json = XLSX.utils.sheet_to_json<DataRow>(sheet);
            parsedFiles.push({ name: file.name, rows: json });
          } catch (err) {
            console.error('Error parsing excel:', err);
          }
          filesProcessed++;
          if (filesProcessed === files.length) {
            finishUpload(parsedFiles);
          }
        };
        reader.readAsArrayBuffer(file);
      } else {
        // CSV / TSV
        Papa.parse(file, {
          header: true,
          dynamicTyping: true,
          skipEmptyLines: true,
          complete: results => {
            parsedFiles.push({ name: file.name, rows: results.data as DataRow[] });
            filesProcessed++;
            if (filesProcessed === files.length) {
              finishUpload(parsedFiles);
            }
          },
          error: err => {
            console.error('CSV parse error:', err);
            filesProcessed++;
            if (filesProcessed === files.length) {
              finishUpload(parsedFiles);
            }
          }
        });
      }
    });
  };

  const finishUpload = (parsedFiles: Array<{ name: string; rows: DataRow[] }>) => {
    if (parsedFiles.length === 0) return;
    setUploadedFilesList(parsedFiles);

    if (parsedFiles.length === 1) {
      onCustomDataUploaded(parsedFiles[0].name, parsedFiles[0].rows);
    } else {
      if (multiFileStrategy === 'merge') {
        const merged: DataRow[] = [];
        parsedFiles.forEach(f => {
          f.rows.forEach(r => merged.push({ ...r, _source_file: f.name }));
        });
        onCustomDataUploaded(`Merged (${parsedFiles.length} files)`, merged);
      } else {
        onCustomDataUploaded(parsedFiles[0].name, parsedFiles[0].rows);
      }
    }
  };

  return (
    <aside className="w-72 bg-white border-r border-slate-200 flex flex-col shrink-0 h-screen overflow-y-auto select-none">
      {/* Brand Header */}
      <div className="p-5 border-b border-slate-100">
        <div className="flex items-center gap-2">
          <div className="w-8 h-8 rounded-lg bg-blue-600 flex items-center justify-center text-white shadow-sm shadow-blue-500/20">
            <Sparkles className="w-4 h-4" />
          </div>
          <div>
            <h1 className="text-base font-extrabold text-slate-900 tracking-tight leading-none">
              InsightAnalyst AI
            </h1>
            <p className="text-xs text-slate-500 mt-1">Simple AI-Powered Insights</p>
          </div>
        </div>
      </div>

      <div className="p-4 space-y-5 flex-1">
        {/* View Mode Radio */}
        <div>
          <label className="text-xs font-bold text-slate-800 uppercase tracking-wider block mb-2">
            View Mode
          </label>
          <div className="grid grid-cols-2 gap-1.5 p-1 bg-slate-100 rounded-lg">
            <button
              type="button"
              onClick={() => setIsSimpleMode(true)}
              className={`py-1.5 px-2 rounded-md text-xs font-semibold transition-all ${
                isSimpleMode
                  ? 'bg-white text-blue-700 shadow-sm'
                  : 'text-slate-600 hover:text-slate-900'
              }`}
            >
              ✨ Simple
            </button>
            <button
              type="button"
              onClick={() => setIsSimpleMode(false)}
              className={`py-1.5 px-2 rounded-md text-xs font-semibold transition-all ${
                !isSimpleMode
                  ? 'bg-white text-blue-700 shadow-sm'
                  : 'text-slate-600 hover:text-slate-900'
              }`}
            >
              🔬 Technical
            </button>
          </div>
        </div>

        {/* Data Source Selector */}
        <div>
          <label className="text-xs font-bold text-slate-800 uppercase tracking-wider block mb-2">
            Data Source
          </label>
          <div className="space-y-1.5">
            <label
              className={`flex items-center gap-2.5 p-2.5 rounded-lg border cursor-pointer transition-all ${
                dataSource === 'practice'
                  ? 'bg-blue-50/70 border-blue-200 text-blue-900'
                  : 'bg-white border-slate-200 text-slate-700 hover:border-slate-300'
              }`}
            >
              <input
                type="radio"
                name="dataSource"
                value="practice"
                checked={dataSource === 'practice'}
                onChange={() => setDataSource('practice')}
                className="text-blue-600 focus:ring-blue-500 h-3.5 w-3.5"
              />
              <Database className="w-4 h-4 text-blue-600" />
              <span className="text-xs font-semibold">Practice Datasets</span>
            </label>

            <label
              className={`flex items-center gap-2.5 p-2.5 rounded-lg border cursor-pointer transition-all ${
                dataSource === 'upload'
                  ? 'bg-blue-50/70 border-blue-200 text-blue-900'
                  : 'bg-white border-slate-200 text-slate-700 hover:border-slate-300'
              }`}
            >
              <input
                type="radio"
                name="dataSource"
                value="upload"
                checked={dataSource === 'upload'}
                onChange={() => setDataSource('upload')}
                className="text-blue-600 focus:ring-blue-500 h-3.5 w-3.5"
              />
              <Upload className="w-4 h-4 text-emerald-600" />
              <span className="text-xs font-semibold">Upload Custom File</span>
            </label>
          </div>
        </div>

        {/* Practice Dataset Picker */}
        {dataSource === 'practice' && (
          <div>
            <label className="text-xs font-semibold text-slate-700 block mb-1.5">
              Choose Practice Dataset:
            </label>
            <select
              value={selectedPracticeId}
              onChange={e => setSelectedPracticeId(e.target.value)}
              className="w-full text-xs font-medium bg-white border border-slate-200 rounded-lg p-2 text-slate-800 focus:outline-none focus:ring-2 focus:ring-blue-500/20 focus:border-blue-500"
            >
              {PRACTICE_DATASETS.map((ds: PracticeDataset) => (
                <option key={ds.id} value={ds.id}>
                  {ds.title}
                </option>
              ))}
            </select>
          </div>
        )}

        {/* Upload Custom File Panel */}
        {dataSource === 'upload' && (
          <div className="space-y-3">
            <input
              ref={fileInputRef}
              type="file"
              accept=".csv,.xlsx,.xls,.tsv,.txt"
              multiple
              onChange={handleFileUpload}
              className="hidden"
            />
            <div
              onClick={() => fileInputRef.current?.click()}
              className="border-2 border-dashed border-slate-200 hover:border-blue-400 bg-slate-50 hover:bg-blue-50/40 rounded-xl p-4 text-center cursor-pointer transition-all"
            >
              <FileSpreadsheet className="w-6 h-6 text-slate-400 mx-auto mb-1.5" />
              <p className="text-xs font-semibold text-slate-700">Click to upload files</p>
              <p className="text-[11px] text-slate-400 mt-0.5">CSV, XLSX, TSV (multi-file supported)</p>
            </div>

            {uploadedFilesList.length > 1 && (
              <div className="bg-slate-50 p-2.5 rounded-lg border border-slate-200 space-y-2">
                <span className="text-xs font-bold text-slate-800 block">
                  {uploadedFilesList.length} Files Uploaded
                </span>
                <div className="grid grid-cols-2 gap-1 text-[11px]">
                  <button
                    type="button"
                    onClick={() => {
                      setMultiFileStrategy('merge');
                      const merged: DataRow[] = [];
                      uploadedFilesList.forEach(f => {
                        f.rows.forEach(r => merged.push({ ...r, _source_file: f.name }));
                      });
                      onCustomDataUploaded(`Merged (${uploadedFilesList.length} files)`, merged);
                    }}
                    className={`p-1.5 rounded font-semibold border ${
                      multiFileStrategy === 'merge'
                        ? 'bg-blue-600 text-white border-blue-600'
                        : 'bg-white text-slate-700 border-slate-200'
                    }`}
                  >
                    🔗 Merge All
                  </button>
                  <button
                    type="button"
                    onClick={() => setMultiFileStrategy('pick')}
                    className={`p-1.5 rounded font-semibold border ${
                      multiFileStrategy === 'pick'
                        ? 'bg-blue-600 text-white border-blue-600'
                        : 'bg-white text-slate-700 border-slate-200'
                    }`}
                  >
                    📄 Pick One
                  </button>
                </div>
                {multiFileStrategy === 'pick' && (
                  <select
                    onChange={e => {
                      const idx = parseInt(e.target.value, 10);
                      onCustomDataUploaded(uploadedFilesList[idx].name, uploadedFilesList[idx].rows);
                    }}
                    className="w-full text-xs bg-white border border-slate-200 rounded p-1.5 mt-1"
                  >
                    {uploadedFilesList.map((f, i) => (
                      <option key={i} value={i}>
                        {f.name} ({f.rows.length} rows)
                      </option>
                    ))}
                  </select>
                )}
              </div>
            )}

            {customDatasetName && (
              <button
                type="button"
                onClick={onRemoveCustomDataset}
                className="w-full py-1.5 px-2 bg-rose-50 hover:bg-rose-100 text-rose-700 border border-rose-200 rounded-lg text-xs font-semibold flex items-center justify-center gap-1.5 transition-all"
              >
                <Trash2 className="w-3.5 h-3.5" />
                Remove Uploaded File
              </button>
            )}
          </div>
        )}

        {/* Analysis Settings Expander */}
        <div className="border border-slate-200 rounded-xl overflow-hidden">
          <button
            type="button"
            onClick={() => setSettingsOpen(!settingsOpen)}
            className="w-full p-3 bg-slate-50 hover:bg-slate-100/80 flex items-center justify-between text-xs font-bold text-slate-800 transition-colors"
          >
            <div className="flex items-center gap-2">
              <Settings className="w-3.5 h-3.5 text-slate-600" />
              <span>Analysis Settings (Optional)</span>
            </div>
            {settingsOpen ? <ChevronDown className="w-3.5 h-3.5" /> : <ChevronRight className="w-3.5 h-3.5" />}
          </button>

          {settingsOpen && (
            <div className="p-3.5 space-y-4 bg-white border-t border-slate-100 text-xs">
              <div>
                <label className="text-[11px] font-bold text-slate-700 block mb-1">
                  Missing Value Strategy:
                </label>
                <select
                  value={imputePolicy}
                  onChange={e => setImputePolicy(e.target.value)}
                  className="w-full bg-slate-50 border border-slate-200 rounded-lg p-1.5 text-xs text-slate-800"
                >
                  <option value="Fill with typical values (Median/Mode)">Fill typical values (Median/Mode)</option>
                  <option value="Fill with averages (Mean)">Fill averages (Mean)</option>
                  <option value="Remove incomplete rows">Remove incomplete rows</option>
                </select>
              </div>

              <div>
                <div className="flex justify-between items-center mb-1">
                  <label className="text-[11px] font-bold text-slate-700">
                    Number of Groups (k):
                  </label>
                  <span className="font-extrabold text-blue-600">{nClusters}</span>
                </div>
                <input
                  type="range"
                  min={2}
                  max={6}
                  value={nClusters}
                  onChange={e => setNClusters(parseInt(e.target.value, 10))}
                  className="w-full accent-blue-600 cursor-pointer"
                />
                <span className="text-[10px] text-slate-400">Natural clusters to detect</span>
              </div>

              <div>
                <div className="flex justify-between items-center mb-1">
                  <label className="text-[11px] font-bold text-slate-700">
                    Outlier Sensitivity:
                  </label>
                  <span className="font-extrabold text-rose-600">{(contamination * 100).toFixed(0)}%</span>
                </div>
                <input
                  type="range"
                  min={0.01}
                  max={0.15}
                  step={0.01}
                  value={contamination}
                  onChange={e => setContamination(parseFloat(e.target.value))}
                  className="w-full accent-rose-600 cursor-pointer"
                />
                <span className="text-[10px] text-slate-400">Expected unusual records boundary</span>
              </div>
            </div>
          )}
        </div>

        {/* API Key Expander */}
        <div className="border border-slate-200 rounded-xl overflow-hidden">
          <button
            type="button"
            onClick={() => setApiKeyOpen(!apiKeyOpen)}
            className="w-full p-3 bg-slate-50 hover:bg-slate-100/80 flex items-center justify-between text-xs font-bold text-slate-800 transition-colors"
          >
            <div className="flex items-center gap-2">
              <Key className="w-3.5 h-3.5 text-slate-600" />
              <span>AI Assistant API Key (Optional)</span>
            </div>
            {apiKeyOpen ? <ChevronDown className="w-3.5 h-3.5" /> : <ChevronRight className="w-3.5 h-3.5" />}
          </button>

          {apiKeyOpen && (
            <div className="p-3 bg-white border-t border-slate-100 space-y-2">
              <p className="text-[11px] text-slate-500">
                Optional: Enables Gemini 2.0 reasoning. Leave blank to use built-in local offline engine.
              </p>
              <input
                type="password"
                placeholder="AIzaSy..."
                value={geminiApiKey}
                onChange={e => setGeminiApiKey(e.target.value)}
                className="w-full bg-slate-50 border border-slate-200 rounded-lg p-2 text-xs font-mono text-slate-800"
              />
            </div>
          )}
        </div>
      </div>

      {/* Dataset Info Chip at Bottom */}
      <div className="p-4 border-t border-slate-200 bg-slate-50/70">
        <div className="flex items-start gap-2.5">
          <Layers className="w-4 h-4 text-blue-600 shrink-0 mt-0.5" />
          <div className="text-xs">
            <span className="font-bold text-slate-800 block truncate" title={activeDatasetTitle}>
              {activeDatasetTitle}
            </span>
            <span className="text-slate-500 text-[11px]">
              {rowCount.toLocaleString()} rows · {colCount} columns
            </span>
          </div>
        </div>
      </div>
    </aside>
  );
};
