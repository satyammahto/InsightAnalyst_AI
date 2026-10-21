import React, { useState, useEffect, useMemo, useRef } from 'react';
import Papa from 'papaparse';
import {
  Sidebar
} from './components/Sidebar';
import { JourneyStepper } from './components/JourneyStepper';
import { TopStatusBar } from './components/TopStatusBar';
import { OverviewTab } from './components/tabs/OverviewTab';
import { QualityTab } from './components/tabs/QualityTab';
import { InsightsTab } from './components/tabs/InsightsTab';
import { ChartsTab } from './components/tabs/ChartsTab';
import { PatternsTab } from './components/tabs/PatternsTab';
import { AskAiTab } from './components/tabs/AskAiTab';

import { PRACTICE_DATASETS } from './sample_data/datasets';
import {
  ChatMessage,
  CleaningReport,
  ColumnClassification,
  DataRow,
  ExecutiveBrief,
  MLResults
} from './types';
import { detectColumnTypes } from './utils/typeDetector';
import { detectAndClean } from './utils/cleaner';
import { runMLAnalysis } from './utils/mlEngine';
import { DatasetRAGEngine } from './utils/ragEngine';
import { generateExecutiveSummary } from './utils/summarizer';
import { AIAssistant } from './utils/aiAssistant';
import { SQLEngine } from './utils/sqlEngine';

export const App: React.FC = () => {
  // Navigation & View mode
  const [activeTab, setActiveTab] = useState<string>('overview');
  const [isSimpleMode, setIsSimpleMode] = useState<boolean>(true);

  // Data Source
  const [dataSource, setDataSource] = useState<'practice' | 'upload'>('practice');
  const [selectedPracticeId, setSelectedPracticeId] = useState<string>('sales');
  const [customDataset, setCustomDataset] = useState<{ name: string; rows: DataRow[] } | null>(null);

  // Analysis settings
  const [imputePolicy, setImputePolicy] = useState<string>('Fill with typical values (Median/Mode)');
  const [nClusters, setNClusters] = useState<number>(4);
  const [contamination, setContamination] = useState<number>(0.05);
  const [geminiApiKey, setGeminiApiKey] = useState<string>('');

  // Active dataset rows
  const [rawRows, setRawRows] = useState<DataRow[]>([]);
  const [datasetTitle, setDatasetTitle] = useState<string>('Retail Sales Data');

  // Pipeline execution state
  const [isAnalyzing, setIsAnalyzing] = useState<boolean>(false);
  const [analysisStep, setAnalysisStep] = useState<string>('');
  const [analysisProgress, setAnalysisProgress] = useState<number>(0);
  const [pipeLatency, setPipeLatency] = useState<string>('0.14s');

  // Computed Analytical results
  const [colTypes, setColTypes] = useState<ColumnClassification>({
    numeric: [],
    categorical: [],
    date: [],
    id_high_cardinality: [],
    text: [],
    details: {},
    telemetry_table: []
  });
  const [cleanedRows, setCleanedRows] = useState<DataRow[]>([]);
  const [cleaningReport, setCleaningReport] = useState<CleaningReport>({
    initial_shape: [0, 0],
    final_shape: [0, 0],
    duplicates_removed: 0,
    total_nulls_imputed: 0,
    nulls_imputed: {},
    string_tokens_sanitized: 0,
    dates_converted: [],
    hygiene_score: '96.5%',
    dag_steps: [],
    execution_ms: '0.0 ms'
  });
  const [mlResults, setMlResults] = useState<MLResults>({
    numeric_cols: [],
    has_enough_numeric: false,
    correlation: null,
    clustering: null,
    outliers: null,
    execution_ms: '0.0 ms',
    status: 'success',
    is_sampled: false,
    sample_size: 0
  });
  const [ragEngine, setRagEngine] = useState<DatasetRAGEngine>(new DatasetRAGEngine());
  const [executiveBrief, setExecutiveBrief] = useState<ExecutiveBrief>({
    what_we_found: [],
    what_should_you_look_at: [],
    actionable_insights: [],
    ingestion_friendly: '',
    driver_friendly: '',
    segmentation_friendly: '',
    outlier_friendly: ''
  });

  // Studio Interactive State
  const [studioDf, setStudioDf] = useState<DataRow[]>([]);
  const [studioHistory, setStudioHistory] = useState<string[]>([]);
  const [undoStack, setUndoStack] = useState<DataRow[][]>([]);

  // Chat Conversation State
  const [chatMessages, setChatMessages] = useState<ChatMessage[]>([]);

  // Initial load or data source switch
  useEffect(() => {
    if (dataSource === 'practice') {
      const practice = PRACTICE_DATASETS.find(p => p.id === selectedPracticeId) || PRACTICE_DATASETS[0];
      Papa.parse(practice.csvContent, {
        header: true,
        dynamicTyping: true,
        skipEmptyLines: true,
        complete: res => {
          setRawRows(res.data as DataRow[]);
          setDatasetTitle(practice.title);
        }
      });
    } else if (customDataset) {
      setRawRows(customDataset.rows);
      setDatasetTitle(customDataset.name);
    }
  }, [dataSource, selectedPracticeId, customDataset]);

  // Run full analysis pipeline when rawRows, settings, or dataset changes
  const prevSignatureRef = useRef<string>('');

  useEffect(() => {
    if (rawRows.length === 0) return;

    const signature = `${datasetTitle}_${rawRows.length}_${imputePolicy}_${nClusters}_${contamination}`;
    if (prevSignatureRef.current === signature) return;
    prevSignatureRef.current = signature;

    const runPipeline = async () => {
      const startTime = performance.now();
      setIsAnalyzing(true);
      setAnalysisProgress(15);
      setAnalysisStep('1. Checking data structure & columns... 🔄');

      await new Promise(r => setTimeout(r, 60));

      // 1. Column Type Detection
      const types = detectColumnTypes(rawRows);
      setColTypes(types);

      setAnalysisProgress(40);
      setAnalysisStep('2. Cleaning and standardizing values... 🔄');
      await new Promise(r => setTimeout(r, 60));

      // 2. 5-Stage Defensive Cleaning DAG
      const { cleanedRows: cRows, report: cReport } = detectAndClean(rawRows, types, imputePolicy);
      setCleanedRows(cRows);
      setCleaningReport(cReport);
      setStudioDf(cRows.map(r => ({ ...r })));
      setStudioHistory([]);
      setUndoStack([]);

      setAnalysisProgress(75);
      setAnalysisStep('3. Finding patterns and unusual records... 🔄');
      await new Promise(r => setTimeout(r, 60));

      // 3. ML Engine (KMeans, Silhouette, PCA, Isolation Forest + Mamdani FIS)
      const ml = runMLAnalysis(cRows, types, nClusters, contamination);
      setMlResults(ml);

      setAnalysisProgress(90);
      setAnalysisStep('4. Creating AI insights and explanations... 🔄');
      await new Promise(r => setTimeout(r, 60));

      // 4. RAG Engine
      const rag = new DatasetRAGEngine();
      rag.buildKnowledgeBase(cRows, types, cReport, ml);
      setRagEngine(rag);

      // 5. Executive Brief
      const brief = generateExecutiveSummary(cRows, types, cReport, ml);
      setExecutiveBrief(brief);

      const latencySec = ((performance.now() - startTime) / 1000).toFixed(2) + 's';
      setPipeLatency(latencySec);

      // Reset Chat Messages to Welcome Greeting
      setChatMessages([
        {
          role: 'assistant',
          content: `Hello! I'm your **AI Data Analyst**. I've analyzed **${datasetTitle}** (${cRows.length.toLocaleString()} rows, ${Object.keys(cRows[0] || {}).length} columns) and I'm ready to answer any questions about your data.\n\nTry clicking one of the sample questions below or ask me anything!`
        }
      ]);

      setAnalysisProgress(100);
      setAnalysisStep('5. Analysis ready! ✅');
      await new Promise(r => setTimeout(r, 120));
      setIsAnalyzing(false);
    };

    runPipeline();
  }, [rawRows, datasetTitle, imputePolicy, nClusters, contamination]);

  // Engines
  const aiAssistant = useMemo(() => {
    return new AIAssistant(
      cleanedRows,
      colTypes,
      ragEngine,
      mlResults,
      cleaningReport,
      datasetTitle,
      geminiApiKey
    );
  }, [cleanedRows, colTypes, ragEngine, mlResults, cleaningReport, datasetTitle, geminiApiKey]);

  const sqlEngine = useMemo(() => {
    return new SQLEngine(studioDf.length > 0 ? studioDf : cleanedRows);
  }, [studioDf, cleanedRows]);

  const handleCustomDataUploaded = (name: string, rows: DataRow[]) => {
    setCustomDataset({ name, rows });
    setDataSource('upload');
  };

  const handleRemoveCustomDataset = () => {
    setCustomDataset(null);
    setDataSource('practice');
    setSelectedPracticeId('sales');
  };

  const tabs = [
    { id: 'overview', label: '📊 Overview' },
    { id: 'quality', label: '🩺 Data Quality' },
    { id: 'insights', label: '💡 Insights' },
    { id: 'charts', label: '📈 Charts' },
    { id: 'patterns', label: '🔍 Patterns & Anomalies' },
    { id: 'ask', label: '💬 Ask AI' }
  ];

  return (
    <div className="flex h-screen w-screen overflow-hidden bg-[#F8FAFC]">
      {/* Left Sidebar */}
      <Sidebar
        isSimpleMode={isSimpleMode}
        setIsSimpleMode={setIsSimpleMode}
        dataSource={dataSource}
        setDataSource={setDataSource}
        selectedPracticeId={selectedPracticeId}
        setSelectedPracticeId={setSelectedPracticeId}
        onCustomDataUploaded={handleCustomDataUploaded}
        customDatasetName={customDataset?.name || null}
        onRemoveCustomDataset={handleRemoveCustomDataset}
        imputePolicy={imputePolicy}
        setImputePolicy={setImputePolicy}
        nClusters={nClusters}
        setNClusters={setNClusters}
        contamination={contamination}
        setContamination={setContamination}
        geminiApiKey={geminiApiKey}
        setGeminiApiKey={setGeminiApiKey}
        activeDatasetTitle={datasetTitle}
        rowCount={cleanedRows.length}
        colCount={cleanedRows.length > 0 ? Object.keys(cleanedRows[0]).length : 0}
      />

      {/* Main Content Area */}
      <main className="flex-1 overflow-y-auto p-6 md:p-8">
        <div className="max-w-[1400px] mx-auto space-y-5">
          {/* Top Journey Stepper Bar */}
          <JourneyStepper activeTab={activeTab} setActiveTab={setActiveTab} />

          {/* Top Status Bar */}
          <TopStatusBar
            isSimpleMode={isSimpleMode}
            datasetTitle={datasetTitle}
            rowCount={cleanedRows.length}
            colCount={cleanedRows.length > 0 ? Object.keys(cleanedRows[0]).length : 0}
            latency={pipeLatency}
          />

          {/* Progress Modal overlay when analyzing */}
          {isAnalyzing && (
            <div className="bg-white border border-blue-200 rounded-2xl p-6 shadow-xl shadow-blue-500/10 space-y-3">
              <div className="flex items-center justify-between text-xs font-bold text-slate-800">
                <span>{analysisStep}</span>
                <span className="text-blue-600 font-extrabold">{analysisProgress}%</span>
              </div>
              <div className="w-full h-2 bg-slate-100 rounded-full overflow-hidden">
                <div
                  className="h-full bg-blue-600 rounded-full transition-all duration-300"
                  style={{ width: `${analysisProgress}%` }}
                />
              </div>
            </div>
          )}

          {/* Navigation Tabs Header */}
          <div className="border-b border-slate-200">
            <div className="flex gap-1.5 overflow-x-auto pb-1 text-xs">
              {tabs.map(tab => {
                const isActive = activeTab === tab.id;
                return (
                  <button
                    key={tab.id}
                    type="button"
                    onClick={() => setActiveTab(tab.id)}
                    className={`py-2 px-4 rounded-lg font-bold transition-all cursor-pointer whitespace-nowrap ${
                      isActive
                        ? 'bg-blue-50 text-blue-700 font-extrabold'
                        : 'text-slate-600 hover:text-slate-900 hover:bg-slate-100/70'
                    }`}
                  >
                    {tab.label}
                  </button>
                );
              })}
            </div>
          </div>

          {/* Tab Panes */}
          <div className="pt-2">
            {activeTab === 'overview' && (
              <OverviewTab
                df={studioDf.length > 0 ? studioDf : cleanedRows}
                colTypes={colTypes}
                cleaningReport={cleaningReport}
                mlResults={mlResults}
                executiveBrief={executiveBrief}
                ragChunks={ragEngine.chunks}
                datasetName={datasetTitle}
                isSimpleMode={isSimpleMode}
                pipeLatency={pipeLatency}
                studioHistory={studioHistory}
              />
            )}

            {activeTab === 'quality' && (
              <QualityTab
                df={cleanedRows}
                colTypes={colTypes}
                cleaningReport={cleaningReport}
                studioDf={studioDf}
                setStudioDf={setStudioDf}
                studioHistory={studioHistory}
                setStudioHistory={setStudioHistory}
                undoStack={undoStack}
                setUndoStack={setUndoStack}
                selectedDatasetName={datasetTitle}
              />
            )}

            {activeTab === 'insights' && (
              <InsightsTab
                df={studioDf.length > 0 ? studioDf : cleanedRows}
                colTypes={colTypes}
                cleaningReport={cleaningReport}
                mlResults={mlResults}
                executiveBrief={executiveBrief}
                datasetName={datasetTitle}
                studioHistory={studioHistory}
              />
            )}

            {activeTab === 'charts' && (
              <ChartsTab
                df={studioDf.length > 0 ? studioDf : cleanedRows}
                colTypes={colTypes}
                mlResults={mlResults}
                isSimpleMode={isSimpleMode}
              />
            )}

            {activeTab === 'patterns' && (
              <PatternsTab
                df={studioDf.length > 0 ? studioDf : cleanedRows}
                colTypes={colTypes}
                mlResults={mlResults}
                isSimpleMode={isSimpleMode}
              />
            )}

            {activeTab === 'ask' && (
              <AskAiTab
                chatMessages={chatMessages}
                setChatMessages={setChatMessages}
                aiAssistant={aiAssistant}
                sqlEngine={sqlEngine}
                df={studioDf.length > 0 ? studioDf : cleanedRows}
                colTypes={colTypes}
                mlResults={mlResults}
                datasetTitle={datasetTitle}
              />
            )}
          </div>
        </div>
      </main>
    </div>
  );
};

export default App;
