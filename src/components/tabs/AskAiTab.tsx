import React, { useState } from 'react';
import {
  Send,
  Sparkles,
  User,
  RotateCcw,
  Play,
  Download,
  Code2,
  ChevronDown,
  ChevronRight,
  Database
} from 'lucide-react';
import { Bar, Line } from 'react-chartjs-2';
import Papa from 'papaparse';
import {
  ChatMessage,
  ColumnClassification,
  DataRow,
  MLResults
} from '../../types';
import { AIAssistant } from '../../utils/aiAssistant';
import { SQLEngine } from '../../utils/sqlEngine';

interface AskAiTabProps {
  chatMessages: ChatMessage[];
  setChatMessages: React.Dispatch<React.SetStateAction<ChatMessage[]>>;
  aiAssistant: AIAssistant;
  sqlEngine: SQLEngine;
  df: DataRow[];
  colTypes: ColumnClassification;
  mlResults: MLResults;
  datasetTitle: string;
}

export const AskAiTab: React.FC<AskAiTabProps> = ({
  chatMessages,
  setChatMessages,
  aiAssistant,
  sqlEngine,
  df: _df,
  colTypes,
  mlResults: _mlResults,
  datasetTitle
}) => {
  const [inputText, setInputText] = useState('');
  const [loading, setLoading] = useState(false);

  // SQL Studio state
  const [sqlStudioOpen, setSqlStudioOpen] = useState(false);
  const sampleQueries = sqlEngine.generateSampleQueries(
    Object.keys(_df[0] || {}),
    colTypes.numeric,
    colTypes.categorical
  );

  const [selectedPreset, setSelectedPreset] = useState<string>(sampleQueries[0]?.title || 'Custom Query');
  const [sqlCode, setSqlCode] = useState<string>(sampleQueries[0]?.sql || 'SELECT * FROM dataset LIMIT 10;');
  const [sqlResults, setSqlResults] = useState<{
    success: boolean;
    rows: DataRow[];
    rowCount: number;
    executionTimeMs: number;
    error?: string;
  } | null>(null);
  const [sqlViewTab, setSqlViewTab] = useState<'table' | 'chart'>('table');

  const handleSendMessage = async (msgToSend?: string) => {
    const text = (msgToSend || inputText).trim();
    if (!text || loading) return;

    const userMsg: ChatMessage = { role: 'user', content: text };
    setChatMessages(prev => [...prev, userMsg]);
    setInputText('');
    setLoading(true);

    try {
      const res = await aiAssistant.processMessage(text);
      const assistantMsg: ChatMessage = {
        role: 'assistant',
        content: res.text,
        chart: res.chart,
        table: res.table,
        code: res.codeSnippet
      };
      setChatMessages(prev => [...prev, assistantMsg]);
    } catch (err: any) {
      setChatMessages(prev => [
        ...prev,
        {
          role: 'assistant',
          content: `Apologies, I encountered an issue analyzing your query: ${err.message || 'Unknown error'}.`
        }
      ]);
    } finally {
      setLoading(false);
    }
  };

  const handleRunSQL = () => {
    const res = sqlEngine.executeQuery(sqlCode);
    setSqlResults(res);
  };

  const downloadSqlCsv = () => {
    if (!sqlResults || sqlResults.rows.length === 0) return;
    const csv = Papa.unparse(sqlResults.rows);
    const blob = new Blob([csv], { type: 'text/csv' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `duckdb_query_result.csv`;
    a.click();
    URL.revokeObjectURL(url);
  };

  return (
    <div className="space-y-6">
      {/* Title */}
      <div>
        <h2 className="text-2xl font-extrabold text-slate-900 tracking-tight">Ask AI</h2>
        <p className="text-sm text-slate-500 mt-1">
          Ask any question about your data in plain English. Your AI analyst will explain it immediately.
        </p>
      </div>

      {/* Quick Prompts */}
      <div>
        <span className="text-[11px] font-bold text-slate-400 uppercase tracking-wider block mb-2">
          SAMPLE QUESTIONS YOU CAN ASK:
        </span>
        <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-6 gap-2">
          <button
            type="button"
            onClick={() => handleSendMessage('Show me the highest critical rows.')}
            className="p-2 bg-white hover:bg-slate-50 border border-slate-200 rounded-xl text-xs font-bold text-slate-800 shadow-sm transition-all text-center truncate cursor-pointer"
            title="Identify records with highest computed anomaly risk scores"
          >
            🚨 Critical Rows
          </button>
          <button
            type="button"
            onClick={() => handleSendMessage('What are the biggest problems in my data?')}
            className="p-2 bg-white hover:bg-slate-50 border border-slate-200 rounded-xl text-xs font-bold text-slate-800 shadow-sm transition-all text-center truncate cursor-pointer"
            title="Check data quality problems, missing values, and outliers"
          >
            ⚠️ Biggest Problems
          </button>
          <button
            type="button"
            onClick={() => handleSendMessage('Show me the most important insights.')}
            className="p-2 bg-white hover:bg-slate-50 border border-slate-200 rounded-xl text-xs font-bold text-slate-800 shadow-sm transition-all text-center truncate cursor-pointer"
            title="Summarize the most important patterns and findings"
          >
            💡 Key Insights
          </button>
          <button
            type="button"
            onClick={() => handleSendMessage('Why are some records unusual?')}
            className="p-2 bg-white hover:bg-slate-50 border border-slate-200 rounded-xl text-xs font-bold text-slate-800 shadow-sm transition-all text-center truncate cursor-pointer"
            title="Understand why specific records were flagged"
          >
            🔍 Why Unusual?
          </button>
          <button
            type="button"
            onClick={() => handleSendMessage('Which products generate the most sales?')}
            className="p-2 bg-white hover:bg-slate-50 border border-slate-200 rounded-xl text-xs font-bold text-slate-800 shadow-sm transition-all text-center truncate cursor-pointer"
            title="See which categories generate the highest volume"
          >
            🏆 Top Drivers
          </button>
          <button
            type="button"
            onClick={() => handleSendMessage('Are there any important trends?')}
            className="p-2 bg-white hover:bg-slate-50 border border-slate-200 rounded-xl text-xs font-bold text-slate-800 shadow-sm transition-all text-center truncate cursor-pointer"
            title="Look at timeline activity and volume trends"
          >
            📈 Trends
          </button>
        </div>
      </div>

      {/* DuckDB In-Memory SQL Query Studio */}
      <div className="border border-slate-200 rounded-xl overflow-hidden bg-white shadow-sm">
        <button
          type="button"
          onClick={() => setSqlStudioOpen(!sqlStudioOpen)}
          className="w-full p-4 bg-slate-50/80 hover:bg-slate-100/80 flex items-center justify-between text-xs font-bold text-slate-800 transition-colors"
        >
          <div className="flex items-center gap-2">
            <Database className="w-4 h-4 text-blue-600" />
            <span>⚡ DuckDB / In-Memory SQL Query Studio</span>
          </div>
          {sqlStudioOpen ? <ChevronDown className="w-4 h-4 text-slate-500" /> : <ChevronRight className="w-4 h-4 text-slate-500" />}
        </button>

        {sqlStudioOpen && (
          <div className="p-5 border-t border-slate-100 space-y-4">
            <p className="text-xs text-slate-500">
              Query your active dataset in-memory using ultra-fast analytical SQL (tables aliased as <code>dataset</code> or <code>df</code>).
            </p>

            <div className="flex flex-col sm:flex-row gap-3">
              <div className="sm:w-64">
                <label className="text-[11px] font-bold text-slate-700 block mb-1">Quick Query Templates:</label>
                <select
                  value={selectedPreset}
                  onChange={e => {
                    setSelectedPreset(e.target.value);
                    const found = sampleQueries.find(q => q.title === e.target.value);
                    if (found) setSqlCode(found.sql);
                    else if (e.target.value === 'Custom Query') setSqlCode('SELECT * FROM dataset LIMIT 10;');
                  }}
                  className="w-full bg-slate-50 border border-slate-200 rounded-lg p-2 text-xs"
                >
                  {sampleQueries.map(q => <option key={q.title} value={q.title}>{q.title}</option>)}
                  <option value="Custom Query">✍️ Custom SQL Query</option>
                </select>
              </div>

              <div className="flex-1">
                <label className="text-[11px] font-bold text-slate-700 block mb-1">SQL Editor:</label>
                <textarea
                  value={sqlCode}
                  onChange={e => setSqlCode(e.target.value)}
                  rows={3}
                  className="w-full font-mono text-xs bg-slate-900 text-slate-100 p-3 rounded-lg border border-slate-700 focus:outline-none focus:ring-2 focus:ring-blue-500"
                />
              </div>
            </div>

            <button
              type="button"
              onClick={handleRunSQL}
              className="py-2 px-4 bg-blue-600 hover:bg-blue-700 text-white font-bold rounded-lg text-xs flex items-center gap-1.5 shadow-sm shadow-blue-500/20 transition-all cursor-pointer"
            >
              <Play className="w-3.5 h-3.5 fill-current" />
              <span>▶ Run SQL Query</span>
            </button>

            {sqlResults && (
              <div className="space-y-3 pt-3 border-t border-slate-100">
                {sqlResults.success ? (
                  <div className="space-y-3">
                    <div className="flex items-center justify-between bg-emerald-50 border border-emerald-200 rounded-xl p-3 text-xs text-emerald-900">
                      <span>
                        ⚡ Query executed successfully in <b>{sqlResults.executionTimeMs.toFixed(2)} ms</b> — returned <b>{sqlResults.rowCount.toLocaleString()}</b> records.
                      </span>
                      <button
                        type="button"
                        onClick={downloadSqlCsv}
                        className="py-1 px-2.5 bg-white border border-emerald-300 text-emerald-800 rounded font-bold text-[11px] flex items-center gap-1 shadow-xs cursor-pointer"
                      >
                        <Download className="w-3 h-3" />
                        Download CSV
                      </button>
                    </div>

                    <div className="flex gap-2 border-b border-slate-200 pb-2 text-xs font-bold">
                      <button
                        type="button"
                        onClick={() => setSqlViewTab('table')}
                        className={`py-1 px-3 rounded ${sqlViewTab === 'table' ? 'bg-blue-50 text-blue-700' : 'text-slate-500'}`}
                      >
                        📋 Result Table
                      </button>
                      <button
                        type="button"
                        onClick={() => setSqlViewTab('chart')}
                        className={`py-1 px-3 rounded ${sqlViewTab === 'chart' ? 'bg-blue-50 text-blue-700' : 'text-slate-500'}`}
                      >
                        📈 Auto-Chart
                      </button>
                    </div>

                    {sqlViewTab === 'table' ? (
                      <div className="bg-white border border-slate-200 rounded-xl overflow-hidden overflow-x-auto shadow-sm max-h-60">
                        <table className="w-full text-left text-xs border-collapse">
                          <thead className="sticky top-0 bg-slate-50 border-b border-slate-200 font-bold text-slate-700">
                            <tr>
                              {Object.keys(sqlResults.rows[0] || {}).map(k => (
                                <th key={k} className="p-2.5">{k}</th>
                              ))}
                            </tr>
                          </thead>
                          <tbody className="divide-y divide-slate-100 font-mono text-[11px]">
                            {sqlResults.rows.slice(0, 20).map((r, i) => (
                              <tr key={i} className="hover:bg-slate-50/80">
                                {Object.values(r).map((v: any, j) => (
                                  <td key={j} className="p-2.5">{String(v ?? '')}</td>
                                ))}
                              </tr>
                            ))}
                          </tbody>
                        </table>
                      </div>
                    ) : (
                      <div className="h-60 bg-white border border-slate-200 rounded-xl p-4">
                        {(() => {
                          const cols = Object.keys(sqlResults.rows[0] || {});
                          if (cols.length >= 2) {
                            const lblCol = cols[0];
                            const valCol = cols[1];
                            const labels = sqlResults.rows.slice(0, 10).map(r => String(r[lblCol]));
                            const data = sqlResults.rows.slice(0, 10).map(r => Number(r[valCol]) || 0);

                            return (
                              <Bar
                                data={{
                                  labels,
                                  datasets: [{ label: valCol, data, backgroundColor: '#0284C7', borderRadius: 6 }]
                                }}
                                options={{ responsive: true, maintainAspectRatio: false }}
                              />
                            );
                          }
                          return <div className="text-xs text-slate-400 text-center py-20">Need at least 2 columns to auto-chart.</div>;
                        })()}
                      </div>
                    )}
                  </div>
                ) : (
                  <div className="bg-rose-50 border border-rose-200 text-rose-800 p-3 rounded-xl text-xs">
                    ❌ {sqlResults.error}
                  </div>
                )}
              </div>
            )}
          </div>
        )}
      </div>

      {/* Conversation Stream */}
      <div className="space-y-4">
        {chatMessages.map((msg, idx) => {
          const isUser = msg.role === 'user';
          return (
            <div
              key={idx}
              className={`flex gap-3 ${isUser ? 'justify-end' : 'justify-start'}`}
            >
              {!isUser && (
                <div className="w-8 h-8 rounded-full bg-blue-50 border border-blue-200 flex items-center justify-center text-blue-700 shrink-0 mt-1 shadow-xs">
                  <Sparkles className="w-4 h-4" />
                </div>
              )}

              <div
                className={`max-w-[85%] rounded-2xl p-4 shadow-sm text-xs leading-relaxed ${
                  isUser
                    ? 'bg-blue-600 text-white rounded-tr-none'
                    : 'bg-white border border-slate-200/90 text-slate-800 rounded-tl-none space-y-3'
                }`}
              >
                <div
                  className="prose prose-xs max-w-none"
                  dangerouslySetInnerHTML={{
                    __html: msg.content
                      .replace(/### (.*?)\n/g, '<h3 class="font-extrabold text-slate-900 text-sm mb-1.5 mt-0">$1</h3>')
                      .replace(/\*\*(.*?)\*\*/g, '<b>$1</b>')
                      .replace(/\*(.*?)\*/g, '<i>$1</i>')
                      .replace(/`([^`]+)`/g, '<code class="bg-slate-100 px-1 py-0.5 rounded text-[11px] text-blue-800 font-mono">$1</code>')
                      .replace(/\n\n/g, '<br/><br/>')
                  }}
                />

                {/* Optional Chart */}
                {msg.chart && (
                  <div className="bg-slate-50 border border-slate-200 rounded-xl p-3 h-56 mt-2">
                    {msg.chart.type === 'line' ? (
                      <Line data={msg.chart} options={{ responsive: true, maintainAspectRatio: false }} />
                    ) : (
                      <Bar data={msg.chart} options={{ responsive: true, maintainAspectRatio: false }} />
                    )}
                  </div>
                )}

                {/* Optional Preview Table */}
                {msg.table && msg.table.length > 0 && (
                  <div className="border border-slate-200 rounded-xl overflow-hidden overflow-x-auto mt-2">
                    <table className="w-full text-left text-[11px] border-collapse">
                      <thead>
                        <tr className="bg-slate-50 font-bold text-slate-700 border-b border-slate-200">
                          {Object.keys(msg.table[0]).map(k => (
                            <th key={k} className="p-2">{k}</th>
                          ))}
                        </tr>
                      </thead>
                      <tbody className="divide-y divide-slate-100">
                        {msg.table.map((r, i) => (
                          <tr key={i} className="hover:bg-slate-50">
                            {Object.values(r).map((v: any, j) => (
                              <td key={j} className="p-2 font-mono">{String(v ?? '')}</td>
                            ))}
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                )}

                {/* Optional Code Snippet */}
                {msg.code && (
                  <div className="bg-slate-900 text-slate-100 rounded-xl p-3 text-[11px] font-mono mt-2 overflow-x-auto">
                    <div className="flex items-center gap-1.5 text-slate-400 text-[10px] mb-1">
                      <Code2 className="w-3 h-3" />
                      <span>SQL / Python Query:</span>
                    </div>
                    <pre className="whitespace-pre-wrap">{msg.code}</pre>
                  </div>
                )}
              </div>

              {isUser && (
                <div className="w-8 h-8 rounded-full bg-slate-900 text-white flex items-center justify-center shrink-0 mt-1 shadow-xs">
                  <User className="w-4 h-4" />
                </div>
              )}
            </div>
          );
        })}

        {loading && (
          <div className="flex gap-3 items-center text-xs text-slate-500">
            <div className="w-8 h-8 rounded-full bg-blue-50 border border-blue-200 flex items-center justify-center text-blue-600 animate-spin">
              <Sparkles className="w-4 h-4" />
            </div>
            <span>Analyzing your data and synthesizing answer...</span>
          </div>
        )}
      </div>

      {/* Input Box */}
      <div className="sticky bottom-4 pt-2">
        <form
          onSubmit={e => {
            e.preventDefault();
            handleSendMessage();
          }}
          className="flex items-center gap-2 bg-white border border-slate-300 rounded-2xl p-2 shadow-lg shadow-slate-900/5 focus-within:border-blue-500 focus-within:ring-2 focus-within:ring-blue-500/20 transition-all"
        >
          <input
            type="text"
            value={inputText}
            onChange={e => setInputText(e.target.value)}
            placeholder="Ask something about your data..."
            className="flex-1 px-3 py-1.5 text-xs text-slate-900 placeholder:text-slate-400 focus:outline-none"
          />
          <button
            type="submit"
            disabled={!inputText.trim() || loading}
            className="w-8 h-8 rounded-xl bg-blue-600 hover:bg-blue-700 disabled:opacity-40 text-white flex items-center justify-center shrink-0 transition-all cursor-pointer shadow-sm shadow-blue-500/20"
          >
            <Send className="w-3.5 h-3.5" />
          </button>
        </form>

        {chatMessages.length > 1 && (
          <div className="text-center mt-2">
            <button
              type="button"
              onClick={() => {
                setChatMessages([
                  {
                    role: 'assistant',
                    content: `Hello! I'm your **AI Data Analyst**. I've analyzed **${datasetTitle}** and I'm ready to answer any questions about your data.\n\nTry clicking one of the sample questions above or ask me anything!`
                  }
                ]);
              }}
              className="text-[11px] text-slate-400 hover:text-slate-600 flex items-center gap-1 mx-auto cursor-pointer"
            >
              <RotateCcw className="w-3 h-3" />
              <span>Reset Conversation</span>
            </button>
          </div>
        )}
      </div>
    </div>
  );
};
