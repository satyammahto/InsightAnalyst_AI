import React, { useState, useMemo } from 'react';
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
  Legend,
  Filler
} from 'chart.js';
import { Bar, Line, Scatter, Pie } from 'react-chartjs-2';
import { Search, ChevronDown, ChevronRight, Palette } from 'lucide-react';
import { ColumnClassification, DataRow, MLResults } from '../../types';
import { cleanName } from '../../utils/cleanName';
import { filterDatasetByNL } from '../../utils/queryEngine';

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
  Legend,
  Filler
);

interface ChartsTabProps {
  df: DataRow[];
  colTypes: ColumnClassification;
  mlResults: MLResults;
  isSimpleMode: boolean;
}

export const ChartsTab: React.FC<ChartsTabProps> = ({
  df,
  colTypes,
  mlResults,
  isSimpleMode
}) => {
  const [nlQuery, setNlQuery] = useState('');
  const [customBuilderOpen, setCustomBuilderOpen] = useState(false);

  // Custom Chart Builder state
  const allColumns = df.length > 0 ? Object.keys(df[0]) : [];
  const numericColumns = colTypes.numeric;
  const categoricalColumns = colTypes.categorical;

  const [chartType, setChartType] = useState<string>('Bar Chart (Aggregated)');
  const [xAxis, setXAxis] = useState<string>(allColumns[0] || '');
  const [yAxis, setYAxis] = useState<string>(numericColumns[0] || 'None');
  const [colorGroup, setColorGroup] = useState<string>('None');
  const [aggMethod, setAggMethod] = useState<'Average' | 'Sum' | 'Count' | 'Median'>('Average');

  // Filter dataset by NL
  const filterResult = useMemo(() => {
    return filterDatasetByNL(df, colTypes, nlQuery);
  }, [df, colTypes, nlQuery]);

  const activeDisplayDf = filterResult.filteredRows;

  // 1. Numeric Distribution Chart Data
  const numDistChart = useMemo(() => {
    if (numericColumns.length === 0) return null;
    const col = numericColumns[0];
    const vals = activeDisplayDf.map(r => Number(r[col])).filter(v => !isNaN(v));
    if (vals.length === 0) return null;

    const min = Math.min(...vals);
    const max = Math.max(...vals);
    const numBins = Math.min(8, Math.max(4, Math.floor(vals.length / 3)));
    const binSize = (max - min) / numBins || 1;

    const bins = new Array(numBins).fill(0);
    const labels: string[] = [];
    for (let i = 0; i < numBins; i++) {
      const bStart = min + i * binSize;
      const bEnd = bStart + binSize;
      labels.push(`${bStart.toFixed(1)} - ${bEnd.toFixed(1)}`);
    }

    vals.forEach(v => {
      const bIdx = Math.min(numBins - 1, Math.floor((v - min) / binSize));
      bins[bIdx]++;
    });

    return {
      title: `${cleanName(col)} Distribution`,
      data: {
        labels,
        datasets: [
          {
            label: 'Record Frequency',
            data: bins,
            backgroundColor: '#0284C7',
            borderRadius: 6
          }
        ]
      }
    };
  }, [activeDisplayDf, numericColumns]);

  // 2. Categorical Counts Chart Data
  const catCountsChart = useMemo(() => {
    if (categoricalColumns.length === 0) return null;
    const col = categoricalColumns[0];
    const counts: Record<string, number> = {};
    activeDisplayDf.forEach(r => {
      const v = String(r[col] ?? 'Unknown');
      counts[v] = (counts[v] || 0) + 1;
    });

    const sorted = Object.entries(counts).sort((a, b) => b[1] - a[1]).slice(0, 8);
    return {
      title: `${cleanName(col)} Breakdown`,
      data: {
        labels: sorted.map(s => s[0]),
        datasets: [
          {
            label: 'Count',
            data: sorted.map(s => s[1]),
            backgroundColor: '#059669',
            borderRadius: 6
          }
        ]
      }
    };
  }, [activeDisplayDf, categoricalColumns]);

  // 3. Date Trend or 2nd Numeric
  const dateTrendChart = useMemo(() => {
    if (colTypes.date.length > 0 && numericColumns.length > 0) {
      const dtCol = colTypes.date[0];
      const numCol = numericColumns[0];
      const dateMap: Record<string, { sum: number; count: number }> = {};

      activeDisplayDf.forEach(r => {
        const dt = String(r[dtCol] ?? '').slice(0, 10);
        if (dt) {
          if (!dateMap[dt]) dateMap[dt] = { sum: 0, count: 0 };
          dateMap[dt].sum += Number(r[numCol]) || 0;
          dateMap[dt].count++;
        }
      });

      const sortedDates = Object.entries(dateMap).sort((a, b) => a[0].localeCompare(b[0]));
      return {
        title: `${cleanName(numCol)} Timeline (${cleanName(dtCol)})`,
        data: {
          labels: sortedDates.map(s => s[0]),
          datasets: [
            {
              label: cleanName(numCol),
              data: sortedDates.map(s => Number((s[1].sum / s[1].count).toFixed(2))),
              borderColor: '#7C3AED',
              backgroundColor: 'rgba(124, 58, 237, 0.1)',
              tension: 0.3,
              fill: true
            }
          ]
        }
      };
    } else if (numericColumns.length >= 2) {
      const col2 = numericColumns[1];
      const vals2 = activeDisplayDf.map(r => Number(r[col2])).filter(v => !isNaN(v));
      const min = Math.min(...vals2);
      const max = Math.max(...vals2);
      const numBins = 6;
      const binSize = (max - min) / numBins || 1;
      const bins = new Array(numBins).fill(0);
      const labels: string[] = [];

      for (let i = 0; i < numBins; i++) {
        const bStart = min + i * binSize;
        const bEnd = bStart + binSize;
        labels.push(`${bStart.toFixed(1)} - ${bEnd.toFixed(1)}`);
      }
      vals2.forEach(v => {
        const idx = Math.min(numBins - 1, Math.floor((v - min) / binSize));
        bins[idx]++;
      });

      return {
        title: `${cleanName(col2)} Distribution`,
        data: {
          labels,
          datasets: [
            {
              label: 'Frequency',
              data: bins,
              backgroundColor: '#EA580C',
              borderRadius: 6
            }
          ]
        }
      };
    }
    return null;
  }, [activeDisplayDf, colTypes.date, numericColumns]);

  // 4. Custom Chart Builder Output
  const customChartData = useMemo(() => {
    if (!xAxis || activeDisplayDf.length === 0) return null;

    if (chartType === 'Scatter Plot' || chartType === '3D Scatter Plot') {
      const targetY = yAxis !== 'None' ? yAxis : (numericColumns[0] || xAxis);
      return {
        type: 'scatter',
        data: {
          datasets: [
            {
              label: `${cleanName(xAxis)} vs ${cleanName(targetY)}`,
              data: activeDisplayDf.map(r => ({
                x: Number(r[xAxis]) || 0,
                y: Number(r[targetY]) || 0
              })),
              backgroundColor: '#0284C7'
            }
          ]
        }
      };
    }

    if (chartType === 'Pie / Donut Chart') {
      const counts: Record<string, number> = {};
      activeDisplayDf.forEach(r => {
        const k = String(r[xAxis] ?? 'Other');
        counts[k] = (counts[k] || 0) + 1;
      });
      const topEntries = Object.entries(counts).slice(0, 6);
      return {
        type: 'pie',
        data: {
          labels: topEntries.map(e => e[0]),
          datasets: [
            {
              data: topEntries.map(e => e[1]),
              backgroundColor: ['#0284C7', '#059669', '#7C3AED', '#EA580C', '#E11D48', '#0D9488']
            }
          ]
        }
      };
    }

    // Default: Bar Chart (Aggregated) or Line
    const groups: Record<string, number[]> = {};
    activeDisplayDf.forEach(r => {
      const key = String(r[xAxis] ?? 'Unknown');
      if (!groups[key]) groups[key] = [];
      if (yAxis !== 'None') {
        const val = Number(r[yAxis]);
        if (!isNaN(val)) groups[key].push(val);
      } else {
        groups[key].push(1);
      }
    });

    const labels = Object.keys(groups).slice(0, 10);
    const values = labels.map(lbl => {
      const arr = groups[lbl];
      if (yAxis === 'None' || aggMethod === 'Count') return arr.length;
      if (arr.length === 0) return 0;
      if (aggMethod === 'Sum') return arr.reduce((a, b) => a + b, 0);
      if (aggMethod === 'Median') {
        const sorted = [...arr].sort((a, b) => a - b);
        return sorted[Math.floor(sorted.length / 2)];
      }
      return Number((arr.reduce((a, b) => a + b, 0) / arr.length).toFixed(2));
    });

    return {
      type: chartType === 'Line Chart' ? 'line' : 'bar',
      data: {
        labels,
        datasets: [
          {
            label: yAxis !== 'None' ? `${aggMethod} of ${cleanName(yAxis)}` : 'Record Count',
            data: values,
            backgroundColor: chartType === 'Line Chart' ? 'rgba(2, 132, 199, 0.1)' : '#0284C7',
            borderColor: '#0284C7',
            borderRadius: 6,
            tension: 0.3
          }
        ]
      }
    };
  }, [activeDisplayDf, chartType, xAxis, yAxis, aggMethod, numericColumns]);

  return (
    <div className="space-y-6">
      {/* Title */}
      <div>
        <h2 className="text-2xl font-extrabold text-slate-900 tracking-tight">Visual Charts & Explorer</h2>
        <p className="text-sm text-slate-500 mt-1">
          Visual representations of your dataset with plain-English explanations.
        </p>
      </div>

      {/* Natural Language Live Data Filter */}
      <div className="space-y-2">
        <label className="text-xs font-bold text-slate-800 block">
          Ask about your data:
        </label>
        <div className="relative">
          <Search className="w-4 h-4 text-slate-400 absolute left-3.5 top-1/2 -translate-y-1/2" />
          <input
            type="text"
            value={nlQuery}
            onChange={e => setNlQuery(e.target.value)}
            placeholder="Type in plain English (e.g. 'show orders above 5000', 'status shipped', 'discount > 0.1')..."
            className="w-full pl-10 pr-4 py-2.5 bg-white border border-slate-200 rounded-xl text-xs text-slate-900 placeholder:text-slate-400 focus:outline-none focus:ring-2 focus:ring-blue-500/20 focus:border-blue-500 shadow-sm"
          />
        </div>
      </div>

      {/* Filter Feedback Banner */}
      {filterResult.appliedRules.length > 0 && (
        <div className="bg-blue-50 border border-blue-200 rounded-xl p-3 px-4 flex items-center justify-between text-xs">
          <div>
            <b className="text-blue-800">FILTER APPLIED:</b>{' '}
            <span className="text-slate-800">{filterResult.summary}</span>
          </div>
          <span className="bg-white border border-blue-200 text-blue-700 font-bold px-2.5 py-0.5 rounded-full">
            Showing {filterResult.filteredCount.toLocaleString()} of {filterResult.originalCount.toLocaleString()} rows ({filterResult.percentage}%)
          </span>
        </div>
      )}

      {/* Custom Interactive Chart Builder Expander */}
      <div className="border border-slate-200 rounded-xl overflow-hidden bg-white shadow-sm">
        <button
          type="button"
          onClick={() => setCustomBuilderOpen(!customBuilderOpen)}
          className="w-full p-4 bg-slate-50/80 hover:bg-slate-100/80 flex items-center justify-between text-xs font-bold text-slate-800 transition-colors"
        >
          <div className="flex items-center gap-2">
            <Palette className="w-4 h-4 text-blue-600" />
            <span>🎨 Custom Chart Builder</span>
          </div>
          {customBuilderOpen ? <ChevronDown className="w-4 h-4 text-slate-500" /> : <ChevronRight className="w-4 h-4 text-slate-500" />}
        </button>

        {customBuilderOpen && (
          <div className="p-5 border-t border-slate-100 space-y-4">
            <p className="text-xs text-slate-500">
              Build any chart by choosing your columns and chart type:
            </p>

            <div className="grid grid-cols-1 md:grid-cols-3 gap-3">
              <div>
                <label className="text-[11px] font-bold text-slate-700 block mb-1">Choose Chart Type</label>
                <select
                  value={chartType}
                  onChange={e => setChartType(e.target.value)}
                  className="w-full bg-slate-50 border border-slate-200 rounded-lg p-2 text-xs"
                >
                  <option value="Bar Chart (Aggregated)">Bar Chart (Aggregated)</option>
                  <option value="Line Chart">Line Chart</option>
                  <option value="Scatter Plot">Scatter Plot</option>
                  <option value="Pie / Donut Chart">Pie / Donut Chart</option>
                  <option value="3D Scatter Plot">3D / Bubble Scatter</option>
                </select>
              </div>

              <div>
                <label className="text-[11px] font-bold text-slate-700 block mb-1">X-Axis Feature</label>
                <select
                  value={xAxis}
                  onChange={e => setXAxis(e.target.value)}
                  className="w-full bg-slate-50 border border-slate-200 rounded-lg p-2 text-xs"
                >
                  {allColumns.map(c => <option key={c} value={c}>{cleanName(c)}</option>)}
                </select>
              </div>

              <div>
                <label className="text-[11px] font-bold text-slate-700 block mb-1">Y-Axis Feature</label>
                <select
                  value={yAxis}
                  onChange={e => setYAxis(e.target.value)}
                  className="w-full bg-slate-50 border border-slate-200 rounded-lg p-2 text-xs"
                >
                  <option value="None">None (Count Only)</option>
                  {allColumns.map(c => <option key={c} value={c}>{cleanName(c)}</option>)}
                </select>
              </div>
            </div>

            <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
              <div>
                <label className="text-[11px] font-bold text-slate-700 block mb-1">Calculation (For Bar Charts)</label>
                <select
                  value={aggMethod}
                  onChange={e => setAggMethod(e.target.value as any)}
                  className="w-full bg-slate-50 border border-slate-200 rounded-lg p-2 text-xs"
                >
                  <option value="Average">Average</option>
                  <option value="Sum">Sum</option>
                  <option value="Count">Count</option>
                  <option value="Median">Median</option>
                </select>
              </div>

              <div>
                <label className="text-[11px] font-bold text-slate-700 block mb-1">Group by Color (Optional)</label>
                <select
                  value={colorGroup}
                  onChange={e => setColorGroup(e.target.value)}
                  className="w-full bg-slate-50 border border-slate-200 rounded-lg p-2 text-xs"
                >
                  <option value="None">None</option>
                  {allColumns.map(c => <option key={c} value={c}>{cleanName(c)}</option>)}
                </select>
              </div>
            </div>

            {/* Custom Chart View */}
            {customChartData && (
              <div className="pt-4 border-t border-slate-100">
                <div className="bg-slate-50 rounded-xl p-4 h-72 flex items-center justify-center">
                  {customChartData.type === 'scatter' ? (
                    <Scatter
                      data={customChartData.data as any}
                      options={{ responsive: true, maintainAspectRatio: false }}
                    />
                  ) : customChartData.type === 'pie' ? (
                    <Pie
                      data={customChartData.data as any}
                      options={{ responsive: true, maintainAspectRatio: false }}
                    />
                  ) : customChartData.type === 'line' ? (
                    <Line
                      data={customChartData.data as any}
                      options={{ responsive: true, maintainAspectRatio: false }}
                    />
                  ) : (
                    <Bar
                      data={customChartData.data as any}
                      options={{ responsive: true, maintainAspectRatio: false }}
                    />
                  )}
                </div>
              </div>
            )}
          </div>
        )}
      </div>

      {/* 4 Automated Dynamic Visual Charts Grid */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-5">
        {/* Chart 1: Numeric Distribution */}
        {numDistChart && (
          <div className="bg-white border border-slate-200/90 rounded-2xl p-5 shadow-sm">
            <h4 className="text-sm font-extrabold text-slate-900 mb-1">{numDistChart.title}</h4>
            <p className="text-xs text-slate-400 mb-3">Continuous distribution across standard bins</p>
            <div className="h-60">
              <Bar
                data={numDistChart.data}
                options={{
                  responsive: true,
                  maintainAspectRatio: false,
                  plugins: { legend: { display: false } }
                }}
              />
            </div>
          </div>
        )}

        {/* Chart 2: Categorical Counts */}
        {catCountsChart && (
          <div className="bg-white border border-slate-200/90 rounded-2xl p-5 shadow-sm">
            <h4 className="text-sm font-extrabold text-slate-900 mb-1">{catCountsChart.title}</h4>
            <p className="text-xs text-slate-400 mb-3">Top categories ranked by volume</p>
            <div className="h-60">
              <Bar
                data={catCountsChart.data}
                options={{
                  responsive: true,
                  maintainAspectRatio: false,
                  plugins: { legend: { display: false } }
                }}
              />
            </div>
          </div>
        )}

        {/* Chart 3: Date Trend or 2nd Numeric */}
        {dateTrendChart && (
          <div className="bg-white border border-slate-200/90 rounded-2xl p-5 shadow-sm">
            <h4 className="text-sm font-extrabold text-slate-900 mb-1">{dateTrendChart.title}</h4>
            <p className="text-xs text-slate-400 mb-3">Timeline trend and chronological variance</p>
            <div className="h-60">
              <Line
                data={dateTrendChart.data}
                options={{
                  responsive: true,
                  maintainAspectRatio: false,
                  plugins: { legend: { display: false } }
                }}
              />
            </div>
          </div>
        )}

        {/* Chart 4: Correlation Matrix Heatmap */}
        {mlResults.correlation && mlResults.correlation.columns.length >= 2 && (
          <div className="bg-white border border-slate-200/90 rounded-2xl p-5 shadow-sm">
            <h4 className="text-sm font-extrabold text-slate-900 mb-1">Correlation Heatmap</h4>
            <p className="text-xs text-slate-400 mb-3">Pearson relationship matrix across metrics</p>
            <div className="overflow-x-auto">
              <table className="w-full text-center text-xs border-collapse">
                <thead>
                  <tr>
                    <th className="p-2 text-left text-slate-400"></th>
                    {mlResults.correlation.columns.map(c => (
                      <th key={c} className="p-2 text-slate-600 font-bold truncate max-w-[80px]" title={cleanName(c)}>
                        {cleanName(c).slice(0, 8)}..
                      </th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {mlResults.correlation.columns.map(rCol => (
                    <tr key={rCol}>
                      <td className="p-2 text-left font-bold text-slate-700 truncate max-w-[80px]" title={cleanName(rCol)}>
                        {cleanName(rCol).slice(0, 8)}..
                      </td>
                      {mlResults.correlation!.columns.map(cCol => {
                        const val = mlResults.correlation!.matrix[rCol]?.[cCol] ?? 0;
                        const isPos = val > 0;
                        const absVal = Math.abs(val);
                        const bgIntensity = Math.min(0.8, absVal * 0.8 + 0.1);
                        const bgColor = rCol === cCol
                          ? 'rgba(2, 132, 199, 0.15)'
                          : isPos
                          ? `rgba(2, 132, 199, ${bgIntensity})`
                          : `rgba(225, 29, 72, ${bgIntensity})`;
                        const textColor = absVal > 0.5 && rCol !== cCol ? '#FFFFFF' : '#0F172A';

                        return (
                          <td
                            key={cCol}
                            style={{ backgroundColor: bgColor, color: textColor }}
                            className="p-2 rounded font-mono text-[11px] font-bold"
                          >
                            {val.toFixed(2)}
                          </td>
                        );
                      })}
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        )}
      </div>
    </div>
  );
};
