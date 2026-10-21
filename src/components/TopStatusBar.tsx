import React from 'react';
import { Folder, Clock } from 'lucide-react';

interface TopStatusBarProps {
  isSimpleMode: boolean;
  datasetTitle: string;
  rowCount: number;
  colCount: number;
  latency: string;
}

export const TopStatusBar: React.FC<TopStatusBarProps> = ({
  isSimpleMode,
  datasetTitle,
  rowCount,
  colCount,
  latency
}) => {
  return (
    <div className="bg-white border border-slate-200/90 rounded-xl p-3 px-5 mb-5 shadow-sm shadow-slate-900/[0.02] flex items-center justify-between flex-wrap gap-3">
      <div className="flex items-center gap-3">
        {isSimpleMode ? (
          <span className="bg-blue-50 text-blue-700 border border-blue-200 text-xs font-bold px-2.5 py-1 rounded-full">
            ✨ Simple View
          </span>
        ) : (
          <span className="bg-slate-100 text-slate-700 border border-slate-200 text-xs font-bold px-2.5 py-1 rounded-md">
            🔬 Technical View
          </span>
        )}

        <div className="flex items-center gap-1.5 text-slate-900 font-extrabold text-sm">
          <Folder className="w-4 h-4 text-blue-600" />
          <span>{datasetTitle}</span>
        </div>

        <span className="text-xs text-slate-500 font-medium">
          ({rowCount.toLocaleString()} rows · {colCount} columns)
        </span>
      </div>

      <div className="flex items-center gap-2 text-xs">
        <Clock className="w-3.5 h-3.5 text-slate-400" />
        <span className="text-slate-500">AI Analyzed in:</span>
        <span className="text-slate-900 font-bold">{latency}</span>
      </div>
    </div>
  );
};
