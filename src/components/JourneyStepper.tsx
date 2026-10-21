import React from 'react';
import { CheckCircle2 } from 'lucide-react';

interface JourneyStepperProps {
  activeTab: string;
  setActiveTab: (tab: string) => void;
}

export const JourneyStepper: React.FC<JourneyStepperProps> = ({ activeTab, setActiveTab }) => {
  const steps = [
    { id: 'overview', label: '1. Overview 📋' },
    { id: 'quality', label: '2. Data Quality 🩺' },
    { id: 'insights', label: '3. Understand Insights 💡' },
    { id: 'charts', label: '4. Explore Charts 📊' },
    { id: 'patterns', label: '5. Patterns & Anomalies 🔍' },
    { id: 'ask', label: '6. Ask AI 💬' },
  ];

  return (
    <div className="bg-white border border-slate-200/90 rounded-xl p-3.5 px-5 mb-5 shadow-sm shadow-slate-900/[0.02] flex items-center justify-between flex-wrap gap-3">
      <div className="flex items-center gap-2">
        <span className="text-xs font-extrabold text-slate-900 tracking-tight">Your Data Journey:</span>
      </div>

      <div className="flex items-center gap-2 flex-wrap text-xs">
        {steps.map((step, idx) => {
          const isActive = activeTab === step.id;
          return (
            <React.Fragment key={step.id}>
              {idx > 0 && <span className="text-slate-300 font-bold select-none">→</span>}
              <button
                type="button"
                onClick={() => setActiveTab(step.id)}
                className={`inline-flex items-center gap-1.5 px-3 py-1 rounded-full text-xs font-bold transition-all cursor-pointer ${
                  isActive
                    ? 'bg-blue-50 text-blue-700 border border-blue-200 shadow-sm'
                    : 'bg-emerald-50 text-emerald-700 border border-emerald-200 hover:bg-emerald-100/70'
                }`}
              >
                {step.label}
              </button>
            </React.Fragment>
          );
        })}
      </div>

      <div className="flex items-center gap-1.5 bg-emerald-50 text-emerald-700 border border-emerald-200 text-xs font-bold px-3 py-1 rounded-full">
        <CheckCircle2 className="w-3.5 h-3.5 text-emerald-600" />
        <span>Ready to Explore ✔</span>
      </div>
    </div>
  );
};
