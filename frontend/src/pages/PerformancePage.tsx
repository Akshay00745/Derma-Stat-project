import React, { useState } from 'react';
import {
  Activity,
  Layers,
  Award,
  PieChart,
  BarChart3,
  CheckCircle2,
  HelpCircle,
} from 'lucide-react';
import { PerformanceMetrics } from '../types/api';

interface PerformancePageProps {
  performance: PerformanceMetrics | null;
  isLoading: boolean;
}

export const PerformancePage: React.FC<PerformancePageProps> = ({
  performance,
  isLoading,
}) => {
  const [showNormalized, setShowNormalized] = useState(false);

  if (isLoading || !performance) {
    return (
      <div className="p-16 max-w-4xl mx-auto text-center space-y-4">
        <div className="w-10 h-10 rounded-full border-4 border-blue-200 border-t-blue-600 animate-spin mx-auto"></div>
        <p className="text-sm font-semibold text-slate-700">Loading benchmark performance data...</p>
      </div>
    );
  }

  // Calculate normalized confusion matrix
  const rowSums = performance.confusion_matrix.map((row) =>
    row.reduce((acc, v) => acc + v, 0)
  );

  return (
    <div className="p-8 max-w-6xl mx-auto space-y-8">
      {/* Title */}
      <div>
        <h2 className="text-2xl font-bold text-slate-900">Model Performance Dashboard</h2>
        <p className="text-xs text-slate-500 mt-1">
          Independent evaluation on the held-out HAM10000 test split (1,494 images, 0% patient leakage)
        </p>
      </div>

      {/* 4 Overall Performance Cards (Panel 6 top) */}
      <section className="space-y-3">
        <h3 className="text-xs font-bold text-slate-800 uppercase tracking-wider">
          Overall Performance
        </h3>

        <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
          {/* Card 1: Accuracy */}
          <div className="bg-white rounded-3xl p-5 border border-slate-200/80 shadow-xs space-y-1">
            <span className="text-[11px] font-semibold text-slate-400 uppercase tracking-wider">
              Accuracy
            </span>
            <div className="text-3xl font-extrabold text-blue-600">
              {performance.test_accuracy_pct}%
            </div>
            <p className="text-[11px] text-slate-400">1,136 of 1,494 correct</p>
          </div>

          {/* Card 2: Macro F1 */}
          <div className="bg-white rounded-3xl p-5 border border-slate-200/80 shadow-xs space-y-1">
            <span className="text-[11px] font-semibold text-slate-400 uppercase tracking-wider">
              Macro F1-Score
            </span>
            <div className="text-3xl font-extrabold text-emerald-600">
              {performance.macro_f1_pct}%
            </div>
            <p className="text-[11px] text-slate-400">Unweighted mean across 7 classes</p>
          </div>

          {/* Card 3: Precision (avg) */}
          <div className="bg-white rounded-3xl p-5 border border-slate-200/80 shadow-xs space-y-1">
            <span className="text-[11px] font-semibold text-slate-400 uppercase tracking-wider">
              Precision (avg)
            </span>
            <div className="text-3xl font-extrabold text-slate-800">
              {performance.macro_precision_pct}%
            </div>
            <p className="text-[11px] text-slate-400">Low false positive rate</p>
          </div>

          {/* Card 4: Recall (avg) */}
          <div className="bg-white rounded-3xl p-5 border border-slate-200/80 shadow-xs space-y-1">
            <span className="text-[11px] font-semibold text-slate-400 uppercase tracking-wider">
              Recall (avg)
            </span>
            <div className="text-3xl font-extrabold text-purple-600">
              {performance.macro_recall_pct}%
            </div>
            <p className="text-[11px] text-slate-400">High diagnostic sensitivity</p>
          </div>
        </div>
      </section>

      {/* Grid: Per-Class Metrics Table + Confusion Matrix (Panel 6 bottom) */}
      <div className="grid lg:grid-cols-2 gap-8">
        {/* Left: Per-Class Metrics */}
        <section className="bg-white rounded-3xl p-6 border border-slate-200/80 shadow-xs space-y-4">
          <div className="flex items-center justify-between border-b border-slate-100 pb-3">
            <h4 className="text-xs font-bold text-slate-800 uppercase tracking-wider">
              Per-Class Metrics
            </h4>
            <span className="text-[11px] text-slate-400 font-medium">Class Weight Balanced</span>
          </div>

          <div className="overflow-x-auto">
            <table className="w-full text-left text-xs">
              <thead>
                <tr className="border-b border-slate-100 text-slate-400 font-semibold uppercase text-[10px]">
                  <th className="py-2">Class</th>
                  <th className="py-2 text-right">Precision</th>
                  <th className="py-2 text-right">Recall</th>
                  <th className="py-2 text-right">F1-Score</th>
                  <th className="py-2 text-right">Support</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-slate-100 font-medium">
                {performance.per_class.map((c) => (
                  <tr key={c.code} className="hover:bg-slate-50 transition-colors">
                    <td className="py-2.5 font-bold uppercase text-slate-800">
                      {c.code}
                      <span className="text-[10px] block font-normal text-slate-400 lowercase capitalize">
                        {c.name.split(' ')[0]}
                      </span>
                    </td>
                    <td className="py-2.5 text-right font-mono">{c.precision_pct}%</td>
                    <td className="py-2.5 text-right font-mono font-semibold text-blue-600">
                      {c.recall_pct}%
                    </td>
                    <td className="py-2.5 text-right font-mono font-bold text-slate-900">
                      {c.f1_pct}%
                    </td>
                    <td className="py-2.5 text-right font-mono text-slate-400">{c.support}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </section>

        {/* Right: Confusion Matrix Heatmap */}
        <section className="bg-white rounded-3xl p-6 border border-slate-200/80 shadow-xs space-y-4">
          <div className="flex items-center justify-between border-b border-slate-100 pb-3">
            <h4 className="text-xs font-bold text-slate-800 uppercase tracking-wider">
              Confusion Matrix
            </h4>
            <div className="flex items-center gap-1 bg-slate-100 p-0.5 rounded-lg text-[10px] font-semibold">
              <button
                onClick={() => setShowNormalized(false)}
                className={`px-2 py-0.5 rounded-md transition-all ${
                  !showNormalized ? 'bg-white shadow-xs text-blue-600 font-bold' : 'text-slate-500'
                }`}
              >
                Counts
              </button>
              <button
                onClick={() => setShowNormalized(true)}
                className={`px-2 py-0.5 rounded-md transition-all ${
                  showNormalized ? 'bg-white shadow-xs text-blue-600 font-bold' : 'text-slate-500'
                }`}
              >
                Normalized %
              </button>
            </div>
          </div>

          {/* Matrix Grid */}
          <div className="overflow-x-auto pt-2">
            <div className="inline-block min-w-full">
              <div className="flex text-[10px] font-bold text-slate-400 mb-1 ml-10">
                <span className="text-center w-full">PREDICTED CATEGORY</span>
              </div>

              <div className="flex">
                {/* Y-Axis Label */}
                <div className="w-10 flex flex-col justify-center items-center text-[10px] font-bold text-slate-400 -rotate-90 select-none">
                  TRUE
                </div>

                <div className="flex-1">
                  {/* Column Headers */}
                  <div className="grid grid-cols-7 gap-1 mb-1 text-center font-bold text-[10px] text-slate-500">
                    {performance.class_labels.map((l) => (
                      <span key={l}>{l}</span>
                    ))}
                  </div>

                  {/* Matrix Rows */}
                  <div className="space-y-1">
                    {performance.confusion_matrix.map((row, rIdx) => {
                      const totalRow = rowSums[rIdx] || 1;
                      return (
                        <div key={rIdx} className="grid grid-cols-7 gap-1 items-center">
                          {row.map((val, cIdx) => {
                            const isDiagonal = rIdx === cIdx;
                            const normPct = Math.round((val / totalRow) * 100);
                            const intensity = Math.min(normPct / 100, 1);

                            return (
                              <div
                                key={cIdx}
                                className={`h-8 rounded-lg flex items-center justify-center font-mono text-[10px] transition-all ${
                                  isDiagonal
                                    ? 'bg-blue-600 text-white font-bold shadow-xs'
                                    : val > 0
                                    ? 'bg-blue-50 text-blue-900 font-medium'
                                    : 'bg-slate-50 text-slate-300'
                                }`}
                                style={{
                                  backgroundColor: isDiagonal
                                    ? `rgba(37, 99, 235, ${Math.max(intensity, 0.4)})`
                                    : val > 0
                                    ? `rgba(219, 234, 254, ${Math.max(intensity * 1.5, 0.2)})`
                                    : undefined,
                                }}
                                title={`True: ${performance.class_labels[rIdx]}, Pred: ${performance.class_labels[cIdx]} (${val} cases, ${normPct}%)`}
                              >
                                {showNormalized ? `${normPct}%` : val}
                              </div>
                            );
                          })}
                        </div>
                      );
                    })}
                  </div>
                </div>
              </div>
            </div>
          </div>

          <p className="text-[11px] text-slate-400 text-center pt-2">
            Diagonal blue cells represent correctly classified test samples.
          </p>
        </section>
      </div>
    </div>
  );
};
