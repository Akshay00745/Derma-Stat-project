import React, { useState } from 'react';
import {
  Download,
  Home,
  CheckCircle2,
  AlertTriangle,
  FileCheck,
  Loader2,
  Sparkles,
} from 'lucide-react';
import { PageId } from '../components/Sidebar';
import { PredictionResult } from '../types/api';
import { api } from '../services/api';

interface DetailedPageProps {
  prediction: PredictionResult;
  onNavigate: (page: PageId) => void;
}

export const DetailedPage: React.FC<DetailedPageProps> = ({
  prediction,
  onNavigate,
}) => {
  const [downloading, setDownloading] = useState(false);
  const isReliable = !prediction.uncertainty.is_uncertain;

  const handleDownload = async () => {
    try {
      setDownloading(true);
      await api.downloadReport(prediction);
    } catch (err) {
      alert('Failed to download diagnostic report.');
    } finally {
      setDownloading(false);
    }
  };

  return (
    <div className="p-8 max-w-5xl mx-auto space-y-8">
      {/* Title */}
      <div>
        <h2 className="text-2xl font-bold text-slate-900">Detailed Analysis</h2>
        <p className="text-xs text-slate-500 mt-0.5">Clinical decision-support breakdown and reporting</p>
      </div>

      {/* 3 Top Summary Pills (Panel 5 top) */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        {/* Pill 1: Predicted Class */}
        <div className="bg-white rounded-2xl p-4 border border-slate-200/80 shadow-xs flex items-center justify-between">
          <div>
            <span className="text-[11px] font-semibold text-slate-400 uppercase tracking-wider block">
              Predicted Class
            </span>
            <span className="text-base font-extrabold text-blue-600 mt-0.5 block truncate">
              {prediction.predicted_name}
            </span>
          </div>
          <span className="text-xs font-mono font-bold bg-blue-50 text-blue-700 px-2 py-1 rounded-lg border border-blue-100">
            {prediction.predicted_code.toUpperCase()}
          </span>
        </div>

        {/* Pill 2: Confidence */}
        <div className="bg-white rounded-2xl p-4 border border-slate-200/80 shadow-xs flex items-center justify-between">
          <div>
            <span className="text-[11px] font-semibold text-slate-400 uppercase tracking-wider block">
              Confidence Score
            </span>
            <span className="text-xl font-extrabold text-slate-900 mt-0.5 block">
              {prediction.confidence_percentage}%
            </span>
          </div>
          <span className="w-2.5 h-2.5 rounded-full bg-blue-500"></span>
        </div>

        {/* Pill 3: Uncertainty */}
        <div className="bg-white rounded-2xl p-4 border border-slate-200/80 shadow-xs flex items-center justify-between">
          <div>
            <span className="text-[11px] font-semibold text-slate-400 uppercase tracking-wider block">
              Uncertainty Status
            </span>
            <span
              className={`text-base font-extrabold mt-0.5 block ${
                isReliable ? 'text-emerald-600' : 'text-amber-600'
              }`}
            >
              {prediction.uncertainty.level} ({prediction.uncertainty.status})
            </span>
          </div>
          {isReliable ? (
            <CheckCircle2 className="w-5 h-5 text-emerald-500" />
          ) : (
            <AlertTriangle className="w-5 h-5 text-amber-500" />
          )}
        </div>
      </div>

      {/* Middle Grid: Scores + Image Preview (Panel 5 middle) */}
      <div className="grid md:grid-cols-3 gap-6">
        {/* Left: Class-wise Scores (2 cols) */}
        <div className="md:col-span-2 bg-white rounded-3xl p-6 border border-slate-200/80 shadow-xs space-y-3">
          <div className="flex items-center justify-between border-b border-slate-100 pb-3">
            <h4 className="text-xs font-bold text-slate-800 uppercase tracking-wider">
              Class-wise Confidence Scores
            </h4>
            <span className="text-[11px] text-slate-400 font-medium">Full Taxonomy (7 Classes)</span>
          </div>

          <div className="space-y-2.5 pt-1">
            {prediction.all_probabilities.map((item) => {
              const isTop = item.code === prediction.predicted_code;
              return (
                <div key={item.code} className="space-y-1">
                  <div className="flex items-center justify-between text-xs">
                    <span className={`font-semibold ${isTop ? 'text-blue-700 font-bold' : 'text-slate-600'}`}>
                      {item.code.toUpperCase()} — {item.name}
                    </span>
                    <span className="font-mono font-bold text-slate-700">
                      {item.percentage}%
                    </span>
                  </div>
                  <div className="h-2 w-full bg-slate-100 rounded-full overflow-hidden">
                    <div
                      className={`h-full rounded-full ${isTop ? 'bg-blue-600' : 'bg-slate-300'}`}
                      style={{ width: `${Math.max(item.percentage, 1)}%` }}
                    ></div>
                  </div>
                </div>
              );
            })}
          </div>
        </div>

        {/* Right: Image Preview */}
        <div className="bg-white rounded-3xl p-6 border border-slate-200/80 shadow-xs flex flex-col items-center justify-center text-center">
          <span className="text-xs font-bold text-slate-700 uppercase tracking-wider mb-4">
            Image Preview
          </span>
          <div className="w-48 h-48 rounded-2xl overflow-hidden border border-slate-100 shadow-md bg-slate-100 mb-3">
            <img
              src={prediction.image_preview_b64 || `/samples/${prediction.predicted_code}.jpg`}
              alt="Lesion Preview"
              className="w-full h-full object-cover"
            />
          </div>
          <span className="text-xs text-slate-500 font-medium">
            {prediction.original_dimensions ? `${prediction.original_dimensions.width} × ${prediction.original_dimensions.height} px` : '224 × 224 px'}
          </span>
        </div>
      </div>

      {/* Key Takeaways Box (Panel 5 bottom) */}
      <div className="bg-white rounded-3xl p-6 border border-slate-200/80 shadow-xs space-y-3">
        <h4 className="text-xs font-bold text-slate-800 uppercase tracking-wider flex items-center gap-2">
          <FileCheck className="w-4 h-4 text-blue-600" />
          <span>Key Takeaways</span>
        </h4>
        <ul className="space-y-2 text-xs text-slate-600 list-disc pl-5">
          <li>
            The model predicted{' '}
            <span className="font-bold text-slate-800">
              {prediction.predicted_name} ({prediction.predicted_code.toUpperCase()})
            </span>{' '}
            with <span className="font-mono font-bold text-blue-600">{prediction.confidence_percentage}%</span>{' '}
            probability.
          </li>
          <li>
            Uncertainty is classified as{' '}
            <span className="font-bold text-slate-800">{prediction.uncertainty.level}</span>. {prediction.uncertainty.message}
          </li>
          <li>
            In HAM10000 benchmark tests, predictions meeting this threshold achieved an empirical accuracy of{' '}
            <span className="font-bold text-slate-800">90.61%</span>.
          </li>
        </ul>
      </div>

      {/* Actions: Download Report & Back to Home */}
      <div className="flex flex-wrap items-center gap-3">
        <button
          onClick={handleDownload}
          disabled={downloading}
          className="px-6 py-3 bg-blue-600 hover:bg-blue-700 text-white rounded-xl text-xs font-semibold shadow-md shadow-blue-500/20 transition-all flex items-center gap-2 active:scale-95 disabled:opacity-50"
        >
          {downloading ? (
            <>
              <Loader2 className="w-4 h-4 animate-spin" />
              <span>Exporting Report...</span>
            </>
          ) : (
            <>
              <Download className="w-4 h-4" />
              <span>Download Report</span>
            </>
          )}
        </button>

        <button
          onClick={() => onNavigate('home')}
          className="px-6 py-3 bg-white hover:bg-slate-50 text-slate-700 border border-slate-200 rounded-xl text-xs font-semibold transition-all flex items-center gap-2 active:scale-95"
        >
          <Home className="w-4 h-4 text-slate-500" />
          <span>Back to Home</span>
        </button>
      </div>
    </div>
  );
};
