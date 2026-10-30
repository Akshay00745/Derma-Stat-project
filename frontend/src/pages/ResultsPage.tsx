import React from 'react';
import {
  CheckCircle2,
  AlertTriangle,
  Eye,
  FileText,
  ShieldCheck,
  RotateCcw,
  Sparkles,
} from 'lucide-react';
import { PageId } from '../components/Sidebar';
import { PredictionResult } from '../types/api';

interface ResultsPageProps {
  prediction: PredictionResult;
  onNavigate: (page: PageId) => void;
  onGradCAM: () => Promise<void>;
  isLoadingGradCAM: boolean;
}

export const ResultsPage: React.FC<ResultsPageProps> = ({
  prediction,
  onNavigate,
  onGradCAM,
  isLoadingGradCAM,
}) => {
  const isReliable = !prediction.uncertainty.is_uncertain;

  const handleOpenGradCAM = async () => {
    await onGradCAM();
    onNavigate('gradcam');
  };

  return (
    <div className="p-8 max-w-5xl mx-auto space-y-6">
      {/* Header bar */}
      <div className="flex items-center justify-between">
        <div>
          <h2 className="text-2xl font-bold text-slate-900">Analysis Result</h2>
          <p className="text-xs text-slate-500 mt-0.5">Automated multiclass dermoscopic classification</p>
        </div>

        <div className="inline-flex items-center gap-1.5 px-3 py-1 rounded-full bg-emerald-50 border border-emerald-200 text-emerald-700 text-xs font-semibold">
          <CheckCircle2 className="w-3.5 h-3.5 text-emerald-600" />
          <span>Prediction Completed</span>
        </div>
      </div>

      {/* Main Grid: Prediction + Uncertainty */}
      <div className="grid md:grid-cols-3 gap-6">
        {/* Left Column (2 spans): Primary Prediction & 7 Confidence Bars */}
        <div className="md:col-span-2 space-y-6">
          {/* Top Prediction Card (Panel 3 top) */}
          <div className="bg-white rounded-3xl p-6 border border-slate-200/80 shadow-xs flex items-center gap-6">
            <div className="w-24 h-24 rounded-2xl overflow-hidden border-2 border-slate-100 shadow-inner bg-slate-100 shrink-0">
              <img
                src={prediction.image_preview_b64 || `/samples/${prediction.predicted_code}.jpg`}
                alt="Analyzed Lesion"
                className="w-full h-full object-cover"
              />
            </div>

            <div className="space-y-2">
              <span className="text-xs font-semibold text-slate-400 uppercase tracking-wider">
                Predicted Category
              </span>
              <h3 className="text-2xl font-extrabold text-slate-900 leading-tight">
                {prediction.predicted_name}{' '}
                <span className="text-blue-600">({prediction.predicted_code.toUpperCase()})</span>
              </h3>
              <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full bg-blue-50 border border-blue-100 text-blue-700 text-xs font-bold">
                <span>Confidence: {prediction.confidence_percentage}%</span>
              </div>
            </div>
          </div>

          {/* 7-Class Confidence Bars (Panel 3 left bottom) */}
          <div className="bg-white rounded-3xl p-6 border border-slate-200/80 shadow-xs space-y-4">
            <div className="flex items-center justify-between border-b border-slate-100 pb-3">
              <h4 className="text-xs font-bold text-slate-800 uppercase tracking-wider">
                Confidence Scores (All 7 Classes)
              </h4>
              <span className="text-[11px] text-slate-400 font-medium">Softmax Normalized</span>
            </div>

            <div className="space-y-3 pt-1">
              {prediction.all_probabilities.map((item) => {
                const isTop = item.code === prediction.predicted_code;
                return (
                  <div key={item.code} className="space-y-1">
                    <div className="flex items-center justify-between text-xs">
                      <div className="flex items-center gap-2">
                        <span className={`w-12 font-bold uppercase ${isTop ? 'text-blue-600' : 'text-slate-600'}`}>
                          {item.code}
                        </span>
                        <span className="text-slate-500 truncate max-w-xs">{item.name}</span>
                      </div>
                      <span className={`font-mono font-bold ${isTop ? 'text-blue-600' : 'text-slate-600'}`}>
                        {item.percentage}%
                      </span>
                    </div>

                    {/* Progress Bar */}
                    <div className="h-2 w-full bg-slate-100 rounded-full overflow-hidden">
                      <div
                        className={`h-full rounded-full transition-all duration-700 ${
                          isTop ? 'bg-blue-600 shadow-sm' : 'bg-slate-300'
                        }`}
                        style={{ width: `${Math.max(item.percentage, 1)}%` }}
                      ></div>
                    </div>
                  </div>
                );
              })}
            </div>
          </div>
        </div>

        {/* Right Column: Uncertainty Card & Navigation CTAs (Panel 3 right) */}
        <div className="space-y-6">
          {/* Uncertainty Assessment Card */}
          <div
            className={`rounded-3xl p-6 border shadow-xs space-y-3 ${
              isReliable
                ? 'bg-emerald-50/50 border-emerald-200/80 text-emerald-950'
                : 'bg-amber-50/60 border-amber-200/80 text-amber-950'
            }`}
          >
            <div className="flex items-center gap-2">
              {isReliable ? (
                <ShieldCheck className="w-5 h-5 text-emerald-600" />
              ) : (
                <AlertTriangle className="w-5 h-5 text-amber-600" />
              )}
              <h4 className="text-xs font-bold uppercase tracking-wider">
                Uncertainty Assessment
              </h4>
            </div>

            <div className="text-sm font-bold">
              {prediction.uncertainty.level} uncertainty ({prediction.uncertainty.status})
            </div>

            <p className="text-xs leading-relaxed text-slate-600">
              {prediction.uncertainty.message}
            </p>

            <div className="pt-2 border-t border-slate-200/60 text-[11px] text-slate-500">
              Validation Threshold: <span className="font-mono font-bold text-slate-700">83.15%</span>
            </div>
          </div>

          {/* Action CTAs (Panel 3 right bottom) */}
          <div className="bg-white rounded-3xl p-6 border border-slate-200/80 shadow-xs space-y-3">
            <h4 className="text-xs font-bold text-slate-800 uppercase tracking-wider mb-2">
              Next Actions
            </h4>

            <button
              onClick={handleOpenGradCAM}
              disabled={isLoadingGradCAM}
              className="w-full py-3 px-4 bg-blue-600 hover:bg-blue-700 text-white rounded-xl text-xs font-semibold shadow-md shadow-blue-500/20 transition-all flex items-center justify-center gap-2 active:scale-95 disabled:opacity-50"
            >
              <Eye className="w-4 h-4" />
              <span>{isLoadingGradCAM ? 'Computing Grad-CAM...' : 'View Grad-CAM'}</span>
            </button>

            <button
              onClick={() => onNavigate('detailed')}
              className="w-full py-3 px-4 bg-white hover:bg-slate-50 text-slate-700 border border-slate-200 rounded-xl text-xs font-semibold transition-all flex items-center justify-center gap-2 active:scale-95"
            >
              <FileText className="w-4 h-4 text-slate-500" />
              <span>View Detailed Results</span>
            </button>

            <button
              onClick={() => onNavigate('upload')}
              className="w-full py-2.5 px-4 text-slate-500 hover:text-slate-800 rounded-xl text-xs font-medium transition-colors flex items-center justify-center gap-1.5"
            >
              <RotateCcw className="w-3.5 h-3.5" />
              <span>Analyze Another Image</span>
            </button>
          </div>
        </div>
      </div>
    </div>
  );
};
