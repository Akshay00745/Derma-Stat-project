import React from 'react';
import { ArrowLeft, Info, Eye, Layers, Sparkles } from 'lucide-react';
import { PageId } from '../components/Sidebar';
import { GradCAMResult } from '../types/api';

interface GradCAMPageProps {
  gradcam: GradCAMResult | null;
  onNavigate: (page: PageId) => void;
  isLoading: boolean;
}

export const GradCAMPage: React.FC<GradCAMPageProps> = ({
  gradcam,
  onNavigate,
  isLoading,
}) => {
  if (isLoading || !gradcam) {
    return (
      <div className="p-16 max-w-4xl mx-auto text-center space-y-4">
        <div className="w-12 h-12 rounded-full border-4 border-blue-200 border-t-blue-600 animate-spin mx-auto"></div>
        <p className="text-sm font-semibold text-slate-700">
          Generating high-resolution Grad-CAM heatmaps...
        </p>
        <p className="text-xs text-slate-400">
          Tracking gradients through EfficientNetB0 final convolutional stage
        </p>
      </div>
    );
  }

  return (
    <div className="p-8 max-w-5xl mx-auto space-y-8">
      {/* Header */}
      <div>
        <h2 className="text-2xl font-bold text-slate-900">Grad-CAM Visualization</h2>
        <p className="text-sm text-slate-500 mt-1">
          This heatmap shows the regions of the image that influenced the model's prediction.
        </p>
      </div>

      {/* 3-Column Display (Panel 4 core) */}
      <div className="grid md:grid-cols-3 gap-6">
        {/* Panel 1: Original Image */}
        <div className="bg-white rounded-3xl p-5 border border-slate-200/80 shadow-xs flex flex-col items-center">
          <div className="w-full aspect-square rounded-2xl overflow-hidden border border-slate-100 shadow-inner bg-slate-100 mb-3">
            <img
              src={gradcam.original_b64}
              alt="Original Lesion"
              className="w-full h-full object-cover"
            />
          </div>
          <span className="text-xs font-bold text-slate-700 uppercase tracking-wide">
            Original Image
          </span>
          <span className="text-[11px] text-slate-400 mt-0.5">Raw Dermoscopy</span>
        </div>

        {/* Panel 2: Heatmap (Grad-CAM) */}
        <div className="bg-white rounded-3xl p-5 border border-slate-200/80 shadow-xs flex flex-col items-center">
          <div className="w-full aspect-square rounded-2xl overflow-hidden border border-slate-100 shadow-inner bg-slate-100 mb-3">
            <img
              src={gradcam.heatmap_b64}
              alt="Grad-CAM Heatmap"
              className="w-full h-full object-cover"
            />
          </div>
          <span className="text-xs font-bold text-slate-700 uppercase tracking-wide">
            Heatmap (Grad-CAM)
          </span>
          <span className="text-[11px] text-slate-400 mt-0.5">Activation Map (Jet)</span>
        </div>

        {/* Panel 3: Overlay */}
        <div className="bg-white rounded-3xl p-5 border border-slate-200/80 shadow-xs flex flex-col items-center">
          <div className="w-full aspect-square rounded-2xl overflow-hidden border border-slate-100 shadow-inner bg-slate-100 mb-3">
            <img
              src={gradcam.overlay_b64}
              alt="Grad-CAM Overlay"
              className="w-full h-full object-cover"
            />
          </div>
          <span className="text-xs font-bold text-slate-700 uppercase tracking-wide">
            Overlay
          </span>
          <span className="text-[11px] text-slate-400 mt-0.5">Alpha-Blended (45%)</span>
        </div>
      </div>

      {/* Interpretation Card (Panel 4 bottom) */}
      <div className="bg-blue-50/60 rounded-3xl p-6 border border-blue-100/80 flex items-start gap-4">
        <div className="w-9 h-9 rounded-xl bg-blue-600 text-white flex items-center justify-center shrink-0 shadow-sm shadow-blue-500/20">
          <Info className="w-5 h-5" />
        </div>
        <div className="space-y-1">
          <h4 className="text-xs font-bold text-slate-900 uppercase tracking-wider">
            Interpretation
          </h4>
          <p className="text-xs text-slate-600 leading-relaxed">
            The highlighted areas (red/yellow) show the regions that contributed most to the
            model's prediction for <span className="font-semibold text-slate-800">{gradcam.predicted_name}</span>.
            Cooler regions (blue/cyan) had negligible impact on the classifier decision.
          </p>
        </div>
      </div>

      {/* Back button */}
      <div className="flex justify-start">
        <button
          onClick={() => onNavigate('results')}
          className="px-5 py-2.5 bg-white hover:bg-slate-50 text-slate-700 border border-slate-200 rounded-xl text-xs font-semibold shadow-xs transition-all flex items-center gap-2 active:scale-95"
        >
          <ArrowLeft className="w-4 h-4" />
          <span>Back to Results</span>
        </button>
      </div>
    </div>
  );
};
