import React from 'react';
import {
  ArrowRight,
  Layers,
  BarChart3,
  Eye,
  AlertTriangle,
  LineChart,
  CheckCircle2,
  ShieldCheck,
} from 'lucide-react';
import { PageId } from '../components/Sidebar';
import { SampleItem } from '../types/api';

interface HomePageProps {
  onNavigate: (page: PageId) => void;
  samples: SampleItem[];
  onSelectSample: (sampleId: string) => void;
}

export const HomePage: React.FC<HomePageProps> = ({
  onNavigate,
  samples,
  onSelectSample,
}) => {
  const featurePills = [
    { title: '7 Skin-Lesion Categories', desc: 'Full taxonomy coverage', icon: Layers },
    { title: 'Confidence Scores for All Classes', desc: 'Normalized distribution', icon: BarChart3 },
    { title: 'Grad-CAM Visual Explanation', desc: 'Morphological hot-spots', icon: Eye },
    { title: 'Uncertainty Assessment', desc: 'Validation-calibrated (83.15%)', icon: AlertTriangle },
    { title: 'Performance Dashboard', desc: '76.04% Test Accuracy', icon: LineChart },
  ];

  return (
    <div className="p-8 max-w-6xl mx-auto space-y-12">
      {/* Hero Section */}
      <section className="bg-gradient-to-br from-white via-white to-blue-50/40 rounded-3xl p-8 md:p-12 border border-slate-200/80 shadow-sm relative overflow-hidden">
        <div className="grid md:grid-cols-2 gap-8 items-center">
          <div className="space-y-6 z-10">
            <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full bg-blue-50 border border-blue-100 text-blue-700 text-xs font-semibold">
              <ShieldCheck className="w-3.5 h-3.5 text-blue-600" />
              <span>Next-Gen Dermatology AI</span>
            </div>

            <h1 className="text-4xl md:text-5xl font-extrabold text-slate-900 tracking-tight leading-tight">
              Smarter AI for <br />
              <span className="text-blue-600">Healthier Skin</span>
            </h1>

            <p className="text-slate-600 text-sm md:text-base leading-relaxed">
              Upload a skin-lesion image and get AI-powered classification, confidence scores,
              visual explanation and more — all in one place.
            </p>

            <div className="flex flex-wrap items-center gap-3 pt-2">
              <button
                onClick={() => onNavigate('upload')}
                className="px-6 py-3 bg-blue-600 hover:bg-blue-700 text-white rounded-xl text-sm font-semibold shadow-md shadow-blue-500/20 transition-all flex items-center gap-2 active:scale-95"
              >
                <span>Upload Image</span>
                <ArrowRight className="w-4 h-4" />
              </button>

              <button
                onClick={() => onNavigate('about')}
                className="px-6 py-3 bg-white hover:bg-slate-50 text-slate-700 border border-slate-200 rounded-xl text-sm font-semibold transition-all active:scale-95"
              >
                <span>Learn More</span>
              </button>
            </div>
          </div>

          {/* Hero Illustration / Sample preview card */}
          <div className="flex justify-center z-10">
            <div className="relative group">
              <div className="w-64 h-64 md:w-72 md:h-72 rounded-3xl overflow-hidden border-4 border-white shadow-xl bg-slate-100">
                <img
                  src="/samples/hero_lesion.jpg"
                  alt="Dermoscopy Lesion"
                  className="w-full h-full object-cover group-hover:scale-105 transition-transform duration-500"
                  onError={(e) => {
                    // Fallback to sample mel if hero not available
                    (e.target as HTMLImageElement).src = '/samples/mel.jpg';
                  }}
                />
              </div>

              {/* Floating Pill Badge */}
              <div className="absolute -bottom-4 right-4 bg-white/95 backdrop-blur-sm px-4 py-2.5 rounded-2xl border border-slate-200/80 shadow-lg flex items-center gap-3">
                <div className="w-7 h-7 rounded-lg bg-blue-50 border border-blue-200 flex items-center justify-center text-blue-600">
                  <CheckCircle2 className="w-4 h-4" />
                </div>
                <div>
                  <div className="text-[11px] font-bold text-slate-800">AI-Powered</div>
                  <div className="text-[10px] text-slate-500">Skin Lesion Classification</div>
                </div>
              </div>
            </div>
          </div>
        </div>

        {/* Subtle background decoration */}
        <div className="absolute -right-20 -bottom-20 w-80 h-80 rounded-full bg-blue-100/40 blur-3xl pointer-events-none"></div>
      </section>

      {/* 5 Feature Pill Cards */}
      <section className="grid grid-cols-2 md:grid-cols-5 gap-3">
        {featurePills.map((pill, idx) => {
          const Icon = pill.icon;
          return (
            <div
              key={idx}
              className="bg-white p-4 rounded-2xl border border-slate-200/70 shadow-xs hover:border-blue-200 hover:shadow-sm transition-all flex flex-col items-center text-center group"
            >
              <div className="w-10 h-10 rounded-xl bg-blue-50 text-blue-600 flex items-center justify-center mb-2.5 group-hover:scale-110 transition-transform">
                <Icon className="w-5 h-5 stroke-[2]" />
              </div>
              <h4 className="text-xs font-bold text-slate-800 leading-tight mb-1">
                {pill.title}
              </h4>
              <p className="text-[10px] text-slate-400 font-medium">
                {pill.desc}
              </p>
            </div>
          );
        })}
      </section>

      {/* The 7 Diagnostic Category Cards */}
      <section className="space-y-4">
        <div className="flex items-center justify-between">
          <div>
            <h3 className="text-lg font-bold text-slate-900">The 7 Skin-Lesion Categories</h3>
            <p className="text-xs text-slate-500">Click any category to test instant inference</p>
          </div>
          <span className="text-xs font-semibold text-blue-600 bg-blue-50 px-2.5 py-1 rounded-full border border-blue-100">
            HAM10000 Dataset
          </span>
        </div>

        <div className="grid grid-cols-2 sm:grid-cols-4 md:grid-cols-7 gap-3">
          {samples.length > 0 ? (
            samples.map((s) => (
              <button
                key={s.code}
                onClick={() => onSelectSample(s.code)}
                className="bg-white p-3 rounded-2xl border border-slate-200/70 shadow-xs hover:border-blue-400 hover:shadow-md transition-all text-center flex flex-col items-center group active:scale-95"
              >
                <div className="w-16 h-16 rounded-full overflow-hidden mb-2.5 border-2 border-slate-100 group-hover:border-blue-500 transition-colors shadow-inner bg-slate-50">
                  <img
                    src={s.thumbnail_b64 || `/samples/${s.code}.jpg`}
                    alt={s.name}
                    className="w-full h-full object-cover group-hover:scale-110 transition-transform duration-300"
                  />
                </div>
                <span className="text-xs font-bold text-slate-800 uppercase tracking-wide group-hover:text-blue-600">
                  {s.code}
                </span>
                <span className="text-[10px] text-slate-500 line-clamp-2 mt-0.5 leading-tight font-medium">
                  {s.name}
                </span>
              </button>
            ))
          ) : (
            ['akiec', 'bcc', 'bkl', 'df', 'mel', 'nv', 'vasc'].map((code) => (
              <button
                key={code}
                onClick={() => onSelectSample(code)}
                className="bg-white p-3 rounded-2xl border border-slate-200/70 shadow-xs hover:border-blue-400 hover:shadow-md transition-all text-center flex flex-col items-center group active:scale-95"
              >
                <div className="w-16 h-16 rounded-full overflow-hidden mb-2.5 border-2 border-slate-100 group-hover:border-blue-500 transition-colors shadow-inner bg-slate-50">
                  <img
                    src={`/samples/${code}.jpg`}
                    alt={code}
                    className="w-full h-full object-cover group-hover:scale-110 transition-transform duration-300"
                  />
                </div>
                <span className="text-xs font-bold text-slate-800 uppercase tracking-wide group-hover:text-blue-600">
                  {code}
                </span>
              </button>
            ))
          )}
        </div>
      </section>

      {/* Educational Prototype Footer */}
      <footer className="pt-6 border-t border-slate-200 text-center text-xs text-slate-400">
        <p>
          Dermastat AI Vision &nbsp;|&nbsp; Educational Prototype &nbsp;|&nbsp; Not a medical diagnostic tool
        </p>
      </footer>
    </div>
  );
};
