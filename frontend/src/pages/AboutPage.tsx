import React from 'react';
import {
  Layers,
  Eye,
  LineChart,
  ShieldCheck,
  ShieldAlert,
  GraduationCap,
  Cpu,
  Database,
  ExternalLink,
} from 'lucide-react';

export const AboutPage: React.FC = () => {
  const goals = [
    {
      title: 'Demonstrate AI classification',
      desc: 'Benchmark modern deep CNN architectures on multiclass dermatological tasks.',
      icon: Layers,
    },
    {
      title: 'Provide visual explanations',
      desc: 'Demystify convolutional feature representations through native Grad-CAM attention heatmaps.',
      icon: Eye,
    },
    {
      title: 'Show model performance',
      desc: 'Present honest, un-cherrypicked metrics evaluated on held-out patient splits.',
      icon: LineChart,
    },
    {
      title: 'Promote responsible AI',
      desc: 'Integrate validation-calibrated uncertainty warnings to flag ambiguous predictions.',
      icon: ShieldCheck,
    },
  ];

  return (
    <div className="p-8 max-w-5xl mx-auto space-y-10">
      {/* Overview */}
      <section className="bg-white rounded-3xl p-8 border border-slate-200/80 shadow-xs space-y-4">
        <div className="flex flex-wrap items-center gap-2">
          <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full bg-blue-50 text-blue-700 text-xs font-semibold">
            <GraduationCap className="w-4 h-4 text-blue-600" />
            <span>Research & Educational Prototype</span>
          </div>
          <span className="text-xs text-slate-400 font-medium hidden sm:inline">•</span>
          <span className="text-xs font-medium text-slate-500">
            Deep Learning Architecture, Data Provenance, and Clinical Mission
          </span>
        </div>

        <h2 className="text-3xl font-extrabold text-slate-900 tracking-tight">
          About Dermastat AI Vision
        </h2>

        <p className="text-sm text-slate-600 leading-relaxed">
          <strong className="text-slate-800">Dermastat AI Vision</strong> is an educational prototype that uses
          deep learning, computer vision, explainable AI, and uncertainty assessment to classify skin-lesion images into seven categories.
          It provides confidence scores, visual explanations and uncertainty assessment to make
          AI predictions more transparent and interpretable.
        </p>

        <div className="grid grid-cols-1 sm:grid-cols-3 gap-4 pt-4 border-t border-slate-100">
          <div className="flex items-center gap-3">
            <Cpu className="w-5 h-5 text-blue-600" />
            <div className="text-xs">
              <span className="font-bold text-slate-800 block">EfficientNet-B0</span>
              <span className="text-slate-400">Transfer Learning</span>
            </div>
          </div>

          <div className="flex items-center gap-3">
            <Database className="w-5 h-5 text-blue-600" />
            <div className="text-xs">
              <span className="font-bold text-slate-800 block">HAM10000</span>
              <span className="text-slate-400">10,015 Dermatoscopic Images</span>
            </div>
          </div>

          <div className="flex items-center gap-3">
            <ShieldCheck className="w-5 h-5 text-blue-600" />
            <div className="text-xs">
              <span className="font-bold text-slate-800 block">0% Patient Leakage</span>
              <span className="text-slate-400">Lesion-Grouped Stratification</span>
            </div>
          </div>
        </div>
      </section>

      {/* Our Goals (Panel 7 middle) */}
      <section className="space-y-4">
        <h3 className="text-sm font-bold text-slate-800 uppercase tracking-wider">
          Our Goals
        </h3>

        <div className="grid sm:grid-cols-2 md:grid-cols-4 gap-4">
          {goals.map((g, idx) => {
            const Icon = g.icon;
            return (
              <div
                key={idx}
                className="bg-white rounded-2xl p-5 border border-slate-200/80 shadow-xs hover:border-blue-200 transition-all space-y-3"
              >
                <div className="w-10 h-10 rounded-xl bg-blue-50 text-blue-600 flex items-center justify-center">
                  <Icon className="w-5 h-5 stroke-[2]" />
                </div>
                <h4 className="text-xs font-bold text-slate-900 leading-snug">
                  {g.title}
                </h4>
                <p className="text-[11px] text-slate-500 leading-relaxed font-medium">
                  {g.desc}
                </p>
              </div>
            );
          })}
        </div>
      </section>

      {/* Clinical Disclaimer Card (Panel 7 bottom) */}
      <section className="bg-blue-50/70 border border-blue-200/80 rounded-3xl p-6 flex items-start gap-4">
        <div className="w-10 h-10 rounded-2xl bg-blue-600 text-white flex items-center justify-center shrink-0 shadow-sm shadow-blue-500/20">
          <ShieldAlert className="w-6 h-6 stroke-[2]" />
        </div>
        <div className="space-y-1.5">
          <h4 className="text-sm font-bold text-slate-900">
            This is not a medical diagnostic tool.
          </h4>
          <p className="text-xs text-slate-600 leading-relaxed">
            Always consult a qualified healthcare professional or certified dermatologist for medical advice,
            diagnosis, and treatment planning. Automated computer vision models provide experimental decision-support
            insights and should never supersede histological confirmation.
          </p>
        </div>
      </section>
    </div>
  );
};
