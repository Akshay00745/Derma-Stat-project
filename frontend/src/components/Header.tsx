import React from 'react';
import { Shield, Sparkles, Wifi } from 'lucide-react';
import { PageId } from './Sidebar';

interface HeaderProps {
  currentPage: PageId;
  onNavigate: (page: PageId) => void;
}

const PAGE_TITLES: Record<PageId, { title: string; subtitle: string }> = {
  home: { title: 'Welcome to Dermastat AI Vision', subtitle: 'AI-assisted dermoscopic classification and explainability' },
  upload: { title: 'Skin Lesion Image Upload', subtitle: 'Analyze dermatoscopic images across 7 diagnostic categories' },
  results: { title: 'Analysis Results', subtitle: 'Deep learning classification scores and uncertainty assessment' },
  gradcam: { title: 'Grad-CAM Visual Explainability', subtitle: 'Feature activation heatmaps and morphological attention regions' },
  detailed: { title: 'Detailed Diagnostic Analysis', subtitle: 'Comprehensive probability breakdown and report export' },
  performance: { title: 'Model Performance Dashboard', subtitle: 'Empirical benchmark metrics on the reserved HAM10000 test set' },
  about: { title: 'About Dermastat AI Vision', subtitle: 'Deep Learning Architecture, Data Provenance, and Clinical Mission' },
  help: { title: 'Help & Knowledge Base', subtitle: 'Frequently asked questions and technical documentation' },
};

export const Header: React.FC<HeaderProps> = ({ currentPage, onNavigate }) => {
  const info = PAGE_TITLES[currentPage];

  return (
    <header className="bg-white border-b border-slate-200/80 px-8 py-4 flex items-center justify-between sticky top-0 z-20 shadow-xs">
      <div>
        <h2 className="text-lg font-bold text-slate-900 tracking-tight">{info.title}</h2>
        <p className="text-xs text-slate-500 font-medium">{info.subtitle}</p>
      </div>

      <div className="flex items-center gap-3">
        <div className="hidden md:flex items-center gap-1.5 px-3 py-1.5 rounded-full bg-blue-50/70 border border-blue-100 text-blue-700 text-xs font-medium">
          <Sparkles className="w-3.5 h-3.5 text-blue-600" />
          <span>HAM10000 Trained</span>
        </div>

        <button
          onClick={() => onNavigate('upload')}
          className="px-4 py-2 bg-blue-600 hover:bg-blue-700 text-white rounded-xl text-xs font-semibold shadow-sm transition-all flex items-center gap-1.5 active:scale-95"
        >
          <span>Analyze Image</span>
        </button>
      </div>
    </header>
  );
};
