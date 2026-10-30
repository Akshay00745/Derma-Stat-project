import React from 'react';
import {
  Home,
  Upload,
  BarChart2,
  Eye,
  FileText,
  Activity,
  Info,
  HelpCircle,
  ShieldAlert,
  Cpu,
  CheckCircle2,
} from 'lucide-react';
import { HealthResponse } from '../types/api';

export type PageId =
  | 'home'
  | 'upload'
  | 'results'
  | 'gradcam'
  | 'detailed'
  | 'performance'
  | 'about'
  | 'help';

interface SidebarProps {
  currentPage: PageId;
  onNavigate: (page: PageId) => void;
  hasResult: boolean;
  health: HealthResponse | null;
}

export const Sidebar: React.FC<SidebarProps> = ({
  currentPage,
  onNavigate,
  hasResult,
  health,
}) => {
  const navItems = [
    { id: 'home' as PageId, label: 'Home', icon: Home },
    { id: 'upload' as PageId, label: 'Upload Image', icon: Upload },
    { id: 'results' as PageId, label: 'Analysis Results', icon: BarChart2, badge: hasResult ? 'Ready' : undefined },
    { id: 'gradcam' as PageId, label: 'Grad-CAM', icon: Eye, disabled: !hasResult },
    { id: 'detailed' as PageId, label: 'Detailed Analysis', icon: FileText, disabled: !hasResult },
    { id: 'performance' as PageId, label: 'Model Performance', icon: Activity },
    { id: 'about' as PageId, label: 'About', icon: Info },
    { id: 'help' as PageId, label: 'Help & Support', icon: HelpCircle },
  ];

  return (
    <aside className="w-64 bg-white border-r border-slate-200/80 flex flex-col h-screen sticky top-0 select-none shadow-sm z-30">
      {/* Brand Header */}
      <div className="p-5 border-b border-slate-100 flex items-center gap-3">
        <div className="w-10 h-10 rounded-xl bg-blue-600 flex items-center justify-center text-white shadow-md shadow-blue-500/20">
          <Activity className="w-6 h-6 stroke-[2.2]" />
        </div>
        <div>
          <h1 className="font-bold text-slate-900 text-base leading-tight tracking-tight">
            Dermastat
          </h1>
          <span className="text-xs font-semibold text-blue-600 tracking-wide">
            AI Vision
          </span>
        </div>
      </div>

      {/* Navigation Menu */}
      <nav className="flex-1 px-3 py-4 space-y-1 overflow-y-auto">
        {navItems.map((item) => {
          const Icon = item.icon;
          const isActive = currentPage === item.id;
          const isDisabled = item.disabled;

          return (
            <button
              key={item.id}
              onClick={() => !isDisabled && onNavigate(item.id)}
              disabled={isDisabled}
              className={`w-full flex items-center justify-between px-3.5 py-2.5 rounded-xl text-sm font-medium transition-all ${
                isActive
                  ? 'bg-blue-50 text-blue-700 shadow-sm border border-blue-100/60 font-semibold'
                  : isDisabled
                  ? 'text-slate-300 cursor-not-allowed'
                  : 'text-slate-600 hover:bg-slate-50 hover:text-slate-900'
              }`}
            >
              <div className="flex items-center gap-3">
                <Icon
                  className={`w-4 h-4 ${
                    isActive ? 'text-blue-600 stroke-[2.2]' : isDisabled ? 'text-slate-300' : 'text-slate-400'
                  }`}
                />
                <span>{item.label}</span>
              </div>
              {item.badge && (
                <span className="text-[10px] uppercase font-bold tracking-wider px-2 py-0.5 rounded-full bg-emerald-50 text-emerald-600 border border-emerald-200">
                  {item.badge}
                </span>
              )}
            </button>
          );
        })}
      </nav>

      {/* Model Engine Status Card */}
      <div className="px-4 py-3 border-t border-slate-100 bg-slate-50/50">
        <div className="flex items-center justify-between mb-1.5">
          <span className="text-[11px] font-semibold text-slate-500 uppercase tracking-wider">
            AI Engine
          </span>
          <span className="inline-flex items-center gap-1 text-[11px] font-medium text-emerald-600">
            <span className="w-1.5 h-1.5 rounded-full bg-emerald-500 animate-pulse"></span>
            {health?.model_loaded ? 'Ready' : 'Checking'}
          </span>
        </div>
        <div className="flex items-center gap-2 text-xs text-slate-600">
          <Cpu className="w-3.5 h-3.5 text-blue-500" />
          <span className="truncate">
            {health?.device ? `EfficientNetB0 (${health.device.toUpperCase()})` : 'EfficientNetB0'}
          </span>
        </div>
      </div>

      {/* Disclaimer Footer (Panel 9 bottom) */}
      <div className="p-4 border-t border-slate-100">
        <div className="flex items-start gap-2 text-[11px] text-slate-400 leading-snug">
          <ShieldAlert className="w-4 h-4 text-slate-400 shrink-0 mt-0.5" />
          <p>
            Educational Prototype.<br />
            <span className="font-semibold text-slate-500">Not a medical diagnostic tool.</span>
          </p>
        </div>
      </div>
    </aside>
  );
};
