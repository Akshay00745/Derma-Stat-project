import React, { useState, useRef } from 'react';
import {
  UploadCloud,
  FileImage,
  CheckCircle2,
  AlertCircle,
  X,
  ArrowRight,
  Loader2,
  Sparkles,
} from 'lucide-react';
import { SampleItem } from '../types/api';

interface UploadPageProps {
  onAnalyzeFile: (file: File) => Promise<void>;
  onAnalyzeSample: (sampleId: string) => Promise<void>;
  isLoading: boolean;
  error: string | null;
  samples: SampleItem[];
}

export const UploadPage: React.FC<UploadPageProps> = ({
  onAnalyzeFile,
  onAnalyzeSample,
  isLoading,
  error,
  samples,
}) => {
  const [dragActive, setDragActive] = useState(false);
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const handleDrag = (e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    if (e.type === 'dragenter' || e.type === 'dragover') {
      setDragActive(true);
    } else if (e.type === 'dragleave') {
      setDragActive(false);
    }
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    setDragActive(false);
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      handleFileSelected(e.dataTransfer.files[0]);
    }
  };

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      handleFileSelected(e.target.files[0]);
    }
  };

  const handleFileSelected = (file: File) => {
    if (!file.type.startsWith('image/')) {
      alert('Please select a valid image file (JPG, PNG, JPEG).');
      return;
    }
    setSelectedFile(file);
    const objectUrl = URL.createObjectURL(file);
    setPreviewUrl(objectUrl);
  };

  const handleClear = () => {
    setSelectedFile(null);
    if (previewUrl) URL.revokeObjectURL(previewUrl);
    setPreviewUrl(null);
    if (fileInputRef.current) fileInputRef.current.value = '';
  };

  const handleStartAnalysis = async () => {
    if (!selectedFile) return;
    await onAnalyzeFile(selectedFile);
  };

  return (
    <div className="p-8 max-w-4xl mx-auto space-y-8">
      {/* Title */}
      <div>
        <h2 className="text-2xl font-bold text-slate-900">Upload Skin-Lesion Image</h2>
        <p className="text-sm text-slate-500 mt-1">
          Select or drag and drop an image. We support JPG, PNG and JPEG formats.
        </p>
      </div>

      {error && (
        <div className="p-4 rounded-2xl bg-red-50 border border-red-200 text-red-700 text-sm flex items-start gap-3">
          <AlertCircle className="w-5 h-5 shrink-0 mt-0.5 text-red-500" />
          <div>
            <span className="font-semibold">Analysis Notice:</span> {error}
          </div>
        </div>
      )}

      {/* Main Upload Dropzone */}
      <div
        onDragEnter={handleDrag}
        onDragLeave={handleDrag}
        onDragOver={handleDrag}
        onDrop={handleDrop}
        className={`bg-white rounded-3xl border-2 border-dashed p-10 transition-all text-center flex flex-col items-center justify-center relative ${
          dragActive
            ? 'border-blue-500 bg-blue-50/30'
            : 'border-slate-200 hover:border-slate-300'
        }`}
      >
        <input
          ref={fileInputRef}
          type="file"
          accept="image/jpeg,image/png,image/jpg"
          onChange={handleFileChange}
          className="hidden"
        />

        {!selectedFile ? (
          <div className="space-y-4 max-w-sm">
            <div className="w-16 h-16 rounded-2xl bg-blue-50 text-blue-600 flex items-center justify-center mx-auto shadow-inner">
              <UploadCloud className="w-8 h-8 stroke-[1.8]" />
            </div>

            <div>
              <p className="text-sm font-semibold text-slate-800">
                Drag and drop your image here
              </p>
              <p className="text-xs text-slate-400 mt-0.5">or</p>
            </div>

            <button
              type="button"
              onClick={() => fileInputRef.current?.click()}
              className="px-6 py-2.5 bg-blue-600 hover:bg-blue-700 text-white rounded-xl text-xs font-semibold shadow-md shadow-blue-500/20 transition-all active:scale-95"
            >
              Choose File
            </button>

            <p className="text-[11px] text-slate-400 pt-2 font-medium">
              Supported formats: JPG, PNG, JPEG &nbsp;|&nbsp; Max size: <span className="font-bold">50MB</span>
            </p>
          </div>
        ) : (
          <div className="w-full space-y-6">
            <div className="flex flex-col md:flex-row items-center gap-6 justify-center">
              {/* Image Preview */}
              <div className="w-48 h-48 rounded-2xl overflow-hidden border border-slate-200 shadow-md bg-slate-100 relative group">
                <img
                  src={previewUrl!}
                  alt="Selected Preview"
                  className="w-full h-full object-cover"
                />
                <button
                  onClick={handleClear}
                  className="absolute top-2 right-2 p-1.5 rounded-full bg-slate-900/60 text-white hover:bg-slate-900 transition-colors"
                  title="Remove image"
                >
                  <X className="w-4 h-4" />
                </button>
              </div>

              {/* Metadata */}
              <div className="text-left space-y-2">
                <div className="flex items-center gap-2 text-sm font-bold text-slate-800">
                  <FileImage className="w-4 h-4 text-blue-600" />
                  <span className="truncate max-w-xs">{selectedFile.name}</span>
                </div>
                <p className="text-xs text-slate-500">
                  Size: {(selectedFile.size / 1024).toFixed(1)} KB &nbsp;|&nbsp; Type: {selectedFile.type}
                </p>

                <div className="flex items-center gap-2 pt-2">
                  <button
                    onClick={() => fileInputRef.current?.click()}
                    className="px-3 py-1.5 text-xs text-slate-600 hover:bg-slate-100 rounded-lg font-medium border border-slate-200 transition-colors"
                  >
                    Replace Image
                  </button>
                  <button
                    onClick={handleClear}
                    className="px-3 py-1.5 text-xs text-red-600 hover:bg-red-50 rounded-lg font-medium transition-colors"
                  >
                    Remove
                  </button>
                </div>
              </div>
            </div>

            {/* Run Analysis CTA */}
            <div>
              <button
                onClick={handleStartAnalysis}
                disabled={isLoading}
                className="px-8 py-3 bg-blue-600 hover:bg-blue-700 text-white rounded-xl text-sm font-semibold shadow-lg shadow-blue-500/25 transition-all flex items-center gap-2 mx-auto disabled:opacity-50 active:scale-95"
              >
                {isLoading ? (
                  <>
                    <Loader2 className="w-4 h-4 animate-spin" />
                    <span>Analyzing Image...</span>
                  </>
                ) : (
                  <>
                    <span>Run AI Analysis</span>
                    <ArrowRight className="w-4 h-4" />
                  </>
                )}
              </button>
            </div>
          </div>
        )}
      </div>

      {/* Image Guidelines (Panel 2 bottom card) */}
      <div className="bg-white rounded-2xl p-6 border border-slate-200/80 shadow-xs space-y-3">
        <h4 className="text-xs font-bold text-slate-800 uppercase tracking-wider flex items-center gap-2">
          <FileImage className="w-4 h-4 text-blue-600" />
          <span>Image Guidelines</span>
        </h4>
        <div className="grid md:grid-cols-3 gap-3 pt-1">
          <div className="flex items-center gap-2.5 text-xs text-slate-600 font-medium">
            <CheckCircle2 className="w-4 h-4 text-emerald-500 shrink-0" />
            <span>Clear and well-lit dermoscopic image</span>
          </div>
          <div className="flex items-center gap-2.5 text-xs text-slate-600 font-medium">
            <CheckCircle2 className="w-4 h-4 text-emerald-500 shrink-0" />
            <span>Recommended size: 224 × 224 or higher</span>
          </div>
          <div className="flex items-center gap-2.5 text-xs text-slate-600 font-medium">
            <CheckCircle2 className="w-4 h-4 text-emerald-500 shrink-0" />
            <span>Avoid blurry, obscured, or dark images</span>
          </div>
        </div>
      </div>

      {/* Quick Preset Samples */}
      <div className="space-y-3">
        <div className="flex items-center justify-between">
          <h4 className="text-xs font-bold text-slate-800 uppercase tracking-wider flex items-center gap-1.5">
            <Sparkles className="w-4 h-4 text-amber-500" />
            <span>Or Test Instant Presets (No Upload Needed)</span>
          </h4>
        </div>

        <div className="grid grid-cols-2 sm:grid-cols-4 md:grid-cols-7 gap-2.5">
          {samples.map((s) => (
            <button
              key={s.code}
              onClick={() => onAnalyzeSample(s.code)}
              disabled={isLoading}
              className="bg-white p-2.5 rounded-xl border border-slate-200 hover:border-blue-400 hover:shadow-sm text-center flex flex-col items-center group transition-all disabled:opacity-50"
            >
              <div className="w-12 h-12 rounded-lg overflow-hidden mb-1.5 border border-slate-100 bg-slate-50">
                <img
                  src={s.thumbnail_b64 || `/samples/${s.code}.jpg`}
                  alt={s.code}
                  className="w-full h-full object-cover group-hover:scale-110 transition-transform"
                />
              </div>
              <span className="text-[11px] font-bold text-slate-800 uppercase group-hover:text-blue-600">
                {s.code}
              </span>
            </button>
          ))}
        </div>
      </div>
    </div>
  );
};
