import React, { useState, useEffect } from 'react';
import { Sidebar, PageId } from './components/Sidebar';
import { Header } from './components/Header';
import { HomePage } from './pages/HomePage';
import { UploadPage } from './pages/UploadPage';
import { ResultsPage } from './pages/ResultsPage';
import { GradCAMPage } from './pages/GradCAMPage';
import { DetailedPage } from './pages/DetailedPage';
import { PerformancePage } from './pages/PerformancePage';
import { AboutPage } from './pages/AboutPage';
import { HelpPage } from './pages/HelpPage';
import { api } from './services/api';
import {
  HealthResponse,
  SampleItem,
  PredictionResult,
  GradCAMResult,
  PerformanceMetrics,
} from './types/api';

export const App: React.FC = () => {
  const [currentPage, setCurrentPage] = useState<PageId>('home');
  const [health, setHealth] = useState<HealthResponse | null>(null);
  const [samples, setSamples] = useState<SampleItem[]>([]);
  const [performance, setPerformance] = useState<PerformanceMetrics | null>(null);

  const [activeFile, setActiveFile] = useState<File | null>(null);
  const [activeSampleId, setActiveSampleId] = useState<string | null>(null);

  const [prediction, setPrediction] = useState<PredictionResult | null>(null);
  const [gradcam, setGradcam] = useState<GradCAMResult | null>(null);

  const [isLoading, setIsLoading] = useState(false);
  const [isLoadingGradCAM, setIsLoadingGradCAM] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // Initial load
  useEffect(() => {
    const initializeData = async () => {
      try {
        const [h, s, p] = await Promise.all([
          api.getHealth().catch(() => null),
          api.getSamples().catch(() => []),
          api.getPerformance().catch(() => null),
        ]);
        if (h) setHealth(h);
        if (s) setSamples(s);
        if (p) setPerformance(p);
      } catch (e) {
        console.error('Failed to initialize app data:', e);
      }
    };
    initializeData();
  }, []);

  // Handle uploaded file analysis
  const handleAnalyzeFile = async (file: File) => {
    setIsLoading(true);
    setError(null);
    setActiveFile(file);
    setActiveSampleId(null);
    setGradcam(null);

    try {
      const result = await api.predictUpload(file);
      setPrediction(result);
      setCurrentPage('results');
    } catch (err: any) {
      setError(err.message || 'Image analysis failed.');
    } finally {
      setIsLoading(false);
    }
  };

  // Handle sample preset analysis
  const handleAnalyzeSample = async (sampleId: string) => {
    setIsLoading(true);
    setError(null);
    setActiveSampleId(sampleId);
    setActiveFile(null);
    setGradcam(null);

    try {
      const result = await api.predictSample(sampleId);
      setPrediction(result);
      setCurrentPage('results');
    } catch (err: any) {
      setError(err.message || 'Sample analysis failed.');
    } finally {
      setIsLoading(false);
    }
  };

  // Compute Grad-CAM visualization
  const handleComputeGradCAM = async () => {
    if (gradcam) return; // Already computed

    setIsLoadingGradCAM(true);
    try {
      let result: GradCAMResult;
      if (activeFile) {
        result = await api.explainUpload(activeFile);
      } else if (activeSampleId) {
        result = await api.explainSample(activeSampleId);
      } else if (prediction) {
        result = await api.explainSample(prediction.predicted_code);
      } else {
        throw new Error('No lesion image currently loaded.');
      }
      setGradcam(result);
    } catch (err: any) {
      alert(`Grad-CAM calculation notice: ${err.message}`);
    } finally {
      setIsLoadingGradCAM(false);
    }
  };

  return (
    <div className="flex min-h-screen bg-[#F8FAFC]">
      {/* Reusable Sidebar (Panel 9) */}
      <Sidebar
        currentPage={currentPage}
        onNavigate={setCurrentPage}
        hasResult={prediction !== null}
        health={health}
      />

      {/* Main Content Area */}
      <div className="flex-1 flex flex-col min-w-0 overflow-y-auto">
        <Header currentPage={currentPage} onNavigate={setCurrentPage} />

        <main className="flex-1 pb-16">
          {currentPage === 'home' && (
            <HomePage
              onNavigate={setCurrentPage}
              samples={samples}
              onSelectSample={handleAnalyzeSample}
            />
          )}

          {currentPage === 'upload' && (
            <UploadPage
              onAnalyzeFile={handleAnalyzeFile}
              onAnalyzeSample={handleAnalyzeSample}
              isLoading={isLoading}
              error={error}
              samples={samples}
            />
          )}

          {currentPage === 'results' && prediction && (
            <ResultsPage
              prediction={prediction}
              onNavigate={setCurrentPage}
              onGradCAM={handleComputeGradCAM}
              isLoadingGradCAM={isLoadingGradCAM}
            />
          )}

          {currentPage === 'gradcam' && (
            <GradCAMPage
              gradcam={gradcam}
              onNavigate={setCurrentPage}
              isLoading={isLoadingGradCAM}
            />
          )}

          {currentPage === 'detailed' && prediction && (
            <DetailedPage
              prediction={prediction}
              onNavigate={setCurrentPage}
            />
          )}

          {currentPage === 'performance' && (
            <PerformancePage
              performance={performance}
              isLoading={performance === null}
            />
          )}

          {currentPage === 'about' && <AboutPage />}

          {currentPage === 'help' && <HelpPage />}
        </main>
      </div>
    </div>
  );
};
