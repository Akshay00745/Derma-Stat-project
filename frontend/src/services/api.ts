import {
  HealthResponse,
  SampleItem,
  PerformanceMetrics,
  PredictionResult,
  GradCAMResult,
} from '../types/api';

const API_BASE = '/api';

export const api = {
  async getHealth(): Promise<HealthResponse> {
    const res = await fetch(`${API_BASE}/health`);
    if (!res.ok) throw new Error(`Health check failed: ${res.statusText}`);
    return res.json();
  },

  async getSamples(): Promise<SampleItem[]> {
    const res = await fetch(`${API_BASE}/samples`);
    if (!res.ok) throw new Error(`Failed to load samples: ${res.statusText}`);
    return res.json();
  },

  async getPerformance(): Promise<PerformanceMetrics> {
    const res = await fetch(`${API_BASE}/performance`);
    if (!res.ok) throw new Error(`Failed to load performance metrics: ${res.statusText}`);
    return res.json();
  },

  async predictUpload(file: File): Promise<PredictionResult> {
    const formData = new FormData();
    formData.append('file', file);

    const res = await fetch(`${API_BASE}/predict/upload`, {
      method: 'POST',
      body: formData,
    });

    if (!res.ok) {
      const err = await res.json().catch(() => ({ detail: res.statusText }));
      throw new Error(err.detail || 'Prediction failed');
    }
    return res.json();
  },

  async predictSample(sampleId: string): Promise<PredictionResult> {
    const res = await fetch(`${API_BASE}/predict/sample/${sampleId}`, {
      method: 'POST',
    });

    if (!res.ok) {
      const err = await res.json().catch(() => ({ detail: res.statusText }));
      throw new Error(err.detail || 'Sample prediction failed');
    }
    return res.json();
  },

  async explainUpload(file: File, targetClass?: string): Promise<GradCAMResult> {
    const formData = new FormData();
    formData.append('file', file);
    if (targetClass) {
      formData.append('target_class', targetClass);
    }

    const res = await fetch(`${API_BASE}/gradcam/upload`, {
      method: 'POST',
      body: formData,
    });

    if (!res.ok) {
      const err = await res.json().catch(() => ({ detail: res.statusText }));
      throw new Error(err.detail || 'Grad-CAM calculation failed');
    }
    return res.json();
  },

  async explainSample(sampleId: string, targetClass?: string): Promise<GradCAMResult> {
    let url = `${API_BASE}/gradcam/sample/${sampleId}`;
    if (targetClass) {
      url += `?target_class=${encodeURIComponent(targetClass)}`;
    }

    const res = await fetch(url, {
      method: 'POST',
    });

    if (!res.ok) {
      const err = await res.json().catch(() => ({ detail: res.statusText }));
      throw new Error(err.detail || 'Sample Grad-CAM calculation failed');
    }
    return res.json();
  },

  async downloadReport(prediction: PredictionResult): Promise<void> {
    const res = await fetch(`${API_BASE}/reports/generate`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(prediction),
    });

    if (!res.ok) throw new Error('Report generation failed');

    const blob = await res.blob();
    const url = window.URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `Dermastat_Report_${prediction.predicted_code.toUpperCase()}_${Math.round(prediction.confidence_percentage)}pct.json`;
    document.body.appendChild(a);
    a.click();
    window.URL.revokeObjectURL(url);
    document.body.removeChild(a);
  },
};
