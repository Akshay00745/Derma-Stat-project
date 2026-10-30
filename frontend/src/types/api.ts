export interface ClassConfidence {
  code: string;
  name: string;
  probability: number;
  percentage: number;
  color: string;
}

export interface UncertaintyInfo {
  threshold: number;
  confidence: number;
  is_uncertain: boolean;
  status: string;
  level: string;
  message: string;
}

export interface PredictionResult {
  predicted_code: string;
  predicted_name: string;
  confidence: number;
  confidence_percentage: number;
  is_confident: boolean;
  uncertainty: UncertaintyInfo;
  all_probabilities: ClassConfidence[];
  image_preview_b64?: string;
  original_dimensions?: {
    width: number;
    height: number;
  };
}

export interface GradCAMResult {
  predicted_code: string;
  predicted_name: string;
  original_b64: string;
  heatmap_b64: string;
  overlay_b64: string;
  target_layer: string;
  description: string;
}

export interface SampleItem {
  id: string;
  filename: string;
  code: string;
  name: string;
  description: string;
  thumbnail_b64: string;
}

export interface ClassMetric {
  code: string;
  name: string;
  precision: number;
  precision_pct: number;
  recall: number;
  recall_pct: number;
  f1_score: number;
  f1_pct: number;
  support: number;
}

export interface PerformanceMetrics {
  test_accuracy: number;
  test_accuracy_pct: number;
  macro_f1: number;
  macro_f1_pct: number;
  macro_precision: number;
  macro_precision_pct: number;
  macro_recall: number;
  macro_recall_pct: number;
  weighted_f1: number;
  weighted_f1_pct: number;
  total_test_samples: number;
  per_class: ClassMetric[];
  confusion_matrix: number[][];
  class_labels: string[];
  training_epochs_completed: number;
  best_val_macro_f1: number;
}

export interface HealthResponse {
  status: string;
  application: string;
  version: string;
  model_loaded: boolean;
  device: string;
  cuda_available: boolean;
}
