"""
Model performance dashboard schemas.
"""

from typing import Dict, List, Any
from pydantic import BaseModel

class ClassMetric(BaseModel):
    code: str
    name: str
    precision: float
    precision_pct: float
    recall: float
    recall_pct: float
    f1_score: float
    f1_pct: float
    support: int

class PerformanceMetrics(BaseModel):
    test_accuracy: float
    test_accuracy_pct: float
    macro_f1: float
    macro_f1_pct: float
    macro_precision: float
    macro_precision_pct: float
    macro_recall: float
    macro_recall_pct: float
    weighted_f1: float
    weighted_f1_pct: float
    total_test_samples: int
    per_class: List[ClassMetric]
    confusion_matrix: List[List[int]]
    class_labels: List[str]
    training_epochs_completed: int
    best_val_macro_f1: float
