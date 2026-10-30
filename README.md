# Dermastat AI Vision: Full-Stack AI Skin Lesion Analysis Platform

[![React](https://img.shields.io/badge/Frontend-React%2018%20%7C%20TypeScript%20%7C%20Vite%20%7C%20Tailwind-61DAFB.svg?logo=react)](https://react.dev/)
[![FastAPI](https://img.shields.io/badge/Backend-FastAPI%20%7C%20Uvicorn-009688.svg?logo=fastapi)](https://fastapi.tiangolo.com/)
[![PyTorch](https://img.shields.io/badge/Deep%20Learning-PyTorch%202.6%20%7C%20CUDA%2012.4-EE4C2C.svg?logo=pytorch)](https://pytorch.org/)
[![EfficientNet-B0](https://img.shields.io/badge/Backbone-EfficientNet--B0-0080FF.svg)](https://arxiv.org/abs/1905.11946)
[![Grad-CAM](https://img.shields.io/badge/Explainability-Grad--CAM-4CAF50.svg)](https://arxiv.org/abs/1610.02391)
[![Dataset](https://img.shields.io/badge/Dataset-HAM10000-orange.svg)](https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/DBW86T)

**Dermastat AI Vision** is a modern, full-stack medical AI web application for multi-class classification, uncertainty quantification, and visual explainability of dermoscopic skin lesions across all 7 diagnostic categories of the **HAM10000** dataset.

- **Subtitle:** Deep Learning Architecture, Data Provenance, and Clinical Mission
- **Project Type:** Research & Educational Prototype
- **Technology:** Deep Learning, Computer Vision, Explainable AI, and Uncertainty Assessment

The system completely replaces legacy Streamlit prototypes with an enterprise-grade architecture:
- **Frontend**: Responsive Single-Page Application (SPA) built with **React 18**, **TypeScript**, **Vite**, **Tailwind CSS**, and **Lucide Icons**, faithfully implementing the 9-panel clinical dashboard design.
- **Backend API**: High-performance asynchronous REST API powered by **FastAPI**, **Pydantic v2**, and **Uvicorn**, serving real-time PyTorch inference with CUDA GPU acceleration.
- **AI Engine**: Pretrained **EfficientNetB0** fine-tuned on lesion-stratified HAM10000 partitions with inverse-frequency class-weighting, native **Grad-CAM visual explainability**, and **validation-calibrated uncertainty quantification** ($\tau = 83.15\%$).

---

## ⚠️ Medical AI Disclaimer
> **IMPORTANT NOTICE:** Dermastat AI Vision is an investigational and educational AI prototype developed strictly for demonstration, benchmarking, and clinical research. It is **NOT** a certified medical diagnostic device, does **NOT** provide medical diagnoses, and must **NEVER** substitute for professional dermatological consultation, dermoscopic biopsy, or histological confirmation by a licensed physician.

---

## 🌟 Architecture & Key Features

1. **Seven Diagnostic Categories**:
   Full taxonomy coverage: Actinic Keratoses (*AKIEC*), Basal Cell Carcinoma (*BCC*), Benign Keratosis (*BKL*), Dermatofibroma (*DF*), Melanoma (*MEL*), Melanocytic Nevi (*NV*), and Vascular Lesions (*VASC*).
2. **Real Model Inference (No Fabricated Metrics)**:
   Powered directly by `checkpoints/best_model.pth` (Epoch 6, Validation Macro-F1: 70.86%).
3. **Leakage-Free Lesion Stratification**:
   Images belonging to the same lesion ID are strictly isolated within a single split ($Train \cap Val \cap Test = \emptyset$).
4. **Native PyTorch Grad-CAM Visual Explainability**:
   Tracks activations and gradients at `EfficientNetB0.features[-1]`, generating side-by-side comparisons of raw dermoscopy, color-mapped activation heatmaps, and 45% alpha-blended overlays.
5. **Validation-Calibrated Uncertainty Engine**:
   Evaluates prediction confidence against an empirically calibrated threshold ($\tau = 0.8315$). Predictions meeting the threshold achieved **90.61% accuracy** in validation testing; ambiguous predictions automatically trigger advisory warnings.
6. **Downloadable Clinical Summary Reports**:
   Exports structured diagnostic JSON reports containing predicted class, confidence, uncertainty flags, complete class probabilities, and model provenance.
7. **Zero Streamlit Dependencies**:
   Clean separation between client and server with robust CORS support, input validation, and file-size guardrails.

---

## 🖥️ UI/UX Design: The 9 Reference Panels

The frontend faithfully recreates the 9 panels from the clinical UI specification:

| # | Panel View | Route / Component | Description & Key Functionality |
| :---: | :--- | :--- | :--- |
| **1** | **Landing Page / Home** | `HomePage.tsx` | • Hero: *"Smarter AI for Healthier Skin"* with sample lesion preview.<br>• Dual CTAs: **"Upload Image"** and **"Learn More"**.<br>• 5 feature pill cards.<br>• Interactive 7-category avatar cards for one-click testing.<br>• Educational disclaimer footer. |
| **2** | **Image Upload Page** | `UploadPage.tsx` | • Drag-and-drop zone accepting `JPG`, `PNG`, `JPEG` (up to 50MB).<br>• Image guidelines card (*Clear & well-lit*, *224×224+*, *Avoid blur*).<br>• 7 quick-preset sample buttons for instant testing.<br>• Image preview with file metadata, replace/remove buttons, and **"Run AI Analysis"**. |
| **3** | **Prediction Result Page** | `ResultsPage.tsx` | • Top predicted category card with green *"Prediction Completed"* badge.<br>• Color-coded progress bars for all 7 diagnostic classes.<br>• **Uncertainty Assessment Card** (Green for Low Uncertainty $\ge 83.15\%$, Amber for High Uncertainty).<br>• Navigation buttons: **"View Grad-CAM"** and **"View Detailed Results"**. |
| **4** | **Grad-CAM Visualization Page** | `GradCAMPage.tsx` | • **3-Column Side-by-Side Comparison**: **Original Image**, **Heatmap (Grad-CAM)** (Jet colormap), and **Overlay** (45% alpha blend).<br>• Clinical interpretation alert box explaining morphological hot-spots.<br>• **"Back to Results"** button. |
| **5** | **Detailed Results Page** | `DetailedPage.tsx` | • **3 Summary Pills**: *Predicted Class*, *Confidence Score*, *Uncertainty Status*.<br>• Class-wise confidence score list and lesion image preview.<br>• **Key Takeaways Box** summarizing clinical findings.<br>• **"Download Report"** button (exports JSON) and **"Back to Home"**. |
| **6** | **Model Performance Dashboard** | `PerformancePage.tsx` | • **4 KPI Cards**: *Accuracy (76.04%)*, *Macro-F1 (69.44%)*, *Precision (66.10%)*, *Recall (77.07%)*.<br>• Full 7-Class Table with precision, recall, F1, and support counts.<br>• Interactive Confusion Matrix with toggle for absolute counts and normalized percentages. |
| **7** | **About Page** | `AboutPage.tsx` | • Educational mission overview, dataset architecture, and model provenance.<br>• **4 Strategic Goal Cards**: *Demonstrate AI*, *Visual Explanations*, *Model Performance*, *Responsible AI*.<br>• Prominent Clinical AI Disclaimer card. |
| **8** | **Help & Support Page** | `HelpPage.tsx` | • **7 Expandable FAQ Accordions** addressing image upload, file formats, the 7 classes, confidence calculation, uncertainty warnings, Macro-F1 vs Accuracy, and diagnostic boundaries.<br>• Support contact channel (`support@dermastat.ai`). |
| **9** | **Reusable Navigation Sidebar** | `Sidebar.tsx` | • Application branding with medical icon.<br>• Active navigation links with Lucide icons.<br>• Live AI Engine badge confirming GPU/CPU model status.<br>• Persistent footer: *"Educational Prototype — Not a medical diagnostic tool"*. |

---

## 📊 Benchmark Evaluation (Reserved Test Set)

Evaluated strictly **once** on the held-out test split (`data/processed/test.csv`, 1,494 images):

| Metric | Score | Percentage |
| :--- | :---: | :---: |
| **Overall Test Accuracy** | **0.7604** | **76.04%** |
| **Macro-Averaged F1-Score** | **0.6944** | **69.44%** |
| **Weighted-Averaged F1-Score** | **0.7825** | **78.25%** |
| **Macro-Averaged Recall** | **0.7707** | **77.07%** |
| **Macro-Averaged Precision** | **0.6610** | **66.10%** |

### Per-Class Performance
| Code | Diagnostic Category | Precision | Recall (Sensitivity) | F1-Score | Test Samples |
| :---: | :--- | :---: | :---: | :---: | :---: |
| **AKIEC** | Actinic Keratoses | 71.43% | 54.35% | 61.73% | 46 |
| **BCC** | Basal Cell Carcinoma | 65.74% | **79.78%** | 72.08% | 89 |
| **BKL** | Benign Keratosis-like | 57.37% | 68.13% | 62.29% | 160 |
| **DF** | Dermatofibroma | 57.14% | 85.71% | 68.57% | 14 |
| **MEL** | Melanoma | 37.50% | **79.59%** | 50.98% | 147 |
| **NV** | Melanocytic Nevi | **97.51%** | 76.92% | **86.00%** | 1,018 |
| **VASC** | Vascular Lesions | 76.00% | **95.00%** | **84.44%** | 20 |

---

## 📁 Repository Structure

```
D:\HAM10000\
├── backend\
│   ├── app\
│   │   ├── main.py                 # FastAPI application, CORS & static file mount
│   │   ├── core\
│   │   │   └── config.py           # Application settings & thresholds
│   │   ├── schemas\
│   │   │   ├── health.py           # Health response schema
│   │   │   ├── prediction.py       # Inference, uncertainty & Grad-CAM schemas
│   │   │   └── performance.py      # Benchmark performance schemas
│   │   ├── services\
│   │   │   ├── model_service.py    # Singleton model loader (CUDA/CPU)
│   │   │   ├── inference_service.py# Preprocessing & inference pipeline
│   │   │   ├── gradcam_service.py  # Grad-CAM heatmap & overlay generator
│   │   │   ├── performance_service.py # Benchmark metrics reader
│   │   │   └── report_service.py   # Downloadable report builder
│   │   └── api\
│   │       ├── router.py           # API router aggregation (/api)
│   │       └── routes\
│   │           ├── health.py       # GET  /api/health
│   │           ├── predictions.py  # POST /api/predict/upload & /sample/{id}
│   │           ├── explainability.py # POST /api/gradcam/upload & /sample/{id}
│   │           ├── performance.py  # GET  /api/performance
│   │           ├── samples.py      # GET  /api/samples
│   │           └── reports.py      # POST /api/reports/generate
│   └── requirements.txt            # Backend Python dependencies
├── frontend\
│   ├── src\
│   │   ├── App.tsx                 # Main application controller
│   │   ├── main.tsx                # React DOM entry point
│   │   ├── index.css               # Tailwind & custom CSS
│   │   ├── types\api.ts            # TypeScript interfaces
│   │   ├── services\api.ts         # Centralized API client
│   │   ├── components\
│   │   │   ├── Sidebar.tsx         # Panel 9: Reusable navigation sidebar
│   │   │   └── Header.tsx          # Top navigation header
│   │   └── pages\
│   │       ├── HomePage.tsx        # Panel 1: Hero & 7 Category cards
│   │       ├── UploadPage.tsx      # Panel 2: Dropzone & image guidelines
│   │       ├── ResultsPage.tsx     # Panel 3: Prediction & confidence bars
│   │       ├── GradCAMPage.tsx     # Panel 4: 3-column Grad-CAM heatmaps
│   │       ├── DetailedPage.tsx    # Panel 5: Detailed analysis & takeaways
│   │       ├── PerformancePage.tsx # Panel 6: Benchmark dashboard & matrix
│   │       ├── AboutPage.tsx       # Panel 7: System goals & clinical disclaimer
│   │       └── HelpPage.tsx        # Panel 8: 7 expandable FAQs
│   ├── public\samples\             # Pre-cached representative sample images
│   ├── dist\                       # Production optimized build bundle
│   ├── package.json
│   ├── vite.config.ts
│   └── tailwind.config.js
├── checkpoints\
│   ├── best_model.pth              # EfficientNetB0 checkpoint (Val F1 70.86%)
│   ├── training_history.json       # Training loss & accuracy records
│   └── uncertainty_calibration.json# Calibrated threshold (83.15%)
├── data\processed\                 # Verified lesion-stratified manifests
├── results\                        # Benchmark evaluation outputs
├── samples\                        # Cached representative lesion photographs
├── src\                            # Core ML modules (dataset, model, gradcam)
├── requirements.txt                # Unified project dependencies
├── .env.example
└── README.md
```

---

## 🚀 How to Run the Application

### Option 1: Unified Server (Recommended)
FastAPI serves both the compiled React frontend and the REST API from a single port (`8000`):

```powershell
# Navigate to project directory
cd D:\HAM10000

# Start FastAPI production server
C:\Users\hp\anaconda3\python.exe -m uvicorn backend.app.main:app --host 0.0.0.0 --port 8000
```
Open your browser at:
```
http://localhost:8000
```
- Web Application: `http://localhost:8000`
- Interactive API Docs (Swagger): `http://localhost:8000/docs`
- Health Check: `http://localhost:8000/api/health`

---

### Option 2: Independent Development Servers
For frontend development with live hot-reloading:

**Terminal 1 — Backend API:**
```powershell
cd D:\HAM10000
C:\Users\hp\anaconda3\python.exe -m uvicorn backend.app.main:app --reload --port 8000
```

**Terminal 2 — React Frontend:**
```powershell
cd D:\HAM10000\frontend
npm run dev
```
Open your browser at:
```
http://localhost:5173
```
*(Vite automatically proxies all `/api` requests to `http://127.0.0.1:8000`)*

---

## 🧪 Testing Verification

All system components have been validated with end-to-end automated tests:
```powershell
C:\Users\hp\anaconda3\python.exe D:\HAM10000\scratch\test_backend_api.py
```
- ✅ **Health Check**: Verified model loaded on CUDA GPU.
- ✅ **Multipart Upload**: Verified PIL validation and inference execution.
- ✅ **Grad-CAM**: Verified gradient tracking on `features[-1]` and base64 rendering.
- ✅ **Uncertainty Calibration**: Verified 83.15% safety threshold flagging.
- ✅ **Performance Endpoint**: Verified actual test metrics (76.04% Acc, 69.44% Macro-F1).
- ✅ **Report Export**: Verified JSON diagnostic attachment generation.
- ✅ **Frontend Build**: `tsc && vite build` completed in 1.71s with 0 errors.

---
*Built with ❤️ for AI in Healthcare & Dermatology Research.*
