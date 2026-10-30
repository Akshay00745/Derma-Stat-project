"""
Uncertainty Assessment Tab Component for HAM10000 Web Application.
"""

import streamlit as st
import os
import json
import numpy as np

from src.uncertainty import load_calibrated_threshold, assess_uncertainty
from src import CLASS_NAMES, CLASS_DESCRIPTIONS


def render_uncertainty_page():
    st.markdown("### Model Prediction Uncertainty Assessment")
    st.markdown(
        "AI systems operating in medical image domains must be capable of identifying when their own predictions "
        "may be unreliable. This module assesses the model's confidence against an empirical validation-calibrated threshold."
    )

    res = st.session_state.get("prediction_results", None)

    # Load calibration metadata if available
    calib_path = r"D:\HAM10000\checkpoints\uncertainty_calibration.json"
    calib_data = {}
    if os.path.exists(calib_path):
        try:
            with open(calib_path, "r", encoding="utf-8") as f:
                calib_data = json.load(f)
        except Exception:
            pass

    calibrated_thresh = calib_data.get("calibrated_threshold", load_calibrated_threshold())

    # Calibration parameters overview
    st.markdown('<div class="med-card">', unsafe_allow_html=True)
    st.markdown('<div class="med-card-title">Empirical Validation Calibration Parameters</div>', unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("Calibrated Threshold", f"{calibrated_thresh * 100:.1f}%")
    with c2:
        st.metric("Calibration Split", "Validation Set Only")
    with c3:
        st.metric("Methodology", "Maximum Softmax Probability (MSP)")
    with c4:
        st.metric("Data Leakage Shield", "Test Set Untouched")

    st.markdown('</div>', unsafe_allow_html=True)

    if res is None:
        st.info("No active image analysis found. Upload an image in the **Skin Lesion Prediction** tab to evaluate prediction uncertainty.")
        return

    unc = res["uncertainty"]
    top_class = res["top_class"]
    top_full = res["top_class_full"]
    top_conf = res["top_confidence"]
    margin = unc.get("prediction_margin", 0.0)

    st.markdown('<div class="med-card">', unsafe_allow_html=True)
    st.markdown(f'<div class="med-card-title">Current Evaluation for {top_class.upper()} ({top_full})</div>', unsafe_allow_html=True)

    # Prominent indicator banner
    if unc["is_reliable"]:
        st.markdown(f"""
        <div style="background: #ecfdf5; border: 1.5px solid #10b981; border-radius: 10px; padding: 1.25rem; margin-bottom: 1rem;">
            <div style="display: flex; align-items: center; gap: 0.75rem;">
                <span class="badge-reliable" style="font-size: 1rem; padding: 0.4rem 1rem;">STATUS: RELIABLE</span>
                <span style="font-weight: 700; color: #065f46; font-size: 1.1rem;">High Model Confidence ({top_conf*100:.1f}%)</span>
            </div>
            <p style="margin: 0.75rem 0 0 0; color: #047857; font-size: 0.95rem; line-height: 1.5;">
                {unc['explanation']}
            </p>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div style="background: #fffbeb; border: 1.5px solid #f59e0b; border-radius: 10px; padding: 1.25rem; margin-bottom: 1rem;">
            <div style="display: flex; align-items: center; gap: 0.75rem;">
                <span class="badge-uncertain" style="font-size: 1rem; padding: 0.4rem 1rem;">STATUS: UNCERTAIN</span>
                <span style="font-weight: 700; color: #92400e; font-size: 1.1rem;">Low / Borderline Model Confidence ({top_conf*100:.1f}%)</span>
            </div>
            <p style="margin: 0.75rem 0 0 0; color: #b45309; font-size: 0.95rem; line-height: 1.5;">
                {unc['explanation']}
            </p>
        </div>
        """, unsafe_allow_html=True)

    # Detailed metrics breakdown
    uc1, uc2, uc3 = st.columns(3)
    with uc1:
        st.markdown(f"""
        <div class="stat-tile">
            <div class="stat-value">{top_conf * 100:.1f}%</div>
            <div class="stat-label">Model Confidence (MSP)</div>
        </div>
        """, unsafe_allow_html=True)
    with uc2:
        st.markdown(f"""
        <div class="stat-tile">
            <div class="stat-value">{calibrated_thresh * 100:.1f}%</div>
            <div class="stat-label">Acceptance Threshold</div>
        </div>
        """, unsafe_allow_html=True)
    with uc3:
        st.markdown(f"""
        <div class="stat-tile">
            <div class="stat-value">+{margin * 100:.1f}%</div>
            <div class="stat-label">Margin Over 2nd Class</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    st.markdown(f"""
    <div style="background: #f8fafc; border: 1px solid #e2e8f0; border-radius: 10px; padding: 1.25rem;">
        <h5 style="margin: 0 0 0.5rem 0; color: #1e3a8a;">Clinical Recommendation & Action Guidelines</h5>
        <p style="margin: 0; font-size: 0.92rem; color: #334155; line-height: 1.5;">{unc['recommendation']}</p>
        <hr style="border: none; border-top: 1px solid #e2e8f0; margin: 0.75rem 0;">
        <p style="margin: 0; font-size: 0.85rem; color: #64748b;">
            <strong>Important Safety Notice:</strong> An output flagged as "Reliable" does NOT equate to clinical certainty. 
            Skin lesion diagnosis is a complex medical procedure requiring biopsy, histological confirmation, and patient medical history.
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)
