"""
Analysis Result View Component for Dermastat Air Vision.
Matching Panel 3 of UI Design Reference.
"""

import streamlit as st
import numpy as np

from src import CLASS_NAMES, CLASS_DESCRIPTIONS


def render_results_page():
    res = st.session_state.get("analysis_result", None)

    if res is None:
        st.markdown("""
        <div class="air-card" style="text-align: center; padding: 3rem 1.5rem;">
            <svg width="48" height="48" viewBox="0 0 24 24" fill="none" stroke="#94a3b8" stroke-width="1.5">
                <circle cx="12" cy="12" r="10"></circle>
                <line x1="12" y1="8" x2="12" y2="12"></line>
                <line x1="12" y1="16" x2="12.01" y2="16"></line>
            </svg>
            <h3 style="margin: 1rem 0 0.5rem 0; color: #0f172a;">No Active Analysis Found</h3>
            <p style="color: #64748b; font-size: 0.95rem; margin-bottom: 1.5rem;">
                Please upload a skin lesion image first to generate diagnostic classification and confidence scores.
            </p>
        </div>
        """, unsafe_allow_html=True)
        if st.button("Go to Image Upload", type="primary"):
            st.session_state["active_page"] = "Upload"
            st.rerun()
        return

    # Header with "Prediction Completed" badge (Matching Panel 3)
    head_c1, head_c2 = st.columns([2, 1])
    with head_c1:
        st.markdown('<h2 style="font-size: 1.6rem; font-weight: 800; color: #0f172a; margin: 0;">Analysis Result</h2>', unsafe_allow_html=True)
    with head_c2:
        st.markdown("""
        <div style="text-align: right;">
            <span class="status-pill-success">
                <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="3"><polyline points="20 6 9 17 4 12"></polyline></svg>
                Prediction Completed
            </span>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    top_class = res["top_class"]
    top_full = res["top_class_full"]
    top_conf = res["top_confidence"]
    probs = res["probabilities"]
    unc = res["uncertainty"]
    pil_image = res["image"]

    # Main Top Card: Image Thumbnail + Diagnosis (Matching Panel 3)
    st.markdown('<div class="air-card" style="padding: 1.25rem;">', unsafe_allow_html=True)
    top_col_img, top_col_info = st.columns([1, 4], gap="medium")
    with top_col_img:
        st.image(pil_image, use_container_width=True)
    with top_col_info:
        st.markdown(f"""
        <div style="padding-left: 0.5rem;">
            <div style="font-size: 0.85rem; font-weight: 700; color: #64748b; text-transform: uppercase; letter-spacing: 0.05em; margin-bottom: 0.25rem;">
                Predicted Category
            </div>
            <div style="font-size: 1.85rem; font-weight: 800; color: #0f172a; margin-bottom: 0.5rem;">
                {top_full} ({top_class.upper()})
            </div>
            <div style="display: inline-block; background: #eff6ff; color: #2563eb; font-weight: 700; font-size: 0.95rem; padding: 0.35rem 0.85rem; border-radius: 9999px; border: 1px solid #bfdbfe;">
                Confidence: {top_conf * 100:.1f}%
            </div>
        </div>
        """, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    # 2-Column Bottom Layout
    col_scores, col_assessment = st.columns([1.5, 1], gap="large")

    with col_scores:
        st.markdown('<div class="air-card">', unsafe_allow_html=True)
        st.markdown('<div style="font-size: 1.05rem; font-weight: 700; color: #0f172a; margin-bottom: 1.25rem;">Confidence Scores (All 7 Classes)</div>', unsafe_allow_html=True)

        # Sort classes descending by probability for clean visual hierarchy
        sorted_indices = np.argsort(probs)[::-1]
        for idx in sorted_indices:
            c_name = CLASS_NAMES[idx]
            p_val = probs[idx] * 100
            st.markdown(f"""
            <div class="confidence-row">
                <span class="conf-label">{c_name.upper()}</span>
                <div class="conf-bar-track">
                    <div class="conf-bar-fill" style="width: {max(p_val, 1.5):.1f}%;"></div>
                </div>
                <span class="conf-value">{p_val:.1f}%</span>
            </div>
            """, unsafe_allow_html=True)

        st.markdown('</div>', unsafe_allow_html=True)

    with col_assessment:
        st.markdown('<div class="air-card">', unsafe_allow_html=True)

        # Uncertainty Card (Matching Panel 3 right box)
        if unc["is_reliable"]:
            st.markdown(f"""
            <div style="background: #ecfdf5; border: 1px solid #a7f3d0; border-radius: 10px; padding: 1.1rem; margin-bottom: 1.25rem;">
                <div style="display: flex; align-items: center; gap: 0.5rem; font-weight: 700; color: #065f46; font-size: 0.95rem; margin-bottom: 0.35rem;">
                    <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5"><polyline points="20 6 9 17 4 12"></polyline></svg>
                    Uncertainty Assessment
                </div>
                <div style="font-weight: 800; font-size: 1.05rem; color: #047857; margin-bottom: 0.25rem;">
                    Low uncertainty
                </div>
                <div style="font-size: 0.85rem; color: #065f46; line-height: 1.45;">
                    This prediction is considered reliable based on our validation criteria ({unc['threshold_used']*100:.1f}% threshold).
                </div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div style="background: #fffbeb; border: 1px solid #fde68a; border-radius: 10px; padding: 1.1rem; margin-bottom: 1.25rem;">
                <div style="display: flex; align-items: center; gap: 0.5rem; font-weight: 700; color: #92400e; font-size: 0.95rem; margin-bottom: 0.35rem;">
                    <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5"><path d="M10.29 3.86L1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0z"></path><line x1="12" y1="9" x2="12" y2="13"></line><line x1="12" y1="17" x2="12.01" y2="17"></line></svg>
                    Uncertainty Assessment
                </div>
                <div style="font-weight: 800; font-size: 1.05rem; color: #b45309; margin-bottom: 0.25rem;">
                    High uncertainty
                </div>
                <div style="font-size: 0.85rem; color: #92400e; line-height: 1.45;">
                    This prediction falls below our validation reliability threshold ({unc['threshold_used']*100:.1f}%). May indicate ambiguous features.
                </div>
            </div>
            """, unsafe_allow_html=True)

        # Action Buttons (Matching Panel 3)
        if st.button("View Grad-CAM", key="btn_goto_gradcam", type="primary", use_container_width=True):
            st.session_state["active_page"] = "Grad-CAM"
            st.rerun()

        st.markdown("<div style='height: 0.5rem;'></div>", unsafe_allow_html=True)

        if st.button("View Detailed Results", key="btn_goto_detailed", type="secondary", use_container_width=True):
            st.session_state["active_page"] = "Detailed Analysis"
            st.rerun()

        st.markdown('</div>', unsafe_allow_html=True)

    # Disclaimer note
    st.markdown("""
    <div class="disclaimer-card">
        <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"></path></svg>
        <span>
            <strong>Medical Disclaimer:</strong> AI predictions and confidence scores reflect probabilistic pattern recognition on the HAM10000 dataset and are not a confirmed clinical diagnosis. Consult a qualified dermatologist.
        </span>
    </div>
    """, unsafe_allow_html=True)
