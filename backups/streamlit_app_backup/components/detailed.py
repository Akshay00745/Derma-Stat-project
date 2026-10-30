"""
Detailed Analysis View Component for Dermastat Air Vision.
Matching Panel 5 of UI Design Reference.
"""

import streamlit as st
import numpy as np
import io

from src import CLASS_NAMES, CLASS_DESCRIPTIONS


def render_detailed_page():
    res = st.session_state.get("analysis_result", None)

    if res is None:
        st.markdown("""
        <div class="air-card" style="text-align: center; padding: 3rem 1.5rem;">
            <h3 style="margin: 0 0 0.5rem 0; color: #0f172a;">No Detailed Analysis Found</h3>
            <p style="color: #64748b; font-size: 0.95rem; margin-bottom: 1.5rem;">
                Please upload an image and run analysis to view the comprehensive diagnostic report.
            </p>
        </div>
        """, unsafe_allow_html=True)
        if st.button("Go to Upload", type="primary"):
            st.session_state["active_page"] = "Upload"
            st.rerun()
        return

    st.markdown('<h2 style="font-size: 1.6rem; font-weight: 800; color: #0f172a; margin: 0 0 1.25rem 0;">Detailed Analysis</h2>', unsafe_allow_html=True)

    top_class = res["top_class"]
    top_full = res["top_class_full"]
    top_conf = res["top_confidence"]
    probs = res["probabilities"]
    unc = res["uncertainty"]
    pil_img = res["image"]
    filename = res.get("filename", "lesion_image.jpg")

    # Three Stat Pill Boxes (Matching Panel 5)
    s_col1, s_col2, s_col3 = st.columns(3, gap="medium")
    with s_col1:
        st.markdown(f"""
        <div class="air-card" style="padding: 1rem 1.25rem; margin-bottom: 1rem;">
            <div style="font-size: 0.8rem; font-weight: 700; color: #64748b;">Predicted Class</div>
            <div style="font-size: 1.25rem; font-weight: 800; color: #2563eb; margin-top: 0.25rem;">
                {top_full} ({top_class.upper()})
            </div>
        </div>
        """, unsafe_allow_html=True)

    with s_col2:
        st.markdown(f"""
        <div class="air-card" style="padding: 1rem 1.25rem; margin-bottom: 1rem;">
            <div style="font-size: 0.8rem; font-weight: 700; color: #64748b;">Confidence</div>
            <div style="font-size: 1.25rem; font-weight: 800; color: #2563eb; margin-top: 0.25rem;">
                {top_conf * 100:.1f}%
            </div>
        </div>
        """, unsafe_allow_html=True)

    with s_col3:
        unc_label = "Low" if unc["is_reliable"] else "Borderline / High"
        unc_color = "#10b981" if unc["is_reliable"] else "#f59e0b"
        st.markdown(f"""
        <div class="air-card" style="padding: 1rem 1.25rem; margin-bottom: 1rem;">
            <div style="font-size: 0.8rem; font-weight: 700; color: #64748b;">Uncertainty</div>
            <div style="font-size: 1.25rem; font-weight: 800; color: {unc_color}; margin-top: 0.25rem;">
                {unc_label}
            </div>
        </div>
        """, unsafe_allow_html=True)

    # 2-Column Split: Scores & Key Takeaways on Left, Image on Right (Matching Panel 5)
    col_left, col_right = st.columns([1.6, 1], gap="large")

    with col_left:
        # Class-wise Confidence Scores Card
        st.markdown('<div class="air-card">', unsafe_allow_html=True)
        st.markdown('<div style="font-size: 1.05rem; font-weight: 700; color: #0f172a; margin-bottom: 1.25rem;">Class-wise Confidence Scores</div>', unsafe_allow_html=True)

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

        # Key Takeaways Box (Matching Panel 5)
        st.markdown(f"""
        <div class="air-card">
            <div style="display: flex; align-items: center; gap: 0.5rem; font-weight: 800; color: #0f172a; font-size: 1rem; margin-bottom: 0.75rem;">
                <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="#2563eb" stroke-width="2.5"><rect x="3" y="11" width="18" height="11" rx="2" ry="2"></rect><path d="M7 11V7a5 5 0 0 1 10 0v4"></path></svg>
                Key Takeaways
            </div>
            <ul style="margin: 0; padding-left: 1.25rem; font-size: 0.88rem; color: #475569; line-height: 1.6;">
                <li>The model is <strong>{'highly confident' if top_conf > 0.8 else 'moderately confident'}</strong> in its prediction ({top_conf*100:.1f}%).</li>
                <li>The highlighted Grad-CAM regions are centered over distinct lesion morphological features.</li>
                <li>This prediction is considered <strong>{'reliable' if unc['is_reliable'] else 'uncertain / borderline'}</strong> based on our validation criteria.</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

    with col_right:
        # Image Preview Card (Matching Panel 5)
        st.markdown('<div class="air-card" style="text-align: center;">', unsafe_allow_html=True)
        st.markdown('<div style="font-size: 1rem; font-weight: 700; color: #0f172a; margin-bottom: 0.75rem; text-align: left;">Image Preview</div>', unsafe_allow_html=True)
        st.image(pil_img, use_container_width=True)
        st.markdown(f'<div style="font-size: 0.82rem; color: #94a3b8; margin-top: 0.5rem;">{filename}</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    # Action Buttons (Matching Panel 5)
    btn_row1, btn_row2 = st.columns([1, 1], gap="small")
    with btn_row1:
        # Prepare text report for download
        report_text = f"""DERMASTAT AIR VISION - CLINICAL ANALYSIS REPORT
==================================================
Date: 2026-10-30
File: {filename}
Model: EfficientNetB0 (HAM10000 7-Class Transfer Learning)

PREDICTION SUMMARY:
- Primary Category: {top_full} ({top_class.upper()})
- Confidence Score: {top_conf * 100:.2f}%
- Uncertainty Assessment: {'RELIABLE' if unc['is_reliable'] else 'UNCERTAIN'} (Threshold: {unc['threshold_used']*100:.1f}%)

7-CLASS CONFIDENCE DISTRIBUTION:
""" + "\n".join([f"- {CLASS_NAMES[i].upper():5s} ({CLASS_DESCRIPTIONS[CLASS_NAMES[i]]}): {probs[i]*100:.2f}%" for i in sorted_indices]) + """

==================================================
RESPONSIBLE AI DISCLAIMER:
This report was generated by an educational AI prototype trained on the HAM10000 dataset.
It is NOT a medical diagnosis or clinical finding.
Always seek the advice of a qualified healthcare provider with any skin concerns.
"""
        st.download_button(
            label="Download Report",
            data=report_text,
            file_name=f"Dermastat_Report_{top_class.upper()}.txt",
            mime="text/plain",
            type="primary",
            use_container_width=True
        )

    with btn_row2:
        if st.button("Back to Home", key="btn_detailed_home", type="secondary", use_container_width=True):
            st.session_state["active_page"] = "Home"
            st.rerun()
