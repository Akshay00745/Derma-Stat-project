"""
Grad-CAM Visualization View Component for Dermastat Air Vision.
Matching Panel 4 of UI Design Reference.
"""

import streamlit as st
from PIL import Image

def render_explainability_page():
    res = st.session_state.get("analysis_result", None)

    if res is None:
        st.markdown("""
        <div class="air-card" style="text-align: center; padding: 3rem 1.5rem;">
            <h3 style="margin: 0 0 0.5rem 0; color: #0f172a;">No Image Under Analysis</h3>
            <p style="color: #64748b; font-size: 0.95rem; margin-bottom: 1.5rem;">
                Please upload an image and generate predictions to inspect Grad-CAM explainability heatmaps.
            </p>
        </div>
        """, unsafe_allow_html=True)
        if st.button("Go to Upload", type="primary"):
            st.session_state["active_page"] = "Upload"
            st.rerun()
        return

    st.markdown("""
    <div style="margin-bottom: 1.5rem;">
        <h2 style="font-size: 1.6rem; font-weight: 800; color: #0f172a; margin: 0 0 0.35rem 0;">Grad-CAM Visualization</h2>
        <p style="color: #64748b; font-size: 0.95rem; margin: 0;">
            This heatmap shows the regions of the image that influenced the model's prediction.
        </p>
    </div>
    """, unsafe_allow_html=True)

    pil_img = res["image"]
    heatmap_colored = res["heatmap_colored"]
    overlay = res["overlay"]

    # 3-Column Image Comparison (Matching Panel 4)
    c1, c2, c3 = st.columns(3, gap="medium")

    with c1:
        st.markdown('<div class="air-card" style="text-align: center; padding: 1rem;">', unsafe_allow_html=True)
        st.image(pil_img, use_container_width=True)
        st.markdown('<div style="font-weight: 700; font-size: 0.95rem; color: #0f172a; margin-top: 0.5rem;">Original Image</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with c2:
        st.markdown('<div class="air-card" style="text-align: center; padding: 1rem;">', unsafe_allow_html=True)
        st.image(heatmap_colored, use_container_width=True)
        st.markdown('<div style="font-weight: 700; font-size: 0.95rem; color: #0f172a; margin-top: 0.5rem;">Heatmap (Grad-CAM)</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with c3:
        st.markdown('<div class="air-card" style="text-align: center; padding: 1rem;">', unsafe_allow_html=True)
        st.image(overlay, use_container_width=True)
        st.markdown('<div style="font-weight: 700; font-size: 0.95rem; color: #0f172a; margin-top: 0.5rem;">Overlay</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    # Blue Interpretation Callout Box (Matching Panel 4)
    st.markdown("""
    <div style="background: #eff6ff; border: 1px solid #bfdbfe; border-radius: 10px; padding: 1.1rem 1.25rem; margin-top: 1.25rem; display: flex; align-items: flex-start; gap: 0.75rem;">
        <div style="color: #2563eb; font-size: 1.2rem; line-height: 1;">ℹ️</div>
        <div>
            <div style="font-weight: 800; color: #1e3a8a; font-size: 0.95rem; margin-bottom: 0.25rem;">Interpretation</div>
            <div style="color: #1e40af; font-size: 0.9rem; line-height: 1.5;">
                The highlighted areas (red/yellow) show the regions that contributed most to the model's prediction. 
                Cooler regions (blue) had minimal influence on the diagnostic classification.
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Button: "Back to Results" (Matching Panel 4)
    if st.button("Back to Results", key="btn_gradcam_back", type="secondary"):
        st.session_state["active_page"] = "Results"
        st.rerun()
