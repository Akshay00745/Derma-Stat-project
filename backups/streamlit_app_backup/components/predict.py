"""
Skin Lesion Prediction Tab Component for HAM10000 Web Application.
"""

import streamlit as st
from PIL import Image
import numpy as np
import torch
import torch.nn.functional as F

from src import CLASS_NAMES, CLASS_DESCRIPTIONS
from src.utils import validate_image_file, preprocess_pil_image_for_model, plot_probability_chart, CLASS_COLORS
from src.uncertainty import assess_uncertainty


def render_predict_page(model, device):
    st.markdown("### Skin Lesion Image Classification")
    st.markdown(
        "Upload a dermoscopic image (JPG, JPEG, or PNG) to generate 7-class prediction probabilities, "
        "confidence scores, and diagnostic assessments."
    )

    col1, col2 = st.columns([1, 1], gap="large")

    with col1:
        st.markdown('<div class="med-card">', unsafe_allow_html=True)
        st.markdown('<div class="med-card-title">Upload Dermoscopic Image</div>', unsafe_allow_html=True)

        uploaded_file = st.file_uploader(
            "Select dermoscopic image file",
            type=["jpg", "jpeg", "png"],
            help="Accepts standard clinical dermoscopy captures (JPG, JPEG, PNG)"
        )

        if uploaded_file is not None:
            file_bytes = uploaded_file.read()
            is_valid, pil_image, msg = validate_image_file(file_bytes)

            if not is_valid:
                st.error(f"Image Validation Error: {msg}")
                st.session_state["current_image"] = None
                st.session_state["prediction_results"] = None
            else:
                st.session_state["current_image"] = pil_image
                st.image(pil_image, caption=f"Uploaded: {uploaded_file.name} ({pil_image.width}x{pil_image.height})", use_container_width=True)

                predict_btn = st.button("Run AI Diagnostic Analysis", type="primary", use_container_width=True)

                if predict_btn:
                    if model is None:
                        st.error("Model weights are not loaded. Train the model first or verify checkpoint availability.")
                    else:
                        with st.spinner("Executing neural network forward pass..."):
                            input_tensor = preprocess_pil_image_for_model(pil_image, device=device)
                            with torch.no_grad():
                                logits = model(input_tensor)
                                probs = F.softmax(logits, dim=1).squeeze().cpu().numpy()

                            top_idx = int(np.argmax(probs))
                            top_class = CLASS_NAMES[top_idx]
                            top_prob = float(probs[top_idx])

                            uncertainty_res = assess_uncertainty(probs)

                            st.session_state["prediction_results"] = {
                                "probabilities": probs,
                                "top_class": top_class,
                                "top_class_full": CLASS_DESCRIPTIONS[top_class],
                                "top_confidence": top_prob,
                                "input_tensor": input_tensor,
                                "uncertainty": uncertainty_res
                            }
        else:
            st.info("Please upload a dermoscopic image to begin analysis.")
            st.session_state["current_image"] = None
            st.session_state["prediction_results"] = None

        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="med-card">', unsafe_allow_html=True)
        st.markdown('<div class="med-card-title">Diagnostic Prediction Output</div>', unsafe_allow_html=True)

        res = st.session_state.get("prediction_results", None)

        if res is not None:
            top_class = res["top_class"]
            top_full = res["top_class_full"]
            top_conf = res["top_confidence"]
            probs = res["probabilities"]
            unc = res["uncertainty"]

            # Main prediction card
            st.markdown(f"""
            <div class="prediction-box">
                <div style="font-size: 0.9rem; text-transform: uppercase; letter-spacing: 0.05em; color: #64748b;">
                    Predicted Diagnostic Category
                </div>
                <div class="prediction-category">{top_class.upper()} &mdash; {top_full}</div>
                <div class="prediction-confidence">{top_conf * 100:.1f}%</div>
                <div style="margin-top: 0.5rem;">
                    <span class="{'badge-reliable' if unc['is_reliable'] else 'badge-uncertain'}">
                        {unc['confidence_level']} ({unc['status']})
                    </span>
                </div>
            </div>
            """, unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)

            # 7-Class probability bar chart
            st.markdown("##### 7-Class Confidence Distribution")
            chart_fig = plot_probability_chart(probs, top_class)
            st.pyplot(chart_fig, use_container_width=True)

            # Clinical context note
            st.markdown(f"""
            <div style="background: #f1f5f9; padding: 0.85rem 1.1rem; border-radius: 8px; font-size: 0.88rem; color: #334155; margin-top: 1rem;">
                <strong>Clinical Interpretation Note:</strong> The neural network identified dermoscopic features most consistent with 
                <em>{top_full}</em> at <strong>{top_conf*100:.1f}%</strong> confidence probability. 
                Navigate to the <strong>Grad-CAM Explainability</strong> tab to inspect the visual feature regions that drove this score.
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div style="text-align: center; padding: 3rem 1rem; color: #94a3b8;">
                <svg width="48" height="48" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round">
                    <circle cx="12" cy="12" r="10"></circle>
                    <line x1="12" y1="8" x2="12" y2="12"></line>
                    <line x1="12" y1="16" x2="12.01" y2="16"></line>
                </svg>
                <p style="margin-top: 1rem; font-size: 0.95rem;">Upload an image and click <strong>Run AI Diagnostic Analysis</strong> to generate results.</p>
            </div>
            """, unsafe_allow_html=True)

        st.markdown('</div>', unsafe_allow_html=True)
