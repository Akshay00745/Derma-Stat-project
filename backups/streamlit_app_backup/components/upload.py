"""
Image Upload View Component for Dermastat Air Vision.
Matching Panel 2 of UI Design Reference.
"""

import streamlit as st
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from src import CLASS_NAMES, CLASS_DESCRIPTIONS
from src.utils import validate_image_file, preprocess_pil_image_for_model
from src.gradcam import GradCAM, create_gradcam_overlay
from src.uncertainty import load_calibrated_threshold, assess_uncertainty


def render_upload_page(model, device):
    st.markdown("""
    <div style="margin-bottom: 1.5rem;">
        <h2 style="font-size: 1.6rem; font-weight: 800; color: #0f172a; margin: 0 0 0.35rem 0;">Upload Skin-Lesion Image</h2>
        <p style="color: #64748b; font-size: 0.95rem; margin: 0;">
            Select or drag and drop an image. We support JPG, PNG and JPEG formats.
        </p>
    </div>
    """, unsafe_allow_html=True)

    col_main, col_side = st.columns([1.6, 1], gap="large")

    with col_main:
        st.markdown('<div class="air-card" style="padding: 1.5rem;">', unsafe_allow_html=True)

        uploaded_file = st.file_uploader(
            "Drag and drop your image here",
            type=["jpg", "jpeg", "png"],
            help="Upload a standard clinical dermoscopy capture (JPG, PNG, JPEG, max 5MB)"
        )

        if uploaded_file is not None:
            file_bytes = uploaded_file.read()
            is_valid, pil_image, msg = validate_image_file(file_bytes, max_size_mb=5.0)

            if not is_valid:
                st.error(f"Image Error: {msg}")
                st.session_state["current_image"] = None
                st.session_state["analysis_result"] = None
            else:
                st.session_state["current_image"] = pil_image
                st.session_state["current_filename"] = uploaded_file.name

                st.markdown("---")
                preview_c1, preview_c2 = st.columns([1, 1.2], gap="medium")
                with preview_c1:
                    st.image(pil_image, use_container_width=True, caption="Selected Preview")
                with preview_c2:
                    st.markdown(f"""
                    <div style="background: #f8fafc; border: 1px solid #e2e8f0; border-radius: 8px; padding: 1rem; font-size: 0.88rem;">
                        <div style="font-weight: 700; color: #0f172a; margin-bottom: 0.35rem;">File Details</div>
                        <div style="color: #64748b;"><strong>Name:</strong> {uploaded_file.name}</div>
                        <div style="color: #64748b;"><strong>Dimensions:</strong> {pil_image.width} &times; {pil_image.height} px</div>
                        <div style="color: #64748b;"><strong>Size:</strong> {len(file_bytes) / 1024:.1f} KB</div>
                        <div style="color: #10b981; font-weight: 700; margin-top: 0.35rem;">&bull; Ready for Analysis</div>
                    </div>
                    """, unsafe_allow_html=True)

                st.markdown("<br>", unsafe_allow_html=True)

                btn_c1, btn_c2 = st.columns([1.5, 1])
                with btn_c1:
                    if st.button("Analyze Image", key="btn_run_analysis", type="primary", use_container_width=True):
                        if model is None:
                            st.error("Model weights are not loaded. Check model checkpoint.")
                        else:
                            with st.spinner("Analyzing skin lesion with EfficientNetB0 & Grad-CAM..."):
                                # 1. Model inference
                                input_tensor = preprocess_pil_image_for_model(pil_image, device=device)
                                with torch.no_grad():
                                    logits = model(input_tensor)
                                    probs = F.softmax(logits, dim=1).squeeze().cpu().numpy()

                                top_idx = int(np.argmax(probs))
                                top_class = CLASS_NAMES[top_idx]
                                top_conf = float(probs[top_idx])

                                # 2. Uncertainty assessment
                                thresh = load_calibrated_threshold()
                                unc = assess_uncertainty(probs, threshold=thresh)

                                # 3. Grad-CAM generation
                                tensor_cam = input_tensor.clone().detach().requires_grad_(True)
                                gradcam = GradCAM(model)
                                heatmap = gradcam.generate(tensor_cam, target_class=top_idx)
                                gradcam.remove_hooks()
                                heatmap_colored, overlay = create_gradcam_overlay(pil_image, heatmap, alpha=0.45)

                                # Save full state
                                st.session_state["analysis_result"] = {
                                    "image": pil_image,
                                    "filename": uploaded_file.name,
                                    "probabilities": probs,
                                    "top_class": top_class,
                                    "top_class_full": CLASS_DESCRIPTIONS[top_class],
                                    "top_confidence": top_conf,
                                    "uncertainty": unc,
                                    "heatmap": heatmap,
                                    "heatmap_colored": heatmap_colored,
                                    "overlay": overlay
                                }

                                st.session_state["active_page"] = "Results"
                                st.rerun()

                with btn_c2:
                    if st.button("Remove Image", key="btn_clear_upload", type="secondary", use_container_width=True):
                        st.session_state["current_image"] = None
                        st.session_state["analysis_result"] = None
                        st.rerun()
        else:
            st.markdown("""
            <div style="text-align: center; padding: 1.5rem 1rem; color: #94a3b8;">
                <p style="margin: 0; font-size: 0.88rem;">Supported formats: JPG, PNG, JPEG &bull; Maximum file size: 5MB</p>
            </div>
            """, unsafe_allow_html=True)

        st.markdown('</div>', unsafe_allow_html=True)

    with col_side:
        # Image Guidelines Card (Matching Panel 2)
        st.markdown("""
        <div class="air-card">
            <div style="display: flex; align-items: center; gap: 0.5rem; margin-bottom: 1rem;">
                <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="#2563eb" stroke-width="2">
                    <circle cx="12" cy="12" r="10"></circle>
                    <line x1="12" y1="16" x2="12" y2="12"></line>
                    <line x1="12" y1="8" x2="12.01" y2="8"></line>
                </svg>
                <span style="font-weight: 700; font-size: 1rem; color: #0f172a;">Image Guidelines</span>
            </div>
            <div style="display: flex; flex-direction: column; gap: 0.75rem; font-size: 0.88rem; color: #475569;">
                <div style="display: flex; align-items: center; gap: 0.6rem;">
                    <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="#10b981" stroke-width="2.5"><polyline points="20 6 9 17 4 12"></polyline></svg>
                    <span>Clear and well-lit image</span>
                </div>
                <div style="display: flex; align-items: center; gap: 0.6rem;">
                    <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="#10b981" stroke-width="2.5"><polyline points="20 6 9 17 4 12"></polyline></svg>
                    <span>Recommended size: 224 &times; 224 or higher</span>
                </div>
                <div style="display: flex; align-items: center; gap: 0.6rem;">
                    <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="#10b981" stroke-width="2.5"><polyline points="20 6 9 17 4 12"></polyline></svg>
                    <span>Avoid blurry or very dark images</span>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
