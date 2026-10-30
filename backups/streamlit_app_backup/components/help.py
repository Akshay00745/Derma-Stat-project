"""
Help & Support View Component for Dermastat Air Vision.
Matching Panel 8 of UI Design Reference.
"""

import streamlit as st


def render_help_page():
    st.markdown("""
    <div style="margin-bottom: 1.5rem;">
        <h2 style="font-size: 1.6rem; font-weight: 800; color: #0f172a; margin: 0 0 0.35rem 0;">Help & Support</h2>
        <p style="color: #64748b; font-size: 0.95rem; margin: 0;">
            Find answers to common questions about using Dermastat Air Vision.
        </p>
    </div>
    """, unsafe_allow_html=True)

    faqs = [
        ("How do I upload an image?", 
         "Navigate to the **Upload** page using the sidebar. You can drag and drop your dermoscopic skin lesion image directly onto the upload area, or click 'Choose File' to browse files from your computer. Once selected, click the blue **Analyze Image** button."),
        
        ("What image formats are supported?", 
         "We support standard image file formats: **JPG**, **JPEG**, and **PNG**. The maximum recommended file size is 5MB, with an optimal input resolution of 224x224 pixels or higher."),
        
        ("What do the seven categories mean?", 
         "The model classifies lesions into the seven clinical diagnostic categories established by the official HAM10000 benchmark: AKIEC (Actinic Keratoses), BCC (Basal Cell Carcinoma), BKL (Benign Keratosis), DF (Dermatofibroma), MEL (Melanoma), NV (Melanocytic Nevi), and VASC (Vascular Lesions)."),
        
        ("How is the prediction confidence calculated?", 
         "The neural network outputs logits for all seven classes, which are converted into calibrated probabilities using the Softmax activation function. The confidence score represents the model's highest predicted probability percentage among the seven classes."),
        
        ("What does the uncertainty warning mean?", 
         "When model confidence falls below our validation-calibrated reliability threshold (83.15%), an amber **Uncertainty Warning** is triggered. This indicates that the image may exhibit overlapping features or unusual artifacts, and the prediction should not be relied upon."),
        
        ("How is Macro F1-score different from accuracy?", 
         "Raw accuracy simply counts total correct predictions, which can be misleadingly high if the model guesses common classes (like NV, which is 67% of the dataset) while failing on rare classes. **Macro F1-score** gives equal weight to all seven categories, ensuring solid diagnostic sensitivity even on rare conditions."),
        
        ("Is this tool a medical diagnostic system?", 
         "**No.** Dermastat Air Vision is strictly an educational AI prototype developed for research and machine learning demonstration. It does not provide clinical diagnoses and must not be used to replace professional dermatologist evaluations or biopsy procedures.")
    ]

    for q, a in faqs:
        with st.expander(q, expanded=False):
            st.markdown(f"<div style='color: #475569; font-size: 0.92rem; line-height: 1.6;'>{a}</div>", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Contact footer (Matching Panel 8)
    st.markdown("""
    <div style="text-align: center; color: #64748b; font-size: 0.9rem; margin-top: 1.5rem;">
        Still have questions? Contact us at <a href="mailto:support@dermastat.ai" style="color: #2563eb; font-weight: 700; text-decoration: none;">support@dermastat.ai</a>
    </div>
    """, unsafe_allow_html=True)
