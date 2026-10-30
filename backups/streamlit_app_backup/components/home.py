"""
Landing Page / Home View Component for Dermastat Air Vision.
Matching Panel 1 of UI Design Reference.
"""

import streamlit as st
import os
import base64
from PIL import Image

def get_base64_image(image_path: str) -> str:
    """Helper to convert local image to base64 string for clean HTML rendering."""
    if os.path.exists(image_path):
        with open(image_path, "rb") as f:
            data = f.read()
        return f"data:image/jpeg;base64,{base64.b64encode(data).decode('utf-8')}"
    return ""


def render_home_page():
    samples_dir = os.path.join(os.path.dirname(__file__), "..", "static", "samples")
    hero_b64 = get_base64_image(os.path.join(samples_dir, "hero_lesion.jpg"))

    # Top Navbar (Matching Panel 1)
    nav_c1, nav_c2, nav_c3 = st.columns([1.5, 3, 1], gap="small")
    with nav_c1:
        st.markdown("""
        <div style="display: flex; align-items: center; gap: 0.6rem; padding-top: 0.25rem;">
            <div class="brand-icon-box" style="width: 32px; height: 32px; font-size: 1rem;">🔬</div>
            <span style="font-weight: 800; font-size: 1.15rem; color: #0f172a;">Dermastat Air Vision</span>
        </div>
        """, unsafe_allow_html=True)
    with nav_c2:
        top_cols = st.columns(4)
        if top_cols[0].button("Home", key="top_nav_home", use_container_width=True):
            st.session_state["active_page"] = "Home"
            st.rerun()
        if top_cols[1].button("About", key="top_nav_about", use_container_width=True):
            st.session_state["active_page"] = "About"
            st.rerun()
        if top_cols[2].button("Dashboard", key="top_nav_dash", use_container_width=True):
            st.session_state["active_page"] = "Dashboard"
            st.rerun()
        if top_cols[3].button("Help", key="top_nav_help", use_container_width=True):
            st.session_state["active_page"] = "Help"
            st.rerun()
    with nav_c3:
        if st.button("Get Started", key="top_get_started", type="primary", use_container_width=True):
            st.session_state["active_page"] = "Upload"
            st.rerun()

    st.markdown("<hr style='border: none; border-top: 1px solid #edf2f7; margin: 0.5rem 0 1.5rem 0;'>", unsafe_allow_html=True)

    # Hero Section
    col_hero_left, col_hero_right = st.columns([1.2, 1], gap="large")

    with col_hero_left:
        st.markdown("""
        <div style="padding-top: 1rem;">
            <h1 class="hero-title">Smarter AI for<br>Healthier Skin</h1>
            <p class="hero-subtitle">
                Upload a skin-lesion image and get AI-powered classification, confidence scores, 
                visual explanation and more &mdash; all in one place.
            </p>
        </div>
        """, unsafe_allow_html=True)

        h_btn1, h_btn2 = st.columns([1, 1], gap="small")
        with h_btn1:
            if st.button("Upload Image", key="hero_btn_upload", type="primary", use_container_width=True):
                st.session_state["active_page"] = "Upload"
                st.rerun()
        with h_btn2:
            if st.button("Learn More", key="hero_btn_learn", type="secondary", use_container_width=True):
                st.session_state["active_page"] = "About"
                st.rerun()

    with col_hero_right:
        st.markdown(f"""
        <div class="hero-img-wrapper" style="padding-top: 0.5rem;">
            <div class="hero-circle-ring"></div>
            <img src="{hero_b64}" class="hero-circle-img" alt="Skin Lesion Dermoscopy" />
            <div class="hero-badge-floating">
                <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                    <circle cx="12" cy="12" r="10"></circle>
                    <path d="M12 16v-4"></path>
                    <path d="M12 8h.01"></path>
                </svg>
                AI-Powered Skin Lesion Classification
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br><br>", unsafe_allow_html=True)

    # 5 Feature Cards Row
    feat_cols = st.columns(5, gap="small")

    features = [
        ("category", "7 Skin-Lesion<br>Categories", "7 clinical classes defined in HAM10000"),
        ("chart", "Confidence Scores<br>for All Classes", "Calibrated multi-class probability output"),
        ("layers", "Grad-CAM<br>Visual Explanation", "Heatmaps showing influential image regions"),
        ("shield", "Uncertainty<br>Assessment", "Validation-calibrated reliability threshold"),
        ("dashboard", "Performance<br>Dashboard", "Accuracy, Macro-F1 & Confusion Matrix")
    ]

    icons = [
        '<svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><rect x="3" y="3" width="7" height="7"></rect><rect x="14" y="3" width="7" height="7"></rect><rect x="14" y="14" width="7" height="7"></rect><rect x="3" y="14" width="7" height="7"></rect></svg>',
        '<svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><line x1="18" y1="20" x2="18" y2="10"></line><line x1="12" y1="20" x2="12" y2="4"></line><line x1="6" y1="20" x2="6" y2="14"></line></svg>',
        '<svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polygon points="12 2 2 7 12 12 22 7 12 2"></polygon><polyline points="2 17 12 22 22 17"></polyline><polyline points="2 12 12 17 22 12"></polyline></svg>',
        '<svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"></path></svg>',
        '<svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><line x1="18" y1="20" x2="18" y2="10"></line><line x1="12" y1="20" x2="12" y2="4"></line><line x1="6" y1="20" x2="6" y2="14"></line></svg>'
    ]

    for idx, (f_key, f_title, f_desc) in enumerate(features):
        with feat_cols[idx]:
            st.markdown(f"""
            <div class="feature-pill-card">
                <div class="feature-pill-icon">{icons[idx]}</div>
                <div class="feature-pill-title">{f_title}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<br><br>", unsafe_allow_html=True)

    # 7 Skin-Lesion Categories Section (Matching Panel 1)
    st.markdown("""
    <div style="margin-bottom: 1.25rem;">
        <h3 style="font-size: 1.35rem; font-weight: 800; color: #0f172a; margin: 0;">The 7 Skin-Lesion Categories</h3>
    </div>
    """, unsafe_allow_html=True)

    cat_cols = st.columns(7, gap="small")

    categories_data = [
        ("akiec", "AKIEC", "Actinic keratoses and intraepithelial carcinoma"),
        ("bcc", "BCC", "Basal cell carcinoma"),
        ("bkl", "BKL", "Benign keratosis-like lesions"),
        ("df", "DF", "Dermatofibroma"),
        ("mel", "MEL", "Melanoma"),
        ("nv", "NV", "Melanocytic nevi"),
        ("vasc", "VASC", "Vascular lesions")
    ]

    for idx, (code_lower, code_upper, full_title) in enumerate(categories_data):
        img_b64 = get_base64_image(os.path.join(samples_dir, f"{code_lower}.jpg"))
        with cat_cols[idx]:
            st.markdown(f"""
            <div class="category-item-card">
                <img src="{img_b64}" class="category-thumb-circle" alt="{code_upper}" />
                <div class="category-code">{code_upper}</div>
                <div class="category-desc">{full_title}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<br><br>", unsafe_allow_html=True)

    # Footer Disclaimer
    st.markdown("""
    <div style="text-align: center; border-top: 1px solid #edf2f7; padding-top: 1.5rem; color: #94a3b8; font-size: 0.85rem;">
        <strong>Dermastat Air Vision</strong> &bull; Educational Prototype &bull; Not a medical diagnostic tool
    </div>
    """, unsafe_allow_html=True)
