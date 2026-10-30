"""
Dermastat Air Vision - AI-Powered Skin Lesion Classification Web Application.
Main Streamlit Application Entry Point.
"""

import os
import sys

# Prevent OpenMP conflict on Windows
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Add project root to sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import streamlit as st
import torch

from src import CLASS_NAMES
from src.model import build_model
from app.components import (
    render_home_page,
    render_upload_page,
    render_results_page,
    render_explainability_page,
    render_detailed_page,
    render_performance_page,
    render_about_page,
    render_help_page,
    render_sidebar
)

# Page configuration
st.set_page_config(
    page_title="Dermastat Air Vision - Smarter AI for Healthier Skin",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Inject custom CSS
css_path = os.path.join(os.path.dirname(__file__), "style.css")
if os.path.exists(css_path):
    with open(css_path, "r", encoding="utf-8") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)


@st.cache_resource
def load_app_model():
    """Caches the trained PyTorch model in memory for instant interactive predictions."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint_path = os.path.join(PROJECT_ROOT, "checkpoints", "best_model.pth")

    if os.path.exists(checkpoint_path):
        try:
            model = build_model(
                num_classes=len(CLASS_NAMES),
                pretrained=False,
                checkpoint_path=checkpoint_path,
                device=device
            )
            model.eval()
            return model, device, True
        except Exception as e:
            st.sidebar.error(f"Error loading checkpoint: {e}")
            return None, device, False
    else:
        return None, device, False


def main():
    model, device, is_loaded = load_app_model()

    # Default state initialization
    if "active_page" not in st.session_state:
        st.session_state["active_page"] = "Home"

    # Render Reusable Sidebar (Matching Panel 9)
    render_sidebar()

    current_page = st.session_state.get("active_page", "Home")

    # Route view rendering
    if current_page == "Home":
        render_home_page()
    elif current_page == "Upload":
        render_upload_page(model, device)
    elif current_page == "Results":
        render_results_page()
    elif current_page == "Grad-CAM":
        render_explainability_page()
    elif current_page == "Detailed Analysis":
        render_detailed_page()
    elif current_page == "Dashboard":
        render_performance_page()
    elif current_page == "About":
        render_about_page()
    elif current_page == "Help":
        render_help_page()
    else:
        render_home_page()


if __name__ == "__main__":
    main()
