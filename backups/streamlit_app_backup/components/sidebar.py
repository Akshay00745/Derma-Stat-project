"""
Reusable Navigation Sidebar Component for Dermastat Air Vision.
Matching Panel 9 of UI Design Reference.
"""

import streamlit as st


def render_sidebar():
    with st.sidebar:
        # App Logo & Name (Matching Panel 9)
        st.markdown("""
        <div class="app-brand">
            <div class="brand-icon-box">
                <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round">
                    <circle cx="12" cy="12" r="3"></circle>
                    <path d="M19.4 15a1.65 1.65 0 0 0 .33 1.82l.06.06a2 2 0 0 1 0 2.83 2 2 0 0 1-2.83 0l-.06-.06a1.65 1.65 0 0 0-1.82-.33 1.65 1.65 0 0 0-1 1.51V21a2 2 0 0 1-2 2 2 2 0 0 1-2-2v-.09A1.65 1.65 0 0 0 9 19.4a1.65 1.65 0 0 0-1.82.33l-.06.06a2 2 0 0 1-2.83 0 2 2 0 0 1 0-2.83l.06-.06a1.65 1.65 0 0 0 .33-1.82 1.65 1.65 0 0 0-1.51-1H3a2 2 0 0 1-2-2 2 2 0 0 1 2-2h.09A1.65 1.65 0 0 0 4.6 9a1.65 1.65 0 0 0-.33-1.82l-.06-.06a2 2 0 0 1 0-2.83 2 2 0 0 1 2.83 0l.06.06a1.65 1.65 0 0 0 1.82.33H9a1.65 1.65 0 0 0 1-1.51V3a2 2 0 0 1 2-2 2 2 0 0 1 2 2v.09a1.65 1.65 0 0 0 1 1.51 1.65 1.65 0 0 0 1.82-.33l.06-.06a2 2 0 0 1 2.83 0 2 2 0 0 1 0 2.83l-.06.06a1.65 1.65 0 0 0-.33 1.82V9a1.65 1.65 0 0 0 1.51 1H21a2 2 0 0 1 2 2 2 2 0 0 1-2 2h-.09a1.65 1.65 0 0 0-1.51 1z"></path>
                </svg>
            </div>
            <div class="brand-name">Dermastat Air Vision</div>
        </div>
        """, unsafe_allow_html=True)

        current_page = st.session_state.get("active_page", "Home")

        # Map internal page states to active sidebar item
        if current_page in ["Results", "Grad-CAM", "Detailed Analysis"]:
            radio_default = "Results"
        elif current_page in ["Dashboard"]:
            radio_default = "Dashboard"
        elif current_page in ["Upload"]:
            radio_default = "Upload"
        elif current_page in ["Help"]:
            radio_default = "Help"
        elif current_page in ["About"]:
            radio_default = "Home"
        else:
            radio_default = "Home"

        nav_options = ["Home", "Upload", "Results", "Dashboard", "Help"]
        nav_icons = {
            "Home": "🏠",
            "Upload": "📤",
            "Results": "📋",
            "Dashboard": "📊",
            "Help": "❓"
        }

        selected_nav = st.radio(
            "Navigation Menu",
            options=nav_options,
            index=nav_options.index(radio_default),
            format_func=lambda x: f"{nav_icons[x]}  {x}",
            label_visibility="collapsed"
        )

        # Trigger page change on selection if different
        if selected_nav != radio_default:
            st.session_state["active_page"] = selected_nav
            st.rerun()

        # Bottom Disclaimer Card (Matching Panel 9)
        st.markdown("""
        <div class="sidebar-disclaimer-box">
            <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="#2563eb" stroke-width="2">
                <path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"></path>
            </svg>
            <div style="font-weight: 700; font-size: 0.82rem; color: #1e3a8a; margin-top: 0.35rem;">Educational Prototype</div>
            <div class="sidebar-disclaimer-text">Not a medical diagnostic tool</div>
        </div>
        """, unsafe_allow_html=True)
