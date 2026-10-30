"""
About Page View Component for Dermastat Air Vision.
Matching Panel 7 of UI Design Reference.
"""

import streamlit as st


def render_about_page():
    st.markdown('<h2 style="font-size: 1.6rem; font-weight: 800; color: #0f172a; margin: 0 0 1rem 0;">About Dermastat Air Vision</h2>', unsafe_allow_html=True)

    # Main intro card (Matching Panel 7)
    st.markdown("""
    <div class="air-card" style="padding: 1.5rem; margin-bottom: 2rem;">
        <p style="font-size: 1.05rem; line-height: 1.7; color: #334155; margin: 0;">
            <strong>Dermastat Air Vision</strong> is an educational prototype that uses deep learning and 
            computer vision to classify skin-lesion images into seven categories. 
            It provides confidence scores, visual explanations and uncertainty assessment 
            to make AI predictions more transparent and interpretable.
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Our Goals Section (Matching Panel 7)
    st.markdown('<h3 style="font-size: 1.25rem; font-weight: 800; color: #0f172a; margin: 0 0 1rem 0;">Our Goals</h3>', unsafe_allow_html=True)

    g_cols = st.columns(4, gap="medium")

    goals = [
        ("Demonstrate AI<br>classification", '<svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="#2563eb" stroke-width="2"><circle cx="12" cy="12" r="3"></circle><path d="M19.4 15a1.65 1.65 0 0 0 .33 1.82l.06.06a2 2 0 0 1 0 2.83 2 2 0 0 1-2.83 0l-.06-.06a1.65 1.65 0 0 0-1.82-.33 1.65 1.65 0 0 0-1 1.51V21a2 2 0 0 1-2 2 2 2 0 0 1-2-2v-.09A1.65 1.65 0 0 0 9 19.4a1.65 1.65 0 0 0-1.82.33l-.06.06a2 2 0 0 1-2.83 0 2 2 0 0 1 0-2.83l.06-.06a1.65 1.65 0 0 0 .33-1.82 1.65 1.65 0 0 0-1.51-1H3a2 2 0 0 1-2-2 2 2 0 0 1 2-2h.09A1.65 1.65 0 0 0 4.6 9a1.65 1.65 0 0 0-.33-1.82l-.06-.06a2 2 0 0 1 0-2.83 2 2 0 0 1 2.83 0l.06.06a1.65 1.65 0 0 0 1.82.33H9a1.65 1.65 0 0 0 1-1.51V3a2 2 0 0 1 2-2 2 2 0 0 1 2 2v.09a1.65 1.65 0 0 0 1 1.51 1.65 1.65 0 0 0 1.82-.33l.06-.06a2 2 0 0 1 2.83 0 2 2 0 0 1 0 2.83l-.06.06a1.65 1.65 0 0 0-.33 1.82V9a1.65 1.65 0 0 0 1.51 1H21a2 2 0 0 1 2 2 2 2 0 0 1-2 2h-.09a1.65 1.65 0 0 0-1.51 1z"></path></svg>'),
        ("Provide visual<br>explanations", '<svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="#2563eb" stroke-width="2"><path d="M1 12s4-8 11-8 11 8 11 8-4 8-11 8-11-8-11-8z"></path><circle cx="12" cy="12" r="3"></circle></svg>'),
        ("Show model<br>performance", '<svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="#2563eb" stroke-width="2"><line x1="18" y1="20" x2="18" y2="10"></line><line x1="12" y1="20" x2="12" y2="4"></line><line x1="6" y1="20" x2="6" y2="14"></line></svg>'),
        ("Promote responsible<br>AI in healthcare", '<svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="#2563eb" stroke-width="2"><path d="M20.84 4.61a5.5 5.5 0 0 0-7.78 0L12 5.67l-1.06-1.06a5.5 5.5 0 0 0-7.78 7.78l1.06 1.06L12 21.23l7.78-7.78 1.06-1.06a5.5 5.5 0 0 0 0-7.78z"></path></svg>')
    ]

    for idx, (g_title, g_icon) in enumerate(goals):
        with g_cols[idx]:
            st.markdown(f"""
            <div class="air-card" style="text-align: center; padding: 1.5rem 1rem; height: 100%;">
                <div style="width: 48px; height: 48px; border-radius: 50%; background: #eff6ff; display: flex; align-items: center; justify-content: center; margin: 0 auto 1rem auto;">
                    {g_icon}
                </div>
                <div style="font-weight: 700; font-size: 0.92rem; color: #0f172a; line-height: 1.35;">
                    {g_title}
                </div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Medical Disclaimer Banner (Matching Panel 7)
    st.markdown("""
    <div style="background: #eff6ff; border: 1px solid #bfdbfe; border-radius: 12px; padding: 1.25rem 1.5rem; display: flex; align-items: center; gap: 1rem;">
        <div style="color: #2563eb; font-size: 1.5rem;">🛡️</div>
        <div>
            <div style="font-weight: 800; color: #1e3a8a; font-size: 0.95rem; margin-bottom: 0.25rem;">
                This is not a medical diagnostic tool.
            </div>
            <div style="color: #1e40af; font-size: 0.9rem;">
                Always consult a qualified healthcare professional for medical advice, skin cancer screening, or diagnosis.
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
