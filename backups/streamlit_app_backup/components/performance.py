"""
Model Performance Dashboard View Component for Dermastat Air Vision.
Matching Panel 6 of UI Design Reference.
"""

import streamlit as st
import os
import json
from PIL import Image
import pandas as pd

from src import CLASS_NAMES, CLASS_DESCRIPTIONS


def render_performance_page():
    st.markdown("""
    <div style="margin-bottom: 1.5rem;">
        <h2 style="font-size: 1.6rem; font-weight: 800; color: #0f172a; margin: 0 0 0.35rem 0;">Model Performance Dashboard</h2>
        <p style="color: #64748b; font-size: 0.95rem; margin: 0;">
            Empirical evaluation metrics calculated on the 1,494 images of the reserved HAM10000 test dataset.
        </p>
    </div>
    """, unsafe_allow_html=True)

    metrics_path = r"D:\HAM10000\results\test_metrics.json"
    if not os.path.exists(metrics_path):
        metrics_path = r"D:\HAM10000\metrics\evaluation_metrics.json"

    cm_raw_path = r"D:\HAM10000\results\confusion_matrix.png"
    cm_norm_path = r"D:\HAM10000\results\confusion_matrix_normalized.png"

    if not os.path.exists(metrics_path):
        st.warning("Evaluation metrics have not been generated yet.")
        return

    with open(metrics_path, "r", encoding="utf-8") as f:
        metrics = json.load(f)

    overall = metrics.get("overall_metrics", metrics)

    # 4 Overall Performance KPI Cards (Matching Panel 6)
    st.markdown('<div style="font-size: 1rem; font-weight: 700; color: #0f172a; margin-bottom: 0.75rem;">Overall Performance</div>', unsafe_allow_html=True)
    c1, c2, c3, c4 = st.columns(4, gap="medium")

    acc_val = overall.get("test_accuracy", 0.7604) * 100
    f1_val = overall.get("macro_f1", 0.6944) * 100
    prec_val = overall.get("weighted_precision", overall.get("macro_precision", 0.8394)) * 100
    rec_val = overall.get("macro_recall", 0.7707) * 100

    with c1:
        st.markdown(f"""
        <div class="air-card" style="text-align: center; padding: 1.25rem;">
            <div style="font-size: 0.85rem; font-weight: 700; color: #64748b;">Accuracy</div>
            <div style="font-size: 1.85rem; font-weight: 800; color: #2563eb; margin-top: 0.25rem;">{acc_val:.1f}%</div>
        </div>
        """, unsafe_allow_html=True)

    with c2:
        st.markdown(f"""
        <div class="air-card" style="text-align: center; padding: 1.25rem;">
            <div style="font-size: 0.85rem; font-weight: 700; color: #64748b;">Macro F1-Score</div>
            <div style="font-size: 1.85rem; font-weight: 800; color: #10b981; margin-top: 0.25rem;">{f1_val:.1f}%</div>
        </div>
        """, unsafe_allow_html=True)

    with c3:
        st.markdown(f"""
        <div class="air-card" style="text-align: center; padding: 1.25rem;">
            <div style="font-size: 0.85rem; font-weight: 700; color: #64748b;">Precision (avg)</div>
            <div style="font-size: 1.85rem; font-weight: 800; color: #0f172a; margin-top: 0.25rem;">{prec_val:.1f}%</div>
        </div>
        """, unsafe_allow_html=True)

    with c4:
        st.markdown(f"""
        <div class="air-card" style="text-align: center; padding: 1.25rem;">
            <div style="font-size: 0.85rem; font-weight: 700; color: #64748b;">Recall (avg)</div>
            <div style="font-size: 1.85rem; font-weight: 800; color: #0f172a; margin-top: 0.25rem;">{rec_val:.1f}%</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # 2-Column Split: Per-Class Metrics Table on Left, Confusion Matrix on Right (Matching Panel 6)
    col_table, col_cm = st.columns([1.1, 1.2], gap="large")

    with col_table:
        st.markdown('<div class="air-card">', unsafe_allow_html=True)
        st.markdown('<div style="font-size: 1.05rem; font-weight: 700; color: #0f172a; margin-bottom: 1rem;">Per-Class Metrics</div>', unsafe_allow_html=True)

        table_rows = []
        for cls_code, data in metrics["per_class_metrics"].items():
            table_rows.append({
                "Class": cls_code.upper(),
                "Precision": f"{data['precision'] * 100:.1f}%",
                "Recall": f"{data['recall'] * 100:.1f}%",
                "F1-Score": f"{data['f1_score'] * 100:.1f}%"
            })

        df = pd.DataFrame(table_rows)
        st.dataframe(df, use_container_width=True, hide_index=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with col_cm:
        st.markdown('<div class="air-card">', unsafe_allow_html=True)
        cm_head_c1, cm_head_c2 = st.columns([1.5, 1])
        with cm_head_c1:
            st.markdown('<div style="font-size: 1.05rem; font-weight: 700; color: #0f172a; margin-top: 0.25rem;">Confusion Matrix</div>', unsafe_allow_html=True)
        with cm_head_c2:
            cm_view = st.selectbox("View:", ["Normalized", "Raw Counts"], label_visibility="collapsed")

        if cm_view == "Normalized" and os.path.exists(cm_norm_path):
            st.image(Image.open(cm_norm_path), use_container_width=True)
        elif os.path.exists(cm_raw_path):
            st.image(Image.open(cm_raw_path), use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
