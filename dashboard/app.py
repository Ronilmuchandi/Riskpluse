import streamlit as st
import requests
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

st.set_page_config(
    page_title="RiskPulse",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .main { background-color: #0f1117; }
    .block-container { padding-top: 2rem; }
    .metric-card {
        background: linear-gradient(135deg, #1e2130, #252840);
        border: 1px solid #2e3250;
        border-radius: 12px;
        padding: 20px;
        text-align: center;
        margin-bottom: 10px;
    }
    .metric-value { font-size: 2rem; font-weight: 700; color: #7c83f5; }
    .metric-label { font-size: 0.85rem; color: #8888aa; margin-top: 4px; }
    .fraud-badge {
        background: linear-gradient(135deg, #ff4b4b, #cc0000);
        color: white; border-radius: 8px;
        padding: 12px 24px; font-size: 1.2rem;
        font-weight: 700; text-align: center;
    }
    .normal-badge {
        background: linear-gradient(135deg, #00c853, #007a33);
        color: white; border-radius: 8px;
        padding: 12px 24px; font-size: 1.2rem;
        font-weight: 700; text-align: center;
    }
    .alert-high {
        background: linear-gradient(135deg, #2d0a0a, #4a1010);
        border-left: 4px solid #ff4b4b;
        border-radius: 8px; padding: 16px; margin: 8px 0;
    }
    .alert-medium {
        background: linear-gradient(135deg, #2d1f0a, #4a3010);
        border-left: 4px solid #ffaa00;
        border-radius: 8px; padding: 16px; margin: 8px 0;
    }
    .sidebar-title {
        font-size: 1.4rem; font-weight: 700;
        color: #7c83f5; margin-bottom: 0.5rem;
    }
    div[data-testid="stTab"] button {
        font-size: 1rem; font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)

API_URL = "http://127.0.0.1:8000"

with st.sidebar:
    st.markdown('<div class="sidebar-title">🛡️ RiskPulse</div>', unsafe_allow_html=True)
    st.caption("Intelligent Financial Risk & Fraud Intelligence Platform")
    st.divider()
    st.markdown("**Models Deployed**")
    st.success("✅ XGBoost — Primary")
    st.info("✅ Autoencoder — Anomaly")
    st.info("✅ PyTorch NN — Comparison")
    st.divider()
    st.markdown("**Dataset**")
    st.caption("284,807 transactions")
    st.caption("Kaggle Credit Card Fraud")
    st.divider()
    st.markdown("**API Status**")
    try:
        r = requests.get(f"{API_URL}/", timeout=2)
        st.success("🟢 API Online")
    except:
        st.error("🔴 API Offline")

st.markdown("## 🛡️ RiskPulse — Fraud Detection & Risk Monitoring")
st.caption("Production-style ML platform | XGBoost · TensorFlow · PyTorch · FastAPI")
st.divider()

tab1, tab2, tab3 = st.tabs(["🔍 Fraud Prediction", "📊 Drift Monitor", "🚨 Alerts"])

with tab1:
    st.markdown("### Transaction Fraud Scorer")
    st.caption("Submit a transaction's 30 feature values to get a real-time fraud probability from the XGBoost model.")
    
    col_input, col_result = st.columns([2, 1])
    
    with col_input:
        features = st.text_area(
            "Feature values (30 comma-separated floats):",
            value="0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0",
            height=120
        )
        predict_btn = st.button("🔍 Run Prediction", use_container_width=True)

    with col_result:
        if predict_btn:
            try:
                values = [float(x.strip()) for x in features.split(",")]
                response = requests.post(f"{API_URL}/predict", json={"features": values})
                result = response.json()
                prob = result['fraud_probability']
                
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-value">{prob:.4f}</div>
                    <div class="metric-label">Fraud Probability</div>
                </div>
                """, unsafe_allow_html=True)
                
                if result['prediction'] == 'FRAUD':
                    st.markdown('<div class="fraud-badge">⚠️ FRAUD DETECTED</div>', unsafe_allow_html=True)
                else:
                    st.markdown('<div class="normal-badge">✅ NORMAL TRANSACTION</div>', unsafe_allow_html=True)

                fig, ax = plt.subplots(figsize=(4, 0.6))
                fig.patch.set_facecolor('#1e2130')
                ax.set_facecolor('#1e2130')
                color = '#ff4b4b' if prob > 0.5 else '#00c853'
                ax.barh(['Risk'], [prob], color=color, height=0.4)
                ax.barh(['Risk'], [1 - prob], left=[prob], color='#2e3250', height=0.4)
                ax.set_xlim(0, 1)
                ax.axis('off')
                st.pyplot(fig)
            except Exception as e:
                st.error(f"Error: {e}")
        else:
            st.markdown("""
            <div class="metric-card">
                <div class="metric-value">—</div>
                <div class="metric-label">Awaiting prediction</div>
            </div>
            """, unsafe_allow_html=True)

with tab2:
    st.markdown("### Feature Drift Monitor")
    st.caption("Compares current transaction distributions against reference data using the KS test. Features with KS > 0.1 are flagged.")
    
    if st.button("📊 Run Drift Analysis", use_container_width=False):
        with st.spinner("Analysing feature distributions..."):
            try:
                response = requests.get(f"{API_URL}/drift")
                report = response.json()
                summary = report['summary']

                c1, c2, c3 = st.columns(3)
                c1.markdown(f"""<div class="metric-card">
                    <div class="metric-value">{summary['total_features']}</div>
                    <div class="metric-label">Total Features</div></div>""", unsafe_allow_html=True)
                c2.markdown(f"""<div class="metric-card">
                    <div class="metric-value" style="color:#ff4b4b">{summary['drifted_features']}</div>
                    <div class="metric-label">Drifted Features</div></div>""", unsafe_allow_html=True)
                c3.markdown(f"""<div class="metric-card">
                    <div class="metric-value" style="color:#ffaa00">{summary['drift_percentage']}%</div>
                    <div class="metric-label">Drift Percentage</div></div>""", unsafe_allow_html=True)

                st.markdown("#### KS Statistics per Feature")
                features_data = {k: v for k, v in report.items() if k != 'summary'}
                ks_stats = [v['ks_statistic'] for v in features_data.values()]
                colors = ['#ff4b4b' if v['drift_detected'] else '#7c83f5' for v in features_data.values()]

                fig, ax = plt.subplots(figsize=(14, 4))
                fig.patch.set_facecolor('#1e2130')
                ax.set_facecolor('#1e2130')
                ax.bar(range(len(ks_stats)), ks_stats, color=colors, width=0.7)
                ax.axhline(y=0.1, color='#ffaa00', linestyle='--', linewidth=1.5, label='Drift threshold (0.1)')
                ax.set_xlabel("Feature Index", color='#8888aa')
                ax.set_ylabel("KS Statistic", color='#8888aa')
                ax.set_title("Feature Drift Analysis", color='white', fontsize=13)
                ax.tick_params(colors='#8888aa')
                for spine in ax.spines.values():
                    spine.set_edgecolor('#2e3250')
                drifted_patch = mpatches.Patch(color='#ff4b4b', label='Drifted')
                normal_patch = mpatches.Patch(color='#7c83f5', label='Stable')
                ax.legend(handles=[drifted_patch, normal_patch, 
                    mpatches.Patch(color='#ffaa00', label='Threshold')],
                    facecolor='#1e2130', labelcolor='white')
                st.pyplot(fig)

            except Exception as e:
                st.error(f"Error: {e}")

with tab3:
    st.markdown("### System Alerts")
    st.caption("Active alerts generated by the drift monitoring system.")
    
    if st.button("🚨 Check Alerts", use_container_width=False):
        try:
            response = requests.get(f"{API_URL}/alerts")
            data = response.json()
            if not data['alerts']:
                st.success("✅ No active alerts — system is healthy")
            else:
                st.warning(f"{len(data['alerts'])} active alert(s) found")
                for alert in data['alerts']:
                    css_class = "alert-high" if alert['severity'] == 'HIGH' else "alert-medium"
                    icon = "🔴" if alert['severity'] == 'HIGH' else "🟡"
                    st.markdown(f"""
                    <div class="{css_class}">
                        <strong>{icon} [{alert['severity']}] {alert['type']}</strong><br>
                        {alert['message']}<br>
                        <small style="color:#888">⏱ {alert['timestamp']}</small>
                    </div>
                    """, unsafe_allow_html=True)
        except Exception as e:
            st.error(f"Error: {e}")

            st.divider()
st.markdown("""
<div style="text-align: center; color: #555577; font-size: 0.8rem; padding: 10px 0;">
    © 2025 Ronil Muchandi · MS Data Science & Analytics · University of Missouri Columbia<br>
    RiskPulse — Intelligent Financial Risk & Fraud Intelligence Platform
</div>
""", unsafe_allow_html=True)