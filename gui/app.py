# app.py
import streamlit as st
import plotly.graph_objects as go
import numpy as np
import json

import config
from utils.model_inference import run_main_inference

# Page settings
st.set_page_config(page_title="ECG Patient Dashboard", layout="wide")

# Light theme card styles
st.markdown("""
    <style>
    .metric-card { background: white; padding: 20px; border-radius: 10px; box-shadow: 0 2px 8px rgba(0,0,0,0.05); margin-bottom: 15px; border: 1px solid #e9ecef; }
    .badge-normal { background-color: #d4edda; color: #155724; padding: 6px 12px; border-radius: 20px; font-weight: bold; display: inline-block; }
    .badge-warning { background-color: #fff3cd; color: #856404; padding: 6px 12px; border-radius: 20px; font-weight: bold; display: inline-block; }
    .badge-critical { background-color: #f8d7da; color: #721c24; padding: 6px 12px; border-radius: 20px; font-weight: bold; display: inline-block; }
    </style>
""", unsafe_allow_html=True)

st.title("Patient ECG Diagnostic Interface")
st.caption("Upload files or manually input demographics for localized waveform analysis.")

# ============================================================
# SIDEBAR: DATA UPLOADS & MANUAL INPUT DEMOGRAPHICS
# ============================================================
st.sidebar.header("Data Input")

uploaded_signal = st.sidebar.file_uploader(
    "1. Upload ECG Signal File (.npy format)", 
    type=["npy"], 
    help="Drag and drop your extracted numpy ECG array here."
)

uploaded_meta = st.sidebar.file_uploader(
    "2. Upload Patient Metadata (.json format - Optional)", 
    type=["json"],
    help="Optional file containing patient data parameters."
)

st.sidebar.markdown("---")
st.sidebar.header("Manual Patient Profile")
st.sidebar.caption("Fill this section out if you do not have a metadata file")

# Added user-editable inputs
manual_name = st.sidebar.text_input("Patient Name", value="")
manual_age = st.sidebar.text_input("Patient Age", value="")
manual_gender = st.sidebar.selectbox("Patient Gender", ["Not Specified", "Male", "Female", "Other"])

# Initialize variables
signal, metadata = None, None

if uploaded_signal is not None:
    signal = np.load(uploaded_signal)

if uploaded_meta is not None:
    metadata = json.load(uploaded_meta)

# ============================================================
# MAIN PANEL CONTENT PRESENTATION
# ============================================================
if signal is not None:
    col1, col2 = st.columns([1, 3])
    
    # --- PATIENT INFO PANEL ---
    with col1:
        st.subheader("Patient Information")
        
        # Priority 1: Read from JSON metadata file if available
        if metadata:
            st.markdown(f"**Age:** {metadata.get('age', 'N/A')} years")
            st.markdown(f"**Sex:** {metadata.get('sex', 'N/A')}")
            st.markdown(f"**Recording Rate:** {metadata.get('sampling_rate', 100)} Hz")
        
        # Priority 2: Read from the sidebar interactive text inputs
        elif manual_name or manual_age or manual_gender != "Not Specified":
            display_name = manual_name if manual_name else "Not Specified"
            display_age = f"{manual_age} years" if manual_age else "N/A"
            
            st.markdown(f"**Name:** {display_name}")
            st.markdown(f"**Age:** {display_age}")
            st.markdown(f"**Gender:** {manual_gender}")
            st.markdown("**Recording Rate:** 100 Hz (Default)")
            
        else:
            st.markdown("""
            <div style='background-color: #f1f3f5; padding: 15px; border-radius: 8px;'>
                <p style='margin:0; color:#6c757d;'>No clinical profile parameters loaded or manually entered.</p>
            </div>
            """, unsafe_allow_html=True)

    # --- INTERACTIVE GRAPH PLOT ---
    with col2:
        st.subheader("ECG Signal Waveform (Lead II)")
        
        fs = metadata.get('sampling_rate', 100) if metadata else 100
        time_axis = np.arange(signal.shape[0]) / fs
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=time_axis, y=signal[:, 1], mode='lines', name='Lead II', line=dict(color='#2b6cb0', width=2)))
        fig.update_layout(
            margin=dict(l=10, r=10, t=10, b=10),
            xaxis_title="Time (seconds)",
            yaxis_title="Amplitude (mV)",
            plot_bgcolor='#f8fafc',
            height=240
        )
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")
    
    # ============================================================
    # ACTION RUNNING SEGMENT
    # ============================================================
    st.subheader("Analysis Interpretation")
    
    # Text-only execution trigger button
    analyze_click = st.button("RUN DIAGNOSTIC ANALYSIS", type="primary")
    
    if analyze_click:
        with st.spinner("Analyzing uploaded waveform charts..."):
            predictions = run_main_inference(signal)
            
        if predictions:
            any_alerts = False
            
            for code, prob in predictions.items():
                if prob >= config.THRESHOLD_MAIN:
                    mapping = config.CLINICAL_MAPPING.get(code, {"name": code, "desc": "Condition flagged.", "severity": "warning"})
                    badge_style = f"badge-{mapping['severity']}"
                    
                    st.markdown(f"""
                    <div class="metric-card">
                        <span class="{badge_style}">{code} — {mapping['name'].upper()} IDENTIFIED</span>
                        <p style="margin-top:12px; margin-bottom:0; color:#4a5568; font-size:15px;">
                            <b>Clinical Context:</b> {mapping['desc']}
                        </p>
                    </div>
                    """, unsafe_allow_html=True)
                    any_alerts = True
                    
            if not any_alerts:
                st.markdown("""
                <div class="metric-card">
                    <span class="badge-normal">NORM — TYPICAL CARDIAC TRACE</span>
                    <p style="margin-top:12px; margin-bottom:0; color:#4a5568; font-size:15px;">
                        The analysis found no major abnormalities or rhythmic deviances outside standard healthy baseline variations.
                    </p>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.error("Could not run analysis. Please verify your background configurations.")

        # Streamlined simple alert text block
        st.markdown("<br>", unsafe_allow_html=True)
        st.warning("Please contact a cardiologist for verification.")

    else:
        st.info("Click the 'RUN DIAGNOSTIC ANALYSIS' button above to run the model on this waveform.")

else:
    st.markdown("""
    <div style="text-align: center; padding: 80px 20px; background: white; border-radius: 12px; border: 2px dashed #cbd5e0; margin-top: 40px;">
        <h3 style="color: #4a5568;">Ready for Data Input</h3>
        <p style="color: #718096; max-width: 500px; margin: 0 auto 20px auto;">
            Please use the upload tools in the sidebar to load your <b>signal.npy</b> file. The option to analyze will appear once the signal is drawn.
        </p>
    </div>
    """, unsafe_allow_html=True)