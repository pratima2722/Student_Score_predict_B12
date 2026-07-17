import streamlit as st
import pandas as pd
import pickle
import numpy as np

# Page configuration
st.set_page_config(
    page_title="Model Inference Console",
    page_icon="⚙️",
    layout="wide"  # Optimized for standardized technical dashboards
)

# Custom CSS for standard technical UI (Slate / Indigo theme)
st.markdown("""
    <style>
        /* Base system font standardization */
        html, body, [class*="css"] {
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
        }
        
        /* Standardized Technical Card */
        .tech-card {
            background-color: #ffffff;
            padding: 24px;
            border-radius: 8px;
            border: 1px solid #e2e8f0;
            box-shadow: 0 1px 3px 0 rgba(0, 0, 0, 0.05), 0 1px 2px 0 rgba(0, 0, 0, 0.03);
            margin-bottom: 20px;
        }
        
        /* Dark-mode compatible styling wrapper */
        @media (prefers-color-scheme: dark) {
            .tech-card {
                background-color: #1e293b;
                border-color: #334155;
            }
        }
        
        /* Code block decoration */
        .metric-header {
            font-size: 0.85rem;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            color: #64748b;
            font-weight: 600;
            margin-bottom: 8px;
        }
    </style>
""", unsafe_allow_html=True)

# Load the model
@st.cache_resource
def load_model():
    with open("model.pkl", "rb") as f:
        model = pickle.load(f)
    return model

try:
    model = load_model()
except FileNotFoundError:
    st.error("⚠️ File Not Found: `model.pkl` must exist in the root execution directory.")
    st.stop()

# --- SIDEBAR CONTROL PANEL ---
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/control-panel.png", width=64)
    st.title("Inference Engine")
    st.caption("v1.0.0 • Production Environment")
    st.markdown("---")
    
    st.subheader("Configuration Parameters")
    
    # Standardized system inputs based on model training features[cite: 1]
    hours_studied = st.slider(
        "Hours Studied", 
        min_value=0.0, 
        max_value=24.0, 
        value=5.0, 
        step=0.5,
        help="Aggregated daily training study hours."
    )
    
    sleep_hours = st.slider(
        "Sleep Hours", 
        min_value=0.0, 
        max_value=24.0, 
        value=7.0, 
        step=0.5,
        help="Rest and recovery period metrics."
    )
    
    attendance_percent = st.slider(
        "Attendance Index (%)", 
        min_value=0.0, 
        max_value=100.0, 
        value=85.0, 
        step=1.0,
        help="Classroom attendance percentage metrics."
    )
    
    previous_scores = st.number_input(
        "Historic Baseline Score", 
        min_value=0.0, 
        max_value=100.0, 
        value=75.0, 
        step=1.0,
        help="Prior evaluation scores."
    )
    
    st.markdown("---")
    execute_inference = st.button("Run Inference Pipeline", type="primary", use_container_width=True)

# --- MAIN WORKSPACE ---
st.header("⚙️ KNN Model Inference Workspace")
st.markdown("Standardized interface for system deployment and real-time inference prediction.")

# Tabbed Layout for modular data presentation
tab_inference, tab_metadata = st.tabs(["🚀 Real-time Prediction", "📋 Model Metadata"])

with tab_inference:
    col_input, col_output = st.columns([1, 1], gap="large")
    
    with col_input:
        st.markdown('<div class="tech-card">', unsafe_allow_html=True)
        st.markdown('<div class="metric-header">Active Payload Configuration</div>', unsafe_allow_html=True)
        
        # Displaying currently configured payload as a standardized JSON structure
        payload = {
            "hours_studied": hours_studied,
            "sleep_hours": sleep_hours,
            "attendance_percent": attendance_percent,
            "previous_scores": previous_scores
        }
        st.json(payload)
        st.markdown('</div>', unsafe_allow_html=True)
        
    with col_output:
        st.markdown('<div class="tech-card">', unsafe_allow_html=True)
        st.markdown('<div class="metric-header">Prediction Registry Pipeline</div>', unsafe_allow_html=True)
        
        if execute_inference:
            input_data = pd.DataFrame([payload])
            
            with st.spinner("Executing calculations..."):
                try:
                    prediction = model.predict(input_data)[0]
                    
                    st.metric(
                        label="Inferred System Metric Output",
                        value=f"{prediction:.2f}%",
                        delta=f"{(prediction - previous_scores):+.2f}% vs Baseline"
                    )
                    
                    st.progress(min(max(float(prediction) / 100.0, 0.0), 1.0))
                    st.success("✔ Pipeline execution completed with code 0.")
                    
                except Exception as e:
                    st.error(f"Inference execution failed: {str(e)}")
        else:
            st.info("💡 Adjust system parameters in the sidebar and select 'Run Inference Pipeline' to process.")
            
        st.markdown('</div>', unsafe_allow_html=True)

with tab_metadata:
    st.markdown('<div class="tech-card">', unsafe_allow_html=True)
    st.markdown('<div class="metric-header">Model Topology & Serialization Summary</div>', unsafe_allow_html=True)
    
    # Display standardized metadata extracted from pickle structures[cite: 1]
    col_meta1, col_meta2 = st.columns(2)
    with col_meta1:
        st.markdown(f"**Model Class:** `{type(model).__name__}`")
        st.markdown(f"**Scikit-Learn Version:** `1.6.1`[cite: 1]")
        st.markdown(f"**Metric Type:** `Minkowski`[cite: 1]")
    with col_meta2:
        st.markdown(f"**Neighbor Nodes ($k$):** `{model.n_neighbors if hasattr(model, 'n_neighbors') else 'N/A'}`[cite: 1]")
        st.markdown(f"**Feature Dimension Count:** `{model.n_features_in_ if hasattr(model, 'n_features_in_') else 'N/A'}`[cite: 1]")
        st.markdown(f"**Registered Features:** `{', '.join(model.feature_names_in_) if hasattr(model, 'feature_names_in_') else 'N/A'}`[cite: 1]")
        
    st.markdown('</div>', unsafe_allow_html=True)
