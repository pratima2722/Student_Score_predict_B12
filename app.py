import streamlit as st
import pandas as pd
import pickle
import numpy as np

# Page configuration
st.set_page_config(
    page_title="Inference Canvas",
    page_icon="✨",
    layout="centered"
)

# Custom CSS for Zen Minimalist Aesthetic
st.markdown("""
    <style>
        /* Sleek Obsidian Canvas */
        .stApp {
            background-color: #0b0f19;
        }
        
        /* Centered Zen Container */
        .zen-card {
            background: #111625;
            border: 1px solid rgba(255, 255, 255, 0.05);
            padding: 35px;
            border-radius: 20px;
            box-shadow: 0 20px 40px rgba(0, 0, 0, 0.5);
            margin-bottom: 25px;
        }
        
        /* Floating Glowing Ring for the output */
        .score-ring {
            display: inline-block;
            padding: 20px 40px;
            border-radius: 50px;
            background: linear-gradient(135deg, #1e1b4b 0%, #0f172a 100%);
            border: 1px solid #6366f1;
            box-shadow: 0 0 30px rgba(99, 102, 241, 0.25);
            margin: 20px 0;
        }
        
        /* Clean, refined typography */
        h1, h2, h3, p, label {
            color: #f1f5f9 !important;
            font-family: system-ui, -apple-system, sans-serif !important;
        }
        
        .accent-text {
            color: #818cf8 !important;
            font-weight: 600;
            letter-spacing: 1px;
            font-size: 0.9rem;
            text-transform: uppercase;
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
    st.error("⚠️ `model.pkl` not found! Please ensure the file is in the same directory as `app.py`.")
    st.stop()

# --- HEADER SECTION ---
st.markdown("<div style='text-align: center; margin-top: 20px;'>", unsafe_allow_html=True)
st.markdown("<p class='accent-text'>🔮 Real-Time Machine Learning</p>", unsafe_allow_html=True)
st.markdown("<h1 style='font-size: 2.3rem; font-weight: 300; margin-top: -5px;'>Inference Canvas</h1>", unsafe_allow_html=True)
st.markdown("<p style='color: #64748b !important; font-size: 0.95rem; max-width: 450px; margin: 0 auto 30px auto;'>A quiet, reactive playground for predicting student scores instantly as habits shift.</p>", unsafe_allow_html=True)
st.markdown("</div>", unsafe_allow_html=True)

# --- INPUT SECTION (Single Zen Card) ---
st.markdown('<div class="zen-card">', unsafe_allow_html=True)

# Sliding inputs map directly to model parameters[cite: 1]
hours_studied = st.slider(
    "Study Intensity (Hours/Day)", 
    min_value=0.0, 
    max_value=24.0, 
    value=5.0, 
    step=0.5
)

sleep_hours = st.slider(
    "Sleep Duration (Hours/Night)", 
    min_value=0.0, 
    max_value=24.0, 
    value=7.0, 
    step=0.5
)

attendance_percent = st.slider(
    "Classroom Attendance (%)", 
    min_value=0.0, 
    max_value=100.0, 
    value=85.0, 
    step=1.0
)

previous_scores = st.slider(
    "Previous Academic Score", 
    min_value=0.0, 
    max_value=100.0, 
    value=75.0, 
    step=1.0
)

st.markdown('</div>', unsafe_allow_html=True) # Close Zen Card

# --- REAL-TIME CALCULATION PIPELINE ---
# Construct data payload mirroring expected features[cite: 1]
input_data = pd.DataFrame([{
    "hours_studied": hours_studied,
    "sleep_hours": sleep_hours,
    "attendance_percent": attendance_percent,
    "previous_scores": previous_scores
}])

try:
    # Model generates instant predictions dynamically without button-clicks[cite: 1]
    prediction = model.predict(input_data)[0]
    
    # --- OUTPUT DISPLAY ---
    st.markdown("<div style='text-align: center; margin-top: 10px;'>", unsafe_allow_html=True)
    st.markdown("<p style='color: #94a3b8 !important; font-size: 0.9rem; margin-bottom: 0;'>ESTIMATED OUTCOME</p>", unsafe_allow_html=True)
    
    # The elegant centered score capsule
    st.markdown(
        f"<div class='score-ring'><h1 style='margin:0; font-size: 3.5rem; font-weight: 200; background: linear-gradient(to right, #ffffff, #a5b4fc); -webkit-background-clip: text; -webkit-text-fill-color: transparent;'>{prediction:.1f}%</h1></div>", 
        unsafe_allow_html=True
    )
    
    # Simple, high-contrast, non-distracting progress bar
    st.progress(min(max(float(prediction) / 100.0, 0.0), 1.0))
    st.markdown("</div>", unsafe_allow_html=True)
    
except Exception as e:
    st.error(f"Inference pipeline paused: {str(e)}")
