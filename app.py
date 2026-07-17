import streamlit as st
import pandas as pd
import pickle
import numpy as np

# Page configuration
st.set_page_config(
    page_title="Performance Analytics AI",
    page_icon="⚡",
    layout="centered"
)

# Custom CSS for Dark Glassmorphism, Neon Accents, and Soft Glow Shadows
st.markdown("""
    <style>
        /* Main background tuning for a dark, premium feel */
        .stApp {
            background-color: #0d1117;
        }
        
        /* Glassmorphic input container */
        .glass-card {
            background: rgba(22, 27, 34, 0.8);
            backdrop-filter: blur(8px);
            -webkit-backdrop-filter: blur(8px);
            padding: 30px;
            border-radius: 16px;
            border: 1px solid rgba(240, 246, 252, 0.1);
            box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37);
            margin-bottom: 24px;
        }
        
        /* High-tech glowing prediction result box */
        .glow-card {
            background: rgba(13, 27, 42, 0.9);
            border: 1px solid #58a6ff;
            box-shadow: 0 0 15px rgba(88, 166, 255, 0.4);
            padding: 25px;
            border-radius: 16px;
            text-align: center;
            margin-top: 25px;
        }
        
        /* Custom typography adjustments */
        h1, h2, h3, p, label {
            color: #f0f6fc !important;
        }
        
        .subheader-text {
            color: #58a6ff !important;
            font-weight: 600;
            letter-spacing: 0.5px;
            margin-bottom: 20px;
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
st.markdown("<h1 style='text-align: center; margin-bottom: 5px;'>⚡ Performance Analytics AI</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; opacity: 0.8; font-size: 1.1rem;'>Predicting academic trajectory through machine learning</p>", unsafe_allow_html=True)
st.markdown("<div style='height: 2px; background: linear-gradient(90deg, transparent, #58a6ff, transparent); margin: 20px 0;'></div>", unsafe_allow_html=True)

# --- INPUT SECTION (Vertical Glass Card) ---
st.markdown('<div class="glass-card">', unsafe_allow_html=True)
st.markdown('<div class="subheader-text">📐 TELEMETRY INPUTS</div>', unsafe_allow_html=True)

hours_studied = st.slider(
    "Hours Studied", 
    min_value=0.0, 
    max_value=24.0, 
    value=5.0, 
    step=0.5,
    help="Number of hours spent studying daily."
)

sleep_hours = st.slider(
    "Sleep Hours", 
    min_value=0.0, 
    max_value=24.0, 
    value=7.0, 
    step=0.5,
    help="Average hours of sleep per night."
)

attendance_percent = st.slider(
    "Attendance Percentage (%)", 
    min_value=0.0, 
    max_value=100.0, 
    value=85.0, 
    step=1.0,
    help="Percentage of classes attended."
)

previous_scores = st.number_input(
    "Previous Exam Score", 
    min_value=0.0, 
    max_value=100.0, 
    value=75.0, 
    step=1.0,
    help="Score obtained in the previous assessment."
)
st.markdown('</div>', unsafe_allow_html=True) # Close Glass Card

# --- PREDICTION ACTION ---
if st.button("🔮 Run Predictive Engine", type="primary", use_container_width=True):
    # Construct the input DataFrame to preserve feature names
    input_data = pd.DataFrame([{
        "hours_studied": hours_studied,
        "sleep_hours": sleep_hours,
        "attendance_percent": attendance_percent,
        "previous_scores": previous_scores
    }])
    
    with st.spinner("Processing telemetry metrics..."):
        try:
            # Predict
            prediction = model.predict(input_data)[0]
            
            # --- RESULT DISPLAY (Glowing Tech Card) ---
            st.markdown('<div class="glow-card">', unsafe_allow_html=True)
            st.markdown("<h3 style='color: #58a6ff; margin: 0; font-weight: bold;'>💻 PREDICTED RESULT</h3>", unsafe_allow_html=True)
            
            # Utilizing st.metric inside the card
            st.metric(
                label="Estimated Academic Output", 
                value=f"{prediction:.2f} %"
            )
            
            # Matching styled progress indicator
            st.progress(min(max(float(prediction) / 100.0, 0.0), 1.0))
            st.markdown('</div>', unsafe_allow_html=True) # Close Glow Card
            
        except Exception as e:
            st.error(f"Prediction Pipeline Error: {str(e)}")
