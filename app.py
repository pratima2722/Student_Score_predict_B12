import streamlit as st
import pandas as pd
import pickle
import numpy as np

# Page configuration
st.set_page_config(
    page_title="Fairy Study Garden",
    page_icon="🌸",
    layout="centered"
)

# Custom CSS for Floral Cartoon Vibes & Bouncy Animations
st.markdown("""
    <style>
        /* Pastel floral sky background */
        .stApp {
            background: linear-gradient(135deg, #FFF0F5 0%, #E6E6FA 50%, #FFF9E6 100%);
        }
        
        /* Cute Cartoon Flower Card */
        .flower-card {
            background-color: rgba(255, 255, 255, 0.95);
            padding: 30px;
            border-radius: 28px;
            border: 4px dashed #FFB7B2;
            box-shadow: 0px 10px 25px rgba(255, 183, 178, 0.4);
            margin-bottom: 25px;
            transition: transform 0.3s ease;
        }
        
        .flower-card:hover {
            transform: translateY(-3px);
        }
        
        /* Puffy 3D Bouncing Button */
        div.stButton > button {
            background: linear-gradient(135deg, #FFB7B2 0%, #FFDAC1 100%) !important;
            color: #5D4037 !important;
            border: 3px solid #FF9AA2 !important;
            border-radius: 30px !important;
            padding: 15px 30px !important;
            font-size: 1.25rem !important;
            font-weight: 900 !important;
            box-shadow: 0px 8px 0px 0px #FF9AA2, 0px 15px 20px rgba(255, 154, 162, 0.3) !important;
            transition: all 0.2s cubic-bezier(0.175, 0.885, 0.32, 1.275) !important;
            width: 100%;
        }
        
        div.stButton > button:hover {
            transform: scale(1.02) translateY(-2px) !important;
            background: linear-gradient(135deg, #FF9AA2 0%, #FFB7B2 100%) !important;
        }

        div.stButton > button:active {
            transform: translateY(6px) !important;
            box-shadow: 0px 2px 0px 0px #FF9AA2 !important;
        }

        /* Floating Flower Result Container */
        .bloom-result-card {
            background: linear-gradient(135deg, #EBFCE5 0%, #FFF3F8 100%);
            border: 4px solid #B5EAD7;
            border-radius: 32px;
            padding: 30px;
            text-align: center;
            margin-top: 30px;
            box-shadow: 0px 12px 25px rgba(181, 234, 215, 0.5);
            animation: sway 4s ease-in-out infinite;
        }

        /* Swaying cartoon stem animation */
        @keyframes sway {
            0%, 100% { transform: rotate(-1.5deg) translateY(0px); }
            50% { transform: rotate(1.5deg) translateY(-8px); }
        }

        /* Typography colors */
        h1 {
            color: #FF6F91 !important;
            font-weight: 900;
            text-shadow: 2px 2px 0px #FFF, 4px 4px 0px #FFD1D1;
            font-family: 'Comic Sans MS', cursive, sans-serif;
        }
        
        h3 {
            color: #FF80AB !important;
            font-family: 'Comic Sans MS', cursive, sans-serif;
        }
        
        p, label {
            color: #6C5B7B !important;
            font-weight: 800;
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
    st.error("⚠️ `model.pkl` not found! Put it in the same folder as `app.py`.")
    st.stop()

# --- HEADER SECTION ---
st.markdown("<h1 style='text-align: center; font-size: 2.8rem; margin-bottom: 5px;'>🌸 FAIRY STUDY MEADOW 🌸</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; font-size: 1.15rem; margin-top: -10px;'>Cast a magical prediction spell on your grades! 🧚‍♀️🌷🐝✨</p>", unsafe_allow_html=True)
st.markdown("<div style='text-align: center; font-size: 2rem; margin-bottom: 20px;'>🌹🌻🍀🦋🌼🍄🎈</div>", unsafe_allow_html=True)

# --- INPUT SECTION (Fairy Meadow Card) ---
st.markdown('<div class="flower-card">', unsafe_allow_html=True)
st.markdown("<h3 style='margin-top: 0;'>🌷 Your Study Garden Journal</h3>", unsafe_allow_html=True)

# Sliders configured to match the features required by the model[cite: 1]
hours_studied = st.slider(
    "🌸 Hours spent in the Study Patch", 
    min_value=0.0, 
    max_value=24.0, 
    value=5.0, 
    step=0.5,
    help="How long did you nurture your knowledge today?"
)

sleep_hours = st.slider(
    "🛌 Dreamy Cloud Sleep (Hours)", 
    min_value=0.0, 
    max_value=24.0, 
    value=7.0, 
    step=0.5,
    help="Recharging your fairy wings!"
)

attendance_percent = st.slider(
    "🎒 Fairy Class Attendance (%)", 
    min_value=0.0, 
    max_value=100.0, 
    value=85.0, 
    step=1.0,
    help="How many magical classes did you fly into?"
)

previous_scores = st.number_input(
    "✨ Past Spellcast Exam Score", 
    min_value=0.0, 
    max_value=100.0, 
    value=75.0, 
    step=1.0,
    help="What was your rating on your last scroll?"
)
st.markdown('</div>', unsafe_allow_html=True) # Close Flower Card

# --- PREDICTION ACTION ---
if st.button("🌷 Make My Grade Bloom! 🌷", use_container_width=True):
    # Constructing input dataframe for model compatibility[cite: 1]
    input_data = pd.DataFrame([{
        "hours_studied": hours_studied,
        "sleep_hours": sleep_hours,
        "attendance_percent": attendance_percent,
        "previous_scores": previous_scores
    }])
    
    with st.spinner("🦋 Fairies are gathering pollen & calculating your grade... 🍯"):
        try:
            # Predict[cite: 1]
            prediction = model.predict(input_data)[0]
            
            # --- FLOATING COZY BLOOM CONTAINER ---
            st.markdown('<div class="bloom-result-card">', unsafe_allow_html=True)
            st.markdown("<h2 style='color: #FF6F91; margin: 0;'>🌸 Your Score has Bloomed! 🌸</h2>", unsafe_allow_html=True)
            st.markdown("<p style='font-size: 1.05rem; color: #6C5B7B; margin-bottom: 0;'>The forest oracle predicts your final outcome at:</p>", unsafe_allow_html=True)
            
            # Massive pink bubble score
            st.markdown(f"<h1 style='color: #FF80AB; font-size: 4rem; margin: 15px 0; text-shadow: 2px 2px 0px #FFF;'>🌼 {prediction:.1f}% 🌼</h1>", unsafe_allow_html=True)
            
            # Pastel-toned progress indicator
            st.progress(min(max(float(prediction) / 100.0, 0.0), 1.0))
            
            st.markdown("<p style='font-size: 0.95rem; font-style: italic; margin-top: 15px; color: #8F819D;'>You're growing beautifully every single day! 🧚‍♀️🍯🌻</p>", unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True) # Close Result Card
            
        except Exception as e:
            st.error(f"Oh dear, a thorn blocked the prediction path: {str(e)} 🥀")
