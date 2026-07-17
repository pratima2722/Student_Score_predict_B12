import streamlit as st
import pandas as pd
import pickle
import numpy as np

# Page configuration
st.set_page_config(
    page_title="Study Buddy • Score Predictor",
    page_icon="🌸",
    layout="centered"
)

# Custom CSS for the Cozy Aesthetic, Warm Pastel Gradients, and Soft Retro Shadows
st.markdown("""
    <style>
        /* Main background color - Soft Warm Cream */
        .stApp {
            background-color: #FAF6F0;
        }
        
        /* Pastel Cozy Card with a chunky retro shadow */
        .cozy-card {
            background-color: #FFFFFF;
            padding: 30px;
            border-radius: 24px;
            border: 2px solid #EADBC8;
            box-shadow: 6px 6px 0px 0px #DAC0A3;
            margin-bottom: 25px;
        }
        
        /* Pastel gradient container for the success result */
        .aesthetic-result-card {
            background: linear-gradient(135deg, #FFF2CC 0%, #F4D9E8 100%);
            border: 2px solid #E1BEE7;
            box-shadow: 6px 6px 0px 0px #C8A2C8;
            padding: 25px;
            border-radius: 24px;
            text-align: center;
            margin-top: 25px;
        }

        /* Styling buttons and headers to fit the vibe */
        h1 {
            color: #4A3E3D !important;
            font-family: 'Comfortaa', cursive, sans-serif;
            font-weight: 700;
        }
        
        p, label {
            color: #6F5E5C !important;
            font-weight: 500;
        }
        
        /* Custom styling for standard buttons to make them rounded */
        div.stButton > button {
            border-radius: 20px !important;
            border: 2px solid #4A3E3D !important;
            background-color: #FFF2D8 !important;
            color: #4A3E3D !important;
            box-shadow: 4px 4px 0px 0px #4A3E3D !important;
            transition: all 0.2s ease;
        }
        
        div.stButton > button:hover {
            transform: translate(2px, 2px);
            box-shadow: 2px 2px 0px 0px #4A3E3D !important;
            background-color: #FFE5B4 !important;
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
st.markdown("<h1 style='text-align: center; font-size: 2.5rem;'>🧸 Study Buddy AI</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; font-size: 1.1rem; margin-top: -10px;'>Predict your score & cozy up to your study goals! ✨🧁📝</p>", unsafe_allow_html=True)
st.markdown("<div style='text-align: center; font-size: 1.5rem; margin-bottom: 20px;'>🌸 📚 🛌 🎈 🎓</div>", unsafe_allow_html=True)

# --- INPUT SECTION (Vertical Cozy Card) ---
st.markdown('<div class="cozy-card">', unsafe_allow_html=True)
st.markdown("<h3 style='color: #6F5E5C; margin-top: 0;'>✍️ Log Your Daily Habits</h3>", unsafe_allow_html=True)

hours_studied = st.slider(
    "✏️ Hours Studied Daily", 
    min_value=0.0, 
    max_value=24.0, 
    value=5.0, 
    step=0.5,
    help="How long did you hit the books today?"
)

sleep_hours = st.slider(
    "🛌 Restful Sleep (Hours)", 
    min_value=0.0, 
    max_value=24.0, 
    value=7.0, 
    step=0.5,
    help="Getting those Zs is super important!"
)

attendance_percent = st.slider(
    "🎒 Attendance Rate (%)", 
    min_value=0.0, 
    max_value=100.0, 
    value=85.0, 
    step=1.0,
    help="Percentage of your classes attended."
)

previous_scores = st.number_input(
    "💯 Last Exam Score", 
    min_value=0.0, 
    max_value=100.0, 
    value=75.0, 
    step=1.0,
    help="What did you get on your last major test?"
)
st.markdown('</div>', unsafe_allow_html=True) # Close Cozy Card

# --- PREDICTION ACTION ---
if st.button("🌟 Reveal My Score Prediction! 🌟", use_container_width=True):
    input_data = pd.DataFrame([{
        "hours_studied": hours_studied,
        "sleep_hours": sleep_hours,
        "attendance_percent": attendance_percent,
        "previous_scores": previous_scores
    }])
    
    # Custom bouncy loading message
    with st.spinner("✨ Mixing magic potions & calculating your grade... 🥞"):
        try:
            # Predict
            prediction = model.predict(input_data)[0]
            
            # --- COZY RESULT DISPLAY ---
            st.markdown('<div class="aesthetic-result-card">', unsafe_allow_html=True)
            st.markdown("<h2 style='color: #4A3E3D; margin: 0; font-family: inherit;'>🎉 Yay! Prediction Ready 🎉</h2>", unsafe_allow_html=True)
            st.markdown("<p style='font-size: 1.1rem; color: #6F5E5C; margin-bottom: 5px;'>Based on your habits, you are on track for:</p>", unsafe_allow_html=True)
            
            # Big beautiful score
            st.markdown(f"<h1 style='color: #8E44AD; font-size: 3rem; margin: 10px 0;'>🍀 {prediction:.1f} / 100 🍀</h1>", unsafe_allow_html=True)
            
            # Progress bar matching the warm vibe
            st.progress(min(max(float(prediction) / 100.0, 0.0), 1.0))
            
            # Tiny motivational signoff
            st.markdown("<p style='font-size: 0.9rem; font-style: italic; margin-top: 10px; color: #8C7B78;'>Keep shining, you're doing great! 🧁🌸✨</p>", unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True) # Close Result Card
            
        except Exception as e:
            st.error(f"Oh no, a little hiccup occurred: {str(e)} 😿")
