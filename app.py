import streamlit as st
import pandas as pd
import pickle
import numpy as np

# Page configuration
st.set_page_config(
    page_title="Study Land • Sweet Predictor",
    page_icon="🍬",
    layout="centered"
)

# Custom CSS for 3D Claymorphism, Soft Inflated Borders, and Bouncy Animations
st.markdown("""
    <style>
        /* Creamy pastel sky background */
        .stApp {
            background: linear-gradient(180deg, #FFF5E4 0%, #FFE3E3 100%);
        }
        
        /* 3D Claymorphic Card: Soft, puffed-up plastic appearance */
        .clay-card {
            background-color: #FFFFFF;
            padding: 30px;
            border-radius: 30px;
            border: 4px solid #FFD1D1;
            /* Inner light + heavy soft outer shadow creates a squishy 3D look */
            box-shadow: 
                inset 4px 4px 0px rgba(255, 255, 255, 0.8),
                inset -4px -4px 0px rgba(0, 0, 0, 0.03),
                8px 12px 0px 0px #FFD1D1;
            margin-bottom: 30px;
            transition: all 0.3s cubic-bezier(0.175, 0.885, 0.32, 1.275);
        }
        
        /* Interactive scale-up when hovering over the card */
        .clay-card:hover {
            transform: scale(1.02) translateY(-5px);
            box-shadow: 
                inset 4px 4px 0px rgba(255, 255, 255, 0.8),
                inset -4px -4px 0px rgba(0, 0, 0, 0.03),
                12px 18px 0px 0px #FFD1D1;
        }
        
        /* 3D "Squishy Toy" Button */
        div.stButton > button {
            background: linear-gradient(135deg, #FF9494 0%, #FFB4B4 100%) !important;
            color: #FFFFFF !important;
            border: 4px solid #FF7B7B !important;
            border-radius: 25px !important;
            padding: 15px 30px !important;
            font-size: 1.2rem !important;
            font-weight: 900 !important;
            text-transform: uppercase !important;
            letter-spacing: 1px !important;
            /* Deep button base shadow */
            box-shadow: 0px 8px 0px 0px #FF7B7B, 0px 15px 20px rgba(255, 123, 123, 0.3) !important;
            transition: all 0.15s cubic-bezier(0.175, 0.885, 0.32, 1.275) !important;
            margin-bottom: 15px;
        }
        
        /* Physical press action */
        div.stButton > button:active {
            transform: translateY(6px) !important;
            box-shadow: 0px 2px 0px 0px #FF7B7B, 0px 5px 10px rgba(255, 123, 123, 0.2) !important;
        }

        /* Claymorphic Bubble for Results */
        .clay-result-card {
            background: linear-gradient(135deg, #E8F9FD 0%, #DFF6FF 100%);
            border: 4px solid #B4E4FF;
            border-radius: 32px;
            padding: 25px;
            text-align: center;
            margin-top: 30px;
            box-shadow: 
                inset 6px 6px 0px rgba(255, 255, 255, 0.9),
                8px 12px 0px 0px #B4E4FF;
            animation: floaty 3s ease-in-out infinite;
        }

        /* 3D Floating animation for the score bubble */
        @keyframes floaty {
            0% { transform: translateY(0px); }
            50% { transform: translateY(-8px); }
            100% { transform: translateY(0px); }
        }

        /* Soft cartoon fonts and colors */
        h1 {
            color: #FF7B7B !important;
            font-weight: 900;
            text-shadow: 2px 2px 0px #FFE3E3, 4px 4px 0px #FFD1D1;
        }
        
        p, label {
            color: #85586F !important;
            font-weight: 700;
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
st.markdown("<h1 style='text-align: center; font-size: 2.8rem; margin-bottom: 5px;'>🍭 STUDY LAND</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; font-size: 1.1rem; margin-top: -10px;'>Your magical 3D companion to test-prep success! 🎪✨🌈</p>", unsafe_allow_html=True)
st.markdown("<div style='text-align: center; font-size: 1.8rem; margin-bottom: 15px;'>🧸🎒🧁🎈🍯</div>", unsafe_allow_html=True)

# --- INPUT SECTION (3D Squishy Clay Card) ---
st.markdown('<div class="clay-card">', unsafe_allow_html=True)
st.markdown("<h3 style='color: #FF7B7B; margin-top: 0; font-weight:800;'>🎨 My Study Teleport</h3>", unsafe_allow_html=True)

hours_studied = st.slider(
    "✏️ Hours Studied Daily", 
    min_value=0.0, 
    max_value=24.0, 
    value=5.0, 
    step=0.5,
    help="How many hours did you study today?"
)

sleep_hours = st.slider(
    "🛌 Sleep Rest Time (Hours)", 
    min_value=0.0, 
    max_value=24.0, 
    value=7.0, 
    step=0.5,
    help="Resting time for your brainy battery!"
)

attendance_percent = st.slider(
    "🎒 Fun School Attendance (%)", 
    min_value=0.0, 
    max_value=100.0, 
    value=85.0, 
    step=1.0,
    help="Percentage of classroom adventures unlocked!"
)

previous_scores = st.number_input(
    "💯 Last Magic Exam Score", 
    min_value=0.0, 
    max_value=100.0, 
    value=75.0, 
    step=1.0,
    help="What was your score on the last exam?"
)
st.markdown('</div>', unsafe_allow_html=True) # Close Clay Card

# --- PREDICTION ACTION ---
if st.button("🎈 Cast Prediction Spell! 🎈", use_container_width=True):
    input_data = pd.DataFrame([{
        "hours_studied": hours_studied,
        "sleep_hours": sleep_hours,
        "attendance_percent": attendance_percent,
        "previous_scores": previous_scores
    }])
    
    with st.spinner("🧸 Baking your sweet results inside the 3D oven... 🧁"):
        try:
            # Predict
            prediction = model.predict(input_data)[0]
            
            # --- CLAYMORPHIC RESULT DISPLAY ---
            st.markdown('<div class="clay-result-card">', unsafe_allow_html=True)
            st.markdown("<h2 style='color: #4bacfe; margin: 0; font-weight: 900;'>⭐️ Super Result Ready! ⭐️</h2>", unsafe_allow_html=True)
            st.markdown("<p style='font-size: 1rem; color: #5885AF; margin-bottom: 0;'>Your predicted future score is:</p>", unsafe_allow_html=True)
            
            # Inflated score value with beautiful shadow
            st.markdown(f"<h1 style='color: #FF7B7B; font-size: 3.8rem; margin: 10px 0; text-shadow: 3px 3px 0px #FFF;'>🍬 {prediction:.1f} % 🍬</h1>", unsafe_allow_html=True)
            
            st.progress(min(max(float(prediction) / 100.0, 0.0), 1.0))
            st.markdown("<p style='font-size: 0.9rem; font-style: italic; margin-top: 10px; color: #7B8FA1;'>Yay! You are doing absolutely amazing! 🦄🧸🍯</p>", unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True) # Close Result Card
            
        except Exception as e:
            st.error(f"Oh oops, something got tangled in the toy box: {str(e)} 😿")
