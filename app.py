import streamlit as st
import pandas as pd
import pickle
import numpy as np

# Page configuration
st.set_page_config(
    page_title="Core Analytics 3D Engine",
    page_icon="🕹️",
    layout="centered"
)

# Custom CSS for 3D extrusion, Neumorphic Depth, and Interactive Transforms
st.markdown("""
    <style>
        /* Base Deep Steel Background */
        .stApp {
            background-color: #111827;
        }
        
        /* 3D Extruded Neumorphic Input Card */
        .extruded-card {
            background-color: #1f2937;
            padding: 32px;
            border-radius: 20px;
            border: 1px solid #374151;
            /* Multi-layered shadows create realistic 3D elevation */
            box-shadow: 
                0px 1px 2px rgba(255, 255, 255, 0.05) inset,
                0px 15px 30px -10px rgba(0, 0, 0, 0.6),
                0px 4px 6px -2px rgba(0, 0, 0, 0.5);
            margin-bottom: 30px;
            transition: transform 0.3s ease, box-shadow 0.3s ease;
        }
        
        /* Slight tilt/raise on hover */
        .extruded-card:hover {
            transform: translateY(-4px);
            box-shadow: 
                0px 1px 2px rgba(255, 255, 255, 0.08) inset,
                0px 20px 40px -10px rgba(0, 0, 0, 0.8),
                0px 6px 12px -2px rgba(0, 0, 0, 0.6);
        }
        
        /* Physical 3D Push Button Style */
        div.stButton > button {
            background: linear-gradient(135deg, #00f2fe 0%, #4facfe 100%) !important;
            color: #0d1b2a !important;
            border: none !important;
            border-radius: 12px !important;
            padding: 16px 30px !important;
            font-size: 1.1rem !important;
            font-weight: 800 !important;
            letter-spacing: 1px !important;
            text-transform: uppercase !important;
            /* Hard bottom-shadow simulates a physical key cap */
            box-shadow: 
                0px 6px 0px 0px #005691, 
                0px 12px 24px 0px rgba(0, 242, 254, 0.3) !important;
            transition: all 0.1s ease-in-out !important;
            margin-bottom: 10px;
        }
        
        /* The physical press down click effect */
        div.stButton > button:active {
            transform: translateY(4px) !important;
            box-shadow: 
                0px 2px 0px 0px #005691, 
                0px 4px 10px 0px rgba(0, 242, 254, 0.2) !important;
        }
        
        /* 3D Isometric Tilt Result Card */
        .isometric-card {
            background: linear-gradient(145deg, #1e1b4b, #311042);
            border-radius: 24px;
            padding: 30px;
            border: 1px solid rgba(255, 255, 255, 0.1);
            text-align: center;
            margin-top: 30px;
            /* Creates depth angle projection */
            transform: perspective(1000px) rotateX(10deg) rotateY(-8deg) rotateZ(1deg);
            box-shadow: 
                -10px 15px 30px rgba(0, 0, 0, 0.8),
                5px -5px 15px rgba(255, 255, 255, 0.03) inset;
            transition: transform 0.5s cubic-bezier(0.175, 0.885, 0.32, 1.275), box-shadow 0.5s ease;
        }
        
        /* Snaps flat and gets shiny when hovered */
        .isometric-card:hover {
            transform: perspective(1000px) rotateX(0deg) rotateY(0deg) rotateZ(0deg);
            box-shadow: 
                0px 20px 40px rgba(0, 0, 0, 0.9),
                0px 0px 25px rgba(168, 85, 247, 0.4);
        }

        /* Dark mode typography styling */
        h1, h2, h3, p, label {
            color: #f3f4f6 !important;
        }
        
        .accent-title {
            background: linear-gradient(90deg, #00f2fe, #a855f7);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            font-weight: 900;
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
st.markdown("<h1 style='text-align: center; font-size: 2.7rem; margin-bottom: 0px;'>🕹️ <span class='accent-title'>PREDICTIVE MATRIX</span></h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: #9ca3af !important; font-size: 1.1rem;'>Experience hardware-accelerated 3D regression analytics.</p>", unsafe_allow_html=True)
st.markdown("<div style='height: 4px; background: linear-gradient(90deg, transparent, #00f2fe, #a855f7, transparent); margin: 25px 0;'></div>", unsafe_allow_html=True)

# --- INPUT SECTION (3D Extruded Steel Card) ---
st.markdown('<div class="extruded-card">', unsafe_allow_html=True)
st.markdown("<h3 style='margin-top: 0; color: #00f2fe !important;'>🎛️ TELEMETRY CONFIGURATION</h3>", unsafe_allow_html=True)

hours_studied = st.slider(
    "✏️ STUDY HOURS", 
    min_value=0.0, 
    max_value=24.0, 
    value=5.0, 
    step=0.5,
    help="Daily active studying hours."
)

sleep_hours = st.slider(
    "🛌 SLEEP CYCLES (HOURS)", 
    min_value=0.0, 
    max_value=24.0, 
    value=7.0, 
    step=0.5,
    help="Average sleep recovery time."
)

attendance_percent = st.slider(
    "🎒 ATTENDANCE INDEX (%)", 
    min_value=0.0, 
    max_value=100.0, 
    value=85.0, 
    step=1.0,
    help="Classroom attendance percentage."
)

previous_scores = st.number_input(
    "⚡ HISTORIC PERFORMANCE SCORE", 
    min_value=0.0, 
    max_value=100.0, 
    value=75.0, 
    step=1.0,
    help="The score recorded in the previous test cycle."
)
st.markdown('</div>', unsafe_allow_html=True) # Close Extruded Card

# --- PREDICTION ACTION ---
if st.button("🔌 INITIALIZE PREDICTION PIPELINE", use_container_width=True):
    input_data = pd.DataFrame([{
        "hours_studied": hours_studied,
        "sleep_hours": sleep_hours,
        "attendance_percent": attendance_percent,
        "previous_scores": previous_scores
    }])
    
    with st.spinner("🤖 Simulating data flow across dimensions..."):
        try:
            # Predict
            prediction = model.predict(input_data)[0]
            
            # --- 3D ISOMETRIC RESULT DISPLAY ---
            st.markdown('<div class="isometric-card">', unsafe_allow_html=True)
            st.markdown("<h2 style='color: #00f2fe; margin: 0; font-weight: 800; letter-spacing: 1px;'>🔮 CALCULATION COMPLETE</h2>", unsafe_allow_html=True)
            st.markdown("<p style='color: #d1d5db !important; font-size: 1rem; margin-top: 5px;'>Our KNN Matrix model estimates your final output at:</p>", unsafe_allow_html=True)
            
            # Massive glowing output
            st.markdown(f"<h1 style='color: #fff; font-size: 4rem; margin: 15px 0; text-shadow: 0 0 20px rgba(0, 242, 254, 0.8), 0 0 40px rgba(168, 85, 247, 0.6);'>{prediction:.2f}%</h1>", unsafe_allow_html=True)
            
            st.progress(min(max(float(prediction) / 100.0, 0.0), 1.0))
            st.markdown("<p style='font-size: 0.8rem; color: #a78bfa !important; margin-top: 15px; letter-spacing: 1.5px;'>💡 PRO TIP: Hover over this card to level it out in 3D space!</p>", unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True) # Close Isometric Card
            
        except Exception as e:
            st.error(f"Failed to execute calculation matrix: {str(e)} 🔌")
