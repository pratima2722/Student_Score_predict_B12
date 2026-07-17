import streamlit as st
import pandas as pd
import pickle
import numpy as np

# Page configuration
st.set_page_config(
    page_title="Student Performance Predictor",
    page_icon="🎓",
    layout="centered"
)

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

# Header Section
st.title("🎓 Student Performance Predictor")
st.write("Enter the student's metrics below to predict their final score using the trained KNN model.")
st.markdown("---")

# Input Layout
st.subheader("📝 Input Student Metrics")

col1, col2 = st.columns(2)

with col1:
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

with col2:
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

st.markdown("---")

# Make Prediction
if st.button("🔮 Predict Grade/Score", type="primary", use_container_width=True):
    # Construct the input DataFrame to preserve feature names
    input_data = pd.DataFrame([{
        "hours_studied": hours_studied,
        "sleep_hours": sleep_hours,
        "attendance_percent": attendance_percent,
        "previous_scores": previous_scores
    }])
    
    with st.spinner("Calculating prediction..."):
        try:
            # Predict
            prediction = model.predict(input_data)[0]
            
            # Display Result
            st.success("🎉 Prediction Calculated Successfully!")
            
            # Visual presentation of the score
            st.metric(
                label="Predicted Final Score", 
                value=f"{prediction:.2f} / 100"
            )
            
            # Contextual progress bar
            st.progress(min(max(float(prediction) / 100.0, 0.0), 1.0))
            
        except Exception as e:
            st.error(f"An error occurred during prediction: {str(e)}")
