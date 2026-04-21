import streamlit as st
import pandas as pd
import numpy as np
import pickle

# Page Configuration
st.set_page_config(
    page_title="Wine Quality AI",
    page_icon="🍷",
    layout="wide"
)

# Custom CSS for a cleaner look
st.markdown("""
    <style>
    .main {
        background-color: #f5f7f9;
    }
    .stButton>button {
        width: 100%;
        border-radius: 5px;
        height: 3em;
        background-color: #722F37;
        color: white;
    }
    .prediction-card {
        padding: 20px;
        border-radius: 10px;
        background-color: white;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    </style>
    """, unsafe_allow_html=True)

# Load the saved model
@st.cache_resource
def load_model():
    # Ensure wine_model.pkl is in your GitHub repo
    with open('wine_model.pkl', 'rb') as f:
        return pickle.load(f)

model = load_model()

# Header Section
st.title("🍷 White Wine Quality Classifier")
st.markdown("---")

# Layout: Two columns for Inputs
st.subheader("Adjust Chemical Properties")
col1, col2, col3 = st.columns(3)

with col1:
    fixed_acidity = st.slider("Fixed Acidity", 3.0, 15.0, 7.0)
    volatile_acidity = st.slider("Volatile Acidity", 0.0, 1.5, 0.27)
    citric_acid = st.slider("Citric Acid", 0.0, 1.0, 0.36)
    residual_sugar = st.slider("Residual Sugar", 0.0, 30.0, 20.7)

with col2:
    chlorides = st.slider("Chlorides", 0.0, 0.5, 0.045)
    free_sulfur_dioxide = st.slider("Free Sulfur Dioxide", 1.0, 100.0, 45.0)
    total_sulfur_dioxide = st.slider("Total Sulfur Dioxide", 5.0, 300.0, 170.0)
    density = st.slider("Density", 0.98, 1.01, 1.001, step=0.001)

with col3:
    ph = st.slider("pH Level", 2.5, 4.0, 3.0)
    sulphates = st.slider("Sulphates", 0.2, 1.5, 0.45)
    alcohol = st.slider("Alcohol (%)", 8.0, 15.0, 8.8)

# Transform inputs to match training preprocessing
input_data = pd.DataFrame({
    'fixed_acidity': [fixed_acidity],
    'volatile_acidity': [np.log1p(volatile_acidity)],
    'citric_acid': [np.log1p(citric_acid)],
    'residual_sugar': [np.log1p(residual_sugar)],
    'chlorides': [np.log1p(chlorides)],
    'free_sulfur_dioxide': [np.log1p(free_sulfur_dioxide)],
    'total_sulfur_dioxide': [total_sulfur_dioxide],
    'density': [np.log1p(density)],
    'pH': [ph],
    'sulphates': [np.log1p(sulphates)],
    'alcohol': [alcohol]
})

st.markdown("---")

# Prediction Section
if st.button("Analyze Wine Quality"):
    prediction = model.predict(input_data)
    probability = model.predict_proba(input_data)[0][1]
    
    res_col1, res_col2 = st.columns([1, 2])
    
    with res_col1:
        st.metric(label="Confidence Score", value=f"{probability:.1%}")
        
    with res_col2:
        if prediction[0] == 1:
            st.success("### Results: Premium Quality (Score > 6)")
            st.balloons()
        else:
            st.info("### Results: Standard Quality (Score ≤ 6)")

    # Display an insight
    st.info(f"**Insight:** Based on the current parameters, this wine has a {probability:.1%} chance of being rated as high quality by experts.")
