import streamlit as st
import pandas as pd
import numpy as np
import pickle
 
# --- UI CONFIGURATION ---
st.set_page_config(page_title="Wine Quality Predictor", page_icon="🍷", layout="centered")
 
st.title("🍷 White Wine Quality Predictor")
st.write("Adjust the chemical properties below to predict if the wine is of High Quality (Good) or Standard Quality.")
 
# --- LOAD MODEL ---
@st.cache_resource
def load_model():
    with open('wine_model.pkl', 'rb') as f:
        return pickle.load(f)
 
model = load_model()
 
# --- USER INPUTS (Double-checked against your notebook) ---
st.subheader("Chemical Properties")
 
col1, col2 = st.columns(2)
 
with col1:
    fixed_acidity = st.slider("Fixed Acidity", 3.8, 14.2, 6.8)
    volatile_acidity = st.slider("Volatile Acidity", 0.08, 1.10, 0.26)
    residual_sugar = st.slider("Residual Sugar", 0.6, 65.8, 5.2)
    chlorides = st.slider("Chlorides", 0.009, 0.346, 0.043)
    free_sulfur_dioxide = st.slider("Free Sulfur Dioxide", 2.0, 289.0, 34.0)
 
with col2:
    total_sulfur_dioxide = st.slider("Total Sulfur Dioxide", 9.0, 440.0, 134.0)
    density = st.slider("Density", 0.987, 1.039, 0.994)
    pH = st.slider("pH", 2.72, 3.82, 3.18)
    sulphates = st.slider("Sulphates", 0.22, 1.08, 0.47)
    alcohol = st.slider("Alcohol (%)", 8.0, 14.2, 10.4)
 
# --- PREDICTION LOGIC ---
if st.button("Predict Wine Quality", type="primary"):
    # 1. Gather inputs
    # Note: citric_acid is intentionally omitted as per your notebook
    raw_data = {
        'fixed_acidity': fixed_acidity,
        'volatile_acidity': volatile_acidity,
        'residual_sugar': residual_sugar,
        'chlorides': chlorides,
        'free_sulfur_dioxide': free_sulfur_dioxide,
        'total_sulfur_dioxide': total_sulfur_dioxide,
        'density': density,
        'pH': pH,
        'sulphates': sulphates,
        'alcohol': alcohol
    }
    input_df = pd.DataFrame([raw_data])
    # 2. Apply the exact Log Transformations used in your Jupyter Notebook
    log_transform_cols = [
        'volatile_acidity', 'residual_sugar', 'chlorides', 
        'free_sulfur_dioxide', 'density', 'sulphates'
    ]
    for col in log_transform_cols:
        input_df[col] = np.log1p(input_df[col])
    # 3. Make Prediction
    prediction = model.predict(input_df)[0]
    st.divider()
    if prediction == 1:
        st.success("🌟 **Prediction: High Quality Wine (Score > 6)**")
    else:
        st.warning("📊 **Prediction: Standard Quality Wine (Score ≤ 6)**")
