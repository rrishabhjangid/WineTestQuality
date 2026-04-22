import streamlit as st
import pandas as pd
import numpy as np
import pickle
 
# --- UI CONFIGURATION ---
st.set_page_config(page_title="Wine Quality Predictor", page_icon="🍷", layout="centered")
 
st.title("🍷 White Wine Quality Predictor")
st.write("Adjust the chemical properties below to predict the wine's quality.")
 
# --- LOAD MODEL & SCALER ---
@st.cache_resource
def load_assets():
    with open('wine_model.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('wine_scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    return model, scaler
 
model, scaler = load_assets()
 
# --- USER INPUTS (Updated to 8 Features) ---
st.subheader("Chemical Properties")
 
col1, col2 = st.columns(2)
 
with col1:
    fixed_acidity = st.slider("Fixed Acidity", 3.8, 14.2, 6.8)
    volatile_acidity = st.slider("Volatile Acidity", 0.08, 1.10, 0.26)
    chlorides = st.slider("Chlorides", 0.009, 0.346, 0.043)
    free_sulfur_dioxide = st.slider("Free Sulfur Dioxide", 2.0, 289.0, 34.0)
 
with col2:
    total_sulfur_dioxide = st.slider("Total Sulfur Dioxide", 9.0, 440.0, 134.0)
    pH = st.slider("pH", 2.72, 3.82, 3.18)
    sulphates = st.slider("Sulphates", 0.22, 1.08, 0.47)
    alcohol = st.slider("Alcohol (%)", 8.0, 14.2, 10.4)
 
# --- PREDICTION LOGIC ---
if st.button("Predict Wine Quality", type="primary"):
    # 1. Gather inputs in the EXACT order the model was trained on
    raw_data = {
        'fixed_acidity': fixed_acidity,
        'volatile_acidity': volatile_acidity,
        'chlorides': chlorides,
        'free_sulfur_dioxide': free_sulfur_dioxide,
        'total_sulfur_dioxide': total_sulfur_dioxide,
        'pH': pH,
        'sulphates': sulphates,
        'alcohol': alcohol
    }
    input_df = pd.DataFrame([raw_data])
    # 2. Apply Log Transformations (only to the features that weren't dropped)
    log_transform_cols = ['volatile_acidity', 'chlorides', 'free_sulfur_dioxide', 'sulphates']
    for col in log_transform_cols:
        input_df[col] = np.log1p(input_df[col])
    # 3. Apply the StandardScaler
    input_scaled = scaler.transform(input_df)
    # 4. Make Prediction
    prediction = model.predict(input_scaled)[0]
    st.divider()
    if prediction == 1:
        st.success("🌟 **Prediction: High Quality Wine (Score > 6)**")
    else:
        st.warning("📊 **Prediction: Standard Quality Wine (Score ≤ 6)**")
