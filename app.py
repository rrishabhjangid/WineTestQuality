import streamlit as st
import pandas as pd
import numpy as np
import pickle

# Load the saved model
@st.cache_resource
def load_model():
    with open('wine_model.pkl', 'rb') as f:
        return pickle.load(f)

model = load_model()

st.title("🍷 White Wine Quality Predictor")
st.markdown("Enter the chemical properties below to check if the wine is 'Good' or 'Average'.")

# Input fields in the sidebar
st.sidebar.header("Input Wine Parameters")

def user_input_features():
    fixed_acidity = st.sidebar.number_input("Fixed Acidity", value=7.0)
    volatile_acidity = st.sidebar.number_input("Volatile Acidity", value=0.27)
    citric_acid = st.sidebar.number_input("Citric Acid", value=0.36)
    residual_sugar = st.sidebar.number_input("Residual Sugar", value=20.7)
    chlorides = st.sidebar.number_input("Chlorides", value=0.045)
    free_sulfur_dioxide = st.sidebar.number_input("Free Sulfur Dioxide", value=45.0)
    total_sulfur_dioxide = st.sidebar.number_input("Total Sulfur Dioxide", value=170.0)
    density = st.sidebar.number_input("Density", value=1.001)
    ph = st.sidebar.number_input("pH", value=3.0)
    sulphates = st.sidebar.number_input("Sulphates", value=0.45)
    alcohol = st.sidebar.number_input("Alcohol Content", value=8.8)
    
    data = {
        'fixed_acidity': fixed_acidity,
        'volatile_acidity': np.log1p(volatile_acidity),
        'citric_acid': np.log1p(citric_acid),
        'residual_sugar': np.log1p(residual_sugar),
        'chlorides': np.log1p(chlorides),
        'free_sulfur_dioxide': np.log1p(free_sulfur_dioxide),
        'total_sulfur_dioxide': total_sulfur_dioxide,
        'density': np.log1p(density),
        'pH': ph,
        'sulphates': np.log1p(sulphates),
        'alcohol': alcohol
    }
    return pd.DataFrame(data, index=[0])

input_df = user_input_features()

# Display inputs
st.subheader("Current Inputs")
st.write(input_df)

# Prediction
if st.button("Predict Quality"):
    prediction = model.predict(input_df)
    probability = model.predict_proba(input_df)

    if prediction[0] == 1:
        st.success("✨ This is a **GOOD** quality wine!")
    else:
        st.warning("🧐 This is an **AVERAGE** quality wine.")
        
    st.write(f"Probability of being high quality: {probability[0][1]:.2%}")
