import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.graph_objects as go
import os

# Page Config
st.set_page_config(page_title="Wine Quality AI", page_icon="🍷", layout="wide")

# Custom Styling
st.markdown("""
    <style>
    .stButton>button { width: 100%; border-radius: 5px; height: 3em; background-color: #722F37; color: white; }
    </style>
    """, unsafe_allow_html=True)

@st.cache_resource
def load_assets():
    # Load model and raw data for comparison
    with open('wine_model.pkl', 'rb') as f:
        model = pickle.load(f)
    df = pd.read_csv('winequality-white.csv', sep=';')
    return model, df

try:
    model, raw_df = load_assets()
except FileNotFoundError:
    st.error("Error: 'wine_model.pkl' or 'winequality-white.csv' not found. Please upload them to your GitHub repo.")
    st.stop()

# --- SIDEBAR: ALL 11 INPUTS ---
st.sidebar.header("🧪 Chemical Analysis")
def get_user_inputs():
    # Ranges based on typical White Wine distribution
    fa = st.sidebar.slider("Fixed Acidity", 3.8, 14.2, 6.8)
    va = st.sidebar.slider("Volatile Acidity", 0.08, 1.1, 0.27)
    ca = st.sidebar.slider("Citric Acid", 0.0, 1.66, 0.33)
    rs = st.sidebar.slider("Residual Sugar", 0.6, 65.8, 6.3)
    cl = st.sidebar.slider("Chlorides", 0.009, 0.34, 0.04)
    fsd = st.sidebar.slider("Free Sulfur Dioxide", 2.0, 289.0, 35.0)
    tsd = st.sidebar.slider("Total Sulfur Dioxide", 9.0, 440.0, 138.0)
    de = st.sidebar.slider("Density", 0.987, 1.038, 0.994, step=0.001)
    ph = st.sidebar.slider("pH Level", 2.7, 3.8, 3.18)
    su = st.sidebar.slider("Sulphates", 0.2, 1.08, 0.49)
    al = st.sidebar.slider("Alcohol (%)", 8.0, 14.2, 10.5)
    
    # Preprocess inputs to match model training (Log 1p)
    processed_data = {
        'fixed_acidity': fa,
        'volatile_acidity': np.log1p(va),
        'citric_acid': np.log1p(ca),
        'residual_sugar': np.log1p(rs),
        'chlorides': np.log1p(cl),
        'free_sulfur_dioxide': np.log1p(fsd),
        'total_sulfur_dioxide': tsd,
        'density': np.log1p(de),
        'pH': ph,
        'sulphates': np.log1p(su),
        'alcohol': al
    }
    return pd.DataFrame(processed_data, index=[0]), [fa, va, ca, rs, cl, fsd, tsd, de, ph, su, al]

input_df, raw_list = get_user_inputs()

# --- MAIN PANEL ---
st.title("🍷 Premium Wine Quality Predictor")
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("Analysis")
    prob = model.predict_proba(input_df)[0][1]
    pred = model.predict(input_df)[0]
    
    # Gauge Chart
    fig = go.Figure(go.Indicator(
        mode = "gauge+number",
        value = prob * 100,
        title = {'text': "Confidence Score (%)"},
        gauge = {'bar': {'color': "#722F37"}, 'axis': {'range': [0, 100]}}
    ))
    fig.update_layout(height=300)
    st.plotly_chart(fig, use_container_width=True)

    if pred == 1:
        st.success("✨ **RESULT: High Quality Wine**")
    else:
        st.info("🧐 **RESULT: Standard Quality Wine**")

with col2:
    st.subheader("Comparison Profile")
    categories = ['Fixed Acid', 'Volatile Acid', 'Citric Acid', 'Sugar', 'Chlorides', 'Free SO2', 'Total SO2', 'Density', 'pH', 'Sulphates', 'Alcohol']
    # Mean of good wines from your dataset for the radar
    avg_good = raw_df[raw_df['quality'] > 6].mean().values[:11]
    
    radar = go.Figure()
    radar.add_trace(go.Scatterpolar(r=raw_list, theta=categories, fill='toself', name='Your Wine', line_color='#722F37'))
    radar.add_trace(go.Scatterpolar(r=avg_good, theta=categories, fill='toself', name='Avg Premium', line_color='#999999'))
    radar.update_layout(polar=dict(radialaxis=dict(visible=False)), height=400)
    st.plotly_chart(radar, use_container_width=True)
