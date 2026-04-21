import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.graph_objects as go
import plotly.express as px

# Page Configuration
st.set_page_config(page_title="Wine AI Pro", page_icon="🍷", layout="wide")

# Load Model & Data (Data is used for comparison charts)
@st.cache_resource
def load_assets():
    with open('wine_model.pkl', 'rb') as f:
        model = pickle.load(f)
    # Load dataset to get averages for the radar chart
    df = pd.read_csv('winequality-white.csv', sep=';')
    return model, df

model, raw_df = load_assets()

# --- SIDEBAR INPUTS ---
st.sidebar.header("🧪 Chemical Analysis")
def user_input_features():
    # Using sliders with ranges based on typical white wine values
    fixed_acidity = st.sidebar.slider("Fixed Acidity", 4.0, 14.0, 6.8)
    volatile_acidity = st.sidebar.slider("Volatile Acidity", 0.1, 1.1, 0.27)
    citric_acid = st.sidebar.slider("Citric Acid", 0.0, 1.0, 0.33)
    residual_sugar = st.sidebar.slider("Residual Sugar", 0.6, 30.0, 6.3)
    chlorides = st.sidebar.slider("Chlorides", 0.01, 0.3, 0.04)
    free_sulfur_dioxide = st.sidebar.slider("Free Sulfur Dioxide", 2.0, 200.0, 35.0)
    total_sulfur_dioxide = st.sidebar.slider("Total Sulfur Dioxide", 9.0, 400.0, 138.0)
    density = st.sidebar.slider("Density", 0.987, 1.003, 0.994, step=0.001)
    ph = st.sidebar.slider("pH Level", 2.7, 3.8, 3.18)
    sulphates = st.sidebar.slider("Sulphates", 0.2, 1.1, 0.49)
    alcohol = st.sidebar.slider("Alcohol (%)", 8.0, 14.0, 10.5)
    
    # Preprocessing (Log transformation as per your notebook)
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
    return pd.DataFrame(data, index=[0]), [fixed_acidity, volatile_acidity, citric_acid, residual_sugar, chlorides, free_sulfur_dioxide, total_sulfur_dioxide, density, ph, sulphates, alcohol]

input_df, raw_values = user_input_features()

# --- MAIN PANEL ---
st.title("🍷 Premium Wine Quality Predictor")
st.write("Adjust the chemical properties in the sidebar to analyze the wine.")

col1, col2 = st.columns([1, 1])

# Column 1: Prediction & Gauge
with col1:
    st.subheader("Prediction Result")
    prediction = model.predict(input_df)
    probability = model.predict_proba(input_df)[0][1]

    # Probability Gauge
    fig_gauge = go.Figure(go.Indicator(
        mode = "gauge+number",
        value = probability * 100,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': "Quality Confidence %", 'font': {'size': 24}},
        gauge = {
            'axis': {'range': [None, 100], 'tickwidth': 1, 'tickcolor': "darkblue"},
            'bar': {'color': "#722F37"},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "gray",
            'steps': [
                {'range': [0, 50], 'color': '#ffcccc'},
                {'range': [50, 80], 'color': '#fff2cc'},
                {'range': [80, 100], 'color': '#d9ead3'}],
        }
    ))
    fig_gauge.update_layout(height=350, margin=dict(l=20, r=20, t=50, b=20))
    st.plotly_chart(fig_gauge, use_container_width=True)

    if prediction[0] == 1:
        st.success("✨ **RESULT: This is a High Quality Wine!**")
    else:
        st.error("🧐 **RESULT: This is a Standard Quality Wine.**")

# Column 2: Comparison Radar Chart
with col2:
    st.subheader("Chemical Profile Comparison")
    
    # Normalize values for a better radar chart view (Simplified)
    categories = ['Fixed Acidity', 'Volatile Acid', 'Citric Acid', 'Residual Sugar', 'Chlorides', 'Free SO2', 'Total SO2', 'Density', 'pH', 'Sulphates', 'Alcohol']
    
    # Calculate Mean of "Good" Wines from dataset for comparison
    good_wines_mean = raw_df[raw_df['quality'] > 6].mean().values[:11]
    
    fig_radar = go.Figure()

    fig_radar.add_trace(go.Scatterpolar(
        r=raw_values,
        theta=categories,
        fill='toself',
        name='Your Wine',
        line_color='#722F37'
    ))
    fig_radar.add_trace(go.Scatterpolar(
        r=good_wines_mean,
        theta=categories,
        fill='toself',
        name='Avg Good Wine',
        line_color='#999999'
    ))

    fig_radar.update_layout(
      polar=dict(radialaxis=dict(visible=False)),
      showlegend=True,
      height=400,
      margin=dict(l=40, r=40, t=40, b=40)
    )
    st.plotly_chart(fig_radar, use_container_width=True)

# Bottom section: Data Summary
with st.expander("See input data summary"):
    st.table(input_df)
