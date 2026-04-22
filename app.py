import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.graph_objects as go

st.set_page_config(page_title="Wine Quality AI", page_icon="🍷", layout="wide")

@st.cache_resource
def load_assets():
    with open('wine_model.pkl', 'rb') as f:
        model = pickle.load(f)
    df = pd.read_csv('winequality-white.csv', sep=';')
    return model, df

model, raw_df = load_assets()

# --- SIDEBAR: 8 INPUTS ---
st.sidebar.header("🧪 Key Chemical Markers")
def get_user_inputs():
    fa = st.sidebar.slider("Fixed Acidity", 3.8, 14.2, 6.8)
    va = st.sidebar.slider("Volatile Acidity", 0.08, 1.1, 0.27)
    rs = st.sidebar.slider("Residual Sugar", 0.6, 65.8, 6.3)
    fsd = st.sidebar.slider("Free Sulfur Dioxide", 2.0, 289.0, 35.0)
    tsd = st.sidebar.slider("Total Sulfur Dioxide", 9.0, 440.0, 138.0)
    de = st.sidebar.slider("Density", 0.987, 1.038, 0.994, step=0.001)
    ph = st.sidebar.slider("pH Level", 2.7, 3.8, 3.18)
    al = st.sidebar.slider("Alcohol (%)", 8.0, 14.2, 10.5)
    
    # Preprocess (Applying Log1p to the specific 4 we selected)
    processed_data = {
        'fixed_acidity': fa,
        'volatile_acidity': np.log1p(va),
        'residual_sugar': np.log1p(rs),
        'free_sulfur_dioxide': np.log1p(fsd),
        'total_sulfur_dioxide': tsd,
        'density': np.log1p(de),
        'pH': ph,
        'alcohol': al
    }
    return pd.DataFrame(processed_data, index=[0]), [fa, va, rs, fsd, tsd, de, ph, al]

input_df, raw_list = get_user_inputs()

# --- MAIN UI ---
st.title("🍷 White Wine Quality Predictor")
st.info("This model focuses on the 8 most significant chemical properties.")

col1, col2 = st.columns([1, 1])

with col1:
    prob = model.predict_proba(input_df)[0][1]
    pred = model.predict(input_df)[0]
    
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
        st.warning("🧐 **RESULT: Standard Quality Wine**")

with col2:
    st.subheader("Profile vs. Premium Average")
    categories = ['Fixed Acid', 'Volatile Acid', 'Sugar', 'Free SO2', 'Total SO2', 'Density', 'pH', 'Alcohol']
    
    # Extracting means for only these 8 columns for the radar chart
    good_wines = raw_df[raw_df['quality'] > 6]
    # Map the CSV columns to our display names
    avg_vals = [
        good_wines['fixed acidity'].mean(),
        good_wines['volatile acidity'].mean(),
        good_wines['residual sugar'].mean(),
        good_wines['free sulfur dioxide'].mean(),
        good_wines['total sulfur dioxide'].mean(),
        good_wines['density'].mean(),
        good_wines['pH'].mean(),
        good_wines['alcohol'].mean()
    ]
    
    radar = go.Figure()
    radar.add_trace(go.Scatterpolar(r=raw_list, theta=categories, fill='toself', name='Your Wine', line_color='#722F37'))
    radar.add_trace(go.Scatterpolar(r=avg_vals, theta=categories, fill='toself', name='Avg Premium', line_color='#999999'))
    radar.update_layout(polar=dict(radialaxis=dict(visible=False)), height=400)
    st.plotly_chart(radar, use_container_width=True)
