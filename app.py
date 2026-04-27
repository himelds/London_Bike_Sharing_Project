"""
Streamlit Web Application (Premium UI).
Provides an interactive, visually stunning UI for users to predict London Bike-Sharing demand.
Utilizes Plotly for gauge charts and custom CSS for a modern, glassmorphism-inspired aesthetic.
"""

import streamlit as st
import pandas as pd
import joblib
import yaml
import os
import datetime
import plotly.graph_objects as go

# Configure the Streamlit page layout and title (Must be the first Streamlit command)
st.set_page_config(page_title="London Bike Demand Predictor", page_icon="🚴‍♂️", layout="wide")

# Custom CSS for Premium Look
st.markdown("""
    <style>
    /* Typography */
    .stApp {
        font-family: 'Inter', sans-serif;
    }
    
    /* Input Container styling for glassmorphism effect */
    .stNumberInput, .stSelectbox, .stDateInput, .stTimeInput {
        background: rgba(128, 128, 128, 0.05);
        border-radius: 10px;
        padding: 5px;
    }
    
    /* Button Styling */
    .stButton>button {
        background: linear-gradient(90deg, #00C9FF 0%, #92FE9D 100%);
        color: #121212 !important;
        font-weight: 700;
        border: none;
        border-radius: 8px;
        padding: 10px 24px;
        transition: all 0.3s ease;
    }
    
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 20px rgba(0, 201, 255, 0.4);
        color: #000000 !important;
    }
    
    /* Subtle neon glow for the main title */
    .premium-title {
        text-align: center;
        background: -webkit-linear-gradient(45deg, #00C9FF, #38A169);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 3rem !important;
        font-weight: 900;
        margin-bottom: 0;
    }
    
    .subtitle {
        text-align: center;
        font-size: 1.2rem;
        margin-bottom: 40px;
        opacity: 0.7;
    }
    </style>
""", unsafe_allow_html=True)


from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer

# Load configuration and model assets
@st.cache_resource
def load_assets():
    try:
        with open("config/config.yaml", 'r') as file:
            config = yaml.safe_load(file)

        model_path = os.path.join(os.getcwd(), config['model']['saved_model_path'])
        preprocessor_path = os.path.join(os.getcwd(), config['model']['preprocessor_path'])

        # If model doesn't exist, train it on the fly!
        if not os.path.exists(model_path) or not os.path.exists(preprocessor_path):
            with st.spinner("Model not found. Training a new Random Forest model automatically (this will take ~10 seconds)..."):
                obj = DataIngestion()
                train_data, test_data = obj.initiate_data_ingestion()
                
                data_transformation = DataTransformation()
                X_train, y_train, X_test, y_test, _ = data_transformation.initiate_data_transformation(train_data, test_data)
                
                model_trainer = ModelTrainer()
                model_trainer.initiate_model_trainer(X_train, y_train, X_test, y_test)

        # Load model and feature columns
        model = joblib.load(model_path)
        model_columns = joblib.load(preprocessor_path)
        return model, model_columns, True
    except Exception as e:
        st.error(f"Error loading or training model: {e}")
        return None, None, False

model, model_columns, model_loaded = load_assets()

# Hero Section
st.markdown("<h1 class='premium-title'>🚴‍♂️ London Bike Demand AI</h1>", unsafe_allow_html=True)
st.markdown("<p class='subtitle'>Predicting city-wide bike rentals in real-time using environmental intelligence.</p>", unsafe_allow_html=True)

if not model_loaded:
    st.error("Error loading model. Please ensure you have trained the model first by running `python train.py`.")
    st.stop()

# Layout: Split into Input Section (Left) and Output Section (Right)
col_input, col_output = st.columns([1.2, 1], gap="large")

with col_input:
    st.markdown("### 🎛️ Scenario Parameters")
    
    # Nested columns for dense but clean input layout
    in_col1, in_col2 = st.columns(2)
    
    with in_col1:
        date_input = st.date_input("Date", datetime.date.today())
        t1 = st.number_input("Temperature (°C)", min_value=-10.0, max_value=40.0, value=15.0, step=0.5)
        wind_speed = st.number_input("Wind Speed (km/h)", min_value=0.0, max_value=100.0, value=10.0, step=1.0)
        is_weekend = st.selectbox("Is Weekend?", options=[0, 1], format_func=lambda x: "No" if x==0 else "Yes")
        
    with in_col2:
        time_input = st.time_input("Time", datetime.time(12, 0))
        hum = st.number_input("Humidity (%)", min_value=0.0, max_value=100.0, value=50.0, step=1.0)
        weather_code = st.selectbox("Weather Code", options=[1, 2, 3, 4, 7, 10, 26], format_func=lambda x: {
            1: 'Clear', 2: 'Scattered clouds', 3: 'Broken clouds', 4: 'Cloudy', 
            7: 'Rain', 10: 'Rain with thunderstorm', 26: 'Snowfall'}[x])
        is_holiday = st.selectbox("Is Holiday?", options=[0, 1], format_func=lambda x: "No" if x==0 else "Yes")
        
    season = st.selectbox("Season", options=[0, 1, 2, 3], format_func=lambda x: {
        0: 'Spring', 1: 'Summer', 2: 'Fall', 3: 'Winter'}[x])

    st.markdown("<br>", unsafe_allow_html=True)
    predict_button = st.button("🚀 Predict Demand", use_container_width=True)

with col_output:
    st.markdown("### 📈 Live Prediction")
    
    if predict_button:
        # Create input DataFrame from user selection
        input_data = {
            't1': [t1],
            'hum': [hum],
            'wind_speed': [wind_speed],
            'weather_code': [weather_code],
            'is_holiday': [is_holiday],
            'is_weekend': [is_weekend],
            'season': [season],
            'hour': [time_input.hour],
            'day': [date_input.day],
            'month': [date_input.month],
            'year': [date_input.year]
        }
        
        input_df = pd.DataFrame(input_data)
        
        # Apply one-hot encoding on categorical features
        input_df = pd.get_dummies(input_df, columns=['weather_code', 'is_holiday', 'is_weekend', 'season'])
        
        # Align with the model's expected columns
        input_df = input_df.reindex(columns=model_columns, fill_value=0)
        
        # Make Prediction
        prediction = int(model.predict(input_df)[0])
        
        # Render Plotly Gauge Chart
        fig = go.Figure(go.Indicator(
            mode = "gauge+number",
            value = prediction,
            domain = {'x': [0, 1], 'y': [0, 1]},
            title = {'text': "Predicted Bikes Rented", 'font': {'size': 24}},
            number = {'font': {'size': 60, 'color': '#00C9FF'}},
            gauge = {
                'axis': {'range': [None, 5000], 'tickwidth': 1},
                'bar': {'color': "#38A169"},
                'bgcolor': "rgba(0,0,0,0)",
                'borderwidth': 2,
                'bordercolor': "gray",
                'steps': [
                    {'range': [0, 1000], 'color': "rgba(0, 201, 255, 0.1)"},
                    {'range': [1000, 3000], 'color': "rgba(0, 201, 255, 0.3)"},
                    {'range': [3000, 5000], 'color': "rgba(0, 201, 255, 0.6)"}],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 4500}
            }
        ))
        
        fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", height=350)
        st.plotly_chart(fig, use_container_width=True)
        
        # Show contextual metrics
        st.markdown("#### Current Scenario")
        m1, m2, m3 = st.columns(3)
        m1.metric("Temperature", f"{t1} °C")
        m2.metric("Wind Speed", f"{wind_speed} km/h")
        m3.metric("Humidity", f"{hum} %")
        
    else:
        # Placeholder before prediction
        st.info("👈 Adjust the parameters on the left and click **Predict Demand** to see the AI forecast.")
        st.image("https://images.unsplash.com/photo-1507035895480-2b3156c31fc8?ixlib=rb-4.0.3&auto=format&fit=crop&w=800&q=80", use_container_width=True, caption="London City Bikes")

st.markdown("---")
with st.expander("ℹ️ About the Model Architecture"):
    st.write("""
    This prediction engine uses a **Random Forest Regressor** trained on historical London Bike-Sharing data. 
    The model analyzes temporal factors (time of day, day of week, seasonality) and environmental conditions 
    (temperature, humidity, wind speed, weather type) to identify non-linear patterns and accurately forecast city-wide demand.
    """)
