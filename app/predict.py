import os
import sys
import streamlit as st
import pandas as pd
import joblib
import numpy as np

# --- Setup Paths ---
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..'))
sys.path.insert(0, project_root)

# --- Load Model and Scaler ---
model_path = os.path.join(project_root, 'model', 'delivery_time_model.pkl')
scaler_path = os.path.join(project_root, 'model', 'scaler.pkl')

try:
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
except Exception as e:
    st.error(f"❌ Failed to load model files: {e}")
    st.stop()

# --- App UI ---
st.title("📦 Delivery Time Predictor - RapidLogistics")
st.markdown("Predict estimated delivery time using shipment details.")


def get_user_input():
    """Collect user input via sidebar"""
    with st.sidebar:
        st.header("Shipment Details")
        transport_mode = st.selectbox("Transport Mode", ['Air', 'Ship', 'Road'])
        origin_city = st.selectbox("Origin City", ['Mumbai', 'Pune', 'Nagpur', 'Delhi', 'Bangalore', 'Kolkata'])
        destination_city = st.selectbox("Destination City",
                                        ['Mumbai', 'Pune', 'Nagpur', 'Delhi', 'Bangalore', 'Kolkata'])
        package_weight_kg = st.slider("Package Weight (kg)", 0.1, 100.0, 10.0)
        package_size_cu_m = st.slider("Package Size (cu.m)", 0.1, 10.0, 1.0)
        historical_avg = st.slider("Historical Avg Delivery (hrs)", 1, 100, 20)
        traffic_delay = st.slider("Traffic Delay (hrs)", 0.0, 10.0, 1.0)
        weather_delay = st.slider("Weather Delay (hrs)", 0.0, 10.0, 1.0)

    return pd.DataFrame({
        'origin_city': [origin_city],
        'destination_city': [destination_city],
        'transport_mode': [transport_mode],
        'package_weight_kg': [package_weight_kg],
        'package_size_cu_m': [package_size_cu_m],
        'historical_avg_delivery_time_hrs': [historical_avg],
        'traffic_delay_hrs': [traffic_delay],
        'weather_delay_hrs': [weather_delay]
    })


def preprocess_input(df, scaler):
    """
    Preprocess input data to match model requirements
    Returns: numpy array ready for prediction
    """
    # Encoding dictionaries
    MODE_MAP = {'Air': 0, 'Ship': 1, 'Road': 2}
    CITY_MAP = {
        'Mumbai': 0, 'Pune': 1, 'Nagpur': 2,
        'Delhi': 3, 'Bangalore': 4, 'Kolkata': 5
    }

    # Create a copy to avoid modifying original
    processed = df.copy()

    # Encode categorical features
    processed['transport_mode'] = processed['transport_mode'].map(MODE_MAP)
    processed['origin_city'] = processed['origin_city'].map(CITY_MAP)
    processed['destination_city'] = processed['destination_city'].map(CITY_MAP)

    # List of features in EXACT order the model expects
    model_features = [
        'package_weight_kg',
        'package_size_cu_m',
        'historical_avg_delivery_time_hrs',
        'traffic_delay_hrs',
        'weather_delay_hrs'
    ]

    # Scale only the numerical features
    scaled_values = scaler.transform(processed[model_features])

    return scaled_values


# --- Main App Logic ---
input_df = get_user_input()

if st.button("🚀 Predict Delivery Time", type="primary"):
    try:
        # Validate input
        if input_df.isnull().any().any():
            st.error("Please fill all fields correctly")
            st.stop()

        # Preprocess and predict
        X_processed = preprocess_input(input_df, scaler)
        prediction = model.predict(X_processed)

        # Display results
        st.success("### Prediction Results")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Estimated Delivery Time", f"{prediction[0]:.1f} hours")
        with col2:
            st.metric("Confidence", "High" if prediction[0] < 24 else "Medium")

        # Show input summary
        with st.expander("Show Input Summary"):
            st.dataframe(input_df.T.style.background_gradient(cmap='Blues'))

    except Exception as e:
        st.error(f"Prediction failed: {str(e)}")

# --- Footer ---
st.markdown("---")
st.caption("© 2023 RapidLogistics | Delivery Time Prediction System")