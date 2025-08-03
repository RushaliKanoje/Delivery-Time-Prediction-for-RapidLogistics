import pandas as pd
import joblib
import os
from sklearn.preprocessing import StandardScaler

# Load the dataset
data_path = r"D:\DevTools\project\DeliveryTimePrediction\data\Delivery Time Prediction for RapidLogistics.csv"
df = pd.read_csv(data_path)

# Select numerical features to scale
features_to_scale = [
    'package_weight_kg',
    'package_size_cu_m',
    'historical_avg_delivery_time_hrs',
    'traffic_delay_hrs',
    'weather_delay_hrs'
]

# Fit the scaler
scaler = StandardScaler()
scaler.fit(df[features_to_scale])

# Ensure the output directory exists
os.makedirs('model', exist_ok=True)

# Save the scaler to a file
joblib.dump(scaler, 'model/scaler.pkl')

print("✅ Scaler saved successfully at: model/scaler.pkl")