import os
import pandas as pd
import joblib
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score

# === Step 1: Load Data ===
df = pd.read_csv(r"D:\DevTools\project\DeliveryTimePrediction\data\Delivery Time Prediction for RapidLogistics.csv")

# === Step 2: Clean Data ===
df.dropna(inplace=True)

# === Step 3: Feature Selection ===
features = ['package_weight_kg', 'package_size_cu_m',
            'historical_avg_delivery_time_hrs',
            'traffic_delay_hrs', 'weather_delay_hrs']
target = 'estimated_delivery_time_hrs'

X = df[features]
y = df[target]

# === Step 4: Train-Test Split ===
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# === Step 5: Train Model ===
model = LinearRegression()
model.fit(X_train, y_train)

# === Step 6: Evaluate Model ===
y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f"Model Evaluation:\n - MAE: {mae:.2f}\n - R2 Score: {r2:.2f}")

# === Step 7: Save Model ===
os.makedirs('model', exist_ok=True)
joblib.dump(model, 'model/delivery_time_model.pkl')
print("\n✅ Model trained and saved to app/model/delivery_time_model.pkl")
