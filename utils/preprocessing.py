import numpy as np

def preprocess_input(df, scaler):
    """
    Preprocess input DataFrame for prediction (8 features).

    Args:
        df (pd.DataFrame): Input DataFrame (8 features)
        scaler: Trained scaler that expects 8 features

    Returns:
        np.array: Scaled and encoded input array
    """
    df = df.copy()

    # Encoding mappings
    mode_map = {'Air': 0, 'Ship': 1, 'Road': 2}
    city_map = {
        'Mumbai': 0, 'Pune': 1, 'Nagpur': 2,
        'Delhi': 3, 'Bangalore': 4, 'Kolkata': 5
    }

    # Map categorical to numeric
    df['transport_mode'] = df['transport_mode'].map(mode_map)
    df['origin_city'] = df['origin_city'].map(city_map)
    df['destination_city'] = df['destination_city'].map(city_map)

    # Ensure correct feature order
    features = [
        'origin_city', 'destination_city', 'package_weight_kg',
        'package_size_cu_m', 'transport_mode',
        'historical_avg_delivery_time_hrs',
        'traffic_delay_hrs', 'weather_delay_hrs'
    ]

    # Check for missing columns
    missing_cols = set(features) - set(df.columns)
    if missing_cols:
        raise ValueError(f"Missing columns in input: {missing_cols}")

    df = df[features]

    # Check expected shape
    if df.shape[1] != 8:
        raise ValueError(f"Expected 8 features, got {df.shape[1]}")

    # Confirm scaler shape match
    if hasattr(scaler, 'n_features_in_') and scaler.n_features_in_ != 8:
        raise ValueError(f"Scaler expects {scaler.n_features_in_} features, but got 8")

    # Apply scaling
    X = df.values
    X_scaled = scaler.transform(X)

    return X_scaled