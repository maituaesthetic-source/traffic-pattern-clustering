
import joblib
import pandas as pd
from pathlib import Path


def load_model():
    model = joblib.load("model/traffic_model.pkl")
    scaler = joblib.load("model/scaler.pkl")

    print("Model and scaler loaded successfully!")

    return model, scaler


def get_new_data():
    print("\nEnter new traffic data:")

    data = {
        "traffic_volume": float(input("Traffic volume: ")),
        "temp": float(input("Temperature: ")),
        "rain_1h": float(input("Rainfall: ")),
        "clouds_all": float(input("Cloud coverage: ")),
        "hour": int(input("Hour (0-23): ")),
        "day": int(input("Day (0=Monday, 6=Sunday): "))
    }

    return pd.DataFrame([data])


def predict_cluster(model, scaler, new_data):
    scaled_data = scaler.transform(new_data)

    prediction = model.predict(scaled_data)

    print("\nPredicted cluster:", prediction[0])

    return prediction


def main():
    model, scaler = load_model()

    new_data = get_new_data()

    predict_cluster(model, scaler, new_data)


if __name__ == "__main__":
    main()
