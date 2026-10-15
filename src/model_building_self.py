import pandas as pd
import matplotlib.pyplot as plt
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


def load_data(file_path):
    data = pd.read_csv(file_path)
    print("Data load sucessfully!")
    print("Shape: ",data.shape)
    return data

def clean_data(data):
    data = data.copy()

    data["date_time"] = pd.to_datetime(
        data["date_time"], errors="coerce"
    )

    data = data.dropna(subset=["date_time"])
    data = data.drop_duplicates()

    numeric_columns = [
            "traffic_volume", "temp", "rain_1h",
            "snow_1h", "clouds_all"
            ]

    for column in numeric_columns:
        data[column] = pd.to_numeric(
            data[column], errors="coerce"
        )
        data[column] = data[column].fillna(
            data[column].median()
        )

    data["weather_main"] = (
        data["weather_main"].fillna("Unknown")
    )
    print("Data cleaned successfully!")
    return data

def create_features(data):
    data = data.copy()

    data["hour"] = data["date_time"].dt.hour
    data["day"] = data["date_time"].dt.dayofweek

    features = data[
        [
                    "traffic_volume",
                    "temp",
                    "rain_1h",
                    "clouds_all",
                    "hour",
                    "day",
        ]
    ]
    return features

def scale_data(features):
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features)

    print("Features scaled successfuly!")
    joblib.dump(scaler, "model/scaler.pkl")
    return scaled_features


def create_cluster(scaled_features, number_of_clusters=3):
    model = KMeans(
        n_clusters=number_of_clusters,
        random_state=42,
        n_init=10
    )

    clusters = model.fit_predict(scaled_features)
    
    joblib.dump(model, "model/traffic_model.pkl")

    print("Clustering completed!")
    return clusters


def evaluate_cluster(scaled_features, clusters):
    score = silhouette_score(
        scaled_features,clusters
    )

    print("silhouette_score: ", round(score, 3))

def show_results(data):
    summary = data.groupby("cluster")[
        "traffic_volume"
    ].mean()

    print("\nAverage traffic volume in each cluster:")
    print(summary)

    summary.plot(kind="bar")
    plt.title("Average Traffic Volume by Cluster")
    plt.xlabel("Cluster")
    plt.ylabel("Average Traffic Volume")
    plt.tight_layout()
    plt.show()



def main():
    file_path = "data/Metro_Interstate_Traffic_Volume.csv"

    data = load_data(file_path)
    data = clean_data(data)

    features = create_features(data)
    scaled_features = scale_data(features)

    clusters = create_cluster(
        scaled_features,
        number_of_clusters=3
    )
    data["cluster"] = "clusters"
    evaluate_cluster(scaled_features, clusters)
    show_results(data)
    data.to_csv(
        "clustered_traffic_2.csv",
        index=False
    )
    
    print("\nResults saved to clustered_traffic.csv")

if __name__ == "__main__":
    main()