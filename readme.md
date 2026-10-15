# Traffic Pattern Clustering Using Machine Learning

## Project Overview

This project uses **unsupervised machine learning** to identify patterns in traffic data using the K-Means clustering algorithm.

The goal is to group traffic observations with similar characteristics based on features such as traffic volume, temperature, rainfall, cloud coverage, hour of the day, and day of the week.

The project also allows users to save the trained model and assign clusters to new traffic observations without retraining the model.

## Objectives

- Load and clean traffic data using Pandas.
- Handle missing values and duplicate records.
- Extract useful time-based features.
- Scale numerical features using StandardScaler.
- Apply K-Means clustering.
- Save the trained model and scaler.
- Predict clusters for new traffic observations.
- Analyze and visualize the discovered traffic patterns.

## Dataset

**Dataset:** Metro Interstate Traffic Volume

The dataset contains hourly traffic observations along with weather and time-related information.

Dataset source: https://archive.ics.uci.edu/dataset/492/metro+interstate+traffic+volume

### Features Used

| Feature | Description |
|---|---|
| `traffic_volume` | Recorded traffic volume |
| `temp` | Temperature |
| `rain_1h` | Rainfall during the previous hour |
| `clouds_all` | Cloud coverage |
| `hour` | Hour of the day, from 0 to 23 |
| `day` | Day of the week, from 0 (Monday) to 6 (Sunday) |

The temperature units and traffic-volume meaning follow the original dataset.

## Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Joblib

## Machine Learning Algorithm

### K-Means Clustering

K-Means is an unsupervised learning algorithm that groups observations into a specified number of clusters.

This project uses three clusters by default:

- Cluster 0
- Cluster 1
- Cluster 2

These are identifiers assigned by the algorithm. Their meaning must be determined by analyzing the characteristics of each cluster.

### Feature Scaling

StandardScaler transforms numerical features to comparable scales. The fitted scaler is saved and reused when predicting clusters for new observations.

## Project Structure

```text
Traffic-clustering/
├── data/
│   └── Metro_Interstate_Traffic_Volume.csv
├── model/
│   ├── traffic_model.pkl
│   └── scaler.pkl
├── src/
│   ├── model_building_self.py
│   └── predict.py
├── clustered_traffic_2.csv
├── requirements.txt
└── README.md
```

The exact file locations may vary depending on your implementation. Generated model files and output CSV files are created when the relevant scripts run.

## Installation

### 1. Clone or download the project

Open the project folder in your terminal.

### 2. Create a virtual environment (recommended)

```bash
python -m venv .venv
```

Activate it on Windows:

```bash
.venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Download the dataset

Download the Metro Interstate Traffic Volume dataset and place its CSV file in the `data/` directory.

## How to Run the Project

### Step 1: Train the model

Run the training script from the project root:

```bash
python src/model_building_self.py
```

The training workflow should:

1. Load the dataset.
2. Clean the data.
3. Create the required features.
4. Scale the features.
5. Train the K-Means model.
6. Save the fitted scaler and trained model.
7. Generate cluster assignments.

### Step 2: Predict a cluster for new data

Run:

```bash
python src/predict.py
```

Enter the required traffic features when prompted.

The program loads the saved model and scaler, transforms the new observation, and returns a predicted cluster.

### Step 3: Analyze the clusters

Use the generated clustered CSV file to calculate average traffic volume and other statistics for each cluster. Visualize the results to understand the differences between groups.

## Example Prediction

Example input:

```text
Traffic volume: 4444
Temperature: 237.20
Rainfall: 0
Cloud coverage: 30
Hour: 15
Day: 6
```

Example output:

```text
Model and scaler loaded successfully!
Predicted cluster: 0
```

This is an illustrative example of the program output, not a guarantee that every training run will produce the same cluster assignment.

## Understanding the Results

Cluster labels do not have predefined meanings.

For example, one cluster might have a higher average traffic volume than another. However, high traffic volume alone does not establish that a road is congested.

Analyze cluster averages, record counts, weather characteristics, and time patterns before drawing conclusions.

## Limitations

- The model groups observations; it does not forecast future traffic volume.
- The cluster labels do not directly measure congestion.
- Results depend on the selected features, preprocessing, and number of clusters.
- New observations should use the same feature definitions, units, order, and fitted scaler as the training data.
- The model should be evaluated to determine whether the discovered clusters are useful.

## Future Improvements

- Compare different values of K using the silhouette score.
- Visualize clusters using PCA.
- Analyze weekday and weekend traffic patterns.
- Compare traffic patterns across weather conditions.
- Build an interactive dashboard using Streamlit.
- Explore a separate supervised-learning project for forecasting traffic volume.

## Author

Created as a practical machine learning project to learn data preprocessing, unsupervised learning, model persistence, prediction, and cluster analysis.

## License

Add a license if you plan to publish and distribute this project on GitHub.
