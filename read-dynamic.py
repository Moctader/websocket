import requests
import pandas as pd
from datetime import datetime, timedelta
import psycopg2
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import ks_2samp
import time
import pandas as pd
import numpy as np
import requests
import zipfile
import io
from sklearn import ensemble
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from datetime import datetime, time
from sklearn import datasets, ensemble
from evidently.metric_preset import DataDriftPreset
from evidently import ColumnMapping
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset, TargetDriftPreset, RegressionPreset
import webbrowser

# Database connection parameters
conn_params = {
    'dbname': 'your_database',
    'user': 'your_user',
    'password': 'your_password',
    'host': 'localhost',
    'port': '5432'
}

def read_data():
    """Read data from PostgreSQL."""
    conn = psycopg2.connect(**conn_params)
    query = "SELECT * FROM intraday_data"
    df = pd.read_sql_query(query, conn)
    conn.close()
    return df

def plot_distributions(reference_data, current_data):
    """Plot datetime vs close for reference and current data."""
    plt.figure(figsize=(14, 7))

    # Plot reference data
    sns.lineplot(x=reference_data['datetime'], y=reference_data['close'], label='Reference', color='blue')

    # Plot current data
    sns.lineplot(x=current_data['datetime'], y=current_data['close'], label='Current', color='red')

    plt.legend()
    plt.title("Datetime vs Close")
    plt.xlabel("Datetime")
    plt.ylabel("Close")
    plt.xticks(rotation=45)
    plt.show()

def ks_test(reference_data, current_data):
    """Apply KS test to each feature to detect drift."""
    drift_results = {}
    for column in reference_data.columns:
        if pd.api.types.is_numeric_dtype(reference_data[column]):
            ks_stat, p_value = ks_2samp(reference_data[column], current_data[column])
            drift_results[column] = {'ks_stat': ks_stat, 'p_value': p_value}
    return drift_results

def get_dataset_drift_report(
    reference: pd.DataFrame, current: pd.DataFrame, column_mapping: ColumnMapping
):
    """
    Returns True if Data Drift is detected, else returns False.
    If get_ratio is True, returns the share of drifted features.
    """
    data_drift_report = Report(metrics=[DataDriftPreset()])
    data_drift_report.run(
        reference_data=reference, current_data=current, column_mapping=column_mapping
    )
    
    return data_drift_report

def detect_dataset_drift(report: Report):
    print(report.as_dict())
    return report.as_dict()["metrics"][0]["result"]["dataset_drift"]

if __name__ == "__main__":
    # Read the data from the database
    df = read_data()

    # Ensure 'datetime' is a datetime object
    df['datetime'] = pd.to_datetime(df['datetime'])

    # Filter the most recent day's data as current data
    df['datetime'] = pd.to_datetime(df['datetime'])
    df.set_index('datetime', inplace=True)

    # Correctly slice the DataFrame with the appropriate date range
    reference = df.loc['2024-04-25 08:01:00':'2024-08-22 23:00:00']
    current = df.loc['2024-08-23 00:00:00':'2024-08-23 23:00:00']

    target='close'
    prediction='prediction'
    numerical_features = ['open', 'high', 'low', 'volume', 'previous_close']
    categorical_features = []

    regressor = ensemble.RandomForestRegressor(random_state = 0, n_estimators = 50)
    regressor.fit(reference[numerical_features + categorical_features], reference[target])
    ref_prediction = regressor.predict(reference[numerical_features + categorical_features])
    current_prediction = regressor.predict(current[numerical_features + categorical_features])
    
    reference['prediction'] = ref_prediction
    current['prediction'] = current_prediction

    column_mapping = ColumnMapping()
    column_mapping.target = target
    column_mapping.prediction = prediction
    column_mapping.numerical_features = numerical_features
    column_mapping.categorical_features = categorical_features

    report = get_dataset_drift_report(reference, current, column_mapping)
    drift_detected = detect_dataset_drift(report)
   
    if drift_detected:
        print("Detect dataset drift between")
    else:
        print("Detect no dataset drift between")

    # Save the report to an HTML file
    report.save_html("data_drift_report.html")

    # Open the report in the default web browser
    webbrowser.open("data_drift_report.html")