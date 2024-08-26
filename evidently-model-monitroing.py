import argparse
import requests
import pandas as pd
from datetime import datetime, timedelta
import psycopg2
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import ks_2samp
import time
import numpy as np
import zipfile
import io
from sklearn import ensemble
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from evidently.metric_preset import DataDriftPreset, TargetDriftPreset, RegressionPreset
from evidently import ColumnMapping
from evidently.report import Report
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

def get_dataset_drift_report(reference: pd.DataFrame, current: pd.DataFrame, column_mapping: ColumnMapping):
    """Returns a data drift report."""
    data_drift_report = Report(metrics=[DataDriftPreset()])
    data_drift_report.run(reference_data=reference, current_data=current, column_mapping=column_mapping)
    return data_drift_report

def get_model_performance_report(reference: pd.DataFrame, current: pd.DataFrame, column_mapping: ColumnMapping):
    """Returns a model performance report."""
    model_performance_report = Report(metrics=[RegressionPreset()])
    model_performance_report.run(reference_data=reference, current_data=current, column_mapping=column_mapping)
 
    return model_performance_report

def detect_dataset_drift(report: Report):
    """Detect dataset drift from the report."""
    print(report.as_dict())
    print(report.as_dict()["metrics"][0]["result"]["share_of_drifted_columns"])
    return report.as_dict()["metrics"][0]["result"]["dataset_drift"]

def main():
    # Read the data from the database
    df = read_data()
    
    # Ensure 'datetime' is a datetime object
    df['datetime'] = pd.to_datetime(df['datetime'])
    df.set_index('datetime', inplace=True)
    
    # Get the latest date in the DataFrame
    latest_date = df.index.max()
    
    # Filter the DataFrame to get the latest day's data
    current = df.loc[latest_date.strftime('%Y-%m-%d')]
    
    # Define the reference date range (e.g., last 365 days)
    reference_start_date = latest_date - timedelta(days=365)
    reference_end_date = latest_date - timedelta(days=1)
    
    # Filter the DataFrame to get the reference data
    reference = df.loc[reference_start_date:reference_end_date]

    target = 'close'
    prediction = 'prediction'
    numerical_features = ['open', 'high', 'low', 'volume', 'previous_close']
    categorical_features = []

    regressor = ensemble.RandomForestRegressor(random_state=0, n_estimators=50)
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

    # Generate and detect dataset drift report
    data_drift_report = get_dataset_drift_report(reference, current, column_mapping)
    drift_detected = detect_dataset_drift(data_drift_report)
   
    if drift_detected:
        print("Dataset drift detected.")
    else:
        print("No dataset drift detected.")

    # Generate and save the dataset drift report
    data_drift_report.save_html("data_drift_report.html")

    # Generate and save the model performance report
    model_performance_report = get_model_performance_report(reference, current, column_mapping)
    model_performance_report.save_html("model_performance_report.html")

    # Open the reports in the default web browser
    webbrowser.open("data_drift_report.html")
    webbrowser.open("model_performance_report.html")

if __name__ == "__main__":
    main()