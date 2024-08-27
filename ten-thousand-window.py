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
    sns.lineplot(x=reference_data.index, y=reference_data['close'], label='Reference', color='blue')

    # Plot current data
    sns.lineplot(x=current_data.index, y=current_data['close'], label='Current', color='red')

    plt.legend()
    plt.title("Datetime vs Close")
    plt.xlabel("Datetime")
    plt.ylabel("Close")
    plt.xticks(rotation=45)
    plt.show()

def bar_plot_distributions(reference_data, current_data):
    """Plot datetime vs close for reference and current data."""
    plt.figure(figsize=(14, 7))

    # Plot reference data
    sns.barplot(x=reference_data.index, y=reference_data['close'], label='Reference', color='blue', alpha=0.6)

    # Plot current data
    sns.barplot(x=current_data.index, y=current_data['close'], label='Current', color='red', alpha=0.6)

    plt.legend()
    plt.title("Datetime vs Close")
    plt.xlabel("Datetime")
    plt.ylabel("Close")
    plt.xticks(rotation=45)
    plt.show()

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

def process_data_in_chunks(df):
    """Process the data in chunks of 10,000 records."""
    chunk_size = 10000
    num_chunks = len(df) // chunk_size

    for i in range(num_chunks - 1):
        start_idx_ref = i * chunk_size
        end_idx_ref = start_idx_ref + chunk_size
        start_idx_cur = end_idx_ref
        end_idx_cur = start_idx_cur + chunk_size

        # Reference data
        reference_data = df.iloc[start_idx_ref:end_idx_ref]
        print(f"Processing reference data chunk {i + 1}")
        print(reference_data.head())

        # Current data
        current_data = df.iloc[start_idx_cur:end_idx_cur]
        print(f"Processing current data chunk {i + 2}")
        print(current_data.head())
        #plot_distributions(reference_data, current_data)
        #bar_plot_distributions(reference_data, current_data)
        
        # Perform drift detection and model performance evaluation
        target = 'close'
        prediction = 'prediction'
        numerical_features = ['open', 'high', 'low', 'volume', 'previous_close']
        categorical_features = []

        regressor = ensemble.RandomForestRegressor(random_state=0, n_estimators=50)
        regressor.fit(reference_data[numerical_features + categorical_features], reference_data[target])
        ref_prediction = regressor.predict(reference_data[numerical_features + categorical_features])
        current_prediction = regressor.predict(current_data[numerical_features + categorical_features])
        
        reference_data['prediction'] = ref_prediction
        current_data['prediction'] = current_prediction

        column_mapping = ColumnMapping()
        column_mapping.target = target
        column_mapping.prediction = prediction
        column_mapping.numerical_features = numerical_features
        column_mapping.categorical_features = categorical_features

        # Generate and detect dataset drift report
        data_drift_report = get_dataset_drift_report(reference_data, current_data, column_mapping)
        drift_detected = detect_dataset_drift(data_drift_report)
    
        if drift_detected:
            print("Dataset drift detected.")
        else:
            print("No dataset drift detected.")

        # Generate and save the dataset drift report
        data_drift_report.save_html(f"data_drift_report_chunk_{i + 1}_to_{i + 2}.html")

        # Generate and save the model performance report
        model_performance_report = get_model_performance_report(reference_data, current_data, column_mapping)
        model_performance_report.save_html(f"model_performance_report_chunk_{i + 1}_to_{i + 2}.html")

        # Open the reports in the default web browser
        webbrowser.open(f"data_drift_report_chunk_{i + 1}_to_{i + 2}.html")
        webbrowser.open(f"model_performance_report_chunk_{i + 1}_to_{i + 2}.html")

def main():
    # Read the data from the database
    df = read_data()
    
    # Ensure 'datetime' is a datetime object
    df['datetime'] = pd.to_datetime(df['datetime'])
    df.set_index('datetime', inplace=True)
    
    # Process the data in chunks
    process_data_in_chunks(df)

if __name__ == "__main__":
    main()