import requests
import pandas as pd
from datetime import datetime, timedelta
import psycopg2
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset
from evidently import ColumnMapping

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

def generate_data_drift_report(reference_data, current_data, column_mapping):
    """Generate a data drift report using Evidently with column mapping."""
    report = Report(metrics=[DataDriftPreset()])
    report.run(reference_data=reference_data, current_data=current_data, column_mapping=column_mapping)
    report.save_html("data_drift_report.html")

if __name__ == "__main__":
    # Read the data from the database
    df = read_data()

    # Ensure 'datetime' is a datetime object
    df['datetime'] = pd.to_datetime(df['datetime'])

    # Filter the most recent day's data as current data
    last_day = df[df['datetime'].dt.date == df['datetime'].dt.date.max()]

    # Filter all data before the most recent day as reference data
    previous_data = df[df['datetime'].dt.date < df['datetime'].dt.date.max()]

    # Define column mapping
    column_mapping = ColumnMapping(
        target=None,  # Set the target column if you have one
        prediction=None,  # Set the prediction column if you have one
        datetime="datetime",  # Specify the datetime column
        numerical_features=['open', 'high', 'low', 'close', 'volume'],  # List of numerical features
        categorical_features=None  # List of categorical features, if any
    )

    # Generate data drift report
    generate_data_drift_report(reference_data=previous_data, current_data=last_day, column_mapping=column_mapping)

    print("Data drift report generated and saved as 'data_drift_report.html'.")
