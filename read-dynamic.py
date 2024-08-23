import requests
import pandas as pd
from datetime import datetime, timedelta
import psycopg2
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import ks_2samp


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


if __name__ == "__main__":
    # Read the data from the database
    df = read_data()

    # Ensure 'datetime' is a datetime object
    df['datetime'] = pd.to_datetime(df['datetime'])

    # Filter the most recent day's data as current data
    last_day = df[df['datetime'].dt.date == df['datetime'].dt.date.max()]
    # print(last_day)
    # print("Last day data filtered.")

    # Filter all data before the most recent day as reference data
    previous_data = df[df['datetime'].dt.date < df['datetime'].dt.date.max()]
    # print(previous_data)
    plot_distributions(previous_data, last_day)
    ks_results = ks_test(previous_data, last_day)
    print("KS Test Results:\n", ks_results)




