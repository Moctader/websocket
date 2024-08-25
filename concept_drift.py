import psycopg2
import pandas as pd
from datetime import datetime, timedelta

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

if __name__ == "__main__":
    # Read the data from the database
    df = read_data()
    print(df.head())
    
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
    
    # Print the current and reference data
    print("Current data:")
    print(current.head())
    print("Reference data:")
    print(reference.tail())