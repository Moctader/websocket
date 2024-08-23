import requests
import pandas as pd
from datetime import datetime, timedelta
import time
import psycopg2

# Database connection parameters
conn_params = {
    'dbname': 'your_database',
    'user': 'your_user',
    'password': 'your_password',
    'host': 'localhost',
    'port': '5432'
}

# Function to create the table if it doesn't exist
def create_table():
    conn = psycopg2.connect(**conn_params)
    cur = conn.cursor()
    create_table_query = """
    CREATE TABLE IF NOT EXISTS intraday_data (
        timestamp BIGINT PRIMARY KEY,
        gmtoffset INT,
        datetime TIMESTAMP,
        open FLOAT,
        high FLOAT,
        low FLOAT,
        close FLOAT,
        volume INT,
        previous_close FLOAT
    );
    """
    cur.execute(create_table_query)
    conn.commit()
    cur.close()
    conn.close()

# Function to insert data into the table
def insert_data(df):
    conn = psycopg2.connect(**conn_params)
    cur = conn.cursor()
   
    # Convert datetime column to proper timestamp format
    df['datetime'] = pd.to_datetime(df['datetime'])

    # Convert datetime to string format
    df['datetime'] = df['datetime'].astype(str)
    
    data_tuples = [tuple(x) for x in df.to_numpy()]
    
    # Print first few data tuples for debugging
    
    insert_query = """
    INSERT INTO intraday_data (timestamp, gmtoffset, datetime, open, high, low, close, volume, previous_close)
    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
    ON CONFLICT (timestamp) DO NOTHING;
    """
    cur.executemany(insert_query, data_tuples)
    conn.commit()
    cur.close()
    conn.close()

# Convert date to Unix timestamp
def to_unix_timestamp(date_str):
    dt = datetime.strptime(date_str, "%Y-%m-%d")
    return int(dt.timestamp())

# Create the table
create_table()

while True:
    # Define date range
    today = datetime.today().strftime('%Y-%m-%d')
    one_year_ago = (datetime.today() - timedelta(days=365)).strftime('%Y-%m-%d')
    start_timestamp = to_unix_timestamp(one_year_ago)
    end_timestamp = to_unix_timestamp(today)

    try:
        # Fetch intraday historical data
        url = f"https://eodhd.com/api/intraday/AAPL.US?interval=1m&api_token=demo&fmt=json"
        response = requests.get(url)
        response.raise_for_status()  # Raise an error for bad status codes
        data = response.json()
        
        # Convert response to pandas DataFrame
        df = pd.DataFrame(data)
        
        # Check if DataFrame is empty
        if df.empty:
            print("No data returned. Please check your API token and subscription level.")
        else:
            # Convert timestamp to datetime
            df['datetime'] = pd.to_datetime(df['timestamp'], unit='s')
            # Add previous close column
            df['previous_close'] = df['close'].shift(1)
               
            # Remove the first row with NaN in previous_close
            df = df.dropna(subset=['previous_close'])
            
            # Check for any NaN or missing values
            if df.isna().any().any():
                print("There are missing values in the DataFrame.")
            else:
                print("No missing values in the DataFrame.")
                # Insert data into the database
                insert_data(df)
    except requests.exceptions.RequestException as e:
        print(f"An error occurred: {e}")

    # Sleep for a specified interval before fetching the data again
    time.sleep(3600)  # Sleep for 1 hour