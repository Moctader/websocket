import time
import requests
import psycopg2
from psycopg2.extras import execute_values
from datetime import datetime

# Your API token
api_token = '66aaaa9072a398.90417905'
# The stock ticker you want to poll
ticker = 'AAPL.US'
# The API endpoint for real-time data
realtime_url = f'https://eodhd.com/api/real-time/{ticker}?api_token={api_token}&fmt=json'
# The API endpoint for historical data
historical_url = f'https://eodhd.com/api/eod/{ticker}?api_token={api_token}&fmt=json'

# Polling interval in seconds (e.g., every 60 seconds)
polling_interval = 60

# PostgreSQL connection details
conn = psycopg2.connect(
    dbname='your_database',
    user='your_user',
    password='your_password',
    host='localhost',
    port='5432'
)

def create_table():
    with conn.cursor() as cursor:
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS stock_data (
            id SERIAL PRIMARY KEY,
            ticker VARCHAR(10),
            timestamp TIMESTAMP,
            open NUMERIC,
            high NUMERIC,
            low NUMERIC,
            close NUMERIC,
            volume NUMERIC,
            previous_close NUMERIC,
            change NUMERIC,
            change_p NUMERIC
        );
        """)
        conn.commit()

def get_historical_data():
    try:
        response = requests.get(historical_url)
        if response.status_code == 200:
            data = response.json()
            # Handle the data (e.g., print it or process it)
            print(f"Received historical data: {data}")
            store_data(data, historical=True)
        else:
            print(f"Failed to retrieve historical data. Status code: {response.status_code}")
    except Exception as e:
        print(f"An error occurred while fetching historical data: {e}")

def get_stock_data():
    try:
        response = requests.get(realtime_url)
        if response.status_code == 200:
            data = response.json()
            # Handle the data (e.g., print it or process it)
            print(f"Received real-time data: {data}")
            store_data(data)
        else:
            print(f"Failed to retrieve real-time data. Status code: {response.status_code}")
    except Exception as e:
        print(f"An error occurred while fetching real-time data: {e}")

def store_data(data, historical=False):
    with conn.cursor() as cursor:
        query = """
        INSERT INTO stock_data (ticker, timestamp, open, high, low, close, volume, previous_close, change, change_p)
        VALUES %s
        """
        if historical:
            values = [
                (ticker, datetime.strptime(record['date'], '%Y-%m-%d'), record['open'], record['high'], record['low'], record['close'], record['volume'], None, None, None)
                for record in data
            ]
        else:
            # Convert Unix timestamp to datetime
            timestamp = datetime.fromtimestamp(data['timestamp'])
            values = [
                (data['code'], timestamp, data['open'], data['high'], data['low'], data['close'], data['volume'], data['previousClose'], data['change'], data['change_p'])
            ]
        execute_values(cursor, query, values)
        conn.commit()

def start_polling():
    while True:
        get_stock_data()
        time.sleep(polling_interval)

if __name__ == "__main__":
    create_table()
    get_historical_data()  # Fetch and store historical data once
    start_polling()  # Start polling for new data