import requests
import pandas as pd
from datetime import datetime, timedelta
import time

# Convert date to Unix timestamp
def to_unix_timestamp(date_str):
    dt = datetime.strptime(date_str, "%Y-%m-%d")
    return int(dt.timestamp())

while True:
    # Define date range
    today = datetime.today().strftime('%Y-%m-%d')
    one_year_ago = (datetime.today() - timedelta(days=365)).strftime('%Y-%m-%d')
    start_timestamp = to_unix_timestamp(one_year_ago)
    end_timestamp = to_unix_timestamp(today)

    # Construct the URL
    api_token = "66aaaa9072a398.90417905"
    symbol = "AAPL.US"
    interval = "1m"
    #url = f"https://eodhd.com/api/intraday/{symbol}?interval={interval}&api_token={api_token}&fmt=json"
    url = f"https://eodhd.com/api/intraday/AAPL.US?interval=1m&api_token=demo&fmt=json"


    try:
        # Fetch intraday historical data
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
            
            # Calculate drift for each interval
            df['drift'] = df['close'] - df['open']
            
            # Extract the last day's data
            last_day = df[df['datetime'].dt.date == df['datetime'].dt.date.max()]
            
            # Calculate the drift for the last day
            last_day_drift = last_day['drift'].sum()
            
            # Calculate the drift for the previous data
            previous_data = df[df['datetime'].dt.date < df['datetime'].dt.date.max()]
            previous_data_drift = previous_data['drift'].sum()
            
            # Print the results
            print(f"Last day's drift: {last_day_drift}")
            print(f"Previous data's drift: {previous_data_drift}")
            
            # Print the DataFrame with drift
            print(df)
    except requests.exceptions.RequestException as e:
        print(f"An error occurred: {e}")

    # Sleep for a specified interval before fetching the data again
    time.sleep(20)  # Sleep for 1 hour