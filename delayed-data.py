import requests
from datetime import datetime
import time
import pandas as pd
import sqlite3

url = 'https://eodhd.com/api/real-time/AAPL.US?api_token=658e841fa1a6f9.30546411&fmt=json'

# Initialize an empty DataFrame
df = pd.DataFrame()

# Connect to SQLite database (or create it if it doesn't exist)
conn = sqlite3.connect('stock_data.db')
c = conn.cursor()

# Create table if it doesn't exist
c.execute('''
    CREATE TABLE IF NOT EXISTS stock_data (
        datetime TEXT PRIMARY KEY,
        code TEXT,
        timestamp INTEGER,
        gmtoffset INTEGER,
        open REAL,
        high REAL,
        low REAL,
        close REAL,
        volume INTEGER,
        previousClose REAL,
        change REAL,
        change_p REAL
    )
''')
conn.commit()

while True:
    try:
        response = requests.get(url)
        response.raise_for_status()  # Check if the request was successful
        data = response.json()

        # Add a current timestamp to the data
        data['timestamp'] = int(time.time())

        # Convert the timestamp to a human-readable datetime format
        data['datetime'] = datetime.fromtimestamp(data['timestamp']).strftime('%Y-%m-%d %H:%M:%S')

        # Convert the data to a DataFrame and set the datetime as the index
        temp_df = pd.DataFrame([data])
        temp_df['datetime'] = pd.to_datetime(temp_df['datetime'])
        temp_df.set_index('datetime', inplace=True)

        # Insert data into the database
        temp_df.to_sql('stock_data', conn, if_exists='append', index=True)

        # Query the latest data from the database for monitoring
        latest_data = pd.read_sql_query('SELECT * FROM stock_data ORDER BY datetime DESC LIMIT 1', conn)
        print(latest_data)

    except requests.exceptions.RequestException as e:
        print(f"Request failed: {e}")
    except ValueError as e:
        print(f"JSON decode failed: {e}")
    except sqlite3.IntegrityError as e:
        print(f"Database error: {e}")

    # Wait for 1 second before making the next API call
    time.sleep(1)

# Close the database connection when done
conn.close()