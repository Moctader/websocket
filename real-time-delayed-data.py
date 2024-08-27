import requests
from datetime import datetime
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

# Function to create the new table if it doesn't exist
def create_new_table():
    conn = psycopg2.connect(**conn_params)
    cur = conn.cursor()
    create_table_query = """
    CREATE TABLE IF NOT EXISTS new_stock_data (
        datetime TIMESTAMP PRIMARY KEY,
        code TEXT,
        open FLOAT,
        high FLOAT,
        low FLOAT,
        close FLOAT,
        volume INTEGER,
        previous_close FLOAT,
        change FLOAT,
        change_p FLOAT
    );
    """
    cur.execute(create_table_query)
    conn.commit()
    cur.close()
    conn.close()

# Create the new table
create_new_table()

# Establish a connection to the database
connection = psycopg2.connect(**conn_params)
cursor = connection.cursor()

# API URL for real-time stock data
symbol = 'AAPL.US'
api_token = '658e841fa1a6f9.30546411'
url = f'https://eodhd.com/api/real-time/{symbol}?api_token={api_token}&fmt=json'

# Variable to store the last fetched data
last_data = None

while True:
    try:
        # Fetch the latest data from the API
        response = requests.get(url)
        response.raise_for_status()  # Check if the request was successful
        data = response.json()

        # Add a current timestamp to the data
        data['timestamp'] = int(time.time())
        data['datetime'] = datetime.fromtimestamp(data['timestamp']).strftime('%Y-%m-%d %H:%M:%S')
        print(data)

        # Check if the data has changed
        if last_data is None or (
            data['high'] != last_data['high'] or
            data['low'] != last_data['low'] or
            data['close'] != last_data['close']
        ):
            # Insert the new data
            insert_query = """
            INSERT INTO new_stock_data (datetime, code, open, high, low, close, volume, previous_close, change, change_p)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            """
            cursor.execute(insert_query, (
                data['datetime'],
                data['code'],
                data['open'],
                data['high'],
                data['low'],
                data['close'],
                data['volume'],
                data['previousClose'],
                data['change'],
                data['change_p']
            ))

            # Commit the transaction
            connection.commit()
            print(f"Inserted new data for {data['code']} at {data['datetime']}")

            # Update the last_data variable
            last_data = data
        else:
            print("Data has not changed, skipping insertion.")

    except requests.exceptions.RequestException as e:
        print(f"Request failed: {e}")
    except ValueError as e:
        print(f"JSON decode failed: {e}")
    except psycopg2.IntegrityError as e:
        print(f"Database error: {e}")
        connection.rollback()  # Rollback in case of error

    # Wait for 1 minute before making the next API call
    time.sleep(60)

# Close the cursor and connection when done
cursor.close()
connection.close()