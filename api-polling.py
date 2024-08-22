import time
import requests

# Your API token
api_token = '66aaaa9072a398.90417905'
# The stock ticker you want to poll
ticker = 'AAPL.US'
# The API endpoint
url = f'https://eodhd.com/api/real-time/{ticker}?api_token={api_token}&fmt=json'

# Polling interval in seconds (e.g., every 60 seconds)
polling_interval = 2

def get_stock_data():
    try:
        response = requests.get(url)
        if response.status_code == 200:
            data = response.json()
            # Handle the data (e.g., print it or process it)
            print(data)
        else:
            print(f"Failed to retrieve data. Status code: {response.status_code}")
    except Exception as e:
        print(f"An error occurred: {e}")

def start_polling():
    while True:
        get_stock_data()
        time.sleep(polling_interval)

if __name__ == "__main__":
    start_polling()
