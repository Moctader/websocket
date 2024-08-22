import yfinance as yf
import time

# Create a ticker object for a specific stock
ticker = yf.Ticker("AAPL")

while True:
    # Fetch intraday data, e.g., 1-minute interval for the last 7 days
    intraday_data = ticker.history(period="5d", interval="1m")

    # Print the intraday data
    print(intraday_data)

    # Wait for a specified interval before fetching the data again
    time.sleep(60)  # Wait for 60 seconds (1 minute)