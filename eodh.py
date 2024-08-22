from eodhd import APIClient
import pandas as pd

api = APIClient("66aaaa9072a398.90417905")
print(api)


# resp = api.get_eod_historical_stock_market_data(symbol = 'AAPL.MX', period='d', from_date = '2023-01-01', to_date = '2023-01-15', order='a')
# print(resp)