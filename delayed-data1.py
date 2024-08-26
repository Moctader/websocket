import pandas as pd

timestamp = 1724678664603
datetime_value = pd.to_datetime(timestamp, unit='ms')
print(datetime_value)