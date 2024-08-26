import requests
import pandas as pd
from datetime import datetime, timedelta
import psycopg2
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import ks_2samp
import time
import pandas as pd
import numpy as np
import requests
import zipfile
import io
from sklearn import ensemble
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from datetime import datetime, time
from sklearn import datasets, ensemble
from evidently.metric_preset import DataDriftPreset
from evidently import ColumnMapping
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset, TargetDriftPreset, RegressionPreset
import webbrowser

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
    AAPL = pd.read_sql_query(query, conn)
    conn.close()
    return AAPL



if __name__ == "__main__":
    # Read the data from the database
    AAPL = read_data()
    day = np.arange(1, len(AAPL) + 1)
    AAPL['day'] = day
    AAPL = AAPL.drop(['timestamp', 'gmtoffset', 'volume', 'previous_close'], axis=1)
    AAPL =AAPL[['day', 'datetime', 'open', 'high', 'low', 'close']]
    AAPL.set_index('datetime', inplace=True)

    #2. Add data/transform data
    AAPL['9-day'] = AAPL['close'].rolling(9).mean()
    AAPL['21-day'] = AAPL['close'].rolling(21).mean()

    #3. Add "signal" column
    AAPL['signal'] = np.where(AAPL['9-day'] > AAPL['21-day'], 1, 0)
    AAPL['signal'] = np.where(AAPL['9-day'] < AAPL['21-day'], -1, AAPL['signal'])
    AAPL.dropna(inplace=True)

    #4 Calculate Instantaneous returns/system returns
    AAPL['return'] = np.log(AAPL['close']).diff()
    AAPL['system_return'] = AAPL['signal'] * AAPL['return']
    AAPL['entry'] = AAPL.signal.diff()

    #5 Plot trades on time series
    plt.rcParams['figure.figsize'] = 12, 6
    plt.grid(True, alpha = .3)
    plt.plot(AAPL.iloc[-252:]['close'], label = 'AAPL')
    plt.plot(AAPL.iloc[-252:]['9-day'], label = '9-day')
    plt.plot(AAPL.iloc[-252:]['21-day'], label = '21-day')
    plt.plot(AAPL[-252:].loc[AAPL.entry == 2].index, AAPL[-252:]['9-day'][AAPL.entry == 2], '^',
            color = 'g', markersize = 12)
    plt.plot(AAPL[-252:].loc[AAPL.entry == -2].index, AAPL[-252:]['21-day'][AAPL.entry == -2], 'v',
            color = 'r', markersize = 12)
    plt.legend(loc=2)
    plt.show()


    #
    plt.plot(np.exp(AAPL['return']).cumprod(), label='Buy/Hold')
    plt.plot(np.exp(AAPL['system_return']).cumprod(), label='System')
    plt.legend(loc=2)
    plt.grid(True, alpha=.3)
    plt.show()

