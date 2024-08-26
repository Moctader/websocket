import pandas as pd
import numpy as np
import joblib
import vectorbt as vbt
from datetime import datetime, timedelta
import psycopg2
from sqlalchemy import create_engine
import matplotlib.pyplot as plt

# Database connection parameters
conn_params = {
    'dbname': 'your_database',
    'user': 'your_user',
    'password': 'your_password',
    'host': 'localhost',
    'port': '5432'
}

def read_data():
    """Read data from PostgreSQL using SQLAlchemy engine."""
    conn_str = f"postgresql://{conn_params['user']}:{conn_params['password']}@{conn_params['host']}:{conn_params['port']}/{conn_params['dbname']}"
    engine = create_engine(conn_str)
    query = "SELECT * FROM intraday_data"
    df = pd.read_sql_query(query, engine)
    return df

def calculate_returns(df, risk_free_rate=0.01):
    """Calculate daily and cumulative returns, and Sharpe Ratio."""
    # Ensure 'datetime' is a datetime object and set as index
    df['datetime'] = pd.to_datetime(df['datetime'])
    df.set_index('datetime', inplace=True)
    
    # Calculate daily returns
    df['daily_return'] = df['close'].pct_change()

    # Drop NaN values that appear due to pct_change
    df = df.dropna(subset=['daily_return'])

    # Calculate average daily return
    average_daily_return = df['daily_return'].mean()

    # Calculate standard deviation of daily returns
    std_dev_daily_return = df['daily_return'].std()

    # Number of trading days in a year
    t = 252

    # Calculate annualized return
    annualized_return = (1 + average_daily_return) ** t - 1

    # Calculate Sharpe Ratio
    sharpe_ratio = (average_daily_return - risk_free_rate / t) / std_dev_daily_return

    # Print the results
    print("Average Daily Return:", average_daily_return)
    print("Annualized Return:", annualized_return)
    print("Sharpe Ratio:", sharpe_ratio)

    # Calculate cumulative returns
    df['cumulative_return'] = (1 + df['daily_return']).cumprod() - 1
    
    return df

def plot_returns(df):
    """Plot daily and cumulative returns in a single figure with two subplots."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 12))

    # Plot daily returns
    ax1.plot(df.index, df['daily_return'], label='Daily Returns')
    ax1.set_xlabel('Date')
    ax1.set_ylabel('Daily Return')
    ax1.set_title('Daily Returns Over Time')
    ax1.legend()

    # Plot cumulative returns
    ax2.plot(df.index, df['cumulative_return'], label='Cumulative Returns', color='orange')
    ax2.set_xlabel('Date')
    ax2.set_ylabel('Cumulative Return')
    ax2.set_title('Cumulative Returns Over Time')
    ax2.legend()

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # Read the data from the database
    df = read_data()
    
    # Calculate returns
    df = calculate_returns(df)
    
    # Plot returns
    plot_returns(df)