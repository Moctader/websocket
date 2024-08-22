import pandas as pd
import vectorbt as vbt
import plotly.graph_objects as go

# Example data
data = {
    'timestamp': [1633072800, 1633076400, 1633080000, 1633083600],
    'close': [145.0, 145.8, 146.2, 146.5]
}
df = pd.DataFrame(data)
df['datetime'] = pd.to_datetime(df['timestamp'], unit='s')
df.set_index('datetime', inplace=True)

# Calculate returns
returns = df['close'].pct_change().dropna()

# Calculate Sharpe ratio
sharpe_ratio = returns.vbt.returns.sharpe_ratio()
print(f"Sharpe Ratio: {sharpe_ratio}")

# Plot closing prices
fig = go.Figure()
fig.add_trace(go.Scatter(x=df.index, y=df['close'], mode='lines', name='Close Price'))
fig.update_layout(title='Closing Prices', xaxis_title='Date', yaxis_title='Price')
fig.show()

# Plot returns
fig = go.Figure()
fig.add_trace(go.Scatter(x=returns.index, y=returns, mode='lines', name='Returns'))
fig.update_layout(title='Returns', xaxis_title='Date', yaxis_title='Returns')
fig.show()