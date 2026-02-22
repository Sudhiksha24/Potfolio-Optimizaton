import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize

tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META']
risk_free_rate = 0.02
initial_capital = 100

prices = yf.download(
    tickers,
    start='2018-01-01',
    end='2025-01-01'
)['Close']

prices.dropna(inplace=True)
returns = prices.pct_change().dropna()

num_assets = len(tickers)

def portfolio_return(weights, mean_returns):
    return np.dot(weights, mean_returns) * 252

def portfolio_volatility(weights, cov_matrix):
    return np.sqrt(np.dot(weights.T, np.dot(cov_matrix * 252, weights)))

def negative_sharpe(weights, mean_returns, cov_matrix):
    return -(portfolio_return(weights, mean_returns) - risk_free_rate) / portfolio_volatility(weights, cov_matrix)

constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
bounds = tuple((0, 1) for _ in range(num_assets))
init_guess = num_assets * [1. / num_assets]

window_years = 2
window_days = 252 * window_years

portfolio_values = [initial_capital]
dates = []

for i in range(window_days, len(returns)):
    train_returns = returns.iloc[i-window_days:i]
    test_return = returns.iloc[i]

    mean_returns = train_returns.mean()
    cov_matrix = train_returns.cov()

    result = minimize(
        negative_sharpe,
        init_guess,
        args=(mean_returns, cov_matrix),
        method='SLSQP',
        bounds=bounds,
        constraints=constraints
    )

    weights = result.x
    daily_return = np.dot(weights, test_return)

    new_value = portfolio_values[-1] * (1 + daily_return)
    portfolio_values.append(new_value)
    dates.append(returns.index[i])

portfolio_series = pd.Series(portfolio_values[1:], index=dates)

plt.figure(figsize=(10,6))
plt.plot(portfolio_series, label='Rolling Optimized Portfolio')
plt.xlabel('Date')
plt.ylabel('Portfolio Value')
plt.title('Rolling Optimization Backtest')
plt.legend()
plt.grid(True)
plt.savefig("plots/rolling_optimization.png", dpi=300, bbox_inches='tight')
plt.show()

total_return = ( (1 + portfolio_series).cumprod().iloc[-1] - 1 ) * 100
ann_return = portfolio_series.pct_change().mean() * 252 * 100
ann_vol = portfolio_series.pct_change().std() * np.sqrt(252) * 100
sharpe = ann_return / ann_vol

print("\nRolling Optimization Performance:")
print(f"Total Return: {total_return:.2f}%")
print(f"Annualized Return: {ann_return:.2f}%")
print(f"Annualized Volatility: {ann_vol:.2f}%")
print(f"Sharpe Ratio: {sharpe:.2f}")

