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
    start='2019-01-01',
    end='2025-01-01'
)['Close']

prices.dropna(inplace=True)

returns = prices.pct_change().dropna()

train_returns = returns.loc['2019-01-01':'2022-12-31']
test_returns = returns.loc['2023-01-01':'2024-12-31']

mean_returns = train_returns.mean()
cov_matrix = train_returns.cov()
num_assets = len(tickers)

def portfolio_return(weights):
    return np.dot(weights, mean_returns) * 252

def portfolio_volatility(weights):
    return np.sqrt(np.dot(weights.T, np.dot(cov_matrix * 252, weights)))

def negative_sharpe_ratio(weights):
    return -(portfolio_return(weights) - risk_free_rate) / portfolio_volatility(weights)

constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
bounds = tuple((0, 1) for _ in range(num_assets))
init_guess = num_assets * [1. / num_assets]

result = minimize(
    negative_sharpe_ratio,
    init_guess,
    method='SLSQP',
    bounds=bounds,
    constraints=constraints
)

optimal_weights = result.x

portfolio_daily_returns = test_returns.dot(optimal_weights)
portfolio_value = (1 + portfolio_daily_returns).cumprod() * initial_capital

total_return = ( (1 + portfolio_daily_returns).cumprod().iloc[-1] - 1 ) * 100
annualized_return = portfolio_daily_returns.mean() * 252 * 100
annualized_volatility = portfolio_daily_returns.std() * np.sqrt(252) * 100
sharpe_ratio = annualized_return / annualized_volatility

print("Optimal Portfolio Weights:")
for ticker, weight in zip(tickers, optimal_weights):
    print(f"{ticker}: {weight:.2f}")

print("\nBacktesting Performance:")
print(f"Total Return: {total_return:.2f}%")
print(f"Annualized Return: {annualized_return:.2f}%")
print(f"Annualized Volatility: {annualized_volatility:.2f}%")
print(f"Sharpe Ratio: {sharpe_ratio:.2f}")

plt.figure(figsize=(10, 6))
plt.plot(portfolio_value, label='Optimized Portfolio')
plt.xlabel('Date')
plt.ylabel('Portfolio Value')
plt.title('Backtesting of Optimized Portfolio (Out-of-Sample)')
plt.legend()
plt.grid(True)
plt.savefig("plots/backtesting_equity_curve.png")
plt.show()