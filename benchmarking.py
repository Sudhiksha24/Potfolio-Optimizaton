import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize

tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META']
benchmark = '^GSPC'  # S&P 500
risk_free_rate = 0.02
initial_capital = 100

prices = yf.download(
    tickers + [benchmark],
    start='2019-01-01',
    end='2025-01-01'
)['Close']

prices.dropna(inplace=True)

returns = prices.pct_change().dropna()

train_returns = returns.loc['2019-01-01':'2022-12-31', tickers]
test_returns = returns.loc['2023-01-01':'2024-12-31', tickers]
benchmark_test_returns = returns.loc['2023-01-01':'2024-12-31', benchmark]

mean_returns = train_returns.mean()
cov_matrix = train_returns.cov()
num_assets = len(tickers)

def portfolio_return(weights):
    return np.dot(weights, mean_returns) * 252

def portfolio_volatility(weights):
    return np.sqrt(np.dot(weights.T, np.dot(cov_matrix * 252, weights)))

def negative_sharpe(weights):
    return -(portfolio_return(weights) - risk_free_rate) / portfolio_volatility(weights)

constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
bounds = tuple((0, 1) for _ in range(num_assets))
init_guess = num_assets * [1. / num_assets]

result = minimize(
    negative_sharpe,
    init_guess,
    method='SLSQP',
    bounds=bounds,
    constraints=constraints
)

optimal_weights = result.x

# -------- Validation --------
print("Weight Sum:", np.sum(optimal_weights))

# -------- Backtest Portfolio --------
portfolio_daily_returns = test_returns.dot(optimal_weights)
portfolio_value = (1 + portfolio_daily_returns).cumprod() * initial_capital

# -------- Backtest Benchmark --------
benchmark_value = (1 + benchmark_test_returns).cumprod() * initial_capital

# -------- Metrics Function --------
def performance_metrics(daily_returns, label):
    total_return = ( (1 + daily_returns).cumprod().iloc[-1] - 1 ) * 100
    ann_return = daily_returns.mean() * 252 * 100
    ann_vol = daily_returns.std() * np.sqrt(252) * 100
    sharpe = ann_return / ann_vol

    print(f"\n{label} Performance:")
    print(f"Total Return: {total_return:.2f}%")
    print(f"Annualized Return: {ann_return:.2f}%")
    print(f"Annualized Volatility: {ann_vol:.2f}%")
    print(f"Sharpe Ratio: {sharpe:.2f}")

performance_metrics(portfolio_daily_returns, "Optimized Portfolio")
performance_metrics(benchmark_test_returns, "S&P 500 Benchmark")

# -------- Plot Comparison --------
plt.figure(figsize=(10, 6))
plt.plot(portfolio_value, label='Optimized Portfolio')
plt.plot(benchmark_value, label='S&P 500 Benchmark', linestyle='--')
plt.xlabel('Date')
plt.ylabel('Portfolio Value')
plt.title('Optimized Portfolio vs S&P 500 (Backtesting)')
plt.legend()
plt.grid(True)
plt.savefig("plots/benchmark_comparison.png", dpi=300, bbox_inches='tight')
plt.show()

# -------- Print Weights --------
print("\nOptimal Weights:")
for t, w in zip(tickers, optimal_weights):
    print(f"{t}: {w:.2f}")
