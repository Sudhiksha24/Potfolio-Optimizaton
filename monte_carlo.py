import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize

tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META']
risk_free_rate = 0.02
initial_capital = 100
num_simulations = 1000
num_days = 252

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

mean_daily_return = portfolio_daily_returns.mean()
daily_volatility = portfolio_daily_returns.std()

simulated_paths = np.zeros((num_days, num_simulations))

for i in range(num_simulations):
    daily_returns = np.random.normal(
        mean_daily_return,
        daily_volatility,
        num_days
    )
    simulated_paths[:, i] = initial_capital * np.cumprod(1 + daily_returns)

plt.figure(figsize=(10, 6))
plt.plot(simulated_paths, linewidth=1, alpha=0.1)
plt.xlabel("Days")
plt.ylabel("Portfolio Value")
plt.title("Monte Carlo Simulation of Portfolio Value")
plt.grid(True)
plt.savefig("plots/monte_carlo_simulation.png", dpi=300, bbox_inches='tight')
plt.show()
