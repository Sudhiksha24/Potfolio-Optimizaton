import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from scipy.optimize import minimize


tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META']

prices = yf.download(
    tickers,
    start='2019-01-01',
    end='2024-01-01'
)['Close']

prices.dropna(inplace=True)

returns = prices.pct_change().dropna()

mean_returns = returns.mean()
cov_matrix = returns.cov()

def portfolio_return(weights):
    return np.dot(weights, mean_returns)

def portfolio_volatility(weights):
    return np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))



num_assets = len(tickers)

# Constraint: sum of weights = 1
constraints = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1})

# Bounds: no short selling (0 ≤ w ≤ 1)
bounds = tuple((0, 1) for _ in range(num_assets))

# Initial guess (equal weights)
init_guess = num_assets * [1. / num_assets]


min_var_result = minimize(
    portfolio_volatility,
    init_guess,
    method='SLSQP',
    bounds=bounds,
    constraints=constraints
)

min_var_weights = min_var_result.x


risk_free_rate = 0.02  # 2%

def negative_sharpe_ratio(weights):
    ret = portfolio_return(weights)
    vol = portfolio_volatility(weights)
    return -(ret - risk_free_rate) / vol

max_sharpe_result = minimize(
    negative_sharpe_ratio,
    init_guess,
    method='SLSQP',
    bounds=bounds,
    constraints=constraints
)

max_sharpe_weights = max_sharpe_result.x


num_portfolios = 5000
results = np.zeros((3, num_portfolios))

for i in range(num_portfolios):
    weights = np.random.random(num_assets)
    weights /= np.sum(weights)

    results[0, i] = portfolio_volatility(weights)
    results[1, i] = portfolio_return(weights)
    results[2, i] = (results[1, i] - risk_free_rate) / results[0, i]


plt.figure(figsize=(10, 6))
plt.scatter(results[0], results[1], c=results[2], cmap='viridis')
plt.colorbar(label='Sharpe Ratio')

plt.scatter(
    portfolio_volatility(min_var_weights),
    portfolio_return(min_var_weights),
    color='red',
    marker='*',
    s=300,
    label='Minimum Variance'
)

plt.scatter(
    portfolio_volatility(max_sharpe_weights),
    portfolio_return(max_sharpe_weights),
    color='blue',
    marker='*',
    s=300,
    label='Maximum Sharpe Ratio'
)

plt.xlabel('Volatility (Risk)')
plt.ylabel('Expected Return')
plt.title('Efficient Frontier')
plt.legend()
plt.savefig("plots/efficient_frontier.png")
plt.show()


weights_df = pd.DataFrame({
    'Stock': tickers,
    'Min Variance Weights': min_var_weights,
    'Max Sharpe Weights': max_sharpe_weights
})

print(weights_df)
