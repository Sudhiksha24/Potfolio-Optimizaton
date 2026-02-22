import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META']
start = '2019-01-01'
end = '2025-01-01'

prices = yf.download(tickers, start=start, end=end)['Close']
prices.dropna(inplace=True)

returns = prices.pct_change().dropna()
portfolio_returns = returns.mean(axis=1)

url = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/F-F_Research_Data_Factors_daily_CSV.zip"
ff = pd.read_csv(url, skiprows=3)
ff.rename(columns={ff.columns[0]: "Date"}, inplace=True)
ff = ff[ff["Date"].str.isnumeric()]
ff["Date"] = pd.to_datetime(ff["Date"], format="%Y%m%d")
ff.set_index("Date", inplace=True)
ff = ff.astype(float) / 100

ff = ff.loc[start:end]

data = pd.concat([portfolio_returns, ff], axis=1).dropna()
data.columns = ['Portfolio', 'MKT_RF', 'SMB', 'HML', 'RF']

y = data['Portfolio'] - data['RF']
X = data[['MKT_RF', 'SMB', 'HML']]
X = np.column_stack([np.ones(len(X)), X])

beta = np.linalg.lstsq(X, y, rcond=None)[0]

alpha = beta[0]
betas = beta[1:]

print("\nFactor Model Results:")
print(f"Alpha: {alpha:.5f}")
print(f"Market Beta: {betas[0]:.3f}")
print(f"SMB Beta: {betas[1]:.3f}")
print(f"HML Beta: {betas[2]:.3f}")

cum_portfolio = (1 + portfolio_returns).cumprod()
cum_market = (1 + data['MKT_RF'] + data['RF']).cumprod()

plt.figure(figsize=(10, 6))
plt.plot(cum_portfolio, label='Portfolio')
plt.plot(cum_market, label='Market (MKT)', linestyle='--')
plt.title("Cumulative Returns: Portfolio vs Market")
plt.xlabel("Date")
plt.ylabel("Growth")
plt.legend()
plt.grid(True)
plt.savefig("plots/factor_cumulative_returns.png", dpi=300, bbox_inches='tight')
plt.show()

factors = ['Market', 'SMB', 'HML']
plt.figure(figsize=(8, 5))
plt.bar(factors, betas)
plt.title("Factor Loadings (Betas)")
plt.ylabel("Beta Value")
plt.grid(True, axis='y')
plt.savefig("plots/factor_betas.png", dpi=300, bbox_inches='tight')
plt.show()
