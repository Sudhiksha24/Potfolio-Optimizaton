Portfolio Optimization with Rolling Strategy and Benchmark Comparison

This project implements portfolio optimization using historical stock market data based on Modern Portfolio Theory (MPT).

The objective is to construct optimal portfolios, evaluate their performance using backtesting and simulation, and compare results against a benchmark index using both static and rolling strategies.

⸻

Project Overview

This project includes:
	•	Construction of optimal portfolios using historical returns
	•	Efficient Frontier visualization
	•	Identification of:
	•	Minimum Variance Portfolio
	•	Maximum Sharpe Ratio Portfolio
	•	Out-of-sample backtesting
	•	Monte Carlo simulation of future portfolio paths
	•	Rolling (walk-forward) portfolio optimization
	•	Performance comparison with S&P 500 benchmark

This project is built for learning and demonstration purposes in Quantitative Finance and Data Science.

⸻

Methodology

1. Data Collection
	•	Historical stock data downloaded from Yahoo Finance
	•	Assets used:
	•	AAPL
	•	MSFT
	•	GOOGL
	•	AMZN
	•	META
	•	Benchmark:
	•	S&P 500 Index (^GSPC)

⸻

2. Data Preprocessing
	•	Daily closing prices converted to daily returns
	•	Missing values removed
	•	Covariance matrix computed for risk estimation

⸻

3. Portfolio Optimization (Static)
	•	Random portfolios generated
	•	Portfolio metrics computed:
	•	Expected return
	•	Volatility
	•	Sharpe ratio
	•	Optimization performed using SciPy (SLSQP)
	•	Constraints:
	•	Weights sum to 1
	•	No short selling (weights between 0 and 1)

⸻

4. Efficient Frontier Visualization
	•	Risk vs return of random portfolios plotted
	•	Highlighted portfolios:
	•	Minimum Variance Portfolio
	•	Maximum Sharpe Ratio Portfolio

⸻

5. Backtesting (Out-of-Sample Evaluation)
	•	Data split into training and testing periods
	•	Optimized weights applied to unseen future data
	•	Portfolio performance evaluated over time
	•	Look-ahead bias avoided

⸻

6. Monte Carlo Simulation
	•	Thousands of simulated return paths generated
	•	Future portfolio value distribution estimated
	•	Used to analyze risk and uncertainty

⸻

7. Rolling Optimization (Walk-Forward Strategy)

Instead of keeping portfolio weights fixed, the strategy:
	•	Uses a rolling historical window (e.g., 2 years)
	•	Re-optimizes portfolio weights periodically
	•	Applies new weights to the next trading period
	•	Repeats this process over time

This simulates how real-world asset managers dynamically adjust allocations based on recent market conditions.

⸻

8. Benchmark Comparison

Portfolio performance is compared with the S&P 500 Index.

Metrics compared:
	•	Total return
	•	Annualized return
	•	Volatility
	•	Sharpe ratio
	•	Equity curve comparison

This allows evaluation of whether active optimization outperforms passive market exposure.

⸻

Output

The project generates:
	•	Efficient Frontier plot
	•	Backtesting equity curve
	•	Monte Carlo simulation paths
	•	Rolling optimization performance curve
	•	Benchmark vs Portfolio comparison plot

⸻

Tools and Libraries
	•	Python
	•	NumPy
	•	Pandas
	•	Matplotlib
	•	SciPy
	•	yFinance

All dependencies are listed in requirements.txt.


portfolio-optimization/
├── optimization.py
├── backtesting.py
├── monte_carlo.py
├── rolling_optimization.py
├── benchmark_comparison.py
├── requirements.txt
├── README.md
├── plots/
│   ├── efficient_frontier.png
│   ├── backtesting_equity_curve.png
│   ├── monte_carlo_simulation.png
│   ├── rolling_optimization.png
│   └── benchmark_comparison.png

Future Enhancements
	•	Transaction cost modeling
	•	Portfolio turnover analysis
	•	Risk metrics (VaR, CVaR)
	•	Regime detection
	•	Factor-based allocation
	•	Interactive dashboard

⸻

Disclaimer

This project is for educational purposes only and does not constitute financial or investment advice.
