Portfolio Optimization using Modern Portfolio Theory (MPT)

This project implements portfolio optimization using historical equity market data based on Modern Portfolio Theory (MPT).

The objective is to:

Identify optimal asset allocations that balance risk and return

Evaluate out-of-sample performance

Benchmark results against a market index

Simulate future portfolio scenarios using Monte Carlo methods

This project is intended for educational and demonstration purposes in quantitative finance and data science.

Key Features
Modern Portfolio Theory Implementation

Efficient Frontier construction

Minimum Variance Portfolio (MVP)

Maximum Sharpe Ratio Portfolio (MSRP)

Constrained optimization (no short selling)

Rolling (Walk-Forward) Optimization

To increase realism and robustness, the project implements rolling portfolio optimization:

Uses rolling training windows

Re-optimizes portfolio weights periodically

Applies optimized weights to subsequent test windows

Simulates systematic portfolio rebalancing

This avoids static allocations and better reflects real-world portfolio management under evolving market conditions.

Benchmarking Against Market Index

Portfolios are benchmarked against the S&P 500 (^GSPC) to evaluate market-relative performance.

Performance metrics compared include:

Cumulative return

Annualized return

Volatility

Sharpe ratio

Maximum drawdown

Comparison plots illustrate:

Growth of $1 invested

Risk-adjusted return comparison

Drawdown analysis

This ensures that performance is evaluated relative to a broad market benchmark rather than in isolation.

Monte Carlo Simulation

Thousands of simulated future return paths

Portfolio value distribution analysis

Downside risk estimation

Expected performance under uncertainty

Methodology
1. Data Collection

Historical data is downloaded from Yahoo Finance.

Assets analyzed:

AAPL

MSFT

GOOGL

AMZN

META

Benchmark:

S&P 500 (^GSPC)

2. Portfolio Optimization

Random portfolios are generated to explore the risk-return space.

Portfolio metrics computed:

Expected return

Volatility

Sharpe ratio

Optimization is performed using SciPy (SLSQP) under the following constraints:

Portfolio weights sum to 1

0 ≤ weight ≤ 1 (no short selling)

3. Efficient Frontier

The Efficient Frontier visualizes the risk-return trade-off and highlights:

Minimum Variance Portfolio

Maximum Sharpe Ratio Portfolio

4. Backtesting (Out-of-Sample Evaluation)

Data is split into training and testing periods

Optimized weights from the training period are applied to unseen data

Look-ahead bias is avoided

5. Rolling Optimization (Walk-Forward Framework)

Instead of optimizing once and holding static weights:

Train on a historical window (e.g., 2–3 years)

Optimize the portfolio

Apply weights to the next testing window

Roll the window forward

Repeat

This framework simulates periodic rebalancing and dynamic portfolio allocation.

6. Benchmarking

The optimized portfolio is compared against the S&P 500.

Outputs include:

Cumulative return comparison chart

Drawdown comparison

Sharpe ratio comparison table

This highlights whether the strategy provides superior risk-adjusted returns relative to the broader market.

7. Monte Carlo Simulation

Future returns are simulated using random sampling:

Thousands of potential return paths

Distribution of final portfolio values

Confidence intervals

Downside risk estimation

Project Structure
portfolio-optimization/
│
├── data/                      # Historical stock and benchmark data
├── notebooks/                 # Jupyter notebooks for exploration
├── src/
│   ├── data_loader.py
│   ├── optimizer.py
│   ├── backtest.py
│   ├── benchmark.py
│   ├── rolling_optimization.py
│   └── monte_carlo.py
│
├── results/                   # Plots and outputs
├── requirements.txt
└── README.md
Example Outputs

Efficient Frontier plot

Optimal portfolio weights (MVP and MSRP)

Rolling backtest performance chart

Portfolio vs benchmark comparison

Monte Carlo simulation distributions

Summary performance statistics

Future Enhancements

Transaction cost modeling

Portfolio turnover analysis

Multiple benchmark comparison

Sector-level diversification

Value at Risk (VaR) and Conditional VaR

Regime-switching models

Tools and Libraries

Python

NumPy

Pandas

Matplotlib

SciPy

yFinance

All dependencies are listed in requirements.txt.

Disclaimer

This project is for educational purposes only and does not constitute financial or investment advice.

If you would like, I can further refine this into a research-style README suitable for quantitative research roles or a concise recruiter-focused version optimized for GitHub presentation.
