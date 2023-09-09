import yfinance as yf
import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt

# 1. Fetch Data
def fetch_data(tickers, start_date, end_date):
    data = yf.download(tickers, start=start_date, end=end_date)['Adj Close']
    return data

# 2. Calculate MPT Allocation with the enhanced approach
# def mpt_portfolio_allocation(returns, num_trials=10000):
#     expected_returns = returns.mean()
#     cov_matrix = returns.cov()
#     num_assets = returns.shape[1]
#     risk_free_rate = 0.02

#     best_weights = None
#     best_objective_value = float('-inf')  # Initialize with a low value

#     for _ in range(num_trials):
#         initial_weights = np.random.random(num_assets)
#         initial_weights /= initial_weights.sum()  # Normalize to make the sum 1
#         constraints = ({'type': 'eq', 'fun': lambda weights: np.sum(weights) - 1})
#         bounds = tuple((0, 1) for asset in range(num_assets))

#         def objective(weights): 
#             portfolio_return = np.dot(expected_returns, weights)
#             portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
#             sharpe_ratio = (portfolio_return - risk_free_rate) / portfolio_volatility
#             return -sharpe_ratio  # Negative since we want to maximize the Sharpe Ratio

#         solution = minimize(objective, initial_weights, method='SLSQP', bounds=bounds, constraints=constraints)

#         # Update best solution if the current one is better
#         if solution.fun > best_objective_value:
#             best_objective_value = solution.fun
#             best_weights = solution.x

#     return best_weights

def mpt_portfolio_allocation(returns, num_trials=100, lambda_reg=1.5):
    expected_returns = returns.mean()
    cov_matrix = returns.cov()
    num_assets = returns.shape[1]
    risk_free_rate = 0.02

    best_weights = None
    best_objective_value = float('-inf')  # Initialize with a low value

    for _ in range(num_trials):
        # initial_weights = np.random.random(num_assets)
        initial_weights = np.full(num_assets, 1/num_assets)
        initial_weights /= initial_weights.sum()  # Normalize to make the sum 1
        constraints = ({'type': 'eq', 'fun': lambda weights: np.sum(weights) - 1})
        bounds = tuple((0.0, 0.2) for asset in range(num_assets))

        def objective(weights): 
            portfolio_return = np.dot(expected_returns, weights)
            portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
            sharpe_ratio = (portfolio_return - risk_free_rate) / portfolio_volatility
            regularization = lambda_reg * np.sum(np.square(weights))
            return -(sharpe_ratio - regularization)  # Negative since we want to maximize the Sharpe Ratio

        solution = minimize(objective, initial_weights, method='SLSQP', bounds=bounds, constraints=constraints)
        # solution = minimize(objective, initial_weights, method='CG', bounds=bounds, constraints=constraints)


        # Update best solution if the current one is better
        if solution.fun > best_objective_value:
            best_objective_value = solution.fun
            best_weights = solution.x

    return best_weights

# 3. Simulate Portfolio Performance
def simulate_portfolio_performance(data, weights):
    returns = data.pct_change().dropna()
    portfolio_returns = returns.dot(weights)
    portfolio_cumulative = (portfolio_returns + 1).cumprod()
    return portfolio_cumulative

# 4. Calculate Performance Indicators (same as in GMVP)
def calculate_performance(data, weights):
    returns = data.pct_change().dropna()
    portfolio_returns = returns.dot(weights)
    avg_daily_return = portfolio_returns.mean()
    annualized_return = (1 + avg_daily_return) ** 252 - 1
    std_dev = portfolio_returns.std()
    risk_free_rate = 0.02
    sharpe_ratio = (annualized_return - risk_free_rate) / (std_dev * np.sqrt(252))
    return annualized_return, std_dev, sharpe_ratio

# Tickers and Data Fetching
tickers = ["XLK", "XLC", "XLY", "XLE", "XLI", "XLB", "XLF", "XLRE", "XLP", "XLU", "XLV", "FXI", "GDX"]
start_date = "2018-08-10"
end_date = "2023-08-10"
data = fetch_data(tickers, start_date, end_date)

# Compute returns
returns = data.pct_change().dropna()

# MPT Allocation
optimal_weights = mpt_portfolio_allocation(returns)
for ticker, weight in zip(tickers, optimal_weights):
    print(f"{ticker}: {weight:.2%}")

# Simulate Portfolio Performance and Plot
portfolio_cumulative = simulate_portfolio_performance(data, optimal_weights)
plt.figure(figsize=(14, 7))
portfolio_cumulative.plot()
plt.title("Portfolio Cumulative Returns (MPT)")
plt.xlabel("Date")
plt.ylabel("Cumulative Returns")
plt.grid(True)
plt.savefig("mpt_portfolio_performance_comparison.png", dpi=300)  # Adjust filename and dpi as needed
plt.close()

# Calculate and Print Performance Indicators
annualized_return, std_dev, sharpe_ratio = calculate_performance(data, optimal_weights)
print(f"Annualized Return: {annualized_return:.2%}")
print(f"Standard Deviation: {std_dev:.2%}")
print(f"Sharpe Ratio: {sharpe_ratio:.2f}")
