import yfinance as yf
import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt

# 1. Fetch Data
def fetch_data(tickers, start_date, end_date):
    data = yf.download(tickers, start=start_date, end=end_date)['Adj Close']
    return data

# 2. Calculate GMVP Allocation with a modified approach
# def global_minimum_variance_portfolio(cov_matrix):
#     num_assets = cov_matrix.shape[0]
    
#     # Different initial guess
#     initial_weights = np.random.random(num_assets)
#     initial_weights /= initial_weights.sum()  # Normalize to make the sum 1
    
#     constraints = ({'type': 'eq', 'fun': lambda weights: np.sum(weights) - 1})
#     bounds = tuple((0, 1) for asset in range(num_assets))
    
#     def objective(weights): 
#         return np.dot(weights.T, np.dot(cov_matrix, weights))
    
#     solution = minimize(objective, initial_weights, method='SLSQP', bounds=bounds, constraints=constraints)
#     return solution.x

def global_minimum_variance_portfolio(cov_matrix, num_trials=10000):
    num_assets = cov_matrix.shape[0]
    best_weights = None
    best_objective_value = float('inf')  # Initialize with a high value

    for _ in range(num_trials):
        initial_weights = np.random.random(num_assets)
        initial_weights /= initial_weights.sum()  # Normalize to make the sum 1
        constraints = ({'type': 'eq', 'fun': lambda weights: np.sum(weights) - 1})
        bounds = tuple((0, 1) for asset in range(num_assets))
        
        def objective(weights): 
            return np.dot(weights.T, np.dot(cov_matrix, weights))
        
        solution = minimize(objective, initial_weights, method='SLSQP', bounds=bounds, constraints=constraints)
        
        # Update best solution if the current one is better
        if solution.fun < best_objective_value:
            best_objective_value = solution.fun
            best_weights = solution.x

    return best_weights

# 3. Simulate Portfolio Performance
def simulate_portfolio_performance(data, weights):
    returns = data.pct_change().dropna()
    portfolio_returns = returns.dot(weights)
    portfolio_cumulative = (portfolio_returns + 1).cumprod()
    return portfolio_cumulative

# 4. Calculate Performance Indicators
def calculate_performance(data, weights):
    returns = data.pct_change().dropna()
    portfolio_returns = returns.dot(weights)
    
    # Annualized Portfolio Return
    avg_daily_return = portfolio_returns.mean()
    annualized_return = (1 + avg_daily_return) ** 252 - 1
    
    # Portfolio Standard Deviation
    std_dev = portfolio_returns.std()
    
    # Risk-free rate (using a generic 2% here, but can be adjusted)
    risk_free_rate = 0.02
    
    # Sharpe Ratio
    sharpe_ratio = (annualized_return - risk_free_rate) / (std_dev * np.sqrt(252))
    
    return annualized_return, std_dev, sharpe_ratio

# Tickers and Data Fetching
tickers = ["XLK", "XLC", "XLY", "XLE", "XLI", "XLB", "XLF", "XLRE", "XLP", "XLU", "XLV", "FXI", "GDX"]
start_date = "2018-08-10"
end_date = "2023-08-10"
data = fetch_data(tickers, start_date, end_date)

# Compute covariance matrix
returns = data.pct_change().dropna()
cov_matrix = returns.cov()

# GMVP Allocation
optimal_weights = global_minimum_variance_portfolio(cov_matrix)
print("Optimal Weights:", optimal_weights)

# Simulate Portfolio Performance and Plot
portfolio_cumulative = simulate_portfolio_performance(data, optimal_weights)
plt.figure(figsize=(14, 7))
portfolio_cumulative.plot()
plt.title("Portfolio Cumulative Returns")
plt.xlabel("Date")
plt.ylabel("Cumulative Returns")
plt.grid(True)

# Save the plot to a file
plt.savefig("portfolio_performance_comparison.png", dpi=300)  # Adjust filename and dpi as needed

# Close the plot figure
plt.close()


# Calculate and Print Performance Indicators
annualized_return, std_dev, sharpe_ratio = calculate_performance(data, optimal_weights)
print(f"Annualized Return: {annualized_return:.2%}")
print(f"Standard Deviation: {std_dev:.2%}")
print(f"Sharpe Ratio: {sharpe_ratio:.2f}")
