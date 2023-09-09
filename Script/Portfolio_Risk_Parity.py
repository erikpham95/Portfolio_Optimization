import numpy as np
import pandas as pd
import yfinance as yf
from scipy.optimize import minimize

def fetch_data(tickers, start, end):
    data = yf.download(tickers, start=start, end=end)['Adj Close']
    return data

def compute_covariance_matrix(data):
    return data.pct_change().dropna().cov()

def risk_contribution(weights, cov_matrix):
    portfolio_variance = np.dot(weights, np.dot(cov_matrix, weights))
    marginal_contribution = np.dot(cov_matrix, weights)
    risk_contribution = np.multiply(weights, marginal_contribution) / portfolio_variance
    return risk_contribution

def risk_parity_objective(weights, cov_matrix):
    contributions = risk_contribution(weights, cov_matrix)
    return np.var(contributions)

def risk_parity_optimization(cov_matrix, num_assets, num_trials=1000):
    results = []
    for _ in range(num_trials):
        # Random initial weights
        init_weights = np.random.random(num_assets)
        init_weights /= init_weights.sum()

        constraints = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1})
        bounds = tuple((0, 1) for asset in range(num_assets))
        result = minimize(risk_parity_objective, init_weights, args=cov_matrix, method='SLSQP', bounds=bounds, constraints=constraints)
        results.append((result.fun, result.x))
    
    # Choose the best solution
    best_solution = min(results, key=lambda x: x[0])
    return best_solution[1]

def main():
    # Define the tickers and date range
    tickers = ["XLK", "XLC", "XLY", "XLE", "XLI", "XLB", "XLF", "XLRE", "XLP", "XLU", "XLV", "FXI", "GDX"]
    start_date = "2018-08-10"
    end_date = "2023-08-10"
    
    # Fetch data
    data = fetch_data(tickers, start_date, end_date)
    
    # Compute covariance matrix
    cov_matrix = compute_covariance_matrix(data)
    
    # Optimize portfolio using Risk Parity
    optimized_weights_rp = risk_parity_optimization(cov_matrix, len(tickers))
    
    # Display results
    print("Optimized Portfolio Weights:")
    for ticker, weight in zip(tickers, optimized_weights_rp):
        print(f"{ticker}: {weight:.4f}")

if __name__ == "__main__":
    main()
