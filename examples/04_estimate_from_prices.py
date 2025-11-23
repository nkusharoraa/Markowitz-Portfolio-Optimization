"""
Example: Estimate Portfolio Data from Historical Prices

Demonstrates how to calculate expected returns and covariance matrix
from real historical price data.

Author: Ankush Arora
Date: 2025-11-24
"""

import sys
import os
import json
import pandas as pd
import numpy as np

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from portfolio_optimization.data_utils import create_portfolio_json_from_prices


def main():
    print("=" * 80)
    print("EXAMPLE: ESTIMATING PORTFOLIO DATA FROM HISTORICAL PRICES")
    print("=" * 80)
    print()
    
    # Example 1: Create sample monthly price data (1 year)
    print("Creating sample price data (12 months)...")
    print()
    
    # Simulate realistic monthly prices for demonstration
    np.random.seed(42)  # For reproducibility
    
    months = 12
    prices = pd.DataFrame({
        'Tech_Stock': [100 * (1 + 0.015)**i + np.random.normal(0, 2) for i in range(months)],
        'Blue_Chip': [200 * (1 + 0.008)**i + np.random.normal(0, 1.5) for i in range(months)],
        'Bond_Fund': [1000 * (1 + 0.003)**i + np.random.normal(0, 0.5) for i in range(months)],
        'REIT': [75 * (1 + 0.010)**i + np.random.normal(0, 1.2) for i in range(months)]
    })
    
    print("Sample Prices (first 5 months):")
    print(prices.head())
    print()
    
    # Calculate returns to show how it works
    returns = prices.pct_change().dropna()
    
    print("Monthly Returns:")
    print(returns.describe())
    print()
    
    # Create portfolio data from prices
    print("Calculating expected returns and covariance matrix...")
    portfolio_data = create_portfolio_json_from_prices(
        prices,
        frequency='monthly',  # Monthly data
        risk_free_rate=0.02,   # 2% annual risk-free rate
        description='Example portfolio estimated from monthly price data'
    )
    
    print("\n" + "=" * 80)
    print("ESTIMATED PORTFOLIO PARAMETERS")
    print("=" * 80)
    
    # Display expected returns
    print("\nExpected Annual Returns:")
    print("-" * 80)
    for name, ret in zip(portfolio_data['asset_names'], portfolio_data['expected_returns']):
        print(f"{name:<20} {ret*100:>6.2f}%")
    
    # Display volatility (from covariance matrix diagonal)
    print("\nAnnual Volatility (Standard Deviation):")
    print("-" * 80)
    cov_matrix = np.array(portfolio_data['covariance_matrix'])
    for i, name in enumerate(portfolio_data['asset_names']):
        vol = np.sqrt(cov_matrix[i, i])
        print(f"{name:<20} {vol*100:>6.2f}%")
    
    # Display correlation matrix
    print("\nCorrelation Matrix:")
    print("-" * 80)
    std_devs = np.sqrt(np.diag(cov_matrix))
    corr_matrix = cov_matrix / np.outer(std_devs, std_devs)
    
    corr_df = pd.DataFrame(
        corr_matrix,
        index=portfolio_data['asset_names'],
        columns=portfolio_data['asset_names']
    )
    print(corr_df.round(3))
    
    # Save to JSON file
    output_file = os.path.join(
        os.path.dirname(__file__), 
        'data', 
        'estimated_portfolio.json'
    )
    
    with open(output_file, 'w') as f:
        json.dump(portfolio_data, f, indent=2)
    
    print()
    print("=" * 80)
    print(f"Portfolio data saved to: {os.path.basename(output_file)}")
    print("=" * 80)
    print()
    print("You can now use this file with optimization scripts:")
    print("  python examples/basic_optimization.py")
    print("  (modify script to load 'estimated_portfolio.json')")
    print()
    
    # Show how to use in optimization
    print("=" * 80)
    print("SAMPLE CODE: Using Estimated Data for Optimization")
    print("=" * 80)
    print("""
from portfolio_optimization import optimize_portfolio
import json
import numpy as np

# Load the estimated data
with open('examples/data/estimated_portfolio.json', 'r') as f:
    data = json.load(f)

# Optimize
result = optimize_portfolio(
    np.array(data['expected_returns']),
    np.array(data['covariance_matrix']),
    target_return=0.10,  # 10% target
    allow_short_selling=False
)

print(f"Optimal weights: {result['weights']}")
print(f"Portfolio risk: {result['risk']*100:.2f}%")
    """)


if __name__ == "__main__":
    main()
