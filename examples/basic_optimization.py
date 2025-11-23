"""
Basic Portfolio Optimization Example

Demonstrates simple portfolio optimization for a target return.
Loads asset data from JSON file.

Author: Ankush Arora
Date: 2025-11-24
"""

import sys
import os
import json
import numpy as np

# Add parent directory to path to import portfolio_optimization package
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from portfolio_optimization import optimize_portfolio_no_short


def load_portfolio_data(json_file):
    """Load portfolio data from JSON file."""
    with open(json_file, 'r') as f:
        data = json.load(f)
    
    return {
        'asset_names': data['asset_names'],
        'expected_returns': np.array(data['expected_returns']),
        'cov_matrix': np.array(data['covariance_matrix']),
        'risk_free_rate': data['risk_free_rate'],
        'description': data.get('description', '')
    }


def main():
    print("=" * 80)
    print("BASIC PORTFOLIO OPTIMIZATION EXAMPLE")
    print("=" * 80)
    print()
    
    # Load data from JSON file
    data_file = os.path.join(os.path.dirname(__file__), 'data', 'assets_5class.json')
    data = load_portfolio_data(data_file)
    
    print(f"Portfolio: {data['description']}")
    print(f"Data source: {os.path.basename(data_file)}")
    print()
    
    # Display asset information
    print("ASSET INFORMATION")
    print("-" * 80)
    print(f"{'Asset':<20} {'Expected Return':<20} {'Volatility':<20}")
    print("-" * 80)
    
    for i, name in enumerate(data['asset_names']):
        ret = data['expected_returns'][i]
        vol = np.sqrt(data['cov_matrix'][i, i])
        print(f"{name:<20} {ret*100:>6.2f}% {'':<14} {vol*100:>6.2f}%")
    print()
    
    # Optimize for target return
    target_return = 0.08  # 8% target return
    print("OPTIMIZATION")
    print("-" * 80)
    print(f"Target Return: {target_return*100:.2f}%")
    print(f"Constraint: No short selling (all weights >= 0)")
    print()
    
    result = optimize_portfolio_no_short(
        data['expected_returns'],
        data['cov_matrix'],
        target_return
    )
    
    # Display results
    print("OPTIMAL PORTFOLIO WEIGHTS")
    print("-" * 80)
    for i, name in enumerate(data['asset_names']):
        weight = result['weights'][i]
        print(f"{name:<20} {weight*100:>6.2f}%")
    print("-" * 80)
    print(f"{'Sum of weights:':<20} {np.sum(result['weights'])*100:>6.2f}%")
    print()
    
    print("PORTFOLIO STATISTICS")
    print("-" * 80)
    print(f"Expected Return:     {result['return']*100:>6.2f}%")
    print(f"Risk (Std Dev):      {result['risk']*100:>6.2f}%")
    print(f"Variance:            {result['variance']:.6f}")
    
    # Calculate Sharpe ratio
    from portfolio_optimization import calculate_sharpe_ratio
    sharpe = calculate_sharpe_ratio(result['return'], result['risk'], data['risk_free_rate'])
    print(f"Sharpe Ratio:        {sharpe:>6.3f}")
    print()
    
    print("=" * 80)
    print("EXAMPLE COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()
