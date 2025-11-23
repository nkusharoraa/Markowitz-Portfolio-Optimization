"""
Maximum Sharpe Ratio Demonstration

Finds and visualizes the portfolio with maximum Sharpe ratio.
Compares results with and without short selling.

Author: Ankush Arora
Date: 2025-11-24
"""

import sys
import os
import json
import numpy as np
import matplotlib.pyplot as plt

# Add parent directory to path to import portfolio_optimization package
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from portfolio_optimization import optimize_max_sharpe, plot_efficient_frontier


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


def display_max_sharpe_portfolio(result, asset_names, scenario_name):
    """Display maximum Sharpe ratio portfolio details."""
    print(f"\n{scenario_name}")
    print("=" * 80)
    
    print("\nOPTIMAL PORTFOLIO WEIGHTS")
    print("-" * 80)
    for i, name in enumerate(asset_names):
        weight = result['weights'][i]
        position = "LONG" if weight >= 0 else "SHORT"
        print(f"{name:<20} {weight*100:>6.2f}% ({position})")
    print("-" * 80)
    print(f"{'Sum of weights:':<20} {np.sum(result['weights'])*100:>6.2f}%")
    
    print("\nPORTFOLIO STATISTICS")
    print("-" * 80)
    print(f"Expected Return:     {result['return']*100:>6.2f}%")
    print(f"Risk (Std Dev):      {result['risk']*100:>6.2f}%")
    print(f"Sharpe Ratio:        {result['sharpe_ratio']:>6.3f}")
    print()


def main():
    print("=" * 80)
    print("MAXIMUM SHARPE RATIO DEMONSTRATION")
    print("=" * 80)
    print()
    
    # Load data from JSON file  
    data_file = os.path.join(os.path.dirname(__file__), 'data', 'assets_5class.json')
    data = load_portfolio_data(data_file)
    
    print(f"Portfolio: {data['description']}")
    print(f"Data source: {os.path.basename(data_file)}")
    print(f"Risk-free rate: {data['risk_free_rate']*100:.2f}%")
    print()
    
    # Find max Sharpe ratio portfolio (No Short Selling)
    print("Computing optimal portfolios...")
    result_no_short = optimize_max_sharpe(
        data['expected_returns'],
        data['cov_matrix'],
        risk_free_rate=data['risk_free_rate'],
        allow_short_selling=False
    )
    
    # Find max Sharpe ratio portfolio (With Short Selling)
    result_with_short = optimize_max_sharpe(
        data['expected_returns'],
        data['cov_matrix'],
        risk_free_rate=data['risk_free_rate'],
        allow_short_selling=True
    )
    
    # Display results
    display_max_sharpe_portfolio(
        result_no_short,
        data['asset_names'],
        "SCENARIO 1: NO SHORT SELLING"
    )
    
    display_max_sharpe_portfolio(
        result_with_short,
        data['asset_names'],
        "SCENARIO 2: SHORT SELLING ALLOWED"
    )
    
    # Visualize on efficient frontier
    print("Generating visualization...")
    
    # Create output directory
    output_dir = os.path.join(os.path.dirname(__file__), 'output')
    os.makedirs(output_dir, exist_ok=True)
    
    output_file = os.path.join(output_dir, 'max_sharpe_portfolio.png')
    
    fig = plot_efficient_frontier(
        data['expected_returns'],
        data['cov_matrix'],
        asset_names=data['asset_names'],
        show_assets=True,
        show_max_sharpe=True,
        risk_free_rate=data['risk_free_rate'],
        allow_short_selling=False,
        output_file=output_file
    )
    
    print()
    print("=" * 80)
    print("DEMONSTRATION COMPLETE")
    print("=" * 80)
    print()
    print(f"Saved visualization to: {output_file}")
    print()
    print("Close the plot window to exit.")
    print()
    
    # Show plot
    plt.show()


if __name__ == "__main__":
    main()
