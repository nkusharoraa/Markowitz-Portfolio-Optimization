"""
Efficient Frontier Demonstration

Generates and visualizes the efficient frontier for a portfolio.
Compares scenarios with and without short selling.

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

from portfolio_optimization import plot_efficient_frontier


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
    print("EFFICIENT FRONTIER DEMONSTRATION")
    print("=" * 80)
    print()
    
    # Load data from JSON file (you can change this to use different data)
    data_file = os.path.join(os.path.dirname(__file__), 'data', 'assets_5class.json')
    data = load_portfolio_data(data_file)
    
    print(f"Portfolio: {data['description']}")
    print(f"Data source: {os.path.basename(data_file)}")
    print(f"Risk-free rate: {data['risk_free_rate']*100:.2f}%")
    print()
    
    # Create output directory if it doesn't exist
    output_dir = os.path.join(os.path.dirname(__file__), 'output')
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate efficient frontier (No Short Selling)
    print("[1/2] Generating efficient frontier (no short selling)...")
    output_file_1 = os.path.join(output_dir, 'efficient_frontier_no_short.png')
    
    fig1 = plot_efficient_frontier(
        data['expected_returns'],
        data['cov_matrix'],
        asset_names=data['asset_names'],
        show_assets=True,
        show_max_sharpe=True,
        risk_free_rate=data['risk_free_rate'],
        allow_short_selling=False,
        output_file=output_file_1
    )
    
    # Generate efficient frontier (With Short Selling)
    print("[2/2] Generating efficient frontier (with short selling allowed)...")
    output_file_2 = os.path.join(output_dir, 'efficient_frontier_with_short.png')
    
    fig2 = plot_efficient_frontier(
        data['expected_returns'],
        data['cov_matrix'],
        asset_names=data['asset_names'],
        show_assets=True,
        show_max_sharpe=True,
        risk_free_rate=data['risk_free_rate'],
        allow_short_selling=True,
        output_file=output_file_2
    )
    
    print()
    print("=" * 80)
    print("VISUALIZATION COMPLETE")
    print("=" * 80)
    print()
    print("Generated files:")
    print(f"  - {output_file_1}")
    print(f"  - {output_file_2}")
    print()
    print("Close the plot windows to exit.")
    print()
    
    # Show plots
    plt.show()


if __name__ == "__main__":
    main()
