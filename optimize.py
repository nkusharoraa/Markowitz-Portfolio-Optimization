"""
Portfolio Optimizer - Main CLI Tool

End-to-end portfolio optimization from price history.
Simply provide a CSV file with historical prices and get optimized portfolios.

Usage:
    python optimize.py prices.csv

Author: Ankush Arora
Date: 2025-11-24
"""

import argparse
import sys
import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from portfolio_optimization import (
    optimize_portfolio,
    optimize_max_sharpe,
    plot_efficient_frontier,
    plot_correlation_matrix
)
from portfolio_optimization.data_utils import (
    estimate_expected_returns,
    estimate_covariance_matrix
)


def detect_frequency(df):
    """Auto-detect data frequency from date index."""
    if len(df) < 2:
        return 'daily'
    
    # Calculate median time difference
    time_diffs = df.index[1:] - df.index[:-1]
    median_diff = time_diffs.median()
    
    # Classify based on median difference
    days = median_diff.days
    if days <= 1:
        return 'daily'
    elif days <= 10:
        return 'weekly'
    else:
        return 'monthly'


def load_prices_from_csv(csv_file):
    """Load price history from CSV file."""
    try:
        # Try to parse with date index
        df = pd.read_csv(csv_file, index_col=0, parse_dates=True)
        print(f"✓ Loaded {len(df)} rows of price data")
        print(f"✓ Assets: {', '.join(df.columns)}")
        print(f"✓ Date range: {df.index[0]} to {df.index[-1]}")
        
        # Auto-detect frequency
        detected_freq = detect_frequency(df)
        print(f"✓ Detected frequency: {detected_freq}")
        
        return df, detected_freq
    except Exception as e:
        print(f"✗ Error loading CSV: {e}")
        print("\nExpected CSV format:")
        print("Date,Asset1,Asset2,Asset3")
        print("2020-01-01,100,50,200")
        print("2020-01-02,102,51,198")
        print("...")
        sys.exit(1)


def get_user_input(prompt, default=None, input_type=float):
    """Get user input with default value."""
    default_str = f" [{default}]" if default is not None else ""
    while True:
        try:
            user_input = input(f"{prompt}{default_str}: ").strip()
            if not user_input and default is not None:
                return default
            if input_type == bool:
                return user_input.lower() in ['y', 'yes', 'true', '1']
            return input_type(user_input)
        except ValueError:
            print(f"Invalid input. Please enter a {input_type.__name__}")


def print_header(text):
    """Print formatted header."""
    print("\n" + "=" * 80)
    print(text)
    print("=" * 80)


def print_section(text):
    """Print formatted section."""
    print("\n" + "-" * 80)
    print(text)
    print("-" * 80)


def main():
    parser = argparse.ArgumentParser(
        description='Portfolio Optimization from Historical Prices',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python optimize.py prices.csv
  python optimize.py prices.csv --target 0.10
  python optimize.py prices.csv --auto --output results/
  
CSV Format:
  Date,Stock_A,Stock_B,Bond
  2020-01-01,100,50,1000
  2020-01-02,102,51,1001
  ...
        """
    )
    
    parser.add_argument('prices_csv', help='CSV file with historical prices')
    parser.add_argument('--target', type=float, help='Target return (e.g., 0.10 for 10%%)')
    parser.add_argument('--risk-free', type=float, default=0.02, help='Risk-free rate (default: 0.02)')
    parser.add_argument('--frequency', choices=['daily', 'weekly', 'monthly'], 
                       default='daily', help='Data frequency (default: daily)')
    parser.add_argument('--short', action='store_true', help='Allow short selling')
    parser.add_argument('--auto', action='store_true', help='Run with defaults (no prompts)')
    parser.add_argument('--output', default='output', help='Output directory (default: output/)')
    parser.add_argument('--save-data', action='store_true', help='Save estimated data to JSON')
    
    args = parser.parse_args()
    
    # Print header
    print_header("PORTFOLIO OPTIMIZATION FROM PRICE HISTORY")
    
    # Load prices
    print_section("1. LOADING PRICE DATA")
    prices, detected_freq = load_prices_from_csv(args.prices_csv)
    
    # Use detected frequency if not specified
    if args.frequency == 'daily' and detected_freq != 'daily':
        args.frequency = detected_freq
        print(f"  Using detected frequency: {args.frequency}")
    
    # Estimate returns and covariance
    print_section("2. ESTIMATING PARAMETERS")
    print("Calculating expected returns and covariance matrix...")
    
    expected_returns = estimate_expected_returns(prices, args.frequency)
    cov_matrix = estimate_covariance_matrix(prices, args.frequency)
    asset_names = list(prices.columns)
    
    print("\n✓ Expected Annual Returns:")
    for name, ret in zip(asset_names, expected_returns):
        print(f"  {name:<20} {ret*100:>6.2f}%")
    
    print("\n✓ Annual Volatility:")
    for i, name in enumerate(asset_names):
        vol = np.sqrt(cov_matrix[i, i])
        print(f"  {name:<20} {vol*100:>6.2f}%")
    
    # Save data if requested
    if args.save_data:
        data_file = os.path.join(args.output, 'estimated_data.json')
        os.makedirs(args.output, exist_ok=True)
        
        portfolio_data = {
            'description': f'Estimated from {args.prices_csv}',
            'asset_names': asset_names,
            'expected_returns': expected_returns.tolist(),
            'covariance_matrix': cov_matrix.tolist(),
            'risk_free_rate': args.risk_free,
            'frequency': args.frequency,
            'source': args.prices_csv
        }
        
        with open(data_file, 'w') as f:
            json.dump(portfolio_data, f, indent=2)
        print(f"\n✓ Saved estimated data to: {data_file}")
    
    # Get target return
    print_section("3. OPTIMIZATION PARAMETERS")
    
    if args.target is not None:
        target_return = args.target
        print(f"Target return: {target_return*100:.2f}% (from command line)")
    elif not args.auto:
        print(f"\nAsset returns range from {expected_returns.min()*100:.2f}% to {expected_returns.max()*100:.2f}%")
        target_return = get_user_input(
            "Enter target annual return (as decimal, e.g., 0.10 for 10%)",
            default=expected_returns.mean()
        )
    else:
        target_return = expected_returns.mean()
        print(f"Target return: {target_return*100:.2f}% (mean of assets)")
    
    # Short selling
    if not args.auto and not args.short:
        allow_short = get_user_input(
            "Allow short selling? (y/n)",
            default='n',
            input_type=lambda x: x.lower() in ['y', 'yes']
        )
    else:
        allow_short = args.short
    
    print(f"\n✓ Target return: {target_return*100:.2f}%")
    print(f"✓ Short selling: {'Allowed' if allow_short else 'Not allowed'}")
    print(f"✓ Risk-free rate: {args.risk_free*100:.2f}%")
    
    # Run optimization
    print_section("4. OPTIMIZATION RESULTS")
    
    # Optimal portfolio for target return
    print("\n[1/2] Finding optimal portfolio for target return...")
    try:
        result = optimize_portfolio(
            expected_returns,
            cov_matrix,
            target_return=target_return,
            allow_short_selling=allow_short
        )
    except ValueError as e:
        if "infeasible" in str(e).lower():
            print(f"\n✗ Error: Target return {target_return*100:.2f}% is not achievable!")
            print(f"  Feasible range: {expected_returns.min()*100:.2f}% to {expected_returns.max()*100:.2f}%")
            print("\nTry one of these:")
            print(f"  - Use a target within the feasible range")
            print(f"  - Enable short selling (--short) to expand possibilities")
            print(f"  - Check if your data frequency is correct (currently: {args.frequency})")
            sys.exit(1)
        else:
            raise
    
    print("\n✓ Optimal Portfolio Weights:")
    for name, weight in zip(asset_names, result['weights']):
        position = ""
        if allow_short and weight < 0:
            position = " (SHORT)"
        print(f"  {name:<20} {weight*100:>6.2f}%{position}")
    
    print(f"\n  Sum of weights: {np.sum(result['weights'])*100:.2f}%")
    
    print("\n✓ Portfolio Statistics:")
    print(f"  Expected Return:    {result['return']*100:>6.2f}%")
    print(f"  Risk (Std Dev):     {result['risk']*100:>6.2f}%")
    print(f"  Variance:           {result['variance']:.6f}")
    
    from portfolio_optimization.metrics import calculate_sharpe_ratio
    sharpe = calculate_sharpe_ratio(result['return'], result['risk'], args.risk_free)
    print(f"  Sharpe Ratio:       {sharpe:>6.3f}")
    
    # Maximum Sharpe ratio
    print("\n[2/2] Finding maximum Sharpe ratio portfolio...")
    max_sharpe_result = optimize_max_sharpe(
        expected_returns,
        cov_matrix,
        risk_free_rate=args.risk_free,
        allow_short_selling=allow_short
    )
    
    print("\n✓ Maximum Sharpe Ratio Portfolio:")
    for name, weight in zip(asset_names, max_sharpe_result['weights']):
        position = ""
        if allow_short and weight < 0:
            position = " (SHORT)"
        print(f"  {name:<20} {weight*100:>6.2f}%{position}")
    
    print(f"\n  Expected Return:    {max_sharpe_result['return']*100:>6.2f}%")
    print(f"  Risk (Std Dev):     {max_sharpe_result['risk']*100:>6.2f}%")
    print(f"  Sharpe Ratio:       {max_sharpe_result['sharpe_ratio']:>6.3f}")
    
    # Visualization
    print_section("5. GENERATING VISUALIZATIONS")
    os.makedirs(args.output, exist_ok=True)
    
    # Efficient frontier
    print("\n[1/3] Efficient frontier plot...")
    ef_file = os.path.join(args.output, 'efficient_frontier.png')
    plot_efficient_frontier(
        expected_returns,
        cov_matrix,
        asset_names=asset_names,
        show_assets=True,
        show_max_sharpe=True,
        risk_free_rate=args.risk_free,
        allow_short_selling=allow_short,
        output_file=ef_file
    )
    print(f"  ✓ Saved to: {ef_file}")
    
    # Correlation matrix
    print("\n[2/3] Correlation matrix plot...")
    corr_file = os.path.join(args.output, 'correlation_matrix.png')
    plot_correlation_matrix(
        cov_matrix,
        asset_names=asset_names,
        output_file=corr_file
    )
    print(f"  ✓ Saved to: {corr_file}")
    
    # Save results to text file
    print("\n[3/3] Saving results to text file...")
    results_file = os.path.join(args.output, 'optimization_results.txt')
    
    with open(results_file, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("PORTFOLIO OPTIMIZATION RESULTS\n")
        f.write("=" * 80 + "\n\n")
        
        f.write(f"Source: {args.prices_csv}\n")
        f.write(f"Data frequency: {args.frequency}\n")
        f.write(f"Risk-free rate: {args.risk_free*100:.2f}%\n")
        f.write(f"Short selling: {'Allowed' if allow_short else 'Not allowed'}\n\n")
        
        f.write("-" * 80 + "\n")
        f.write("OPTIMAL PORTFOLIO FOR TARGET RETURN\n")
        f.write("-" * 80 + "\n")
        f.write(f"Target Return: {target_return*100:.2f}%\n\n")
        
        f.write("Weights:\n")
        for name, weight in zip(asset_names, result['weights']):
            f.write(f"  {name:<20} {weight*100:>6.2f}%\n")
        
        f.write(f"\nExpected Return:  {result['return']*100:>6.2f}%\n")
        f.write(f"Risk (Std Dev):   {result['risk']*100:>6.2f}%\n")
        f.write(f"Sharpe Ratio:     {sharpe:>6.3f}\n\n")
        
        f.write("-" * 80 + "\n")
        f.write("MAXIMUM SHARPE RATIO PORTFOLIO\n")
        f.write("-" * 80 + "\n\n")
        
        f.write("Weights:\n")
        for name, weight in zip(asset_names, max_sharpe_result['weights']):
            f.write(f"  {name:<20} {weight*100:>6.2f}%\n")
        
        f.write(f"\nExpected Return:  {max_sharpe_result['return']*100:>6.2f}%\n")
        f.write(f"Risk (Std Dev):   {max_sharpe_result['risk']*100:>6.2f}%\n")
        f.write(f"Sharpe Ratio:     {max_sharpe_result['sharpe_ratio']:>6.3f}\n")
    
    print(f"  ✓ Saved to: {results_file}")
    
    # Summary
    print_header("OPTIMIZATION COMPLETE")
    print(f"\nAll results saved to: {args.output}/")
    print("\nGenerated files:")
    print(f"  - {os.path.basename(ef_file)}")
    print(f"  - {os.path.basename(corr_file)}")
    print(f"  - {os.path.basename(results_file)}")
    if args.save_data:
        print(f"  - estimated_data.json")
    
    # Ask to show plots
    if not args.auto:
        show_plots = get_user_input(
            "\nShow plots now? (y/n)",
            default='y',
            input_type=lambda x: x.lower() in ['y', 'yes']
        )
        if show_plots:
            plt.show()
    
    print("\n✓ Done!\n")


if __name__ == "__main__":
    main()
