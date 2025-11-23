"""
Portfolio Data Estimation Utility

Helper functions to estimate expected returns and covariance matrix
from historical price data.

Author: Ankush Arora
Date: 2025-11-24
"""

import numpy as np
import pandas as pd
from typing import Tuple, Optional


def calculate_returns_from_prices(prices: pd.DataFrame, 
                                  frequency: str = 'daily') -> pd.DataFrame:
    """
    Calculate returns from price data.
    
    Parameters
    ----------
    prices : pd.DataFrame
        DataFrame with asset prices (columns = assets, rows = dates)
    frequency : str, optional
        Frequency of data: 'daily', 'weekly', 'monthly' (default: 'daily')
        
    Returns
    -------
    pd.DataFrame
        DataFrame of returns
    """
    returns = prices.pct_change().dropna()
    return returns


def estimate_expected_returns(prices: pd.DataFrame,
                              frequency: str = 'daily',
                              method: str = 'mean') -> np.ndarray:
    """
    Estimate expected annual returns from historical prices.
    
    Parameters
    ----------
    prices : pd.DataFrame
        DataFrame with asset prices
    frequency : str, optional
        Data frequency: 'daily' (252 days), 'weekly' (52 weeks), 
        'monthly' (12 months)
    method : str, optional
        Estimation method: 'mean' (historical average), 
        'capm' (future enhancement)
        
    Returns
    -------
    np.ndarray
        Array of annualized expected returns
        
    Example
    -------
    >>> prices = pd.DataFrame({'Stock': [100, 102, 101, 103]})
    >>> returns = estimate_expected_returns(prices, frequency='daily')
    """
    # Annualization factors
    periods_per_year = {
        'daily': 252,
        'weekly': 52,
        'monthly': 12
    }
    
    if frequency not in periods_per_year:
        raise ValueError(f"Frequency must be one of {list(periods_per_year.keys())}")
    
    # Calculate returns
    returns = calculate_returns_from_prices(prices, frequency)
    
    # Annualize mean returns
    if method == 'mean':
        expected_returns = returns.mean() * periods_per_year[frequency]
    else:
        raise ValueError(f"Method '{method}' not implemented")
    
    return expected_returns.values


def estimate_covariance_matrix(prices: pd.DataFrame,
                               frequency: str = 'daily',
                               method: str = 'sample') -> np.ndarray:
    """
    Estimate covariance matrix from historical prices.
    
    Parameters
    ----------
    prices : pd.DataFrame
        DataFrame with asset prices
    frequency : str, optional
        Data frequency for annualization
    method : str, optional
        Estimation method: 'sample' (sample covariance),
        'shrinkage' (Ledoit-Wolf, future enhancement)
        
    Returns
    -------
    np.ndarray
        Annualized covariance matrix
        
    Example
    -------
    >>> prices = pd.DataFrame({'A': [100, 102], 'B': [50, 51]})
    >>> cov = estimate_covariance_matrix(prices)
    """
    # Annualization factors
    periods_per_year = {
        'daily': 252,
        'weekly': 52,
        'monthly': 12
    }
    
    if frequency not in periods_per_year:
        raise ValueError(f"Frequency must be one of {list(periods_per_year.keys())}")
    
    # Calculate returns
    returns = calculate_returns_from_prices(prices, frequency)
    
    # Calculate and annualize covariance
    if method == 'sample':
        cov_matrix = returns.cov() * periods_per_year[frequency]
    else:
        raise ValueError(f"Method '{method}' not implemented")
    
    return cov_matrix.values


def create_portfolio_json_from_prices(prices: pd.DataFrame,
                                      asset_names: Optional[list] = None,
                                      frequency: str = 'daily',
                                      risk_free_rate: float = 0.02,
                                      description: str = '') -> dict:
    """
    Create a complete portfolio JSON structure from price data.
    
    Parameters
    ----------
    prices : pd.DataFrame
        Historical price data
    asset_names : list, optional
        Asset names (uses column names if None)
    frequency : str, optional
        Data frequency
    risk_free_rate : float, optional
        Annual risk-free rate (default: 0.02)
    description : str, optional
        Portfolio description
        
    Returns
    -------
    dict
        Portfolio data in JSON-compatible format
        
    Example
    -------
    >>> prices = pd.DataFrame({
    ...     'Stock': [100, 102, 103],
    ...     'Bond': [50, 50.5, 51]
    ... })
    >>> portfolio_data = create_portfolio_json_from_prices(
    ...     prices, 
    ...     description="My Portfolio"
    ... )
    >>> import json
    >>> with open('my_portfolio.json', 'w') as f:
    ...     json.dump(portfolio_data, f, indent=2)
    """
    if asset_names is None:
        asset_names = list(prices.columns)
    
    expected_returns = estimate_expected_returns(prices, frequency)
    cov_matrix = estimate_covariance_matrix(prices, frequency)
    
    portfolio_data = {
        'description': description,
        'asset_names': asset_names,
        'expected_returns': expected_returns.tolist(),
        'covariance_matrix': cov_matrix.tolist(),
        'risk_free_rate': risk_free_rate,
        'notes': f'Estimated from {len(prices)} periods of {frequency} data'
    }
    
    return portfolio_data


# Example usage functions
def example_from_csv():
    """Example: Load prices from CSV and create portfolio data."""
    # Load your price data (CSV with dates and prices)
    prices = pd.read_csv('historical_prices.csv', index_col='Date', parse_dates=True)
    
    # Create portfolio data
    portfolio_data = create_portfolio_json_from_prices(
        prices,
        frequency='daily',
        risk_free_rate=0.02,
        description='My custom portfolio from historical data'
    )
    
    # Save to JSON
    import json
    with open('examples/data/my_portfolio.json', 'w') as f:
        json.dump(portfolio_data, f, indent=2)
    
    print("Portfolio data saved to examples/data/my_portfolio.json")
    return portfolio_data


def example_from_manual_data():
    """Example: Create portfolio from manually entered prices."""
    # Manual price data (e.g., monthly closing prices)
    prices = pd.DataFrame({
        'Stock_A': [100, 105, 103, 108, 112, 115, 118, 120, 119, 122, 125, 128],
        'Stock_B': [50, 51, 52, 51, 53, 54, 55, 56, 55, 57, 58, 59],
        'Bond': [1000, 1002, 1001, 1003, 1005, 1004, 1006, 1008, 1007, 1009, 1010, 1011]
    })
    
    portfolio_data = create_portfolio_json_from_prices(
        prices,
        frequency='monthly',
        description='Example portfolio from monthly data'
    )
    
    print("\nExpected Returns (annualized):")
    for name, ret in zip(portfolio_data['asset_names'], portfolio_data['expected_returns']):
        print(f"  {name}: {ret*100:.2f}%")
    
    print("\nCovariance Matrix:")
    print(np.array(portfolio_data['covariance_matrix']))
    
    return portfolio_data


if __name__ == "__main__":
    print("=" * 80)
    print("PORTFOLIO DATA ESTIMATION - EXAMPLE")
    print("=" * 80)
    print()
    
    # Run example with manual data
    example_from_manual_data()
