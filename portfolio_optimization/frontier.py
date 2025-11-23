"""
Efficient Frontier Module

Functions for computing the efficient frontier.

Author: Ankush Arora
Date: 2025-11-24
"""

import numpy as np
from typing import Tuple

from .optimization import optimize_portfolio


def compute_efficient_frontier(expected_returns: np.ndarray,
                               cov_matrix: np.ndarray,
                               n_points: int = 100,
                               allow_short_selling: bool = False) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute the Efficient Frontier - the set of optimal portfolios.
    
    The efficient frontier represents all portfolios that offer the maximum
    expected return for a given level of risk, or minimum risk for a given
    level of expected return.
    
    Parameters
    ----------
    expected_returns : np.ndarray
        Expected returns for each asset
    cov_matrix : np.ndarray
        Covariance matrix of asset returns
    n_points : int, optional
        Number of points to compute on the frontier (default: 100)
    allow_short_selling : bool, optional
        If True, allows short selling (default: False)
        
    Returns
    -------
    tuple of (np.ndarray, np.ndarray)
        Returns array and corresponding risk (std dev) array for frontier
        
    Example
    -------
    >>> returns = np.array([0.10, 0.08, 0.12])
    >>> cov = np.array([[0.04, 0.01, 0.02],
    ...                 [0.01, 0.03, 0.01],
    ...                 [0.02, 0.01, 0.05]])
    >>> ef_returns, ef_risks = compute_efficient_frontier(returns, cov)
    >>> len(ef_returns) > 0
    True
    """
    # Determine range of possible returns
    min_return = np.min(expected_returns)
    max_return = np.max(expected_returns)
    
    # Generate target returns across the feasible range
    target_returns = np.linspace(min_return, max_return, n_points)
    
    frontier_returns = []
    frontier_risks = []
    
    for target_return in target_returns:
        try:
            result = optimize_portfolio(expected_returns, cov_matrix, 
                                      target_return, allow_short_selling)
            frontier_returns.append(result['return'])
            frontier_risks.append(result['risk'])
        except:
            # Skip infeasible points
            continue
    
    return np.array(frontier_returns), np.array(frontier_risks)
