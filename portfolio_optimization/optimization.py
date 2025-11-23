"""
Portfolio Optimization Module

Functions for optimizing portfolio weights using Markowitz mean-variance framework.

Author: Ankush Arora
Date: 2025-11-24
"""

import numpy as np
import cvxpy as cp
from typing import Dict, Optional
import warnings

from .metrics import calculate_portfolio_return, calculate_portfolio_variance, calculate_sharpe_ratio

# Suppress CVXPY warnings for cleaner output
warnings.filterwarnings('ignore', category=FutureWarning)


def optimize_portfolio(expected_returns: np.ndarray, 
                      cov_matrix: np.ndarray,
                      target_return: Optional[float] = None,
                      allow_short_selling: bool = False) -> Dict[str, any]:
    """
    Optimize portfolio weights to minimize variance for a given target return.
    
    This is the core Markowitz optimization problem:
    
    Minimize: w^T * Σ * w  (portfolio variance)
    Subject to:
        - w^T * μ = target_return  (if specified)
        - w^T * 1 = 1  (weights sum to 1)
        - w >= 0  (if short selling not allowed)
    
    Parameters
    ----------
    expected_returns : np.ndarray
        Expected returns for each asset
    cov_matrix : np.ndarray
        Covariance matrix of asset returns
    target_return : float, optional
        Target portfolio return. If None, minimizes variance without return constraint.
    allow_short_selling : bool, optional
        If True, allows negative weights (short positions). Default: False
        
    Returns
    -------
    dict
        Dictionary containing:
        - 'weights': optimal portfolio weights
        - 'return': portfolio return
        - 'risk': portfolio standard deviation
        - 'variance': portfolio variance
        - 'status': optimization status
        
    Raises
    ------
    ValueError
        If optimization problem is infeasible
        
    Example
    -------
    >>> returns = np.array([0.10, 0.08, 0.12])
    >>> cov = np.array([[0.04, 0.01, 0.02],
    ...                 [0.01, 0.03, 0.01],
    ...                 [0.02, 0.01, 0.05]])
    >>> result = optimize_portfolio(returns, cov, target_return=0.09)
    >>> result['weights'].sum()
    1.0
    """
    n_assets = len(expected_returns)
    
    # Define optimization variable
    weights = cp.Variable(n_assets)
    
    # Objective: Minimize portfolio variance
    # Portfolio variance = w^T * Σ * w, use quadratic form for CVXPY
    portfolio_variance = cp.quad_form(weights, cov_matrix)
    objective = cp.Minimize(portfolio_variance)
    
    # Constraints
    constraints = []
    
    # Constraint 1: Weights must sum to 1 (fully invested)
    constraints.append(cp.sum(weights) == 1)
    
    # Constraint 2: Target return (if specified)
    if target_return is not None:
        portfolio_return = expected_returns @ weights  # Matrix multiplication
        constraints.append(portfolio_return == target_return)
    
    # Constraint 3: No short selling (if required)
    if not allow_short_selling:
        constraints.append(weights >= 0)
    
    # Solve the optimization problem
    problem = cp.Problem(objective, constraints)
    
    try:
        problem.solve(solver=cp.ECOS)  # ECOS is good for quadratic problems
    except:
        # Fallback to another solver if ECOS fails
        try:
            problem.solve(solver=cp.SCS)
        except:
            problem.solve()
    
    # Check if solution is valid
    if problem.status not in ['optimal', 'optimal_inaccurate']:
        raise ValueError(f"Optimization failed with status: {problem.status}")
    
    # Extract results
    optimal_weights = weights.value
    optimal_return = calculate_portfolio_return(optimal_weights, expected_returns)
    optimal_variance = calculate_portfolio_variance(optimal_weights, cov_matrix)
    optimal_std = np.sqrt(optimal_variance)
    
    return {
        'weights': optimal_weights,
        'return': optimal_return,
        'risk': optimal_std,
        'variance': optimal_variance,
        'status': problem.status
    }


def optimize_portfolio_no_short(expected_returns: np.ndarray, 
                                cov_matrix: np.ndarray,
                                target_return: float) -> Dict[str, any]:
    """
    Convenience function: Optimize portfolio without short selling.
    
    Parameters
    ----------
    expected_returns : np.ndarray
        Expected returns for each asset
    cov_matrix : np.ndarray
        Covariance matrix of asset returns
    target_return : float
        Target portfolio return
        
    Returns
    -------
    dict
        Optimization results (see optimize_portfolio)
    """
    return optimize_portfolio(expected_returns, cov_matrix, target_return, 
                            allow_short_selling=False)


def optimize_portfolio_with_short(expected_returns: np.ndarray, 
                                  cov_matrix: np.ndarray,
                                  target_return: float) -> Dict[str, any]:
    """
    Convenience function: Optimize portfolio allowing short selling.
    
    Parameters
    ----------
    expected_returns : np.ndarray
        Expected returns for each asset
    cov_matrix : np.ndarray
        Covariance matrix of asset returns
    target_return : float
        Target portfolio return
        
    Returns
    -------
    dict
        Optimization results (see optimize_portfolio)
    """
    return optimize_portfolio(expected_returns, cov_matrix, target_return, 
                            allow_short_selling=True)


def optimize_max_sharpe(expected_returns: np.ndarray,
                        cov_matrix: np.ndarray,
                        risk_free_rate: float = 0.02,
                        allow_short_selling: bool = False) -> Dict[str, any]:
    """
    Find portfolio with maximum Sharpe Ratio.
    
    This searches across the efficient frontier to find the portfolio
    with the best risk-adjusted return.
    
    Formula: SR = (R_p - R_f) / σ_p
    
    Parameters
    ----------
    expected_returns : np.ndarray
        Expected returns for each asset
    cov_matrix : np.ndarray
        Covariance matrix of asset returns
    risk_free_rate : float, optional
        Risk-free rate (default: 0.02)
    allow_short_selling : bool, optional
        If True, allows short positions (default: False)
        
    Returns
    -------
    dict
        Dictionary with optimal portfolio details including 'sharpe_ratio' key
        
    Raises
    ------
    ValueError
        If maximum Sharpe ratio portfolio cannot be found
        
    Example
    -------
    >>> returns = np.array([0.10, 0.08, 0.12])
    >>> cov = np.array([[0.04, 0.01, 0.02],
    ...                 [0.01, 0.03, 0.01],
    ...                 [0.02, 0.01, 0.05]])
    >>> result = optimize_max_sharpe(returns, cov, risk_free_rate=0.02)
    >>> 'sharpe_ratio' in result
    True
    """
    # Create a range of target returns
    min_return = np.min(expected_returns)
    max_return = np.max(expected_returns)
    target_returns = np.linspace(min_return, max_return, 100)
    
    best_sharpe = -np.inf
    best_result = None
    
    # Search for maximum Sharpe ratio across efficient frontier
    for target_ret in target_returns:
        try:
            result = optimize_portfolio(expected_returns, cov_matrix, 
                                      target_ret, allow_short_selling)
            sharpe = calculate_sharpe_ratio(result['return'], result['risk'], 
                                          risk_free_rate)
            
            if sharpe > best_sharpe:
                best_sharpe = sharpe
                best_result = result
                best_result['sharpe_ratio'] = sharpe
        except:
            continue
    
    if best_result is None:
        raise ValueError("Could not find maximum Sharpe ratio portfolio")
    
    return best_result
