"""
Portfolio Metrics Module

Functions for calculating portfolio return, variance, risk, and Sharpe ratio.

Author: Ankush Arora
Date: 2025-11-24
"""

import numpy as np
from typing import Union


def calculate_portfolio_return(weights: np.ndarray, expected_returns: np.ndarray) -> float:
    """
    Calculate expected portfolio return.
    
    Formula: E[R_p] = w^T * μ
    where w is the weight vector and μ is the expected returns vector.
    
    Parameters
    ----------
    weights : np.ndarray
        Portfolio weights for each asset (must sum to 1)
    expected_returns : np.ndarray
        Expected returns for each asset
        
    Returns
    -------
    float
        Expected portfolio return
        
    Example
    -------
    >>> weights = np.array([0.4, 0.3, 0.3])
    >>> returns = np.array([0.10, 0.12, 0.08])
    >>> calculate_portfolio_return(weights, returns)
    0.102
    """
    return np.dot(weights, expected_returns)


def calculate_portfolio_variance(weights: np.ndarray, cov_matrix: np.ndarray) -> float:
    """
    Calculate portfolio variance.
    
    Formula: σ²_p = w^T * Σ * w
    where w is the weight vector and Σ is the covariance matrix.
    
    Parameters
    ----------
    weights : np.ndarray
        Portfolio weights for each asset
    cov_matrix : np.ndarray
        Covariance matrix of asset returns (n x n)
        
    Returns
    -------
    float
        Portfolio variance
        
    Example
    -------
    >>> weights = np.array([0.5, 0.5])
    >>> cov = np.array([[0.04, 0.01], [0.01, 0.09]])
    >>> calculate_portfolio_variance(weights, cov)
    0.03
    """
    return np.dot(weights, np.dot(cov_matrix, weights))


def calculate_portfolio_std(weights: np.ndarray, cov_matrix: np.ndarray) -> float:
    """
    Calculate portfolio standard deviation (volatility/risk).
    
    Formula: σ_p = sqrt(w^T * Σ * w)
    
    Parameters
    ----------
    weights : np.ndarray
        Portfolio weights for each asset
    cov_matrix : np.ndarray
        Covariance matrix of asset returns
        
    Returns
    -------
    float
        Portfolio standard deviation (risk)
        
    Example
    -------
    >>> weights = np.array([0.5, 0.5])
    >>> cov = np.array([[0.04, 0.01], [0.01, 0.09]])
    >>> calculate_portfolio_std(weights, cov)
    0.173...
    """
    return np.sqrt(calculate_portfolio_variance(weights, cov_matrix))


def calculate_sharpe_ratio(portfolio_return: float, 
                          portfolio_std: float, 
                          risk_free_rate: float = 0.02) -> float:
    """
    Calculate Sharpe Ratio - risk-adjusted return metric.
    
    Formula: SR = (R_p - R_f) / σ_p
    where R_p is portfolio return, R_f is risk-free rate, σ_p is portfolio std dev.
    
    Parameters
    ----------
    portfolio_return : float
        Expected portfolio return
    portfolio_std : float
        Portfolio standard deviation (risk)
    risk_free_rate : float, optional
        Risk-free rate of return (default: 0.02 or 2%)
        
    Returns
    -------
    float
        Sharpe Ratio
        
    Example
    -------
    >>> calculate_sharpe_ratio(0.10, 0.15, 0.02)
    0.533...
    """
    if portfolio_std == 0:
        raise ValueError("Portfolio standard deviation cannot be zero")
    
    return (portfolio_return - risk_free_rate) / portfolio_std
