"""
Portfolio Optimization Library

A modular implementation of Markowitz Mean-Variance Portfolio Optimization.

Author: Ankush Arora
Date: 2025-11-24
"""

from .metrics import (
    calculate_portfolio_return,
    calculate_portfolio_variance,
    calculate_portfolio_std,
    calculate_sharpe_ratio
)

from .optimization import (
    optimize_portfolio,
    optimize_portfolio_no_short,
    optimize_portfolio_with_short,
    optimize_max_sharpe
)

from .frontier import (
    compute_efficient_frontier
)

from .visualization import (
    plot_efficient_frontier,
    plot_correlation_matrix,
    plot_covariance_matrix
)

__all__ = [
    # Metrics
    'calculate_portfolio_return',
    'calculate_portfolio_variance',
    'calculate_portfolio_std',
    'calculate_sharpe_ratio',
    
    # Optimization
    'optimize_portfolio',
    'optimize_portfolio_no_short',
    'optimize_portfolio_with_short',
    'optimize_max_sharpe',
    
    # Frontier
    'compute_efficient_frontier',
    
    # Visualization
    'plot_efficient_frontier',
    'plot_correlation_matrix',
    'plot_covariance_matrix'
]

__version__ = '2.0.0'
__author__ = 'Ankush Arora'
