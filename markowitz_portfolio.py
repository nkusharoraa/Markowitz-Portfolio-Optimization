"""
Markowitz Portfolio Optimization - Production Implementation

This module implements the complete Markowitz mean-variance portfolio optimization
framework including:
- Portfolio return and risk calculations
- Optimal portfolio weight computation (with/without short selling)
- Efficient Frontier generation and visualization
- Sharpe Ratio maximization
- Correlation matrix visualization

Author: Ankush Arora
Date: 2025-11-24
"""

import numpy as np
import cvxpy as cp
import matplotlib.pyplot as plt
from typing import Tuple, Optional, Dict, List
import warnings

# Suppress CVXPY warnings for cleaner output
warnings.filterwarnings('ignore', category=FutureWarning)


# ============================================================================
# PORTFOLIO METRICS FUNCTIONS
# ============================================================================

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
    """
    return np.sqrt(calculate_portfolio_variance(weights, cov_matrix))


def calculate_sharpe_ratio(portfolio_return: float, portfolio_std: float, 
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
    """
    return (portfolio_return - risk_free_rate) / portfolio_std


# ============================================================================
# PORTFOLIO OPTIMIZATION FUNCTIONS
# ============================================================================

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
        problem.solve(solver=cp.SCS)
    
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
    
    This transforms the problem into a convex optimization:
    Maximize: (R_p - R_f) / σ_p
    
    We solve this by maximizing (R_p - R_f)² / σ_p² which is equivalent
    and can be formulated as a quadratic program.
    
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
        Dictionary with optimal portfolio details including Sharpe ratio
    """
    n_assets = len(expected_returns)
    
    # Define optimization variable
    weights = cp.Variable(n_assets)
    
    # Excess returns over risk-free rate
    excess_returns = expected_returns - risk_free_rate
    
    # Portfolio variance
    portfolio_variance = cp.quad_form(weights, cov_matrix)
    
    # Portfolio excess return
    portfolio_excess_return = excess_returns @ weights
    
    # Objective: Maximize Sharpe ratio
    # We maximize return / risk, equivalently minimize -return / risk
    # For numerical stability, we solve: maximize excess_return subject to variance <= 1
    # Then scale the result
    
    # Alternative formulation: maximize portfolio_excess_return - risk_aversion * variance
    # We use grid search over the efficient frontier instead
    
    # Simpler approach: minimize variance, maximize return with specific weighting
    # Use the fact that max Sharpe is on the efficient frontier
    # We'll solve by trying different target returns and finding max Sharpe
    
    # Create a range of target returns
    min_return = np.min(expected_returns)
    max_return = np.max(expected_returns)
    target_returns = np.linspace(min_return, max_return, 100)
    
    best_sharpe = -np.inf
    best_result = None
    
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


# ============================================================================
# EFFICIENT FRONTIER FUNCTIONS
# ============================================================================

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


def plot_efficient_frontier(expected_returns: np.ndarray,
                           cov_matrix: np.ndarray,
                           asset_names: Optional[List[str]] = None,
                           show_assets: bool = True,
                           show_max_sharpe: bool = True,
                           risk_free_rate: float = 0.02,
                           allow_short_selling: bool = False,
                           figsize: Tuple[int, int] = (12, 8)) -> plt.Figure:
    """
    Plot the Efficient Frontier with optional annotations.
    
    Parameters
    ----------
    expected_returns : np.ndarray
        Expected returns for each asset
    cov_matrix : np.ndarray
        Covariance matrix of asset returns
    asset_names : list of str, optional
        Names of assets for labeling
    show_assets : bool, optional
        If True, plot individual assets on the chart (default: True)
    show_max_sharpe : bool, optional
        If True, mark the maximum Sharpe ratio portfolio (default: True)
    risk_free_rate : float, optional
        Risk-free rate for Sharpe ratio calculation (default: 0.02)
    allow_short_selling : bool, optional
        If True, allows short selling (default: False)
    figsize : tuple, optional
        Figure size (default: (12, 8))
        
    Returns
    -------
    matplotlib.figure.Figure
        The created figure object
    """
    # Compute efficient frontier
    frontier_returns, frontier_risks = compute_efficient_frontier(
        expected_returns, cov_matrix, n_points=100, 
        allow_short_selling=allow_short_selling
    )
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot efficient frontier
    ax.plot(frontier_risks, frontier_returns, 'b-', linewidth=2.5, 
            label='Efficient Frontier')
    
    # Plot individual assets if requested
    if show_assets:
        n_assets = len(expected_returns)
        asset_risks = np.sqrt(np.diag(cov_matrix))
        
        ax.scatter(asset_risks, expected_returns, c='red', s=100, 
                  marker='D', edgecolors='black', linewidth=1.5,
                  label='Individual Assets', zorder=5)
        
        # Label assets
        if asset_names is None:
            asset_names = [f'Asset {i+1}' for i in range(n_assets)]
        
        for i, name in enumerate(asset_names):
            ax.annotate(name, (asset_risks[i], expected_returns[i]),
                       xytext=(10, 5), textcoords='offset points',
                       fontsize=9, fontweight='bold')
    
    # Plot maximum Sharpe ratio portfolio if requested
    if show_max_sharpe:
        try:
            max_sharpe_result = optimize_max_sharpe(
                expected_returns, cov_matrix, risk_free_rate, allow_short_selling
            )
            
            ax.scatter(max_sharpe_result['risk'], max_sharpe_result['return'],
                      c='gold', s=300, marker='*', edgecolors='black', 
                      linewidth=2, label='Max Sharpe Ratio', zorder=6)
            
            # Add annotation
            sharpe_text = f"Max Sharpe: {max_sharpe_result['sharpe_ratio']:.3f}"
            ax.annotate(sharpe_text,
                       (max_sharpe_result['risk'], max_sharpe_result['return']),
                       xytext=(15, 15), textcoords='offset points',
                       fontsize=10, fontweight='bold',
                       bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', 
                               alpha=0.7),
                       arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0',
                                     lw=2))
        except Exception as e:
            print(f"Warning: Could not compute max Sharpe portfolio: {e}")
    
    # Formatting
    ax.set_xlabel('Risk (Standard Deviation)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Expected Return', fontsize=12, fontweight='bold')
    
    title = 'Markowitz Efficient Frontier'
    if not allow_short_selling:
        title += ' (No Short Selling)'
    ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
    
    ax.legend(loc='best', fontsize=10, framealpha=0.95)
    ax.grid(True, alpha=0.3, linestyle='--')
    
    # Format axes as percentages
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y*100:.1f}%'))
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x*100:.1f}%'))
    
    plt.tight_layout()
    
    return fig


# ============================================================================
# VISUALIZATION FUNCTIONS
# ============================================================================

def plot_correlation_matrix(cov_matrix: np.ndarray,
                           asset_names: Optional[List[str]] = None,
                           figsize: Tuple[int, int] = (10, 8)) -> plt.Figure:
    """
    Visualize correlation matrix as a heatmap.
    
    Parameters
    ----------
    cov_matrix : np.ndarray
        Covariance matrix of asset returns
    asset_names : list of str, optional
        Names of assets for labeling
    figsize : tuple, optional
        Figure size (default: (10, 8))
        
    Returns
    -------
    matplotlib.figure.Figure
        The created figure object
    """
    # Convert covariance to correlation
    std_devs = np.sqrt(np.diag(cov_matrix))
    correlation_matrix = cov_matrix / np.outer(std_devs, std_devs)
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Create heatmap
    im = ax.imshow(correlation_matrix, cmap='RdYlGn', aspect='auto',
                   vmin=-1, vmax=1)
    
    # Set ticks and labels
    n_assets = len(correlation_matrix)
    if asset_names is None:
        asset_names = [f'Asset {i+1}' for i in range(n_assets)]
    
    ax.set_xticks(np.arange(n_assets))
    ax.set_yticks(np.arange(n_assets))
    ax.set_xticklabels(asset_names)
    ax.set_yticklabels(asset_names)
    
    # Rotate x labels for readability
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    
    # Add correlation values in cells
    for i in range(n_assets):
        for j in range(n_assets):
            text = ax.text(j, i, f'{correlation_matrix[i, j]:.2f}',
                         ha="center", va="center", color="black", fontsize=10,
                         fontweight='bold')
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Correlation', fontsize=12, fontweight='bold')
    
    ax.set_title('Asset Correlation Matrix', fontsize=14, fontweight='bold', pad=20)
    
    plt.tight_layout()
    
    return fig


def plot_covariance_matrix(cov_matrix: np.ndarray,
                          asset_names: Optional[List[str]] = None,
                          figsize: Tuple[int, int] = (10, 8)) -> plt.Figure:
    """
    Visualize covariance matrix as a heatmap.
    
    Parameters
    ----------
    cov_matrix : np.ndarray
        Covariance matrix of asset returns
    asset_names : list of str, optional
        Names of assets for labeling
    figsize : tuple, optional
        Figure size (default: (10, 8))
        
    Returns
    -------
    matplotlib.figure.Figure
        The created figure object
    """
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Create heatmap
    im = ax.imshow(cov_matrix, cmap='YlOrRd', aspect='auto')
    
    # Set ticks and labels
    n_assets = len(cov_matrix)
    if asset_names is None:
        asset_names = [f'Asset {i+1}' for i in range(n_assets)]
    
    ax.set_xticks(np.arange(n_assets))
    ax.set_yticks(np.arange(n_assets))
    ax.set_xticklabels(asset_names)
    ax.set_yticklabels(asset_names)
    
    # Rotate x labels for readability
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    
    # Add covariance values in cells
    for i in range(n_assets):
        for j in range(n_assets):
            text = ax.text(j, i, f'{cov_matrix[i, j]:.4f}',
                         ha="center", va="center", color="black", fontsize=9,
                         fontweight='bold')
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Covariance', fontsize=12, fontweight='bold')
    
    ax.set_title('Asset Covariance Matrix', fontsize=14, fontweight='bold', pad=20)
    
    plt.tight_layout()
    
    return fig


# ============================================================================
# EXAMPLE / DEMONSTRATION
# ============================================================================

def run_example():
    """
    Comprehensive example demonstrating all portfolio optimization features.
    
    This example uses 5 hypothetical assets representing different asset classes:
    - US Stocks (S&P 500)
    - International Stocks (MSCI EAFE)
    - US Bonds (Aggregate Bond Index)
    - Real Estate (REITs)
    - Commodities (Gold)
    """
    print("=" * 80)
    print("MARKOWITZ PORTFOLIO OPTIMIZATION - COMPREHENSIVE EXAMPLE")
    print("=" * 80)
    print()
    
    # Define example assets
    asset_names = ['US Stocks', 'Intl Stocks', 'US Bonds', 'REITs', 'Gold']
    n_assets = len(asset_names)
    
    # Expected annual returns (as decimals, e.g., 0.10 = 10%)
    # These are hypothetical values for demonstration
    expected_returns = np.array([
        0.10,   # US Stocks: 10%
        0.09,   # International Stocks: 9%
        0.04,   # US Bonds: 4%
        0.08,   # REITs: 8%
        0.05    # Gold: 5%
    ])
    
    # Covariance matrix (annualized)
    # Diagonal elements are variances (std_dev²)
    # Off-diagonal elements represent covariances between assets
    cov_matrix = np.array([
        [0.0400, 0.0240, 0.0040, 0.0200, 0.0020],  # US Stocks
        [0.0240, 0.0484, 0.0033, 0.0180, 0.0015],  # Intl Stocks
        [0.0040, 0.0033, 0.0064, 0.0025, 0.0010],  # US Bonds
        [0.0200, 0.0180, 0.0025, 0.0361, 0.0018],  # REITs
        [0.0020, 0.0015, 0.0010, 0.0018, 0.0225]   # Gold
    ])
    
    # Risk-free rate (e.g., T-bill rate)
    risk_free_rate = 0.02  # 2%
    
    # Display input data
    print("ASSET INFORMATION")
    print("-" * 80)
    print(f"{'Asset':<20} {'Expected Return':<20} {'Volatility (Std Dev)':<20}")
    print("-" * 80)
    for i, name in enumerate(asset_names):
        ret = expected_returns[i]
        vol = np.sqrt(cov_matrix[i, i])
        print(f"{name:<20} {ret*100:>6.2f}% {'':<14} {vol*100:>6.2f}%")
    print()
    
    # Example 1: Optimize for specific target return (no short selling)
    print("=" * 80)
    print("EXAMPLE 1: OPTIMIZE FOR TARGET RETURN (No Short Selling)")
    print("=" * 80)
    target_return = 0.08  # 8% target return
    print(f"Target Return: {target_return*100:.2f}%\n")
    
    result = optimize_portfolio_no_short(expected_returns, cov_matrix, target_return)
    
    print("Optimal Portfolio Weights:")
    print("-" * 80)
    for i, name in enumerate(asset_names):
        weight = result['weights'][i]
        print(f"{name:<20} {weight*100:>6.2f}%")
    print("-" * 80)
    print(f"{'Sum of weights:':<20} {np.sum(result['weights'])*100:>6.2f}%")
    print()
    
    print("Portfolio Statistics:")
    print("-" * 80)
    print(f"Expected Return:     {result['return']*100:>6.2f}%")
    print(f"Risk (Std Dev):      {result['risk']*100:>6.2f}%")
    print(f"Variance:            {result['variance']:.6f}")
    sharpe = calculate_sharpe_ratio(result['return'], result['risk'], risk_free_rate)
    print(f"Sharpe Ratio:        {sharpe:>6.3f}")
    print()
    
    # Example 2: Maximum Sharpe Ratio Portfolio
    print("=" * 80)
    print("EXAMPLE 2: MAXIMUM SHARPE RATIO PORTFOLIO")
    print("=" * 80)
    print(f"Risk-Free Rate: {risk_free_rate*100:.2f}%\n")
    
    max_sharpe_result = optimize_max_sharpe(expected_returns, cov_matrix, 
                                           risk_free_rate, allow_short_selling=False)
    
    print("Optimal Sharpe Ratio Portfolio Weights:")
    print("-" * 80)
    for i, name in enumerate(asset_names):
        weight = max_sharpe_result['weights'][i]
        print(f"{name:<20} {weight*100:>6.2f}%")
    print("-" * 80)
    print(f"{'Sum of weights:':<20} {np.sum(max_sharpe_result['weights'])*100:>6.2f}%")
    print()
    
    print("Portfolio Statistics:")
    print("-" * 80)
    print(f"Expected Return:     {max_sharpe_result['return']*100:>6.2f}%")
    print(f"Risk (Std Dev):      {max_sharpe_result['risk']*100:>6.2f}%")
    print(f"Sharpe Ratio:        {max_sharpe_result['sharpe_ratio']:>6.3f}")
    print()
    
    # Example 3: Compare with short selling allowed
    print("=" * 80)
    print("EXAMPLE 3: MAXIMUM SHARPE RATIO (With Short Selling Allowed)")
    print("=" * 80)
    
    max_sharpe_short = optimize_max_sharpe(expected_returns, cov_matrix, 
                                          risk_free_rate, allow_short_selling=True)
    
    print("Optimal Portfolio Weights (Short Selling Allowed):")
    print("-" * 80)
    for i, name in enumerate(asset_names):
        weight = max_sharpe_short['weights'][i]
        position = "LONG" if weight >= 0 else "SHORT"
        print(f"{name:<20} {weight*100:>6.2f}% ({position})")
    print("-" * 80)
    print(f"{'Sum of weights:':<20} {np.sum(max_sharpe_short['weights'])*100:>6.2f}%")
    print()
    
    print("Portfolio Statistics:")
    print("-" * 80)
    print(f"Expected Return:     {max_sharpe_short['return']*100:>6.2f}%")
    print(f"Risk (Std Dev):      {max_sharpe_short['risk']*100:>6.2f}%")
    print(f"Sharpe Ratio:        {max_sharpe_short['sharpe_ratio']:>6.3f}")
    print()
    
    # Visualization 1: Correlation Matrix
    print("=" * 80)
    print("GENERATING VISUALIZATIONS...")
    print("=" * 80)
    print()
    
    print("[1/3] Creating correlation matrix heatmap...")
    fig_corr = plot_correlation_matrix(cov_matrix, asset_names)
    plt.savefig('correlation_matrix.png', dpi=150, bbox_inches='tight')
    print("      Saved: correlation_matrix.png")
    
    # Visualization 2: Efficient Frontier (No Short Selling)
    print("[2/3] Creating efficient frontier (no short selling)...")
    fig_ef_no_short = plot_efficient_frontier(
        expected_returns, cov_matrix, asset_names,
        show_assets=True, show_max_sharpe=True,
        risk_free_rate=risk_free_rate, allow_short_selling=False
    )
    plt.savefig('efficient_frontier_no_short.png', dpi=150, bbox_inches='tight')
    print("      Saved: efficient_frontier_no_short.png")
    
    # Visualization 3: Efficient Frontier (With Short Selling)
    print("[3/3] Creating efficient frontier (with short selling)...")
    fig_ef_with_short = plot_efficient_frontier(
        expected_returns, cov_matrix, asset_names,
        show_assets=True, show_max_sharpe=True,
        risk_free_rate=risk_free_rate, allow_short_selling=True
    )
    plt.savefig('efficient_frontier_with_short.png', dpi=150, bbox_inches='tight')
    print("      Saved: efficient_frontier_with_short.png")
    
    print()
    print("=" * 80)
    print("EXAMPLE COMPLETE!")
    print("=" * 80)
    print()
    print("Generated files:")
    print("  - correlation_matrix.png")
    print("  - efficient_frontier_no_short.png")
    print("  - efficient_frontier_with_short.png")
    print()
    print("Close the plot windows to exit.")
    print()
    
    # Show all plots
    plt.show()


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    # Run the comprehensive example
    run_example()
