"""
Visualization Module

Functions for plotting efficient frontiers and correlation matrices.

Author: Ankush Arora
Date: 2025-11-24
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, List, Tuple

from .frontier import compute_efficient_frontier
from .optimization import optimize_max_sharpe


def plot_efficient_frontier(expected_returns: np.ndarray,
                           cov_matrix: np.ndarray,
                           asset_names: Optional[List[str]] = None,
                           show_assets: bool = True,
                           show_max_sharpe: bool = True,
                           risk_free_rate: float = 0.02,
                           allow_short_selling: bool = False,
                           figsize: Tuple[int, int] = (12, 8),
                           output_file: Optional[str] = None) -> plt.Figure:
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
    output_file : str, optional
        If specified, save plot to this file path
        
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
    
    # Save if output file specified
    if output_file:
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        print(f"Saved plot to: {output_file}")
    
    return fig


def plot_correlation_matrix(cov_matrix: np.ndarray,
                           asset_names: Optional[List[str]] = None,
                           figsize: Tuple[int, int] = (10, 8),
                           output_file: Optional[str] = None) -> plt.Figure:
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
    output_file : str, optional
        If specified, save plot to this file path
        
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
    
    # Save if output file specified
    if output_file:
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        print(f"Saved plot to: {output_file}")
    
    return fig


def plot_covariance_matrix(cov_matrix: np.ndarray,
                          asset_names: Optional[List[str]] = None,
                          figsize: Tuple[int, int] = (10, 8),
                          output_file: Optional[str] = None) -> plt.Figure:
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
    output_file : str, optional
        If specified, save plot to this file path
        
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
    
    # Save if output file specified
    if output_file:
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        print(f"Saved plot to: {output_file}")
    
    return fig
