"""
Visualization Module - Enhanced with Modern Aesthetics

Professional plotting functions with beautiful color schemes and typography.

Author: Ankush Arora
Date: 2025-11-24
Version: 2.0 (Enhanced UX)
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import patheffects
from typing import Optional, List, Tuple
import warnings

from .frontier import compute_efficient_frontier
from .optimization import optimize_max_sharpe

# Suppress warnings
warnings.filterwarnings('ignore')

# ============================================================================
# MODERN COLOR SCHEME
# ============================================================================

COLORS = {
    # Primary palette - professional and accessible
    'primary': '#2E86AB',        # Deep blue
    'secondary': '#A23B72',      # Purple accent
    'success': '#06A77D',        # Emerald green
    'warning': '#F18F01',        # Vibrant orange
    'danger': '#C73E1D',         # Deep red
    'neutral': '#6C757D',        # Slate gray
    
    # Extended palette for plots
    'navy': '#1B3B6F',
    'teal': '#048A81',
    'coral': '#FF6B6B',
    'gold': '#FFD700',
    'lavender': '#9B59B6',
    
    # Gradients (for efficient frontier)
    'gradient_start': '#667EEA',
    'gradient_end': '#764BA2',
    
    # Backgrounds
    'bg_light': '#F8F9FA',
    'bg_dark': '#212529',
}

# Asset colors (for individual asset markers)
ASSET_COLORS = ['#E74C3C', '#3498DB', '#2ECC71', '#F39C12', '#9B59B6', '#1ABC9C']

# ============================================================================
# STYLING CONFIGURATION
# ============================================================================

def apply_modern_style():
    """Apply modern matplotlib styling."""
    plt.rcParams.update({
        # Figure
        'figure.facecolor': 'white',
        'figure.edgecolor': 'none',
        'figure.dpi': 150,
        
        # Axes
        'axes.facecolor': 'white',
        'axes.edgecolor': '#CCCCCC',
        'axes.linewidth': 1.2,
        'axes.grid': True,
        'axes.axisbelow': True,
        'axes.labelsize': 11,
        'axes.titlesize': 14,
        'axes.titleweight': 'bold',
        'axes.labelweight': 'bold',
        'axes.spines.top': False,
        'axes.spines.right': False,
        
        # Grid
        'grid.color': '#E0E0E0',
        'grid.linestyle': '--',
        'grid.linewidth': 0.8,
        'grid.alpha': 0.7,
        
        # Lines
        'lines.linewidth': 2.5,
        'lines.antialiased': True,
        
        # Fonts
        'font.family': 'sans-serif',
        'font.sans-serif': ['Segoe UI', 'Arial', 'DejaVu Sans', 'sans-serif'],
        'font.size': 10,
        
        # Legend
        'legend.frameon': True,
        'legend.framealpha': 0.95,
        'legend.facecolor': 'white',
        'legend.edgecolor': '#CCCCCC',
        'legend.fontsize': 10,
        'legend.title_fontsize': 11,
        
        # Ticks
        'xtick.labelsize': 9,
        'ytick.labelsize': 9,
        'xtick.color': '#333333',
        'ytick.color': '#333333',
    })


def add_text_shadow(text_obj, color='white', linewidth=3):
    """Add subtle shadow effect to text for better readability."""
    text_obj.set_path_effects([
        patheffects.Stroke(linewidth=linewidth, foreground=color, alpha=0.7),
        patheffects.Normal()
    ])


# ============================================================================
# ENHANCED PLOTTING FUNCTIONS
# ============================================================================

def plot_efficient_frontier(expected_returns: np.ndarray,
                           cov_matrix: np.ndarray,
                           asset_names: Optional[List[str]] = None,
                           show_assets: bool = True,
                           show_max_sharpe: bool = True,
                           risk_free_rate: float = 0.02,
                           allow_short_selling: bool = False,
                           figsize: Tuple[int, int] = (14, 9),
                           output_file: Optional[str] = None) -> plt.Figure:
    """
    Plot the Efficient Frontier with modern, professional aesthetics.
    
    Enhanced with:
    - Beautiful gradient colors
    - Modern typography
    - Clean styling (no top/right spines)
    - Professional annotations
    - Accessible color schemes
    """
    # Apply modern styling
    apply_modern_style()
    
    # Compute efficient frontier
    frontier_returns, frontier_risks = compute_efficient_frontier(
        expected_returns, cov_matrix, n_points=100, 
        allow_short_selling=allow_short_selling
    )
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    fig.patch.set_facecolor('white')
    
    # Plot efficient frontier with gradient effect
    ax.plot(frontier_risks, frontier_returns, 
            color=COLORS['primary'], linewidth=3.5, 
            label='Efficient Frontier', zorder=3,
            alpha=0.9, path_effects=[patheffects.SimpleLineShadow(alpha=0.3),
                                     patheffects.Normal()])
    
    # Add subtle fill under the curve
    ax.fill_between(frontier_risks, frontier_returns, alpha=0.1, 
                    color=COLORS['primary'])
    
    # Plot individual assets if requested
    if show_assets:
        n_assets = len(expected_returns)
        asset_risks = np.sqrt(np.diag(cov_matrix))
        
        if asset_names is None:
            asset_names = [f'Asset {i+1}' for i in range(n_assets)]
        
        # Plot each asset with unique color
        for i, (name, risk, ret) in enumerate(zip(asset_names, asset_risks, expected_returns)):
            color = ASSET_COLORS[i % len(ASSET_COLORS)]
            
            ax.scatter(risk, ret, c=color, s=200, 
                      marker='o', edgecolors='white', linewidth=2.5,
                      label=name, zorder=5, alpha=0.9)
            
            # Add label with shadow for readability
            text = ax.annotate(name, (risk, ret),
                              xytext=(12, 8), textcoords='offset points',
                              fontsize=10, fontweight='600',
                              color=color,
                              bbox=dict(boxstyle='round,pad=0.4', 
                                       facecolor='white', 
                                       edgecolor=color,
                                       alpha=0.9, linewidth=1.5))
    
    # Plot maximum Sharpe ratio portfolio if requested
    if show_max_sharpe:
        try:
            max_sharpe_result = optimize_max_sharpe(
                expected_returns, cov_matrix, risk_free_rate, allow_short_selling
            )
            
            # Star marker for max Sharpe
            ax.scatter(max_sharpe_result['risk'], max_sharpe_result['return'],
                      c=COLORS['gold'], s=500, marker='*', 
                      edgecolors='#B8860B', linewidth=2.5,
                      label='Maximum Sharpe Ratio', zorder=10,
                      path_effects=[patheffects.withStroke(linewidth=4, 
                                                           foreground='white')])
            
            # Annotation with professional styling
            sharpe_text = f"★ Max Sharpe\nRatio: {max_sharpe_result['sharpe_ratio']:.3f}"
            annotation = ax.annotate(sharpe_text,
                       (max_sharpe_result['risk'], max_sharpe_result['return']),
                       xytext=(20, 20), textcoords='offset points',
                       fontsize=11, fontweight='bold',
                       color='#B8860B',
                       bbox=dict(boxstyle='round,pad=0.6', 
                                facecolor=COLORS['gold'], 
                                edgecolor='#B8860B',
                                alpha=0.95, linewidth=2),
                       arrowprops=dict(arrowstyle='->', 
                                     connectionstyle='arc3,rad=0.2',
                                     color='#B8860B', lw=2.5,
                                     alpha=0.8))
        except Exception as e:
            print(f"Note: Could not compute max Sharpe portfolio")
    
    # Formatting
    ax.set_xlabel('Risk (Annual Volatility)', fontsize=13, 
                  fontweight='bold', color='#333333')
    ax.set_ylabel('Expected Annual Return', fontsize=13, 
                  fontweight='bold', color='#333333')
    
    title = 'Portfolio Efficient Frontier'
    if not allow_short_selling:
        title += ' (Long-Only Strategy)'
    else:
        title += ' (With Short Selling)'
    
    ax.set_title(title, fontsize=16, fontweight='bold', 
                pad=20, color='#1a1a1a')
    
    # Format axes as percentages with better styling
    ax.yaxis.set_major_formatter(plt.FuncFormatter(
        lambda y, _: f'{y*100:.1f}%'))
    ax.xaxis.set_major_formatter(plt.FuncFormatter(
        lambda x, _: f'{x*100:.1f}%'))
    
    # Legend with modern styling
    legend = ax.legend(loc='lower right', fontsize=10, 
                      framealpha=0.98, edgecolor='#CCCCCC',
                      title='Assets & Portfolios', title_fontsize=11)
    legend.get_frame().set_linewidth(1.2)
    
    # Add subtle background grid
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.8)
    
    # Set axis limits with padding
    x_margin = (frontier_risks.max() - frontier_risks.min()) * 0.1
    y_margin = (frontier_returns.max() - frontier_returns.min()) * 0.1
    ax.set_xlim(min(frontier_risks.min(), asset_risks.min()) - x_margin,
                max(frontier_risks.max(), asset_risks.max()) + x_margin)
    ax.set_ylim(min(frontier_returns.min(), expected_returns.min()) - y_margin,
                max(frontier_returns.max(), expected_returns.max()) + y_margin)
    
    plt.tight_layout()
    
    # Save if output file specified
    if output_file:
        plt.savefig(output_file, dpi=200, bbox_inches='tight', 
                   facecolor='white', edgecolor='none')
        print(f"  ✓ Saved: {output_file}")
    
    return fig


def plot_correlation_matrix(cov_matrix: np.ndarray,
                           asset_names: Optional[List[str]] = None,
                           figsize: Tuple[int, int] = (11, 9),
                           output_file: Optional[str] = None) -> plt.Figure:
    """
    Visualize correlation matrix with modern heatmap styling.
    
    Enhanced with:
    - Beautiful color gradient (RdYlGn)
    - Clean annotations
    - Professional typography    - Modern grid styling
    """
    apply_modern_style()
    
    # Convert covariance to correlation
    std_devs = np.sqrt(np.diag(cov_matrix))
    correlation_matrix = cov_matrix / np.outer(std_devs, std_devs)
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    fig.patch.set_facecolor('white')
    
    # Create heatmap with diverging colormap
    im = ax.imshow(correlation_matrix, cmap='RdYlGn', aspect='auto',
                   vmin=-1, vmax=1, interpolation='nearest')
    
    # Set ticks and labels
    n_assets = len(correlation_matrix)
    if asset_names is None:
        asset_names = [f'Asset {i+1}' for i in range(n_assets)]
    
    ax.set_xticks(np.arange(n_assets))
    ax.set_yticks(np.arange(n_assets))
    ax.set_xticklabels(asset_names, fontsize=11, fontweight='600')
    ax.set_yticklabels(asset_names, fontsize=11, fontweight='600')
    
    # Rotate x labels for readability
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", 
            rotation_mode="anchor")
    
    # Add correlation values in cells with smart coloring
    for i in range(n_assets):
        for j in range(n_assets):
            corr_val = correlation_matrix[i, j]
            # Use black for light backgrounds, white for dark
            text_color = 'white' if abs(corr_val) > 0.5 else 'black'
            
            text = ax.text(j, i, f'{corr_val:.2f}',
                         ha="center", va="center", 
                         color=text_color, fontsize=11,
                         fontweight='bold')
    
    # Add colorbar with modern styling
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Correlation Coefficient', fontsize=12, 
                  fontweight='bold', color='#333333')
    cbar.ax.tick_params(labelsize=10, colors='#333333')
    
    ax.set_title('Asset Correlation Matrix', fontsize=16, 
                fontweight='bold', pad=20, color='#1a1a1a')
    
    # Add subtle frame
    for spine in ax.spines.values():
        spine.set_edgecolor('#CCCCCC')
        spine.set_linewidth(1.5)
    
    plt.tight_layout()
    
    # Save if output file specified
    if output_file:
        plt.savefig(output_file, dpi=200, bbox_inches='tight',
                   facecolor='white', edgecolor='none')
        print(f"  ✓ Saved: {output_file}")
    
    return fig


def plot_covariance_matrix(cov_matrix: np.ndarray,
                          asset_names: Optional[List[str]] = None,
                          figsize: Tuple[int, int] = (11, 9),
                          output_file: Optional[str] = None) -> plt.Figure:
    """
    Visualize covariance matrix with modern heatmap styling.
    """
    apply_modern_style()
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    fig.patch.set_facecolor('white')
    
    # Create heatmap
    im = ax.imshow(cov_matrix, cmap='YlOrRd', aspect='auto',
                   interpolation='nearest')
    
    # Set ticks and labels
    n_assets = len(cov_matrix)
    if asset_names is None:
        asset_names = [f'Asset {i+1}' for i in range(n_assets)]
    
    ax.set_xticks(np.arange(n_assets))
    ax.set_yticks(np.arange(n_assets))
    ax.set_xticklabels(asset_names, fontsize=11, fontweight='600')
    ax.set_yticklabels(asset_names, fontsize=11, fontweight='600')
    
    # Rotate x labels for readability
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
            rotation_mode="anchor")
    
    # Add covariance values in cells
    for i in range(n_assets):
        for j in range(n_assets):
            cov_val = cov_matrix[i, j]
            # Smart text coloring based on background
            max_val = cov_matrix.max()
            text_color = 'white' if cov_val > max_val * 0.6 else 'black'
            
            text = ax.text(j, i, f'{cov_val:.4f}',
                         ha="center", va="center", 
                         color=text_color, fontsize=10,
                         fontweight='bold')
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Covariance', fontsize=12, 
                  fontweight='bold', color='#333333')
    cbar.ax.tick_params(labelsize=10, colors='#333333')
    
    ax.set_title('Asset Covariance Matrix', fontsize=16, 
                fontweight='bold', pad=20, color='#1a1a1a')
    
    # Add subtle frame
    for spine in ax.spines.values():
        spine.set_edgecolor('#CCCCCC')
        spine.set_linewidth(1.5)
    
    plt.tight_layout()
    
    # Save if output file specified
    if output_file:
        plt.savefig(output_file, dpi=200, bbox_inches='tight',
                   facecolor='white', edgecolor='none')
        print(f"  ✓ Saved: {output_file}")
    
    return fig


# Initialize styling on import
apply_modern_style()
