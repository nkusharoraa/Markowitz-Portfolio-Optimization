# Markowitz Portfolio Optimization

A complete, production-quality Python implementation of the **Markowitz Mean-Variance Portfolio Optimization** framework.

## 📋 Features

✅ **Portfolio Metrics**
- Expected return calculation
- Portfolio variance and risk (standard deviation)
- Sharpe ratio computation

✅ **Optimization Algorithms**
- Minimize variance for target return
- No short selling constraint option
- Short selling allowed option
- Maximum Sharpe ratio optimization

✅ **Efficient Frontier**
- Complete frontier computation
- Publication-quality visualization
- Individual asset plotting
- Maximum Sharpe ratio highlighting

✅ **Visualization Tools**
- Correlation matrix heatmap
- Covariance matrix heatmap
- Professional matplotlib charts

✅ **Production Quality**
- Type hints throughout
- Comprehensive docstrings
- Extensive inline comments
- Error handling
- Working examples

---

## 🚀 Quick Start

### 1. Setup Virtual Environment

Run the automated setup script (one-time setup):

```bash
.\setup_env.bat
```

This will:
- Create a Python virtual environment
- Install all required dependencies (numpy, scipy, cvxpy, matplotlib, pandas)
- Prepare the environment for running the optimization

### 2. Run the Example

```bash
# Activate the virtual environment
venv\Scripts\activate

# Run the portfolio optimization script
python markowitz_portfolio.py
```

### 3. View Results

The script will:
- Display detailed portfolio statistics in the console
- Generate three PNG visualization files:
  - `correlation_matrix.png` - Asset correlation heatmap
  - `efficient_frontier_no_short.png` - Efficient frontier without short selling
  - `efficient_frontier_with_short.png` - Efficient frontier with short selling allowed
- Show interactive matplotlib plots (close windows to exit)

---

## 📊 Example Output

The included example optimizes a portfolio of 5 asset classes:
- **US Stocks** (S&P 500) - 10% expected return
- **International Stocks** (MSCI EAFE) - 9% expected return
- **US Bonds** (Aggregate Bond Index) - 4% expected return
- **REITs** (Real Estate) - 8% expected return
- **Gold** (Commodities) - 5% expected return

### Sample Results

**Optimized Portfolio for 8% Target Return:**
- Optimal weights calculated to minimize risk
- Portfolio statistics displayed (return, risk, Sharpe ratio)
- All weights sum to 100%

**Maximum Sharpe Ratio Portfolio:**
- Best risk-adjusted return combination
- Highlighted on efficient frontier plot

---

## 📖 Documentation

### Main Functions

#### Portfolio Metrics
```python
calculate_portfolio_return(weights, expected_returns)
calculate_portfolio_variance(weights, cov_matrix)
calculate_portfolio_std(weights, cov_matrix)
calculate_sharpe_ratio(portfolio_return, portfolio_std, risk_free_rate)
```

#### Optimization
```python
optimize_portfolio(expected_returns, cov_matrix, target_return, allow_short_selling)
optimize_portfolio_no_short(expected_returns, cov_matrix, target_return)
optimize_portfolio_with_short(expected_returns, cov_matrix, target_return)
optimize_max_sharpe(expected_returns, cov_matrix, risk_free_rate, allow_short_selling)
```

#### Efficient Frontier
```python
compute_efficient_frontier(expected_returns, cov_matrix, n_points, allow_short_selling)
plot_efficient_frontier(expected_returns, cov_matrix, asset_names, ...)
```

#### Visualization
```python
plot_correlation_matrix(cov_matrix, asset_names)
plot_covariance_matrix(cov_matrix, asset_names)
```

---

## 🔧 Customization

### Use Your Own Data

Edit the `run_example()` function in `markowitz_portfolio.py`:

```python
# Define your assets
asset_names = ['Asset A', 'Asset B', 'Asset C']

# Set expected returns (annual, as decimals)
expected_returns = np.array([0.08, 0.10, 0.06])

# Define covariance matrix
cov_matrix = np.array([
    [0.04, 0.01, 0.00],
    [0.01, 0.09, 0.02],
    [0.00, 0.02, 0.04]
])

# Run optimization
result = optimize_portfolio_no_short(expected_returns, cov_matrix, target_return=0.08)
```

### Adjust Parameters

```python
# Change target return
target_return = 0.09  # 9%

# Change risk-free rate
risk_free_rate = 0.03  # 3%

# Allow short selling
optimize_portfolio(returns, cov, target, allow_short_selling=True)

# Adjust efficient frontier resolution
compute_efficient_frontier(returns, cov, n_points=200)
```

---

## 📦 Dependencies

All dependencies are automatically installed via `setup_env.bat`:

- **numpy** (≥1.24.0) - Numerical computations
- **scipy** (≥1.10.0) - Statistical functions
- **cvxpy** (≥1.4.0) - Convex optimization solver
- **matplotlib** (≥3.7.0) - Visualization
- **pandas** (≥2.0.0) - Data structures (optional)

---

## 📐 Mathematical Background

### Mean-Variance Optimization

**Objective:** Minimize portfolio variance for a target return

```
Minimize: σ²_p = w^T * Σ * w

Subject to:
  w^T * μ = R_target    (target return constraint)
  w^T * 1 = 1           (weights sum to 1)
  w ≥ 0                 (no short selling, optional)
```

Where:
- `w` = portfolio weights
- `Σ` = covariance matrix
- `μ` = expected returns vector
- `R_target` = target portfolio return

### Sharpe Ratio

```
SR = (R_p - R_f) / σ_p
```

Where:
- `R_p` = portfolio return
- `R_f` = risk-free rate
- `σ_p` = portfolio standard deviation

---

## 🎯 Use Cases

- **Portfolio Management** - Optimize asset allocation for institutional or retail portfolios
- **Risk Analysis** - Understand risk-return tradeoffs across different asset mixes
- **Research** - Study effects of constraints (short selling, position limits) on optimal portfolios
- **Education** - Learn and visualize modern portfolio theory concepts
- **Backtesting** - Historical analysis of optimal portfolio construction

---

## 📝 File Structure

```
Markowitz Portfolio Optimization/
│
├── markowitz_portfolio.py      # Main implementation (all functions)
├── requirements.txt            # Python dependencies
├── setup_env.bat              # Virtual environment setup script
├── venv/                      # Virtual environment (created by setup)
│
└── Generated outputs:
    ├── correlation_matrix.png
    ├── efficient_frontier_no_short.png
    └── efficient_frontier_with_short.png
```

---

## 🔬 Technical Details

**Optimization Solver:** CVXPY with ECOS/SCS backend

**Numerical Precision:** 64-bit floating point (NumPy default)

**Constraint Handling:** Quadratic programming with linear constraints

**Error Handling:** Graceful handling of infeasible problems and numerical issues

---

## 🎓 Next Steps

### For Production Use
1. Replace example data with real historical returns and covariances
2. Add functions to estimate parameters from price data
3. Implement transaction cost modeling
4. Add position size limits or sector constraints
5. Create backtesting framework

### Advanced Features
- Black-Litterman model (incorporate investor views)
- Risk parity portfolios
- Monte Carlo simulation for uncertainty analysis
- Multi-period rebalancing strategies
- Factor-based optimization

---

## 📄 License

This code is provided for educational and professional use. Adapt and modify as needed for your specific applications.

---

## 🙋 Support

All functions include comprehensive docstrings with:
- Detailed parameter descriptions
- Return value specifications
- Mathematical formulas
- Usage examples

Review the inline comments in `markowitz_portfolio.py` for implementation details.

---

## ✨ Author

Ankush Arora

**Version:** 1.0  
**Date:** 2025-11-24
