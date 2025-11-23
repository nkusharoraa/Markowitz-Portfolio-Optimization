# Estimating Covariance Matrix from Historical Data

## Quick Answer

In practice, you calculate the covariance matrix from **historical price data**. Here's the simple version:

```python
import pandas as pd

# 1. Get historical prices
prices = pd.DataFrame({
    'Stock_A': [100, 102, 101, 103, 105],
    'Stock_B': [50, 51, 50.5, 52, 53]
})

# 2. Calculate returns
returns = prices.pct_change().dropna()

# 3. Calculate covariance (annualized)
cov_matrix = returns.cov() * 252  # 252 trading days/year
```

---

## Detailed Methods

### Method 1: Using the Data Utils Module

We've created a helper module for you:

```python
from portfolio_optimization.data_utils import create_portfolio_json_from_prices
import pandas as pd

# Load your historical prices
prices = pd.read_csv('prices.csv', index_col='Date', parse_dates=True)

# Automatically calculate everything
portfolio_data = create_portfolio_json_from_prices(
    prices,
    frequency='daily',  # or 'weekly', 'monthly'
    risk_free_rate=0.02,
    description='My Portfolio'
)

# Save to JSON
import json
with open('examples/data/my_portfolio.json', 'w') as f:
    json.dump(portfolio_data, f, indent=2)
```

This calculates:
- **Expected returns**: Annualized mean of historical returns
- **Covariance matrix**: Annualized covariance of returns
- Saves in the correct JSON format

---

### Method 2: Step-by-Step Calculation

```python
import pandas as pd
import numpy as np

# Step 1: Load historical prices (example)
prices = pd.DataFrame({
    'US_Stocks': [100, 102, 101, 103, 105, 107],
    'Bonds': [1000, 1001, 1002, 1001, 1003, 1004],
    'Gold': [1800, 1795, 1800, 1805, 1810, 1815]
})

# Step 2: Calculate daily returns
returns = prices.pct_change().dropna()

# Step 3: Calculate expected returns (annualized)
expected_returns = returns.mean() * 252  # 252 trading days

# Step 4: Calculate covariance matrix (annualized)
cov_matrix = returns.cov() * 252

print("Expected Returns:")
print(expected_returns)

print("\nCovariance Matrix:")
print(cov_matrix)
```

---

### Method 3: From Real Data Sources

#### Using Yahoo Finance (requires `yfinance` package)

```python
import yfinance as yf
import pandas as pd
from portfolio_optimization.data_utils import create_portfolio_json_from_prices

# Download historical data
tickers = ['SPY', 'AGG', 'GLD']  # ETFs for stocks, bonds, gold
data = yf.download(tickers, start='2020-01-01', end='2024-01-01')['Adj Close']

# Create portfolio data
portfolio_data = create_portfolio_json_from_prices(
    data,
    asset_names=['US Stocks', 'US Bonds', 'Gold'],
    frequency='daily',
    description='Portfolio from Yahoo Finance data'
)
```

---

## Running the Example

We've created a complete example script:

```bash
python examples/estimate_from_prices.py
```

This will:
1. Create sample monthly price data
2. Calculate returns
3. Estimate expected returns and covariance matrix
4. Display the results
5. Save to `examples/data/estimated_portfolio.json`

---

## Understanding Annualization

**Why multiply by 252 (or 52, or 12)?**

- Daily returns → Annual statistics: multiply by 252 (trading days/year)
- Weekly returns → Annual: multiply by 52 (weeks/year)
- Monthly returns → Annual: multiply by 12 (months/year)

This converts your historical period statistics to annualized values.

---

## Where the Example Data Came From

The covariance matrices in `assets_5class.json` and `assets_tech.json` were created using:

1. **Assumed annual volatilities** for each asset
2. **Assumed correlations** between assets
3. **Formula**: `Cov(A,B) = Corr(A,B) × Vol(A) × Vol(B)`

For example, in `assets_5class.json`:
- US Stocks: 20% volatility → variance = 0.04
- Correlation with Intl Stocks: 0.60
- Covariance = 0.60 × 0.20 × 0.22 = 0.0264

But in practice, you'd calculate these from real data as shown above!

---

## Files Created

- **`portfolio_optimization/data_utils.py`** - Utility functions for data estimation
- **`examples/estimate_from_prices.py`** - Complete working example

Try running the example to see it in action!
