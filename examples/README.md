# Portfolio Optimization Examples

This folder contains example scripts demonstrating how to use the portfolio optimization library.

## Data Files

All example data is stored in the `data/` folder as JSON files:

- **`assets_5class.json`** - Traditional 5-asset class portfolio (US Stocks, Intl Stocks, US Bonds, REITs, Gold)
- **`assets_tech.json`** - Technology sector portfolio (Cloud, AI/ML, Semiconductors, Cybersecurity)

### JSON Format

Each data file contains:
```json
{
  "description": "Portfolio description",
  "asset_names": ["Asset 1", "Asset 2", ...],
  "expected_returns": [0.10, 0.09, ...],
  "covariance_matrix": [[...], [...], ...],
  "risk_free_rate": 0.02,
  "notes": "Additional information"
}
```

## Example Scripts

### 1. basic_optimization.py

Simple portfolio optimization for a target return.

**Usage:**
```bash
python examples/basic_optimization.py
```

**What it does:**
- Loads asset data from JSON
- Optimizes portfolio for 8% target return
- Displays optimal weights and statistics
- No short selling constraint

---

### 2. efficient_frontier_demo.py

Generates and visualizes the efficient frontier.

**Usage:**
```bash
python examples/efficient_frontier_demo.py
```

**What it does:**
- Loads asset data from JSON  
- Generates efficient frontier with and without short selling
- Saves plots to `output/` folder
- Shows maximum Sharpe ratio portfolios

**Output files:**
- `output/efficient_frontier_no_short.png`
- `output/efficient_frontier_with_short.png`

---

### 3. max_sharpe_demo.py

Finds the portfolio with maximum Sharpe ratio.

**Usage:**
```bash
python examples/max_sharpe_demo.py
```

**What it does:**
- Loads asset data from JSON
- Computes max Sharpe portfolios with/without short selling
- Compares weights and statistics
- Visualizes the optimal portfolio on efficient frontier

**Output files:**
- `output/max_sharpe_portfolio.png`

---

## Creating Your Own Examples

1. **Create a new JSON data file** in the `data/` folder:
   ```json
{
     "description": "My portfolio",
     "asset_names": ["Asset A", "Asset B", "Asset C"],
     "expected_returns": [0.08, 0.10, 0.06],
     "covariance_matrix": [
       [0.04, 0.01, 0.00],
       [0.01, 0.09, 0.02],
       [0.00, 0.02, 0.04]
     ],
     "risk_free_rate": 0.02
   }
   ```

2. **Create a new Python script** and load your data:
   ```python
   import json
   import numpy as np
   from portfolio_optimization import optimize_portfolio
   
   # Load data
   with open('data/my_portfolio.json', 'r') as f:
       data = json.load(f)
   
   # Optimize
   result = optimize_portfolio(
       np.array(data['expected_returns']),
       np.array(data['covariance_matrix']),
       target_return=0.08
   )
   
   print(f"Optimal weights: {result['weights']}")
   ```

---

## Requirements

All examples require the virtual environment to be activated:

```bash
# Windows
venv\Scripts\activate

# Run any example
python examples/basic_optimization.py
```

---

## Output Directory

The visualization scripts create an `output/` folder to store generated plots:
- `efficient_frontier_no_short.png`
- `efficient_frontier_with_short.png`
- `max_sharpe_portfolio.png`

All plots are saved at 150 DPI for high quality.
