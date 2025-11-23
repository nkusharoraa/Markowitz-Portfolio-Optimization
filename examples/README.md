# Portfolio Optimization Examples

This folder contains ready-to-run example scripts demonstrating different aspects of portfolio optimization.

---

## 🎯 Quick Start

Run any example:

```bash
# Activate virtual environment first
venv\Scripts\activate

# Run any example
python examples/01_basic_optimization.py
python examples/02_efficient_frontier.py
python examples/03_max_sharpe.py
python examples/04_estimate_from_prices.py
```

---

## 📁 Example Scripts

### 01. Basic Optimization

**File:** `01_basic_optimization.py`

**What it does:**
- Loads portfolio data from JSON
- Optimizes for 8% target return (no short selling)
- Displays optimal weights and statistics

**Output:**
```
Optimal weights: [40.72%, 14.68%, 6.79%, 14.82%, 22.99%]
Expected Return: 8.00%
Risk (Std Dev): 12.93%
Sharpe Ratio: 0.464
```

**Best for:** Understanding basic portfolio optimization

---

### 02. Efficient Frontier Visualization

**File:** `02_efficient_frontier.py`

**What it does:**
- Generates efficient frontier plots
- Compares scenarios with/without short selling
- Saves high-quality visualizations

**Output files:**
- `output/efficient_frontier_no_short.png`
- `output/efficient_frontier_with_short.png`

**Best for:** Visualizing the risk-return trade-off


---

### 03. Maximum Sharpe Ratio

**File:** `03_max_sharpe.py`

**What it does:**
- Finds portfolio with maximum Sharpe ratio
- Compares strategies with/without short selling
- Shows detailed portfolio composition

**Output:**
```
Max Sharpe Portfolio (No Shorts):
  Sharpe Ratio: 3.035
  Expected Return: 7.69%
  Risk: 1.88%
```

**Best for:** Finding the best risk-adjusted portfolio

---

### 04. Estimate from Price History

**File:** `04_estimate_from_prices.py`

**What it does:**
- Creates sample monthly price data
- Calculates expected returns and covariance
- Generates new JSON portfolio file
- Shows the complete estimation process

**Output file:**
- `data/estimated_portfolio.json`

**Best for:** Learning how to calculate parameters from price data

---

## 📊 Data Files

### Sample Price History

**File:** `data/sample_prices.csv`

4 years of monthly historical prices for 5 assets:
- US Stocks
- International Stocks
- US Bonds
- REITs
- Gold

Use with CLI tool:
```bash
python optimize.py examples/data/sample_prices.csv --auto
```

---

### Pre-Configured Portfolios

#### `data/assets_5class.json`

Traditional 5-asset diversified portfolio:

| Asset | Expected Return | Volatility |
|-------|----------------|------------|
| US Stocks | 10.0% | 20.0% |
| Intl Stocks | 9.0% | 22.0% |
| US Bonds | 4.0% | 8.0% |
| REITs | 8.0% | 19.0% |
| Gold | 5.0% | 15.0% |

**Risk-free rate:** 2.0%

---

#### `data/assets_tech.json`

High-growth technology sector portfolio:

| Asset | Expected Return | Volatility |
|-------|----------------|------------|
| Cloud Computing | 18.0% | 35.0% |
| AI/ML | 20.0% | 40.0% |
| Semiconductors | 16.0% | 32.0% |
| Cybersecurity | 15.0% | 30.0% |

**Risk-free rate:** 2.5%

---

## 🎨 Output Examples

All visualization examples create plots in `output/` folder with:
- ✨ Modern color schemes
- 📊 Professional formatting
- 🎯 High resolution (200 DPI)
- 📈 Publication-ready quality

---

## 🛠️ Creating Your Own Examples

### Step 1: Create Data File

Create `data/my_portfolio.json`:

```json
{
  "description": "My Custom Portfolio",
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

### Step 2: Create Script

```python
import json
import numpy as np
from portfolio_optimization import optimize_portfolio

# Load data
with open('examples/data/my_portfolio.json', 'r') as f:
    data = json.load(f)

# Optimize
result = optimize_portfolio(
    np.array(data['expected_returns']),
    np.array(data['covariance_matrix']),
    target_return=0.08
)

# Display results
for name, weight in zip(data['asset_names'], result['weights']):
    print(f"{name}: {weight*100:.2f}%")
```

---

## 💡 Tips

1. **Start with 01:** Begin with basic optimization to understand the workflow
2. **Visualize:** Run example 02 to see the efficient frontier
3. **Customize:** Modify JSON files to test your own portfolios
4. **Learn:** Read the code - it's well-commented and clear
5. **Experiment:** Try different target returns and constraints

---

## 📚 Next Steps

- **[CLI Tool](../docs/cli-guide.md)** - For quick analysis
- **[API Reference](../README.md#-api-reference)** - For custom integration
- **[Quick Start](../QUICK_START.md)** - Complete getting started guide

---

## ❓ FAQ

**Q: Can I use my own data?**  
A: Yes! Either create a JSON file or use the CLI with CSV prices.

**Q: How do I change the target return?**  
A: Edit the `target_return` parameter in the script.

**Q: Where are the plots saved?**  
A: In `examples/output/` directory (auto-created).

**Q: Can I save results to a file?**  
A: Yes! Use `output_file` parameter in plot functions.

---

<div align="center">

**Ready to try the examples?**

```bash
python examples/01_basic_optimization.py
```

</div>

