# 🚀 Quick Start Guide

**Get your portfolio optimized in under 5 minutes!**

---

## Step 1: Installation (2 minutes)

### Option A: Automated Setup (Windows)

```bash
# Clone/download the repository
cd "Markowitz Portfolio Optimization"

# Run automatic setup
.\setup_env.bat
```

This script will:
- ✅ Create Python virtual environment
- ✅ Install all dependencies
- ✅ Verify installation

### Option B: Manual Setup (All Platforms)

```bash
# Create virtual environment
python -m venv venv

# Activate it
venv\Scripts\activate  # Windows
# or
source venv/bin/activate  # Linux/Mac

# Install dependencies
pip install -r requirements.txt
```

---

## Step 2: Prepare Your Data (1 minute)

Create a CSV file with your historical prices:

```csv
Date,Stock_A,Stock_B,Bond_Fund
2020-01-01,100,50,1000
2020-02-01,102,51,1001
2020-03-01,101,52,1002
...continues...
```

**Requirements:**
- First column: Dates (any format)
- Other columns: Price history for each asset
- Header row with asset names

**Or use our sample data:** `examples/data/sample_prices.csv`

---

## Step 3: Run Optimization (30 seconds)

### Quickest: Auto Mode

```bash
python optimize.py examples/data/sample_prices.csv --auto
```

✅ Uses smart defaults  
✅ No prompts  
✅ Instant results

### Interactive Mode

```bash
python optimize.py your_prices.csv
```

You'll be prompted for:
- Target return (e.g., 10%)
- Short selling preference
- Other options

### Custom Parameters

```bash
python optimize.py your_prices.csv \
    --target 0.12 \
    --short \
    --output my_results
```

---

## Step 4: View Results (<1 minute)

All outputs saved to `output/` (or your specified folder):

### 📊 Visualizations

**`efficient_frontier.png`**
- Beautiful plot showing risk-return trade-off
- Optimal portfolios highlighted
- Professional, publication-ready

**`correlation_matrix.png`**
- Asset correlation heatmap
- Understand diversification benefits

### 📄 Results File

**`optimization_results.txt`**
```
OPTIMAL PORTFOLIO WEIGHTS
------------------------
US Stocks     40.72%
Bonds          6.79%
REITs         37.08%
Gold          15.41%

Expected Return:  15.54%
Risk (Std Dev):    5.24%
Sharpe Ratio:     2.585
```

### 💾 Data Export (Optional)

**`estimated_data.json`** (with `--save-data` flag)
- Reusable parameters
- Can be loaded by other scripts

---

## 🎯 Common Use Cases

### Use Case 1: Quick Analysis

**Goal:** Fast portfolio analysis with defaults

```bash
python optimize.py my_prices.csv --auto
```

**Time:** 30 seconds  
**Best for:** Initial exploration

---

### Use Case 2: Specific Target Return

**Goal:** Optimize for 10% annual return

```bash
python optimize.py my_prices.csv --target 0.10
```

**Prompts:** Short selling preference  
**Best for:** Specific return goals

---

### Use Case 3: Aggressive Strategy

**Goal:** Maximum returns with leverage

```bash
python optimize.py my_prices.csv --target 0.15 --short
```

**Note:** Allows negative weights (short positions)  
**Best for:** Sophisticated investors

---

### Use Case 4: Complete Analysis

**Goal:** Full report with all data saved

```bash
python optimize.py my_prices.csv \
    --auto \
    --save-data \
    --output detailed_analysis
```

**Output:** All visualizations + JSON data  
**Best for:** Detailed documentation

---

## 📚 Next Steps

### 🎓 Learn More

- **[Full CLI Guide](docs/cli-guide.md)** - All options and features
- **[API Reference](README.md#-api-reference)** - Use as Python library
- **[Examples](examples/README.md)** - More use cases

### 💻 Run Examples

```bash
# Try different scenarios
python examples/01_basic_optimization.py
python examples/02_efficient_frontier.py
python examples/03_max_sharpe.py
```

### 🛠️ Use as Library

```python
from portfolio_optimization import optimize_portfolio
import numpy as np

# Your code here
result = optimize_portfolio(returns, cov_matrix, target=0.10)
```

---

## ❓ Troubleshooting

### Issue: "Module not found"

**Solution:** Activate virtual environment
```bash
venv\Scripts\activate  # Windows
```

### Issue: "Target return not achievable"

**Solution:** Check feasible range in output, adjust target or use `--short`

### Issue: "CSV parsing error"

**Solution:** Verify CSV format:
- First column is dates
- Has header row
- No missing values

### Issue: "No module named 'colorama'"

**Solution:** Install dependencies
```bash
pip install -r requirements.txt
```

---

## 🎉 Success Checklist

- [x] Environment set up
- [x] Dependencies installed
- [x] CSV data prepared (or using sample)
- [x] First optimization completed
- [x] Visualizations generated
- [x] Results reviewed

**Congratulations! You're ready to optimize portfolios!** 🚀

---

## 💡 Pro Tips

1. **Data Quality Matters:** More historical data = better estimates
2. **Diversification:** Include different asset classes
3. **Review Correlations:** Check correlation matrix for diversification
4. **Sharpe Ratio:** Higher is better for risk-adjusted returns
5. **Regular Updates:** Re-optimize as market conditions change

---

## 🆘 Need Help?

- Review **[Full Documentation](README.md)**
- Check **[CLI Guide](docs/cli-guide.md)**
- See **[FAQ](docs/covariance_estimation.md)**

---

<div align="center">

**Ready to start?**

```bash
python optimize.py your_prices.csv
```

</div>
