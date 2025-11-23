<div align="center">

# 📊 Markowitz Portfolio Optimization

### *Professional Mean-Variance Portfolio Optimization in Python*

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg?logo=python&logoColor=white)](https://www.python.org)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Code style](https://img.shields.io/badge/Code%20style-Professional-brightgreen)]()
[![Status](https://img.shields.io/badge/Status-Production%20Ready-success)]()
[![GitHub Pages](https://img.shields.io/badge/Demo-Live-orange)](https://nkusharoraa.github.io/Markowitz-Portfolio-Optimization/)

**Clean** • **Modular** • **User-Friendly** • **Beautiful Visualizations** • **Web App Available**

[Quick Start](#-quick-start) •
[Web App](https://nkusharoraa.github.io/portfolio-optimization/) •
[Features](#-features) •
[Documentation](#-documentation) •
[Examples](#-examples)

---

</div>

## 🎯 What This Does

Transform your **CSV price data** into **optimized investment portfolios**:

**💻 Python CLI:**
```bash
python optimize.py your_prices.csv
```

**🌐 Web App (No Installation):**
Visit: [Markowitz Portfolio Optimization](https://nkusharoraa.github.io/Markowitz-Portfolio-Optimization/)

Get professional-quality:
- ✨ **Optimal portfolio weights** minimizing risk for your target return
- 📈 **Beautiful visualizations** (efficient frontier, correlation matrices)
- 📊 **Maximum Sharpe ratio** portfolios
- 💾 **Complete analysis reports** ready for presentation

---

## ⚡ Quick Start

### 🌐 Option A: Web App (Fastest - No Installation!)

1. Visit **[Web Application](https://nkusharoraa.github.io/Markowitz-Portfolio-Optimization/)**
2. Enter your asset data or upload CSV
3. Click "Optimize Portfolio"
4. Get instant results!

**Perfect for:**  quick analysis, demonstrations, learning

---

### 💻 Option B: Python CLI (Most Powerful)

### 1️⃣ Setup (One-Time)

```bash
git clone <repository>
cd "Markowitz Portfolio Optimization"
.\setup_env.bat  # Creates virtual environment & installs dependencies
```

### 2️⃣ Run Optimization

**Option A: Use Your Own Price Data**
```bash
python optimize.py your_prices.csv
```

**Option B: Try the Example**
```bash
python optimize.py examples/data/sample_prices.csv --auto
```

**Option C: Advanced Usage**
```bash
python optimize.py prices.csv --target 0.12 --short --save-data
```

### 3️⃣ View Results

All outputs saved to `output/` folder:
- 🖼️ `efficient_frontier.png` - Professional visualization
- 🔢 `correlation_matrix.png` - Asset relationships  
- 📄 `optimization_results.txt` - Complete analysis
- 💾 `estimated_data.json` - Reusable parameters (optional)

---

## ✨ Features

<table>
<tr>
<td width="50%">

### 🎨 **Beautiful Visualizations**
- Modern color schemes
- Professional typography
- Publication-ready plots
- Accessible color palettes
- Clean, minimalist design

</td>
<td width="50%">

### 🧮 **Comprehensive Analytics**
- Expected returns calculation
- Covariance matrix estimation
- Efficient frontier generation
- Sharpe ratio optimization
- Risk-return trade-off analysis

</td>
</tr>
<tr>
<td>

### 🛠️ **Easy to Use**
- Simple CLI tool
- Interactive prompts
- Auto-detects data frequency
- Helpful error messages
- Step-by-step guidance

</td>
<td>

### 📦 **Modular Architecture**
- Clean code structure
- Importable as library
- Well-documented API
- Type hints throughout
- Professional standards

</td>
</tr>
</table>

---

## 🚀 Three Ways to Use

### Method 1: CLI Tool (Easiest!)

Perfect for quick analysis:

```bash
python optimize.py prices.csv
```

**Supports:**
- ✅ Interactive mode (prompts for options)
- ✅ Automatic mode (`--auto` flag)
- ✅ Custom target returns (`--target 0.10`)
- ✅ Short selling (`--short` flag)
- ✅ Custom output location (`--output results/`)

📖 **[Complete CLI Guide](CLI_GUIDE.md)**

---

### Method 2: Python Scripts

Ready-to-run examples in `examples/` folder:

```bash
cd examples
python 01_basic_optimization.py
python 02_efficient_frontier.py  
python 03_max_sharpe.py
```

Each script demonstrates different aspects of portfolio optimization.

---

### Method 3: Import as Library

Use in your own Python code:

```python
from portfolio_optimization import (
    optimize_portfolio,
    plot_efficient_frontier
)

# Optimize
result = optimize_portfolio(
    returns, cov_matrix,
    target_return=0.10
)

# Visualize
plot_efficient_frontier(
    returns, cov_matrix,
    asset_names=['Stock', 'Bond', 'Gold']
)
```

📖 **[API Documentation](#-api-reference)**

---

## 📂 Project Structure

```
📁 Markowitz Portfolio Optimization/
│
├── 📁 portfolio_optimization/    # Core library
│   ├── metrics.py               # Portfolio calculations
│   ├── optimization.py          # Optimization algorithms
│   ├── frontier.py              # Efficient frontier
│   ├── visualization.py         # Beautiful plots ✨
│   └── data_utils.py            # Parameter estimation
│
├── 📁 examples/                 # Example scripts & data
│   ├── 📁 data/                 # Sample datasets
│   │   ├── sample_prices.csv   # Example price history
│   │   ├── assets_5class.json  # 5-asset portfolio
│   │   └── assets_tech.json    # Tech sector portfolio
│   └── 📁 scripts/              # Ready-to-run examples
│
├── 📁 docs/                     # Documentation
│   ├── covariance_estimation.md
│   └── (more guides)
│
├── optimize.py                  # 🎯 Main CLI tool
├── CLI_GUIDE.md                 # Complete CLI documentation
├── README.md                    # You are here!
├── requirements.txt             # Python dependencies
└── setup_env.bat                # Environment setup
```

---

## 📊 Example Output

**Input:** CSV file with historical prices

**What You Get:**

1. **Optimal Portfolio Weights**
   ```
   US Stocks      40.72%
   Bonds           6.79%
   REITs          37.08%
   Gold           15.41%
   ```

2. **Portfolio Statistics**
   ```
   Expected Return:  15.54%
   Risk (Std Dev):    5.24%
   Sharpe Ratio:     2.585
   ```

3. **Professional Visualizations**
   - Efficient frontier with gradient styling
   - Color-coded asset markers
   - Maximum Sharpe ratio highlighted
   - Modern, clean design

---

## 🎓 How It Works

```
┌─────────────┐
│ Price CSV   │
└──────┬──────┘
       │
       ▼
┌─────────────────────┐
│  Auto-Calculate:    │
│  • Returns          │
│  • Covariance       │
│  • Correlations     │
└──────┬──────────────┘
       │
       ▼
┌─────────────────────┐
│  Optimize Markowitz:│
│  • Min variance     │
│  • Target return    │
│  • Max Sharpe       │
└──────┬──────────────┘
       │
       ▼
┌─────────────────────┐
│  Professional       │
│  Visualizations     │
└─────────────────────┘
```

---

## 💡 CSV Format

Your price file should look like this:

```csv
Date,Stock_A,Stock_B,Bond,Gold
2020-01-01,100,50,1000,1800
2020-02-01,102,51,1001,1795
2020-03-01,101,52,1002,1810
...
```

**Requirements:**
- ✅ First column: Dates
- ✅ Other columns: Asset prices
- ✅ Header row with names

The tool **automatically detects** whether your data is daily, weekly, or monthly!

---

## 🛠️ Installation

### Requirements

- Python 3.8 or higher
- Windows (batch script) or any OS with Python

### Quick Install

```bash
# Clone repository
git clone <repository>
cd "Markowitz Portfolio Optimization"

# Run setup
.\setup_env.bat  # Windows
# or manually: python -m venv venv && pip install -r requirements.txt
```

### Dependencies

All automatically installed by setup script:
- `numpy` - Numerical computation
- `scipy` - Statistical functions
- `cvxpy` - Convex optimization
- `matplotlib` - Visualizations
- `pandas` - Data handling
- `colorama` - Colored terminal output

---

## 📚 Documentation

| Guide | Description |
|-------|-------------|
| **[CLI Guide](CLI_GUIDE.md)** | Complete CLI tool documentation |
| **[Covariance Estimation](docs/covariance_estimation.md)** | How to calculate parameters |
| **[Examples README](examples/README.md)** | Example scripts guide |
| **[API Reference](#-api-reference)** | Library function documentation |

---

## 📖 API Reference

### Core Modules

```python
from portfolio_optimization import (
    # Metrics
    calculate_portfolio_return,
    calculate_portfolio_variance,
    calculate_sharpe_ratio,
    
    # Optimization
    optimize_portfolio,
    optimize_max_sharpe,
    
    # Visualization
    plot_efficient_frontier,
    plot_correlation_matrix,
    
    # Data Utilities
    estimate_expected_returns,
    estimate_covariance_matrix,
)
```

### Quick API Example

```python
import numpy as np
from portfolio_optimization import optimize_portfolio

# Your data
returns = np.array([0.10, 0.08, 0.12])
cov_matrix = np.array([
    [0.04, 0.01, 0.02],
    [0.01, 0.03, 0.01],
    [0.02, 0.01, 0.05]
])

# Optimize
result = optimize_portfolio(
    expected_returns=returns,
    cov_matrix=cov_matrix,
    target_return=0.09,
    allow_short_selling=False
)

print(f"Weights: {result['weights']}")
print(f"Risk: {result['risk']*100:.2f}%")
print(f"Sharpe: {result['sharpe_ratio']:.3f}")
```

---

## 💻 Examples

### Example 1: Basic Optimization

```bash
python examples/basic_optimization.py
```

Demonstrates simple portfolio optimization for a target return.

### Example 2: Efficient Frontier

```bash
python examples/efficient_frontier_demo.py
```

Generates beautiful efficient frontier visualizations.

### Example 3: Maximum Sharpe Ratio

```bash
python examples/max_sharpe_demo.py
```

Finds the best risk-adjusted portfolio.

### Example 4: Estimate from Prices

```bash
python examples/estimate_from_prices.py
```

Shows how to calculate parameters from historical data.

---

## 🎨 Visual Enhancements

### Modern Plot Aesthetics

- **Professional color schemes** (blues, purples, golds)
- **Clean typography** (Segoe UI, modern fonts)
- **Accessible colors** (colorblind-friendly palettes)
- **Gradient effects** on plots
- **Shadow effects** for depth
- **Clean boundaries** (no top/right spines)
- **High DPI** (200 DPI for crisp output)

### Styled Terminal Output

- ✅ **Green checks** for success
- ⚠️ **Yellow warnings**
- ✗ **Red errors**
- ℹ️ **Blue info**
- **Aligned tables** for better readability

---

## 🤝 Contributing

Contributions welcome! This is a clean, professional codebase with:
- Type hints throughout
- Comprehensive docstrings
- Modular architecture
- Well-organized structure

---

## 📄 License

MIT License - feel free to use in your projects!

---

## 👤 Author

**Ankush Arora**

---

## 🌟 Key Highlights

✨ **Production-Ready** - Clean, modular, professional code  
✨ **User-Friendly** - Simple CLI, great documentation  
✨ **Beautiful** - Modern visualizations, professional aesthetics  
✨ **Comprehensive** - Full Markowitz framework implementation  
✨ **Flexible** - Use as CLI, library, or modify examples  
✨ **Well-Documented** - Extensive guides and API docs  

---

<div align="center">

### Ready to optimize your portfolio?

```bash
python optimize.py your_prices.csv
```

**[Get Started](#-quick-start)** • **[View Examples](#-examples)** • **[Read Docs](#-documentation)**

</div>
