# Portfolio Optimization CLI - Quick Start Guide

## Overview

The `optimize.py` script provides a complete end-to-end solution for portfolio optimization from historical price data. Simply provide a CSV file with prices, and the tool will:

1. **Calculate** expected returns and covariance matrix automatically
2. **Optimize** portfolio for your target return
3. **Find** the maximum Sharpe ratio portfolio
4. **Generate** visualizations (efficient frontier, correlation matrix)
5. **Save** all results to text and image files

---

## Quick Start

### Basic Usage

```bash
python optimize.py prices.csv
```

This will:
- Load your price data
- Auto-detect data frequency (daily/weekly/monthly)
- Calculate parameters
- **Prompt you interactively** for target return and options
- Generate all outputs

### Fully Automatic (No Prompts)

```bash
python optimize.py prices.csv --auto
```

Uses defaults:
- Target return = mean of asset returns
- No short selling
- Risk-free rate = 2%

### Specify Target Return

```bash
python optimize.py prices.csv --target 0.10
```

Sets a 10% annual target return.

---

## CSV Format

Your price file should look like this:

```csv
Date,Asset1,Asset2,Asset3
2020-01-01,100,50,200
2020-02-01,102,51,198
2020-03-01,101,52,205
...
```

**Requirements:**
- First column: Dates (any format pandas can parse)
- Other columns: Prices for each asset
- Header row with asset names

**Sample file provided:** `examples/data/sample_prices.csv`

---

## Command Line Options

### Data Options

```
--frequency {daily,weekly,monthly}
```
Specify data frequency (auto-detected if not provided)

```
--risk-free RATE
```
Set risk-free rate (default: 0.02 = 2%)

### Optimization Options

```
--target RETURN
```
Target annual return (e.g., 0.10 for 10%)

```
--short
```
Allow short selling (negative weights)

```
--auto
```
Run without interactive prompts (use defaults)

### Output Options

```
--output DIRECTORY
```
Output directory (default: `output/`)

```
--save-data
```
Save estimated returns and covariance to JSON

---

## Complete Example

```bash
python optimize.py examples/data/sample_prices.csv \
    --target 0.12 \
    --short \
    --risk-free 0.025 \
    --output my_results \
    --save-data
```

This will:
- Load `sample_prices.csv`
- Optimize for 12% target return
- Allow short selling
- Use 2.5% risk-free rate
- Save to `my_results/` folder
- Save estimated data as JSON

---

## Output Files

The tool generates:

### 1. `efficient_frontier.png`
- Visual plot showing the efficient frontier
- Individual assets marked
- Maximum Sharpe ratio portfolio highlighted

### 2. `correlation_matrix.png`
- Heatmap showing correlations between assets
- Useful for understanding diversification

### 3. `optimization_results.txt`
- Complete text summary
- Optimal weights for both portfolios
- All statistics (returns, risk, Sharpe ratio)

### 4. `estimated_data.json` (if `--save-data` used)
- Calculated expected returns
- Covariance matrix
- Can be used with other scripts

---

## Example Output

```
================================================================================
PORTFOLIO OPTIMIZATION FROM PRICE HISTORY
================================================================================

--------------------------------------------------------------------------------
1. LOADING PRICE DATA
--------------------------------------------------------------------------------
✓ Loaded 48 rows of price data
✓ Assets: US_Stocks, Intl_Stocks, US_Bonds, REITs, Gold
✓ Date range: 2020-01-01 to 2023-12-01
✓ Detected frequency: monthly

--------------------------------------------------------------------------------
2. ESTIMATING PARAMETERS
--------------------------------------------------------------------------------
✓ Expected Annual Returns:
  US_Stocks             21.71%
  Intl_Stocks           20.63%
  US_Bonds               3.92%
  REITs                 16.92%
  Gold                  14.54%

✓ Annual Volatility:
  US_Stocks             16.97%
  ...

--------------------------------------------------------------------------------
3. OPTIMIZATION PARAMETERS
--------------------------------------------------------------------------------
Target return: 15.54% (mean of assets)
✓ Short selling: Not allowed
✓ Risk-free rate: 2.00%

--------------------------------------------------------------------------------
4. OPTIMIZATION RESULTS
--------------------------------------------------------------------------------

✓ Optimal Portfolio Weights:
  US_Stocks              1.66%
  Intl_Stocks           -0.00%
  US_Bonds               0.00%
  REITs                 37.08%
  Gold                  61.25%

  Expected Return:     15.54%
  Risk (Std Dev):       5.24%
  Sharpe Ratio:        2.585

✓ Maximum Sharpe Ratio Portfolio:
  US_Bonds              67.50%
  REITs                 13.48%
  Gold                  19.03%

  Sharpe Ratio:        3.035

All results saved to: output/
```

---

##Common Use Cases

### 1. Quick Analysis

```bash
python optimize.py mydata.csv --auto
```

Fast automated analysis with defaults.

### 2. Conservative Portfolio (No Shorts)

```bash
python optimize.py mydata.csv --target 0.08
```

Conservative 8% target, long-only positions.

### 3. Aggressive Portfolio (With Shorts)

```bash
python optimize.py mydata.csv --target 0.15 --short
```

15% target with leverage/shorts allowed.

### 4. Maximum Sharpe (Focus on Risk-Adjusted Returns)

```bash
python optimize.py mydata.csv --auto
```

The tool always computes max Sharpe automatically.

---

## Troubleshooting

### "Target return not achievable"

**Error:** Target return outside feasible range

**Solution:**
- Check feasible range shown in output
- Use a target between min and max asset returns
- Consider enabling `--short` for more flexibility
- Verify data frequency is correct

### "Error loading CSV"

**Problem:** CSV format not recognized

**Solution:**
- Ensure first column is dates
- Use header row with column names
- Check date format is parseable

### Returns seem too high/low

**Problem:** Frequency detection wrong

**Solution:**
- Manually specify: `--frequency monthly`
- Check your data actually matches the frequency

---

## Advanced Usage

### Save Configuration for Reuse

```bash
# First run: estimate and save
python optimize.py prices.csv --save-data --output run1

# Later: use the saved JSON with other scripts
python examples/basic_optimization.py
# (modify to load run1/estimated_data.json)
```

### Batch Processing

```bash
# Process multiple files
for file in data/*.csv; do
    python optimize.py "$file" --auto --output "results/$(basename $file .csv)"
done
```

---

## Next Steps

1. **Try the sample data:**
   ```bash
   python optimize.py examples/data/sample_prices.csv --auto
   ```

2. **Use your own data:**
   - Export price history from your broker/platform
   - Format as CSV
   - Run optimization!

3. **Explore the library:**
   - Check `examples/` for more advanced usage
   - Import specific functions for custom analysis
   - See `README.md` for full API reference

---

## Help

View all options:
```bash
python optimize.py --help
```

For more detailed documentation, see:
- `README.md` - Full project documentation
- `examples/README.md` - Example scripts
- `docs/covariance_estimation.md` - Parameter estimation details
