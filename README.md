# Risk-Neutral Density Estimation from Equity Options

[![CI](https://github.com/rxj0102/Risk-Neutral-Density/actions/workflows/ci.yml/badge.svg)](https://github.com/rxj0102/Risk-Neutral-Density/actions/workflows/ci.yml)

Extract and compare the market-implied risk-neutral density (RND) from live AAPL option chains using three methods - Breeden-Litzenberger, SVI, and SSVI - and benchmark it against the historical return distribution.

---

## Problem & Methodology

Options prices embed the market's full probability distribution over future returns - the **risk-neutral density** (RND) under the risk-neutral measure Q.  
This project recovers that distribution via three increasingly structured approaches:

| Method | Type | Reference |
|---|---|---|
| **Breeden-Litzenberger** | Model-free | Breeden & Litzenberger (1978) |
| **SVI** | 5-parameter parametric smile | Gatheral (2004) |
| **SSVI** | Arbitrage-free surface (butterfly constraint) | Gatheral & Jacquier (2014) |

The pipeline:
1. Fetch live option chain from Yahoo Finance; filter by open interest and bid-ask spread.
2. Stitch OTM puts (K < F) and OTM calls (K ≥ F) for maximum liquidity.
3. Compute Black-Scholes implied vols via Brent's method.
4. **B-L**: PCHIP-spline call-equivalent prices → second derivative → RND.
5. **SVI / SSVI**: Fit total-variance parametrization → recover RND via B-L identity in log-moneyness space.
6. Construct historical (P-measure) density from 5-year rolling 30-day log-returns + KDE.
7. Compute tail metrics (VaR, CVaR, skewness, crash probabilities) and the **Variance Risk Premium (VRP)**.

---

## Key Results (AAPL, 30-day expiry, April 2026)

| Metric | B-L | SVI | SSVI | Historical (P) |
|---|---|---|---|---|
| ATM IV | - | 30.81% | 30.61% | - |
| Mean S_T | $254.31 | $249.69 | $249.73 | $259.19 |
| Std S_T | $19.59 | $29.05 | $28.92 | $22.47 |
| Skewness | −0.59 | −0.34 | −0.37 | −0.02 |
| 95% VaR | $35.58 | $55.68 | $55.87 | $30.68 |
| 95% CVaR | $47.47 | $59.70 | $59.80 | $41.13 |

**VRP**: Implied vol (SSVI) = 30.61% vs. realised = 27.37%, a gap of 3.24 points (1.25x variance ratio).

> **Caveats.** These figures come from one live option-chain snapshot taken in April 2026 and stored in `notebooks/04_risk_neutral_density.ipynb`; a rerun uses a new snapshot and will give different numbers. A single positive variance risk premium is the normal state of equity options, so it is not by itself a trading signal, and no backtest of a short-volatility rule is included here. The Breeden-Litzenberger standard deviation ($19.59) is well below the SVI/SSVI value ($29) because the model-free density is truncated to the observed strike range, so tail metrics are not directly comparable across methods.  
The implied Q-density is left-skewed and fat-tailed relative to the empirical P-density, quantifying the crash risk premium embedded in options prices.

---

## Tech Stack

- **Python 3.10+**
- `numpy`, `scipy` - numerical integration, optimisation, spline interpolation
- `pandas` - option chain handling
- `yfinance` - live market data (options, rates, dividends)
- `matplotlib` - visualisation
- `pytest` - unit testing

---

## Repository Structure

```
Risk-Neutral-Density/
├── src/
│   └── rnd/
│       ├── __init__.py       # package exports
│       ├── data.py           # yfinance wrappers: expiry, rates, chain
│       ├── preprocessing.py  # OTM stitching, Black-Scholes implied vol
│       ├── models.py         # SVI and SSVI parametrizations + calibration
│       ├── density.py        # B-L, SVI, SSVI RND extraction
│       ├── metrics.py        # VaR, CVaR, tail probs, VRP
│       └── plotting.py       # reusable figure helpers
├── notebooks/
│   └── 01_risk_neutral_density.ipynb   # end-to-end analysis
├── tests/
│   ├── conftest.py
│   ├── test_models.py        # SVI/SSVI calibration round-trips
│   ├── test_density.py       # density normalisation & shape tests
│   └── test_metrics.py       # moment and risk-metric accuracy
├── data/                     # reserved for cached chain snapshots
├── requirements.txt
└── LICENSE
```

---

## How to Run

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Run the notebook

```bash
cd notebooks
jupyter notebook 01_risk_neutral_density.ipynb
```

The notebook fetches live data from Yahoo Finance on execution - no local data files required.

### 3. Run the test suite

```bash
pytest tests/ -v
```

---

## Research-Level Improvements

Three directions that would materially advance this work:

### 1. Jump-diffusion / rough-volatility RND
The B-L and SVI densities underfit the left tail for short expiries because diffusive models cannot produce sufficient negative skewness in ≤30 days. Implementing a Kou double-exponential jump model or the rough Bergomi model (Bayer et al., 2016) would capture the volatility smile's power-law explosion near expiry and produce more realistic crash probabilities.

### 2. Cross-sectional SSVI surface & calendar-spread arbitrage detection
The current analysis fits a single expiry slice in isolation. Extending to a joint calibration across multiple expiries using the power-law SSVI surface (Gatheral & Jacquier, 2014, §4.3) would enforce no-calendar-spread arbitrage across the term structure, enabling forward variance curve extraction and a richer view of the term structure of variance risk premium.

### 3. Entropic / Bayesian RND regularisation
Model-free B-L densities can be noisy and oscillatory with real market data. A Tikhonov-regularised or maximum-entropy approach (Buchen & Kelly, 1996; Jackwerth & Rubinstein, 1996) would impose smoothness as a prior while remaining non-parametric, providing confidence intervals on the density and making crash-probability estimates more robust to bid-ask noise.
