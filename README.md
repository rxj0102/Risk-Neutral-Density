# Risk-Neutral Density Estimation from Equity Options

Extract and compare the market-implied risk-neutral density (RND) from live AAPL option chains using three methods — Breeden-Litzenberger, SVI, and SSVI — and benchmark it against the historical return distribution.

---

## Beginner-Friendly Explanation

### The Core Idea: What Does the Market Think Will Happen?

Imagine you want to know what professional traders think the price of Apple stock (AAPL) will be in one month. You could ask them, but they won't tell you their private forecasts. However, they *do* trade options — and option prices secretly encode their collective probability beliefs about the future stock price.

This project **reverse-engineers those beliefs** from live option prices, producing a full probability curve called the **Risk-Neutral Density (RND)**. Think of the RND as a histogram showing how likely each future stock price is, according to the options market.

---

### Key Concepts (Plain English)

#### What is an Option?

An **option** is a contract that gives you the *right* (but not the obligation) to buy or sell a stock at a predetermined price (the **strike price**, K) on a future date (the **expiry**).

- A **call option** = right to *buy* at strike K. It's valuable if the stock rises above K.
- A **put option** = right to *sell* at strike K. It's valuable if the stock falls below K.

The price you pay today for this right (the **option premium**) depends on how likely the market thinks that outcome is. This is the key insight: **option prices are bets on the future**, and their prices reflect the market's probability assessments.

#### What is Implied Volatility?

The **Black-Scholes model** is the standard formula for pricing options. It takes inputs like current stock price, strike, time, and interest rate — plus one unknown: **volatility** (how wildly the stock might move).

If you *observe* the market price of an option and *solve backwards* for the volatility that the Black-Scholes formula would need to produce that price, you get the **implied volatility (IV)**. This is the market's forecast of future price turbulence.

Plotting IV across different strikes gives you the **volatility smile** or **volatility skew** — which looks like a curved line rather than a flat one. That curvature tells us the market assigns *more* probability to large drops than a simple normal distribution would predict (crash risk premium).

#### What is the Forward Price?

The **forward price** (F) is what the stock is *expected* to be worth at expiry under the risk-free assumption:

```
F = S₀ × e^((r - q) × τ)
```

Where:
- `S₀` = current stock price
- `r` = risk-free interest rate (e.g. 3-month Treasury yield)
- `q` = dividend yield (Apple pays dividends)
- `τ` = time to expiry in years (e.g. 30 days = 30/365 ≈ 0.082)

#### What is Log-Moneyness?

Instead of working with raw strike prices, we often use **log-moneyness**:

```
k = log(K / F)
```

- `k = 0`: the strike equals the forward price (at-the-money)
- `k < 0`: the strike is below the forward (out-of-the-money put)
- `k > 0`: the strike is above the forward (out-of-the-money call)

This centering makes the math cleaner and symmetrical.

#### What is Total Variance?

**Total variance** is `w = σ² × τ` — it combines implied volatility `σ` and time `τ` into one number. Fitting models in total-variance space is numerically more stable than fitting in volatility space directly.

---

### The Pipeline Step by Step

Here is what the code does from start to finish:

```
1. Fetch live option chain (yfinance)
         ↓
2. Filter for quality (open interest, bid-ask spread)
         ↓
3. Stitch OTM puts + OTM calls (preprocessing.py)
         ↓
4. Compute Black-Scholes implied vols (preprocessing.py)
         ↓
5. Extract Risk-Neutral Density via three methods (density.py + models.py)
         ↓
6. Build historical (real-world) density from 5-year stock returns
         ↓
7. Compute tail risk metrics and Variance Risk Premium (metrics.py)
```

**Step 3 — OTM Stitching:** Out-of-the-money options have tighter bid-ask spreads (cheaper, more liquid). So we use OTM *puts* for strikes below the forward, and OTM *calls* for strikes above. This gives the cleanest price signal.

**Step 4 — Implied Vol:** For each option, the code uses **Brent's method** (a robust root-finding algorithm) to solve for the `σ` that makes the Black-Scholes formula match the observed market price.

---

### The Three Methods for Extracting the RND

#### Method 1: Breeden-Litzenberger (Model-Free)

**The insight (1978):** The risk-neutral density is literally the **second derivative of the call price with respect to strike**:

```
f^Q(K) = e^(rτ) × ∂²C/∂K²
```

**In plain English:** If you had a smooth curve of call prices plotted against strikes, the curvature (how fast the slope is changing) at each point tells you the probability density at that strike. Steep curvature = high probability.

**How the code does it:**
1. Takes the market call prices at discrete strikes.
2. Fits a smooth **PCHIP spline** (a type of curve that never goes below zero and is shape-preserving) through those prices.
3. Takes the second derivative of that spline.
4. Multiplies by `e^(rτ)` to discount back from expiry.
5. Clips negative values to zero (probabilities can't be negative) and normalises so the total area = 1.

**Pros:** No model assumptions — purely driven by market prices.  
**Cons:** Noisy. Sparse market data means the spline can wiggle.

#### Method 2: SVI (Stochastic Volatility Inspired)

**The idea (Gatheral, 2004):** Instead of fitting call prices directly, fit a smooth parametric curve to the *implied volatility smile* across strikes. Then use that smooth curve to derive the density.

The SVI formula for total variance is:

```
w(k) = a + b × [ρ(k − m) + √((k − m)² + σ²)]
```

**What each parameter means:**
- `a`: the overall level of variance (shifts the whole curve up/down)
- `b`: controls how steeply the wings of the smile rise
- `ρ` (rho): the **skew** — negative values tilt the smile leftward, meaning puts are more expensive (crash fear)
- `m`: the log-moneyness where the minimum occurs
- `σ` (sigma): how curved/smooth the bottom of the smile is

**How calibration works:** The code uses **L-BFGS-B** (a numerical optimiser) to find the 5 parameter values that minimise the squared error between the SVI curve and the observed market implied vols. This is a standard machine learning-style optimisation.

**Once fitted:** The smooth total-variance curve is used to compute call prices on a fine grid, then the Breeden-Litzenberger second-derivative formula is applied to get the RND.

#### Method 3: SSVI (Surface SVI — Arbitrage-Free)

**The idea (Gatheral & Jacquier, 2014):** SVI has 5 free parameters, which gives flexibility but no guarantee the resulting density is arbitrage-free (i.e., it could theoretically produce negative probabilities across expiries). SSVI is a restricted version with only **3 parameters** that is mathematically guaranteed to be **free of butterfly arbitrage**:

```
w(k) = (θ/2) × [1 + ρφk + √((φk + ρ)² + 1 − ρ²)]
```

**What each parameter means:**
- `θ` (theta): the ATM total variance (exactly `w(0) = θ` by construction)
- `φ` (phi): controls the curvature of the smile wings
- `ρ` (rho): the skew correlation (same interpretation as in SVI)

**Calibration:** `θ` is pinned directly from market data (ATM interpolation), then `φ` and `ρ` are jointly optimised. The constraint `φ × θ ≤ 4` enforces no-butterfly-arbitrage.

**In plain English:** SSVI is a tighter, more well-behaved version of SVI. It trades some flexibility for mathematical cleanliness.

---

### The Historical (Real-World) Density

The RND above is measured under the **risk-neutral (Q) measure** — the world where all assets earn the risk-free rate (a mathematical fiction used for pricing). But investors actually live in the **real-world (P) measure**.

To compare, the code also builds a **historical density** from 5 years of actual daily stock returns, computing overlapping 30-day log-returns and smoothing them with **Kernel Density Estimation (KDE)** — essentially a smooth histogram.

The gap between Q and P reveals the **crash risk premium**: options traders demand extra payment for insuring against tail events, making the Q-density more left-skewed and fat-tailed than historical experience alone would justify.

---

### The Risk Metrics

After computing each density, `metrics.py` computes:

| Metric | What it means |
|--------|--------------|
| **Mean** | Expected future stock price (probability-weighted average) |
| **Std** | Spread of outcomes (uncertainty) |
| **Skewness** | Asymmetry. Negative = left tail is heavier (crash risk) |
| **Excess Kurtosis** | Fat tails beyond a normal distribution |
| **Prob Down 10%** | Probability stock falls more than 10% |
| **Prob Up 10%** | Probability stock rises more than 10% |
| **Prob Crash 20%** | Probability of a crash worse than −20% |
| **VaR 95%** | Dollar loss that is exceeded only 5% of the time |
| **CVaR 95%** | *Average* dollar loss in the worst 5% of scenarios |

**VaR** (Value at Risk) asks: "What is the worst loss I'd expect 95% of the time?" CVaR (Conditional VaR, also called Expected Shortfall) asks the harder question: "If I *am* in that bad 5%, how bad is it on average?" CVaR is more conservative and preferred by regulators.

---

### The Variance Risk Premium (VRP)

The **VRP** is the single most important output for a trader:

```
VRP (additive) = Implied Vol − Realised Vol
VRP (ratio)    = (Implied Vol / Realised Vol)²
```

If implied vol (30.61%) > realised vol (27.37%), the market is *overpaying* for volatility insurance. This is a signal to **sell volatility** (e.g. sell straddles, write options). The VRP is the structural reason why options sellers earn a risk premium over time — they bear the risk of rare crashes.

---

### Code Structure at a Glance

| File | What it does |
|------|-------------|
| `data.py` | Downloads live option chains, interest rates, and dividends from Yahoo Finance |
| `preprocessing.py` | Filters options, stitches OTM puts/calls, computes implied vols via Brent's method |
| `models.py` | Defines the SVI and SSVI total-variance formulas and calibrates them with L-BFGS-B optimisation |
| `density.py` | Extracts the RND from each model using the Breeden-Litzenberger formula (direct spline or via total-variance grid) |
| `metrics.py` | Computes VaR, CVaR, tail probabilities, moments, and the Variance Risk Premium |
| `plotting.py` | Helpers for plotting the densities and smile overlays |

---

## Problem & Methodology

Options prices embed the market's full probability distribution over future returns — the **risk-neutral density** (RND) under the risk-neutral measure Q.  
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
| ATM IV | — | 30.81% | 30.61% | — |
| Mean S_T | $254.31 | $249.69 | $249.73 | $259.19 |
| Std S_T | $19.59 | $29.05 | $28.92 | $22.47 |
| Skewness | −0.59 | −0.34 | −0.37 | −0.02 |
| 95% VaR | $35.58 | $55.68 | $55.87 | $30.68 |
| 95% CVaR | $47.47 | $59.70 | $59.80 | $41.13 |

**VRP**: Implied vol (SSVI) = 30.61% vs. realised = 27.37% → **+3.24 pp (1.25× variance ratio) → SELL VOL signal.**  
The implied Q-density is left-skewed and fat-tailed relative to the empirical P-density, quantifying the crash risk premium embedded in options prices.

---

## Tech Stack

- **Python 3.10+**
- `numpy`, `scipy` — numerical integration, optimisation, spline interpolation
- `pandas` — option chain handling
- `yfinance` — live market data (options, rates, dividends)
- `matplotlib` — visualisation
- `pytest` — unit testing

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

The notebook fetches live data from Yahoo Finance on execution — no local data files required.

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
