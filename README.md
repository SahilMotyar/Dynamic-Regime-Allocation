# Dynamic-Regime-Allocation 

### A Fee-Aware, Regime-Switching Asset Allocation Strategy using Gaussian HMMs

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![License](https://img.shields.io/badge/License-MIT-green)
![Status](https://img.shields.io/badge/Status-Active-success)

## 📖 Overview

**Dynamic-Regime-Allocation** is a quantitative trading strategy designed for the **Nifty 50** index. Unlike traditional forecasting models that attempt to predict exact prices, this system uses **Unsupervised Learning (Gaussian Hidden Markov Models)** to decode latent market regimes (Bull, Bear, and Sideways).

The primary objective is **Risk-Adjusted Return**. The strategy dynamically adjusts portfolio leverage based on the identified regime, aiming to capture market upside while preserving capital during structural downtrends (e.g., 2008, 2020).

> **Key Differentiator:** This is not a theoretical model. It includes a rigorous **Transaction Cost Analysis (TCA)**, factoring in brokerage fees, slippage, and the yield on idle cash (Liquid BeEs), making the results audit-ready and deployable.

---

## Performance (2007–2025)

The strategy was backtested over an 18-year period covering multiple market cycles.

| Metric | Market (Buy & Hold) | HMM Strategy |
| :--- | :--- | :--- |
| **Net Return** | **319.27%** | **311.01%** |
| **Drawdown Risk** | High (-60% in 2008) | Significantly Reduced |
| **Total Trades** | N/A | **57** (Low Churn) |
| **Fees Paid** | 0.0% | ~5.6% of Capital |

**Visual Analysis:**
The strategy matches the market's long-term compounding but avoids the "emotional torture" of deep drawdowns. Note the flat-lining (protection) during the 2008 and 2020 crashes.

<img width="2559" height="1514" alt="image" src="https://github.com/user-attachments/assets/78c80e27-fb17-455d-8f0b-9da460899ad8" />

---

## 🛠️ Methodology

### 1. Feature Engineering
We feed the HMM three distinct features to separate market states:
* **Volatility:** 20-day annualized rolling standard deviation (Risk Proxy).
* **Trend (Z-Score):** Price distance from the 50-day Moving Average (Trend Proxy).
* **Momentum:** 14-day price difference (Velocity Proxy).

### 2. Unsupervised Learning (Gaussian HMM)
The model assumes the market operates in three hidden states. It calculates the probability of being in each state daily:
* 🟢 **Bull Regime:** Low Volatility, Positive Trend.
* 🟠 **Sideways Regime:** Mixed Volatility, Mean Reverting.
* 🔴 **Bear Regime:** High Volatility, Negative Trend.

### 3. Signal Smoothing (The "Churn Killer")
To prevent "whipsaw" losses (buying/selling every day), we apply a **10-day Moving Average** to the regime probabilities. The model must sustain a signal for ~2 weeks before capital is committed. This reduced total trades from **441** to **57**.

### 4. Fee-Aware Backtest Engine
* **Transaction Costs:** 0.1% per trade (Brokerage + STT + Slippage).
* **Cash Yield:** Idle cash (when not invested) earns a **6% annualized return**, simulating parking capital in Liquid BeEs or Overnight Funds.

---

## 🚀 Strategy Logic

| Market Regime | Signal Confidence | Action | Leverage |
| :--- | :--- | :--- | :--- |
| **Strong Bull** | P(Bull) > 80% | **Aggressive Buy** | 1.5x |
| **Bull** | P(Bull) > 60% | **Standard Buy** | 1.0x |
| **Sideways** | P(Side) > 60% & Dip | **Buy the Dip** | 1.0x |
| **Bear** | P(Bear) > 60% | **Sell / Cash** | 0x |

---

## 💻 Installation & Usage

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/SahilMotyar/Dynamic-Regime-Allocation.git](https://github.com/SahilMotyar/Dynamic-Regime-Allocation.git)
    cd Dynamic-Regime-Allocation
    ```

2.  **Install dependencies:**
    ```bash
    pip install yfinance pandas numpy matplotlib hmmlearn scikit-learn
    ```

3.  **Run the strategy:**
    ```bash
    python hmm.py
    ```

---

## 📡 Live Trade Signal

The script concludes by generating a **Live "Trade Card"** for the current day. It retrains the model on the most recent data to provide an actionable recommendation.

**Example Output (2026-01-23):**
<img width="753" height="881" alt="image" src="https://github.com/user-attachments/assets/2e771263-da3c-4cfc-b3c1-dc7a983bb2d2" />
---

## 📜 Disclaimer

*This project is for educational and research purposes only. It does not constitute financial advice. Algorithmic trading involves significant risk. Past performance is not indicative of future results.*
