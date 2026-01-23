import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from hmmlearn.hmm import GaussianHMM
from sklearn.preprocessing import StandardScaler
import warnings

# Suppress warnings
warnings.filterwarnings("ignore")

def get_nifty_data(ticker="^NSEI", start_date="2007-01-01"):
    print(f"Fetching data for {ticker}...")
    df = yf.download(ticker, start=start_date, interval="1d", progress=False)
    
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    
    df = df.ffill()
    price_col = 'Adj Close' if 'Adj Close' in df.columns else 'Close'

    df['Returns'] = df[price_col].pct_change()

    df['Volatility'] = df['Returns'].rolling(window=20).std() * np.sqrt(252)

    sma = df[price_col].rolling(50).mean()
    std = df[price_col].rolling(50).std()
    df['Z_Score'] = (df[price_col] - sma) / std

    df['Momentum'] = df[price_col].diff(14)
    
    df.dropna(inplace=True)
    return df, price_col

def fit_stable_hmm(df, train_window=1250): # 5 Years Memory
    print(f"Training HMM on rolling {train_window}-day windows...")
    
    feature_cols = ['Volatility', 'Z_Score', 'Momentum']
    beliefs_list = []
    
    # Re-train every 10 days
    step = 10
    
    for i in range(train_window, len(df), step):
        train_data = df.iloc[i-train_window : i]
        X_train = train_data[feature_cols].values
        
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_train)

        model = GaussianHMM(n_components=3, covariance_type="diag", n_iter=100, random_state=42)
        try:
            model.fit(X_scaled)
        except:
            beliefs_list.extend([[1/3, 1/3, 1/3]] * min(step, len(df)-i))
            continue

        end_idx = min(i + step, len(df))
        future_X = df[feature_cols].iloc[i : end_idx].values
        future_scaled = scaler.transform(future_X)
        future_probs = model.predict_proba(future_scaled)

        regime_means = []
        preds = model.predict(X_scaled)
        for r in range(3):
            mask = preds == r
            if mask.sum() > 0:
                score = train_data['Z_Score'].iloc[mask].mean()
                regime_means.append((r, score))
            else:
                regime_means.append((r, -999))
                
        regime_means.sort(key=lambda x: x[1])
        map_dict = {regime_means[0][0]: 0, regime_means[1][0]: 1, regime_means[2][0]: 2}
        
        for prob_vec in future_probs:
            reordered = [prob_vec[k] for k, v in sorted(map_dict.items(), key=lambda item: item[1])]
            beliefs_list.append(reordered)
            
        if i % 1000 == 0:
            print(f"Processed {i}/{len(df)} days")
            
    needed_len = len(df)
    current_len = len(beliefs_list)
    if current_len < needed_len:
        padding = [[1/3, 1/3, 1/3]] * (needed_len - current_len)
        beliefs_list = padding + beliefs_list
    else:
        beliefs_list = beliefs_list[:needed_len]
        
    probs = np.array(beliefs_list)
    
    # SMOOTHING (10-day Moving Average)
    df_out = df.copy()
    df_out['P_Bear'] = pd.Series(probs[:, 0], index=df.index).rolling(10).mean()
    df_out['P_Sideways'] = pd.Series(probs[:, 1], index=df.index).rolling(10).mean()
    df_out['P_Bull'] = pd.Series(probs[:, 2], index=df.index).rolling(10).mean()
    
    df_out.dropna(inplace=True)
    return df_out

def run_strategy_smoothed(df, leverage_base=1.0, leverage_max=1.5, 
                          risk_free_rate=0.06, tx_cost=0.001):
    
    positions = [0]
    current_pos = 0
    
    for i in range(len(df)):
        if pd.isna(df['P_Bull'].iloc[i]):
            positions.append(0)
            continue

        p_bull = df['P_Bull'].iloc[i]
        p_bear = df['P_Bear'].iloc[i]
        p_side = df['P_Sideways'].iloc[i]
        z_score = df['Z_Score'].iloc[i]
    
        if current_pos == 0:
            if p_bull > 0.60:
                current_pos = leverage_base
            elif p_bull > 0.80:
                current_pos = leverage_max
            elif p_side > 0.6 and z_score > -0.5:
                current_pos = leverage_base
        
        else:
            if p_bear > 0.60:
                current_pos = 0
            elif current_pos == leverage_max and p_bull < 0.75:
                current_pos = leverage_base
                
        positions.append(current_pos)
        
    df = df.copy()
    df['Position'] = positions[:-1]
    
    # Returns
    df['Strat_Raw'] = df['Position'] * df['Returns']
    daily_rfr = risk_free_rate / 252
    df['Cash_Yield'] = np.maximum(0, (1 - df['Position'])) * daily_rfr
    pos_change = df['Position'].diff().abs().fillna(0)
    df['Tx_Costs'] = pos_change * tx_cost
    
    df['Net_Ret'] = df['Strat_Raw'] + df['Cash_Yield'] - df['Tx_Costs']
    
    df['Cum_Strat'] = (1 + df['Net_Ret']).cumprod()
    df['Cum_Mkt'] = (1 + df['Returns']).cumprod()
    
    return df

def print_live_status(df, price_col):
    """
    Reads the FINAL processed row to give today's recommendation.
    """
    last_row = df.iloc[-1]
    last_date = df.index[-1]
    current_price = last_row[price_col]
    
    # Get Smoothed Probabilities
    p_bear = last_row['P_Bear']
    p_side = last_row['P_Sideways']
    p_bull = last_row['P_Bull']
    z_score = last_row['Z_Score']

    status = "WAIT / CASH"
    if p_bull > 0.80:
        status = "STRONG BULL (Aggressive Buy)"
    elif p_bull > 0.60:
        status = "BULL (Standard Buy)"
    elif p_side > 0.6 and z_score > -0.5:
        status = "SIDEWAYS RECOVERY (Dip Buy)"
    elif p_bear > 0.60:
        status = "BEAR MARKET (Sell/Avoid)"
    else:
        status = "NEUTRAL / NO SIGNAL"

    print("\n" + "="*50)
    print(f"LIVE STRATEGY SIGNAL ({last_date.date()})")
    print("="*50)
    print(f"Current Nifty Price:  {current_price:,.2f}")
    print("-" * 50)
    print(f"Market Regime (Smoothed 10-Day Avg):")
    print(f"  [Bear]:      {p_bear*100:.1f}%")
    print(f"  [Sideways]:  {p_side*100:.1f}%")
    print(f"  [Bull]:      {p_bull*100:.1f}%")
    print("-" * 50)
    print(f"Technical Context:")
    print(f"  Trend (Z-Score): {z_score:.2f} sigma")
    print(f"  Volatility:      {last_row['Volatility']*100:.1f}%")
    print("-" * 50)
    print(f"FINAL VERDICT:   [{status}]")
    print("="*50 + "\n")

if __name__ == "__main__":
    nifty, price_col = get_nifty_data("^NSEI")
    
    # Train
    nifty_hmm = fit_stable_hmm(nifty)
    
    # Backtest
    print("\nRunning Fee-Aware Backtest...")
    results = run_strategy_smoothed(nifty_hmm, leverage_base=1.0, leverage_max=1.5, tx_cost=0.001)
    
    # Metrics
    strat_ret = (results['Cum_Strat'].iloc[-1] - 1) * 100
    mkt_ret = (results['Cum_Mkt'].iloc[-1] - 1) * 100
    trades = (results['Position'].diff() != 0).sum()
    
    print(f"\n--- FINAL AUDITED RESULTS ---")
    print(f"Strategy Return: {strat_ret:.2f}%")
    print(f"Market Return:   {mkt_ret:.2f}%")
    print(f"Total Trades:    {trades} (Low Churn)")
    
    # Plot
    plt.figure(figsize=(12, 6))
    plt.plot(results.index, results['Cum_Mkt'], label='Market (Buy&Hold)', color='gray', alpha=0.5)
    plt.plot(results.index, results['Cum_Strat'], label='HMM Strategy (Net)', color='blue', linewidth=2)
    plt.yscale('log')
    plt.title(f"HMM Strategy: {strat_ret:.0f}% vs Market: {mkt_ret:.0f}% (Net of Fees)")
    plt.legend()
    plt.grid(True, which="both", alpha=0.3)
    plt.show()
    
    # PRINT LIVE STATUS
    print_live_status(results, price_col)
