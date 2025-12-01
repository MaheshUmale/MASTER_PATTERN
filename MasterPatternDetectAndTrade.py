import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# --- Added Imports for Real-Time Data Fetching ---
from tvDatafeed import Interval, TvDatafeed
# from tvDatafeed.main import TvDatafeedBadCredentialsException # Import specific exception for handling

# --- Configuration Constants ---
# Lookback period for the Average Price (The horizontal line in the chart)
# This 200 MA acts as the HTF (Higher Timeframe) directional bias.
MA_PERIOD = 200
# Lookback period for Average True Range (used to define 'significant' moves)
ATR_PERIOD = 14
# How far below/above the MA the price must dip/peak to confirm Phase 2 (Expansion/False Move).
# We require the Low/High to be at least 1.5 * ATR away from the MA.
EXPANSION_ATR_MULTIPLIER = 1.5
# How many periods of consecutive close-to-close change must occur after the low/high
# to confirm the sharp reversal and start of Phase 3 (Green/Red Phase).
CONSECUTIVE_MOMENTUM_CANDLES = 3

def calculate_indicators(df):
    """Calculates the Simple Moving Average (Average Price) and ATR."""
    
    # 1. Average Price (SMA) - Used as the HTF directional bias
    df['Average_Price'] = df['Close'].rolling(window=MA_PERIOD).mean()
    
    # 2. Average True Range (ATR) - used for dynamic volatility threshold
    def calculate_tr(df):
        # High - Low
        range1 = df['High'] - df['Low']
        # |High - Previous Close|
        range2 = np.abs(df['High'] - df['Close'].shift(1))
        # |Low - Previous Close|
        range3 = np.abs(df['Low'] - df['Close'].shift(1))
        return np.maximum.reduce([range1, range2, range3])

    df['TR'] = calculate_tr(df)
    df['ATR'] = df['TR'].rolling(window=ATR_PERIOD).mean()
    
    return df

def detect_master_pattern(df):
    """
    Detects the Master Pattern (Contraction -> Expansion -> Profit Taking) for
    both Long (Buy) and Short (Sell) setups. This represents the LTF entry pattern.
    
    Detection Logic (LTF):
    1. Look for Phase 2 (Expansion/Stop-Hunt): Price significantly moves away from the Average Price.
    2. Look for Phase 3 (Sharp Reversal): Price aggressively reverses from the Phase 2 extreme 
       with sustained momentum (consecutive candles in the direction of the trade).
    """
    
    # Check if indicators are calculated
    if 'Average_Price' not in df.columns or 'ATR' not in df.columns:
        df = calculate_indicators(df)

    # Initialize columns for the alert signals
    df['Alert_Buy_Entry'] = False
    df['Alert_Sell_Entry'] = False
    
    # Lookback window for Phase 2 (recent extreme price action)
    phase_2_lookback = 10 
    
    # Iterate through the data starting after the initial indicator calculation period
    for i in range(MA_PERIOD, len(df)):
        
        # Calculate recent indicators based on the previous period (i-1)
        recent_ma = df['Average_Price'].iloc[i-1]
        recent_atr = df['ATR'].iloc[i-1]
        
        # --- 1. Look for Phase 2 (Expansion/Stop-Hunt) Confirmation ---
        
        # Long Setup Check: Look for a low significantly below the MA
        recent_low = df['Low'].iloc[max(0, i - phase_2_lookback):i].min()
        is_expansion_low = (recent_low < (recent_ma - (recent_atr * EXPANSION_ATR_MULTIPLIER)))
        
        # Short Setup Check: Look for a high significantly above the MA
        recent_high = df['High'].iloc[max(0, i - phase_2_lookback):i].max()
        is_expansion_high = (recent_high > (recent_ma + (recent_atr * EXPANSION_ATR_MULTIPLIER)))
        
        
        # --- 2. Look for Phase 3 (Sharp Reversal/Momentum) Confirmation ---
        
        has_buy_momentum = False
        has_sell_momentum = False

        if i >= CONSECUTIVE_MOMENTUM_CANDLES: 
            # Check for Buy Momentum (Consecutive Green Candles: Close > Prev Close)
            is_green_streak = all(df['Close'].iloc[i - j] > df['Close'].iloc[i - j - 1] 
                                  for j in range(CONSECUTIVE_MOMENTUM_CANDLES))
            if is_green_streak:
                 has_buy_momentum = True

            # Check for Sell Momentum (Consecutive Red Candles: Close < Prev Close)
            is_red_streak = all(df['Close'].iloc[i - j] < df['Close'].iloc[i - j - 1] 
                                for j in range(CONSECUTIVE_MOMENTUM_CANDLES))
            if is_red_streak:
                 has_sell_momentum = True
                 
        # --- Final LTF Entry Signal ---
        
        # BUY Alert: Expansion Low occurred AND sharp upward reversal confirmed (Pattern within pattern)
        if is_expansion_low and has_buy_momentum:
            df.loc[df.index[i], 'Alert_Buy_Entry'] = True
        
        # SELL Alert: Expansion High occurred AND sharp downward reversal confirmed (Pattern within pattern)
        if is_expansion_high and has_sell_momentum:
            df.loc[df.index[i], 'Alert_Sell_Entry'] = True
                
    return df

def generate_mock_data():
    """Generates mock OHLC data simulating both long and short patterns (Fallback)."""
    np.random.seed(42)
    
    # 1. Contraction (Phase 1) - Price near 100
    base_price = 100.0
    prices = [base_price] * 250
    prices = [p + np.random.normal(0, 0.2) for p in prices]
    
    # 2. Long Setup: Expansion Low (Phase 2) - False dip
    for _ in range(50):
        # Sharp drop below the mean
        prices.append(prices[-1] - np.random.uniform(0.5, 1.5) + np.random.normal(0, 0.1))
        
    # 3. Long Setup: Profit Taking (Phase 3) - Sharp rally
    for _ in range(50):
        # Aggressive move back up
        prices.append(prices[-1] + np.random.uniform(1.0, 2.5) + np.random.normal(0, 0.2))

    # 4. Contraction/Peak - Price around 150
    for _ in range(50):
         prices.append(prices[-1] + np.random.normal(0, 0.2))
    
    # 5. Short Setup: Expansion High (Phase 2) - False peak
    for _ in range(50):
        # Sharp jump above the mean
        prices.append(prices[-1] + np.random.uniform(0.5, 1.5) + np.random.normal(0, 0.1))

    # 6. Short Setup: Profit Taking (Phase 3) - Sharp drop
    for _ in range(50):
        # Aggressive move back down
        prices.append(prices[-1] - np.random.uniform(1.0, 2.5) + np.random.normal(0, 0.2))


    # Create the DataFrame
    data = pd.DataFrame({
        'Close': prices,
    })
    
    # Generate High/Low/Open based on Close, adding some realistic noise
    data['Open'] = data['Close'].shift(1).fillna(data['Close'].iloc[0])
    data['High'] = data[['Open', 'Close']].max(axis=1) + np.random.uniform(0.1, 0.5)
    data['Low'] = data[['Open', 'Close']].min(axis=1) - np.random.uniform(0.1, 0.5)
    data.index.name = 'Candle'
    
    return data

def getTVData():
    """Fetches real-time OHLC data from TradingView."""
    try:
        # Initialize TvDatafeed
        tv = TvDatafeed()
        
        # Fetch NIFTY 1-minute data, last 1500 candles
        df = tv.get_hist(symbol="TCS", exchange="NSE", interval=Interval.in_5_minute, n_bars=1500)
        
        # Check if data was returned
        if df is None or df.empty:
            print("--- Data fetch failed or returned empty DataFrame. ---")
            return None

        # 1. Promote Datetime Index to a column. It will be named 'datetime' by default.
        df.reset_index(inplace=True) 
        
        # 2. Explicitly rename columns for clarity and Title Case consistency:
        # Columns names from tvDatafeed: ['datetime', 'symbol', 'open', 'high', 'low', 'close', 'volume']
        df.columns = ['Datetime', 'Symbol', 'Open', 'High', 'Low', 'Close', 'Volume']

        # 3. Rename the index to 'Candle' for consistency with plotting
        df.index.name = 'Candle'
        
        print("--- Data fetched successfully from TradingView ---")
        print(df.head())
        
        # Return the necessary columns, including the new 'Datetime' column
        return df[['Datetime', 'Open', 'High', 'Low', 'Close']]
    
    # except TvDatafeedBadCredentialsException:
    #     print("Error: TvDatafeed Bad Credentials. Ensure you are logged in if necessary.")
    #     return None
    except Exception as e:
        print(f"Error fetching data from TvDatafeed: {e}")
        return None


def plot_pattern(df, buy_alerts, sell_alerts):
    """Plots the Close Price, Average Price (MA), and marks the alerts."""
    
    plt.figure(figsize=(14, 7))
    
    # Plot the Closing Price
    plt.plot(df.index, df['Close'], label='Close Price', color='#4F46E5', linewidth=1.5)
    
    # Plot the Average Price (MA)
    plt.plot(df.index, df['Average_Price'], label=f'HTF Bias ({MA_PERIOD} MA)', 
             color='#9CA3AF', linestyle='--', linewidth=2)
    
    
    # --- Mark BUY Alerts (Green Phase Start) ---
    if not buy_alerts.empty:
        buy_prices = df.loc[buy_alerts.index, 'Close']
        
        plt.scatter(buy_alerts.index, buy_prices, 
                    label='BUY Entry (LTF Reversal)', 
                    color='#10B981', 
                    marker='^', # Upward-pointing triangle
                    s=150,      
                    zorder=5)   
        
        # Annotate the first BUY alert for context
        first_buy_index = buy_alerts.index.min()
        first_buy_price = df.loc[first_buy_index, 'Close']
        plt.annotate('LTF Buy Entry', 
                     xy=(first_buy_index, first_buy_price), 
                     xytext=(first_buy_index - 150, first_buy_price + df['ATR'].iloc[-1]*3),
                     arrowprops=dict(facecolor='green', shrink=0.05, width=1.5, headwidth=8),
                     fontsize=10, 
                     color='#10B981')

    # --- Mark SELL Alerts (Red Phase Start) ---
    if not sell_alerts.empty:
        sell_prices = df.loc[sell_alerts.index, 'Close']
        
        plt.scatter(sell_alerts.index, sell_prices, 
                    label='SELL Entry (LTF Reversal)', 
                    color='#EF4444', 
                    marker='v', # Downward-pointing triangle
                    s=150,      
                    zorder=5)   
        
        # Annotate the first SELL alert for context
        first_sell_index = sell_alerts.index.min()
        first_sell_price = df.loc[first_sell_index, 'Close']
        plt.annotate('LTF Sell Entry', 
                     xy=(first_sell_index, first_sell_price), 
                     xytext=(first_sell_index - 150, first_sell_price - df['ATR'].iloc[-1]*3),
                     arrowprops=dict(facecolor='red', shrink=0.05, width=1.5, headwidth=8),
                     fontsize=10, 
                     color='#EF4444')
        
    # --- Annotate Phase 2 Low/High for Visual Context (using only the last 500 bars) ---
    
    if not buy_alerts.empty or not sell_alerts.empty:
        # Find the overall recent low/high for contextual drawing
        recent_low_index = df.iloc[-500:]['Low'].idxmin()
        recent_low_price = df.loc[recent_low_index, 'Low']
        
        recent_high_index = df.iloc[-500:]['High'].idxmax()
        recent_high_price = df.loc[recent_high_index, 'High']
        
        # Annotate the Expansion Low (Phase 2 Low) 
        if not buy_alerts.empty and abs(recent_low_index - buy_alerts.index.min()) > 5:
            plt.scatter(recent_low_index, recent_low_price, 
                        label='Phase 2 Low (Stop-Hunt)', 
                        color='#EF4444', 
                        marker='P', 
                        s=150, zorder=5) 
            
            plt.annotate('Phase 2 Low', 
                         xy=(recent_low_index, recent_low_price), 
                         xytext=(recent_low_index + 10, recent_low_price - df['ATR'].iloc[-1]*1),
                         fontsize=10, 
                         color='#EF4444')
                         
        # Annotate the Expansion High (Phase 2 High)
        if not sell_alerts.empty and abs(recent_high_index - sell_alerts.index.min()) > 5:
            plt.scatter(recent_high_index, recent_high_price, 
                        label='Phase 2 High (Stop-Hunt)', 
                        color='#10B981', 
                        marker='P', 
                        s=150, zorder=5) 
            
            plt.annotate('Phase 2 High', 
                         xy=(recent_high_index, recent_high_price), 
                         xytext=(recent_high_index + 10, recent_high_price + df['ATR'].iloc[-1]*1),
                         fontsize=10, 
                         color='#10B981')


    # Add labels, title, and legend
    plt.title('Master Pattern Detection - MTFA (HTF Bias + LTF Entry)', fontsize=16)
    plt.xlabel('Candle Index', fontsize=12)
    plt.ylabel('Price', fontsize=12)
    plt.legend()
    plt.grid(True, linestyle=':', alpha=0.6)
    
    # Show the plot
    plt.show()

# --- Simulation and Execution ---

if __name__ == "__main__":
    
    # 1. Get/Generate Data
    historical_data = getTVData()
    
    if historical_data is None:
        print("\n--- Falling back to Mock Data for demonstration. ---")
        historical_data = generate_mock_data()


    # 2. Calculate Indicators and Detect Pattern
    analyzed_data = detect_master_pattern(historical_data.copy())
    
    # 3. Find Alerts (The start of the Green/Red Phase)
    buy_alerts = analyzed_data[analyzed_data['Alert_Buy_Entry'] == True]
    sell_alerts = analyzed_data[analyzed_data['Alert_Sell_Entry'] == True]
    
    print("\n--- 2. Analysis Complete (Last 5 Rows) ---")
    print(analyzed_data.tail())

    print("\n--- 3. Alert Summary (MTFA Entry) ---")
    
    total_alerts = len(buy_alerts) + len(sell_alerts)

    if total_alerts > 0:
        print(f"Master Pattern MTFA Alert triggered {total_alerts} time(s).")
        
        # Helper to print alert details
        def print_alert_details(alerts, action):
            if alerts.empty: return
            
            # Determine if we have a Datetime column (only present with real data)
            has_datetime = 'Datetime' in alerts.columns

            for index, row in alerts.iterrows():
                if has_datetime:
                    timestamp_str = row['Datetime'].strftime('%Y-%m-%d %H:%M:%S')
                    print(f"\n{action.upper()} ALERT TRIGGERED at Candle {index} ({timestamp_str}):")
                else:
                    # If no Datetime column, just print the Candle index
                    print(f"\n{action.upper()} ALERT TRIGGERED at Candle {index}:")

                print(f"Close Price: {row['Close']:.2f}")
                print(f"HTF Bias (MA): {row['Average_Price']:.2f}")
                print(f"ATR Volatility: {row['ATR']:.2f}")
                print(f"Action: Potential {action} opportunity based on LTF REVERSAL (Pattern within Pattern).")
                # Only print the first alert for brevity
                if index != alerts.index[-1]:
                    print("... (More alerts may follow the initial signal)")
                    break

        print_alert_details(buy_alerts, "BUY")
        print_alert_details(sell_alerts, "SELL")

    else:
        print("No Master Pattern MTFA Alerts detected in the historical data.")

    # 4. Generate the Plot for Visual Confirmation
    plot_pattern(analyzed_data, buy_alerts, sell_alerts)
