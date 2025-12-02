import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# --- Added Imports for Real-Time Data Fetching ---
from tvDatafeed import TvDatafeed, Interval
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
    df['ATR_StdDev'] = df['ATR'].rolling(window=ATR_PERIOD).std()
    
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
        dynamic_atr = recent_atr + df['ATR_StdDev'].iloc[i-1]
        recent_low = df['Low'].iloc[max(0, i - phase_2_lookback):i].min()
        is_expansion_low = (recent_low < (recent_ma - dynamic_atr))
        
        # Short Setup Check: Look for a high significantly above the MA
        recent_high = df['High'].iloc[max(0, i - phase_2_lookback):i].max()
        is_expansion_high = (recent_high > (recent_ma + dynamic_atr))
        
        
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
                 
        # --- 3. Check for recent Volume Spike ---
        has_recent_spike = df['is_volume_spike'].iloc[max(0, i - phase_2_lookback):i].any()

        # --- Final LTF Entry Signal ---
        
        # BUY Alert: Expansion Low occurred AND sharp upward reversal confirmed (Pattern within pattern)
        if is_expansion_low and has_buy_momentum and df['Close'].iloc[i] > df['HTF_MA'].iloc[i] and has_recent_spike:
            df.loc[df.index[i], 'Alert_Buy_Entry'] = True
        
        # SELL Alert: Expansion High occurred AND sharp downward reversal confirmed (Pattern within pattern)
        if is_expansion_high and has_sell_momentum and df['Close'].iloc[i] < df['HTF_MA'].iloc[i] and has_recent_spike:
            df.loc[df.index[i], 'Alert_Sell_Entry'] = True
                
    return df


def detect_accumulation_pattern(df, range_lookback=20, volume_lookback=20):
    """
    Detects the accumulation pattern based on the user's description.

    1) Strong Move Down with increase in volume
    2) price stays in RANGE
    3) small down move Not far from range ( just trying to break) with decreasing volume, fails to break , shows seller not interested to sell at lower prices
    4)small upmove within range , with increasing volume, still withing range shows buyer are ready to buy lower prices and increasing volume shows increasing interest
    5) next small down move , fails to go down further with less and decreasing volume shows seller not interested,
    6) break of this range with larger volume and any fractional down move with very small volume
    """
    df['Alert_Accumulation_Buy'] = False
    df['Volume_MA'] = df['Volume'].rolling(window=volume_lookback).mean()

    for i in range(range_lookback, len(df)):
        # Define the lookback period for pattern detection
        window = df.iloc[i-range_lookback:i]

        # 1. Strong Move Down with increase in volume
        down_move_window = df.iloc[i-range_lookback-10:i-range_lookback]
        if down_move_window.empty:
            continue

        price_diff = down_move_window['Close'].iloc[-1] - down_move_window['Close'].iloc[0]
        volume_increase = down_move_window['Volume'].iloc[-1] > down_move_window['Volume'].mean() * 1.5
        is_strong_down_move = price_diff < 0 and abs(price_diff) > window['ATR'].mean() and volume_increase

        if not is_strong_down_move:
            continue

        # 2. Price stays in RANGE
        range_high = window['High'].max()
        range_low = window['Low'].min()
        is_in_range = (df['Close'].iloc[i-1] > range_low) and (df['Close'].iloc[i-1] < range_high)

        if not is_in_range:
            continue

        # 3. Small down move with decreasing volume
        recent_low = df['Low'].iloc[i-5:i].min()
        recent_volume = df['Volume'].iloc[i-5:i].mean()
        is_failed_breakout = (recent_low < range_low) and (recent_low > range_low - window['ATR'].mean() * 0.5)
        is_decreasing_volume = recent_volume < window['Volume_MA'].mean()

        if not (is_failed_breakout and is_decreasing_volume):
            continue

        # 4. Small upmove with increasing volume
        upmove_window = df.iloc[i-3:i]
        is_upmove_in_range = (upmove_window['High'].max() < range_high) and (upmove_window['Low'].min() > range_low)
        is_increasing_volume_upmove = upmove_window['Volume'].iloc[-1] > upmove_window['Volume'].mean()

        if not (is_upmove_in_range and is_increasing_volume_upmove):
            continue

        # 5. Next small down move with less and decreasing volume
        downmove_window_2 = df.iloc[i-2:i]
        is_downmove_failed = downmove_window_2['Low'].iloc[-1] > downmove_window_2['Low'].iloc[-2]
        is_decreasing_volume_downmove_2 = downmove_window_2['Volume'].iloc[-1] < downmove_window_2['Volume'].mean()

        if not (is_downmove_failed and is_decreasing_volume_downmove_2):
            continue

        # 6. Break of this range with larger volume
        is_breakout = df['Close'].iloc[i] > range_high
        is_breakout_volume = df['Volume'].iloc[i] > window['Volume_MA'].mean() * 2

        if is_breakout and is_breakout_volume:
            df.loc[df.index[i], 'Alert_Accumulation_Buy'] = True

    return df


def detect_volume_spikes(df, lookback_period=20, multiplier=3.0):
    """
    Detects significant volume spikes in the data.

    A spike is defined as a candle where the volume is a certain multiplier
    times greater than the recent average volume.

    Args:
        df (pd.DataFrame): DataFrame with a 'Volume' column.
        lookback_period (int): The rolling window for the volume moving average.
        multiplier (float): The threshold multiplier for detecting a spike.

    Returns:
        pd.DataFrame: The original DataFrame with an added 'is_volume_spike' column.
    """
    df['Volume_MA'] = df['Volume'].rolling(window=lookback_period).mean()
    df['is_volume_spike'] = df['Volume'] > (df['Volume_MA'] * multiplier)
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

def getTVData(symbol, exchange, interval):
    """Fetches real-time OHLC data from TradingView."""
    try:
        # Initialize TvDatafeed
        tv = TvDatafeed()
        
        # Fetch NIFTY 1-minute data, last 1500 candles
        df = tv.get_hist(symbol=symbol, exchange=exchange, interval=interval, n_bars=1500)
        
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
        return df[['Datetime', 'Open', 'High', 'Low', 'Close', 'Volume']]
    
    # except TvDatafeedBadCredentialsException:
    #     print("Error: TvDatafeed Bad Credentials. Ensure you are logged in if necessary.")
    #     return None
    except Exception as e:
        print(f"Error fetching data from TvDatafeed: {e}")
        return None


def plot_pattern(df, buy_alerts, sell_alerts, accumulation_buy_alerts, symbol, interval_str, trades=None, bubble_data=None):
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

    # --- Mark ACCUMULATION BUY Alerts ---
    if not accumulation_buy_alerts.empty:
        buy_prices = df.loc[accumulation_buy_alerts.index, 'Close']

        plt.scatter(accumulation_buy_alerts.index, buy_prices,
                    label='Accumulation BUY Entry',
                    color='#FFD700', # Gold
                    marker='*', # Star
                    s=200,
                    zorder=6)
        
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


    # --- Plot Trades ---
    if trades:
        for trade in trades:
            # Entry point
            plt.axvline(x=trade['entry_index'], color='blue', linestyle='--', alpha=0.7)
            # Exit point
            plt.axvline(x=trade['exit_index'], color='purple', linestyle='--', alpha=0.7)
            # SL and TP lines
            plt.axhline(y=trade['stop_loss'], color='red', linestyle=':', alpha=0.7)
            plt.axhline(y=trade['take_profit'], color='green', linestyle=':', alpha=0.7)

    # --- Plot Volume Spikes (Bubbles) ---
    if bubble_data is not None:
        spikes = bubble_data[bubble_data['is_volume_spike']]
        if not spikes.empty:
            # Map spike datetimes to the integer index of the main df
            spike_indices = [df.index[df['Datetime'] == dt][0] for dt in spikes.index if dt in df['Datetime'].values]
            spike_closes = [spikes.loc[dt, 'Close'] for dt in spikes.index if dt in df['Datetime'].values]
            spike_volumes = [spikes.loc[dt, 'Volume'] for dt in spikes.index if dt in df['Datetime'].values]

            if spike_indices:
                plt.scatter(spike_indices, spike_closes, s=[v/100 for v in spike_volumes],
                            label='Volume Spike', color='orange', alpha=0.6, zorder=10)

    # Add labels, title, and legend
    plt.title(f'Master Pattern Detection for {symbol} ({interval_str})', fontsize=16)
    plt.xlabel('Candle Index', fontsize=12)
    plt.ylabel('Price', fontsize=12)
    plt.legend()
    plt.grid(True, linestyle=':', alpha=0.6)
    
    # Save the plot to a file
    filename = f"{symbol}_{interval_str}.png"
    plt.savefig(filename)
    print(f"--- Plot saved to {filename} ---")
    plt.close()

def simulate_trades(df, bubble_data, risk_reward_ratio=2.0):
    """Simulates trades based on entry signals and calculates performance."""
    trades = []
    in_trade = False

    for i in range(1, len(df)):
        # --- TRADE EXIT LOGIC ---
        if in_trade:
            current_low = df['Low'].iloc[i]
            current_high = df['High'].iloc[i]

            # Check for SL/TP hit
            if trade['type'] == 'BUY':
                if current_low <= trade['stop_loss']:
                    trade['exit_price'] = trade['stop_loss']
                    trade['exit_reason'] = 'SL'
                    in_trade = False
                elif current_high >= trade['take_profit']:
                    trade['exit_price'] = trade['take_profit']
                    trade['exit_reason'] = 'TP'
                    in_trade = False

            elif trade['type'] == 'SELL':
                if current_high >= trade['stop_loss']:
                    trade['exit_price'] = trade['stop_loss']
                    trade['exit_reason'] = 'SL'
                    in_trade = False
                elif current_low <= trade['take_profit']:
                    trade['exit_price'] = trade['take_profit']
                    trade['exit_reason'] = 'TP'
                    in_trade = False

            # If trade is exited, calculate P/L and add to list
            if not in_trade:
                trade['pnl'] = (trade['exit_price'] - trade['entry_price']) if trade['type'] == 'BUY' else (trade['entry_price'] - trade['exit_price'])
                trade['exit_index'] = i
                trades.append(trade)

        # --- TRADE ENTRY LOGIC ---
        if not in_trade:
            is_buy_signal = df['Alert_Buy_Entry'].iloc[i-1]
            is_sell_signal = df['Alert_Sell_Entry'].iloc[i-1]

            entry_price = df['Open'].iloc[i]

            if is_buy_signal:
                # Find the most recent volume spike in the bubble data
                recent_bubble_data = bubble_data[bubble_data.index < df['Datetime'].iloc[i]]
                last_spike = recent_bubble_data[recent_bubble_data['is_volume_spike']].tail(1)
                if not last_spike.empty:
                    stop_loss = last_spike['Low'].iloc[0]
                    take_profit = entry_price + ((entry_price - stop_loss) * risk_reward_ratio)
                    trade = {'type': 'BUY', 'entry_price': entry_price, 'stop_loss': stop_loss, 'take_profit': take_profit, 'entry_index': i}
                    in_trade = True

            elif is_sell_signal:
                # Find the most recent volume spike in the bubble data
                recent_bubble_data = bubble_data[bubble_data.index < df['Datetime'].iloc[i]]
                last_spike = recent_bubble_data[recent_bubble_data['is_volume_spike']].tail(1)
                if not last_spike.empty:
                    stop_loss = last_spike['High'].iloc[0]
                    take_profit = entry_price - ((stop_loss - entry_price) * risk_reward_ratio)
                    trade = {'type': 'SELL', 'entry_price': entry_price, 'stop_loss': stop_loss, 'take_profit': take_profit, 'entry_index': i}
                    in_trade = True

    return trades

# --- Simulation and Execution ---

def run_analysis(symbol, exchange, interval):
    """Runs the full analysis for a given symbol and interval."""
    interval_str = interval.value # e.g., '5m'
    print(f"\n--- Running Analysis for {symbol} ({interval_str}) ---")
    
    # 1. Get/Generate Data
    ltf_data = getTVData(symbol, exchange, interval)
    htf_interval = Interval.in_15_minute
    htf_data = getTVData(symbol, exchange, htf_interval)
    bubble_data = getTVData(symbol, exchange, Interval.in_1_minute)
    
    if ltf_data is None or htf_data is None or bubble_data is None:
        print(f"--- No data for {symbol}, skipping. ---")
        return

    # Align HTF data to LTF
    htf_data.set_index('Datetime', inplace=True)
    ltf_data.set_index('Datetime', inplace=True)
    htf_ma = htf_data['Close'].rolling(window=MA_PERIOD).mean()
    ltf_data['HTF_MA'] = htf_ma.reindex(ltf_data.index, method='ffill')
    ltf_data.reset_index(inplace=True)

    # 2. Process bubble data and align with ltf_data
    bubble_data = detect_volume_spikes(bubble_data)
    bubble_data.set_index('Datetime', inplace=True)
    ltf_data['is_volume_spike'] = bubble_data['is_volume_spike'].reindex(ltf_data.set_index('Datetime').index, method='ffill', fill_value=False)

    # 3. Calculate Indicators and Detect Pattern
    analyzed_data = detect_master_pattern(ltf_data.copy())
    analyzed_data = detect_accumulation_pattern(analyzed_data)

    # 3. Find Alerts (The start of the Green/Red Phase)
    buy_alerts = analyzed_data[analyzed_data['Alert_Buy_Entry'] == True]
    sell_alerts = analyzed_data[analyzed_data['Alert_Sell_Entry'] == True]
    accumulation_buy_alerts = analyzed_data[analyzed_data['Alert_Accumulation_Buy'] == True]

    print("\n--- 2. Analysis Complete (Last 5 Rows) ---")
    print(analyzed_data.tail())

    print("\n--- 3. Alert Summary (MTFA Entry) ---")

    total_alerts = len(buy_alerts) + len(sell_alerts) + len(accumulation_buy_alerts)

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
                if 'Average_Price' in row:
                    print(f"HTF Bias (MA): {row['Average_Price']:.2f}")
                if 'ATR' in row:
                    print(f"ATR Volatility: {row['ATR']:.2f}")
                print(f"Action: Potential {action} opportunity based on LTF REVERSAL (Pattern within Pattern).")
                # Only print the first alert for brevity
                if index != alerts.index[-1]:
                    print("... (More alerts may follow the initial signal)")
                    break

        print_alert_details(buy_alerts, "BUY")
        print_alert_details(sell_alerts, "SELL")
        print_alert_details(accumulation_buy_alerts, "ACCUMULATION BUY")

    else:
        print("No Master Pattern MTFA Alerts detected in the historical data.")

    # 4. Simulate Trades
    trades = simulate_trades(analyzed_data, bubble_data)

    # 5. Generate the Plot for Visual Confirmation
    plot_pattern(analyzed_data, buy_alerts, sell_alerts, accumulation_buy_alerts, symbol, interval_str, trades, bubble_data)

    # 6. Print Backtesting Results
    if trades:
        total_pnl = sum(t['pnl'] for t in trades)
        wins = [t for t in trades if t['pnl'] > 0]
        win_rate = (len(wins) / len(trades)) * 100 if trades else 0

        print("\n--- 4. Backtesting Results ---")
        print(f"Total Trades: {len(trades)}")
        print(f"Win Rate: {win_rate:.2f}%")
        print(f"Total P/L: {total_pnl:.2f}")

if __name__ == "__main__":

    symbols_to_run = {
        "NSE": ["TCS", "INFY", "NIFTY", "HINDZINC", "GRANULES"],
    }

    timeframe = Interval.in_5_minute

    for exchange, symbols in symbols_to_run.items():
        for symbol in symbols:
            run_analysis(symbol, exchange, timeframe)
