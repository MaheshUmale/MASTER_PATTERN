# import pandas as pd
# import json
# import matplotlib.pyplot as plt
# from matplotlib.dates import date2num
# import numpy as np

# # --- USER PARAMETERS ---
# FILENAME = 'xxxupstox_WSS_output_2024_08_02_08_51_12.txt'
# INSTRUMENT_CHOICE = 'NSE_INDEX|Nifty 50' # <-- Change this to select your security
# HIGH_QUANTITY_THRESHOLD = 20 # <-- Set your minimum quantity threshold for highlighting
# INTERVAL = '30S' # <-- Set the chart interval ('30S' or '1T')
# # -----------------------

# # (File Reading and Data Cleaning Section)
# # ... The code reads the file and creates a DataFrame 'df' with millisecond precision index ...

# # --- Data Cleaning and Indexing (using full millisecond timestamp) ---
# all_trades = []
# # [File reading and JSON parsing block as provided in the execution step]

# df = pd.DataFrame(all_trades)
# if not df.empty:
#     df['timestamp'] = pd.to_numeric(df['timestamp']).astype(np.int64)
#     df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms') # MS for exact match
#     df = df.set_index('timestamp')
#     df['price'] = pd.to_numeric(df['price'], errors='coerce')
#     df['quantity'] = pd.to_numeric(df['quantity'], errors='coerce')
#     df = df.dropna(subset=['price', 'quantity'])
# else:
#     print(f"No valid trade data found for instrument: {INSTRUMENT_CHOICE}")
#     # exit() # Exit for terminal use
# print(df.head())
# # --- 1. Resample to OHLCV (Candlestick Background) ---
# ohlc = df['price'].resample(INTERVAL).ohlc().dropna()
# ohlc.columns = ['Open', 'High', 'Low', 'Close']
# volume = df['quantity'].resample(INTERVAL).sum().fillna(0)
# ohlc['Volume'] = volume[ohlc.index]

# # --- 2. NEW Bubble Logic: Aggregate by EXACT Timestamp ---
# # Group by the exact timestamp index and sum the quantity
# aggregated_trades = df.groupby(df.index).agg({
#     'price': 'last', # Last price at the exact timestamp for bubble position
#     'quantity': 'sum'  # Sum all quantities at this exact timestamp
# })

# # Filter this aggregated data based on the total quantity threshold
# highlight_data = aggregated_trades[aggregated_trades['quantity'] >= HIGH_QUANTITY_THRESHOLD].copy()

# # --- 3. Visualization (using Matplotlib) ---
# def create_candlestick_plot(ohlc_data, highlight_data, instrument_name, interval_label, high_q_threshold):
#     ohlc_data['Date'] = ohlc_data.index.to_series().apply(date2num)
#     fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(12, 8),
#                                    gridspec_kw={'height_ratios': [3, 1]})
#     fig.suptitle(f'Candlestick Chart for {instrument_name} ({interval_label} Intervals)', fontsize=16)
    
#     bar_width = 0.0005 # Relative width for candlestick bodies
    
#     # Candlestick Plotting
#     for index, row in ohlc_data.iterrows():
#         # ... (Candlestick drawing logic) ...
#         color = 'green' if row['Close'] >= row['Open'] else 'red'
#         ax1.plot([row['Date'], row['Date']], [row['Low'], row['High']], color='black', linewidth=1)
#         rect_height = abs(row['Close'] - row['Open'])
#         rect_bottom = min(row['Open'], row['Close'])
#         rect = plt.Rectangle((row['Date'] - bar_width/2, rect_bottom), bar_width, rect_height, facecolor=color, edgecolor='black', linewidth=0.5)
#         ax1.add_patch(rect)
        
#     ax1.set_ylabel('Price', fontsize=12)

#     # High-Quantity Bubbles (Exact Timestamp)
#     if not highlight_data.empty:
#         highlight_data['Date_num'] = highlight_data.index.to_series().apply(date2num)
#         marker_size = highlight_data['quantity'] * 20 # Size proportional to summed quantity
        
#         ax1.scatter(
#             highlight_data['Date_num'], 
#             highlight_data['price'], 
#             s=marker_size, 
#             color='gold', 
#             alpha=0.7, 
#             edgecolors='red', 
#             linewidth=1.5,
#             label=f'Aggregated Trades (Sum Q $\geq$ {high_q_threshold})'
#         )
#         ax1.legend(loc='upper left', frameon=True)

#     # Volume Plot
#     ax2.bar(ohlc_data['Date'], ohlc_data['Volume'], color='blue', alpha=0.6, width=bar_width)
#     ax2.set_ylabel('Volume', fontsize=12)
#     ax2.tick_params(axis='x', rotation=45)
    
#     ax1.xaxis_date()
#     ax2.xaxis_date()
#     fig.autofmt_xdate()
    
#     plt.tight_layout(rect=[0, 0, 1, 0.96])
#     plt.savefig('candlestick_chart_exact_timestamp.png')
#     plt.close(fig)

# create_candlestick_plot(ohlc, highlight_data, INSTRUMENT_CHOICE, INTERVAL, HIGH_QUANTITY_THRESHOLD)



import pandas as pd
import json
import matplotlib.pyplot as plt
from matplotlib.dates import date2num
import numpy as np

# --- USER PARAMETERS ---
FILENAME = 'wss_data_new.txt'
FILENAME = './/OutFiles//upstox_WSS_output_2024_07_29_06_54_02.txt' # upstox_WSS_output_2024_07_29_06_54_02
INSTRUMENT_CHOICE = 'NSE_FO|35165' ##'NSE_INDEX|Nifty Bank' #'NSE_FO|35415' # <-- Set to your security of interest
HIGH_QUANTITY_THRESHOLD = 2000   # <-- Set the minimum quantity sum for a bubble
INTERVAL = '30S'                # <-- Set the chart interval ('30S' or '1T')
# -----------------------

# --- (Mock File Setup with NEW Data) ---
data_string = """
{"feeds": {"NSE_INDEX|Nifty Bank": {"ltpc": {"ltp": 51493.25, "ltt": "1722588673000", "cp": 51564.0}}, "NSE_FO|35165": {"ltpc": {}}, "NSE_FO|35415": {"ltpc": {"ltp": 24790.0, "ltt": "1722588672206", "ltq": "75", "cp": 25032.25}}, "NSE_FO|35080": {"ltpc": {}}, "NSE_INDEX|Nifty 50": {"ltpc": {"ltp": 24777.05, "ltt": "1722588673000", "cp": 25010.9}}, "NSE_FO|35006": {"ltpc": {"ltp": 52185.0, "ltt": "1722588671464", "ltq": "15", "cp": 52359.0}}}}
{"type": "live_feed", "feeds": {"NSE_INDEX|Nifty Bank": {"ltpc": {"ltp": 51493.85, "ltt": "1722588673000", "cp": 51564.0}}, "NSE_INDEX|Nifty 50": {"ltpc": {"ltp": 24778.0, "ltt": "1722588673000", "cp": 25010.9}}, "NSE_FO|35006": {"ltpc": {"ltp": 52185.0, "ltt": "1722588671464", "ltq": "15", "cp": 52359.0}}}}
{"type": "live_feed", "feeds": {"NSE_INDEX|Nifty 50": {"ltpc": {"ltp": 24776.95, "ltt": "1722588674000", "cp": 25010.9}}}}
"""
# # Write to file for demonstration of file reading
# with open(FILENAME, 'w') as f:
#     f.write(data_string)
# # -------------------------------------------------------------------------

# --- Read from file and Parse Data ---
all_trades = []
try:
    with open(FILENAME, 'r') as f:
        for line in f:
            try:
                data = json.loads(line)
            except json.JSONDecodeError:
                continue

            if 'feeds' in data and INSTRUMENT_CHOICE in data['feeds']:
                feed_data = data['feeds'][INSTRUMENT_CHOICE]
                # Ensure all required keys are present
                if all(k in feed_data.get('ltpc', {}) for k in ['ltp', 'ltt', 'ltq']):
                    trade = {
                        'timestamp': feed_data['ltpc']['ltt'],
                        'price': feed_data['ltpc']['ltp'],
                        'quantity': feed_data['ltpc']['ltq']
                    }
                    all_trades.append(trade)
except FileNotFoundError:
    print(f"Error: The file '{FILENAME}' was not found.")
    exit()

df = pd.DataFrame(all_trades)

# --- Data Cleaning and Indexing ---
if df.empty:
    # Explicitly catch the case where no data was filtered for the instrument
    print(f"No trade data found for instrument: {INSTRUMENT_CHOICE}. Cannot generate chart.")
    exit()

df['timestamp'] = pd.to_numeric(df['timestamp']).astype(np.int64)
df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
df = df.set_index('timestamp')
df['price'] = pd.to_numeric(df['price'], errors='coerce')
df['quantity'] = pd.to_numeric(df['quantity'], errors='coerce')
df = df.dropna(subset=['price', 'quantity']) # Final clean-up

# --- 1. Resample to OHLCV (Candlestick Background) ---
ohlc = df['price'].resample(INTERVAL).ohlc().dropna()
ohlc.columns = ['Open', 'High', 'Low', 'Close']
volume = df['quantity'].resample(INTERVAL).sum().fillna(0)
ohlc['Volume'] = volume[ohlc.index]

# --- 2. Bubble Logic: Aggregate by EXACT Timestamp ---
# Group by the exact timestamp index and sum the quantity
aggregated_trades = df.groupby(df.index).agg({
    'price': 'last', 
    'quantity': 'sum'
})

# Filter this aggregated data based on the total quantity threshold
highlight_data = aggregated_trades[aggregated_trades['quantity'] >= HIGH_QUANTITY_THRESHOLD].copy()

# --- 3. Visualization (using Matplotlib) ---

def create_candlestick_plot(ohlc_data, highlight_data, instrument_name, interval_label, high_q_threshold):
    ohlc_data['Date'] = ohlc_data.index.to_series().apply(date2num)

    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(12, 8),
                                   gridspec_kw={'height_ratios': [3, 1]})
    
    fig.suptitle(f'Candlestick Chart for {instrument_name} ({interval_label} Intervals)', fontsize=16)
    bar_width = 0.0005 
    
    # Candlestick Plotting
    for index, row in ohlc_data.iterrows():
        color = 'green' if row['Close'] >= row['Open'] else 'red'
        ax1.plot([row['Date'], row['Date']], [row['Low'], row['High']], color='black', linewidth=1)
        rect_height = abs(row['Close'] - row['Open'])
        rect_bottom = min(row['Open'], row['Close'])
        rect = plt.Rectangle((row['Date'] - bar_width/2, rect_bottom), bar_width, rect_height, facecolor=color, edgecolor='black', linewidth=0.5)
        ax1.add_patch(rect)
        
    ax1.set_ylabel('Price', fontsize=12)
    ax1.grid(True, linestyle='--', alpha=0.7)

    # High-Quantity Bubbles (Exact Timestamp)
    if not highlight_data.empty:
        highlight_data['Date_num'] = highlight_data.index.to_series().apply(date2num)
        marker_size = highlight_data['quantity'] * 1
        
        ax1.scatter(
            highlight_data['Date_num'], 
            highlight_data['price'], 
            s=marker_size, 
            color='gold', 
            alpha=0.7, 
            edgecolors='red', 
            linewidth=1.5,
            label=f'Aggregated Trades (Sum Q $\geq$ {high_q_threshold})'
        )
        ax1.legend(loc='upper left', frameon=True)

    # Volume Plot
    ax2.bar(ohlc_data['Date'], ohlc_data['Volume'], color='blue', alpha=0.6, width=bar_width)
    ax2.set_ylabel('Volume', fontsize=12)
    ax2.grid(True, linestyle='--', alpha=0.7)
    ax2.tick_params(axis='x', rotation=45)
    
    ax1.xaxis_date()
    ax2.xaxis_date()
    fig.autofmt_xdate()
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig('candlestick_chart_new_data.png')
    plt.close(fig)

create_candlestick_plot(ohlc, highlight_data, INSTRUMENT_CHOICE, INTERVAL, HIGH_QUANTITY_THRESHOLD)
print("Filtered Candlestick chart saved as 'candlestick_chart_new_data.png'")