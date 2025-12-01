# MASTER_PATTERN

This Python script implements and backtests a sophisticated multi-timeframe analysis (MTFA) strategy known as the "Master Pattern." The strategy is designed to detect and trade market reversals that occur after a period of market manipulation or "stop-hunting."

## The Core Strategy: The Master Pattern

The strategy is based on the idea that the market often moves in three repeating phases. This script identifies these phases to generate entry signals for trend-reversal trades.

*   **Phase 1 (Contraction/HTF Bias):** The market is in a state of equilibrium, often consolidating around a long-term moving average.
*   **Phase 2 (Expansion/Manipulation/Stop-Hunt):** The price makes a sharp, aggressive move away from the mean, designed to trigger the stop-loss orders of retail traders.
*   **Phase 3 (Profit Taking/Reversal - The LTF Entry):** Following the stop-hunt, the price sharply reverses and begins to move back toward the mean. This reversal is the entry signal for the strategy.

## Key Indicators and Refinements

The script has undergone several iterations of refinement to improve its performance. The final version of the strategy incorporates the following key components:

*   **Multi-Timeframe Analysis (MTFA):** The script uses a higher timeframe (15-minute) 200-period Simple Moving Average (SMA) as a trend filter for the lower timeframe (5-minute) entry signals. A trade is only taken if the price action on the lower timeframe is aligned with the trend on the higher timeframe.
*   **Dynamic Volatility Threshold:** Instead of a fixed ATR multiplier, the strategy uses a dynamic threshold to detect the stop-hunt. This threshold is calculated by adding the rolling standard deviation of the ATR to the mean ATR. This allows the strategy to adapt to changing market volatility and identify more significant price expansions.
*   **Three-Candle Reversal Confirmation:** The entry signal is confirmed by three consecutive candles closing in the direction of the trade, providing a more robust confirmation of the reversal.

## Detection Logic (`detect_master_pattern`)

The core logic combines the MTFA filter with the dynamic volatility threshold to identify high-probability reversal trades:

1.  **Phase 2 Confirmation (The Setup):** The script looks for a recent price extreme that is significantly above or below the 200-period SMA on the 5-minute chart. The significance of this move is determined by the dynamic ATR threshold.
2.  **Phase 3 Confirmation (The Entry Trigger):** After a valid setup is identified, the script waits for a three-candle confirmation of the reversal.
3.  **MTFA Filter:** A trade is only triggered if the entry signal is aligned with the trend on the 15-minute timeframe (i.e., the price is above the 15-minute 200-period SMA for a long trade, and below it for a short trade).

## Backtesting Engine

The script includes a simple backtesting engine to simulate trades and evaluate the strategy's performance. The backtesting engine uses the following logic:

*   **Entry:** Trades are entered on the open of the candle following the entry signal.
*   **Stop-Loss:** An ATR-based stop-loss is placed at a multiple of the ATR from the entry price.
*   **Take-Profit:** A fixed 2:1 risk-to-reward ratio is used to set the take-profit level.
*   **Performance Metrics:** The backtesting engine calculates and displays the following key performance metrics for each symbol:
    *   Total Profit/Loss (P/L)
    *   Win Rate (%)
    *   Total Number of Trades

## Data Handling and Visualization

*   **Data Source:** The script uses the `tvdatafeed` library to fetch historical data from TradingView for multiple symbols and timeframes.
*   **Plotting:** The script generates and saves a chart for each symbol, visualizing the following:
    *   The 5-minute closing price and 200-period SMA.
    *   The buy and sell entry signals.
    *   The entry, exit, stop-loss, and take-profit levels for each trade.

## Installation and Usage

1.  **Install Dependencies:**
    ```bash
    pip install pandas numpy matplotlib
    pip install --upgrade --no-cache-dir git+https://github.com/rongardF/tvdatafeed.git
    ```

2.  **Configure Symbols:**
    Modify the `symbols_to_run` dictionary in the main execution block of the script to include the symbols you want to backtest.

3.  **Run the Script:**
    ```bash
    python3 MasterPatternDetectAndTrade.py
    ```
    The script will then run the backtest for each configured symbol, print the performance metrics to the console, and save the charts as PNG files in the same directory.
