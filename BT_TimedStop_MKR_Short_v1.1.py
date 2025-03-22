'''
Note: For what it's worth, on 10/23/23 I did a comparison between the polygon live 
data timestamps, and the firstrate. It was virtually identical.

10/27/23:
Ideas to consider for exit strategies:
    - Volatility based exit (when price is a % of ATR)
    - Scaling Out
    - MA Crossing
    
v1.1:
    This is an experimental version to test new exit strategies.
'''

import requests
import json
import os
import pandas as pd
import numpy as np
from datetime import timedelta, date
import time
import datetime
import yfinance as yf
import talib
import math
import pandas_ta
from polygon import RESTClient
pd.options.mode.chained_assignment = None
os.chdir(r'C:\Users\pchil\Dropbox\Stocks\Back-Testing\Multi-Kernel Regression')




client = RESTClient('_')

global SLIPPAGE_PCT
SLIPPAGE_PCT = .0004

'''
resp = client.get_aggs('TQQQ', multiplier=1, timespan='minute', from_='2023-08-31', to='2023-09-04', limit=50000)
tqqq2 = pd.DataFrame(resp)
tqqq2['Datetime'] = pd.to_datetime(tqqq2['timestamp'],unit='ms')

# Subset the datetime column to times between 9:30 AM and 4:00 PM
start_time = pd.to_datetime('09:30:00').time()
end_time = pd.to_datetime('16:00:00').time()
subset = (tqqq2['Datetime'].dt.time >= start_time) & (tqqq2['Datetime'].dt.time <= end_time)
df_subset = tqqq2[subset]
df_subset.drop(['transactions','otc','timestamp'], axis=1, inplace=True)
df_subset.columns = ['Open', 'High', 'Low', 'Close', 'Volume', 'VWAP', 'Datetime']

tqqq = df_subset.copy()
'''

# To Test a specific day:
# tqqq = pd.read_csv(r"C:\Users\flip6\Dropbox\Stocks\Back-Testing\Live Automation\Data Timestamps\timestamp_2023-12-06 15-59-26.csv")
# tqqq.columns = ['index','Open','High','Low','Close','volume','vwap','timestamp','transactions','otc','Datetime']
# tqqq['Datetime'] = pd.to_datetime(tqqq['timestamp'],unit='ms')
# tqqq['Datetime'] = tqqq['Datetime'] - pd.Timedelta(hours=4)



#tqqq = yf.download(tickers="TQQQ", period="7d", interval="1m")
#tqqq = tqqq.reset_index()
tqqq = pd.read_csv('TQQQ_full_1hour_adjsplit_930Adjusted.csv')

# Subset to 2022 forward as 2021 has very bad data quality
tqqq['Datetime'] = pd.to_datetime(tqqq['Datetime'])
tqqq['Date'] = tqqq['Datetime'].dt.date

tqqq = tqqq[(tqqq['Datetime'].dt.year >= 2015) & (tqqq['Datetime'].dt.year < 2024)]
tqqq = tqqq.reset_index(drop=True)

df_full = tqqq.copy()
df_2mo = tqqq[(tqqq['Datetime'].dt.year >= 2023) & (tqqq['Datetime'].dt.month >= 5)]
df_5mo = tqqq[(tqqq['Datetime'].dt.year >= 2023)]

df_full = df_full.reset_index()
df_2mo = df_2mo.reset_index()
df_5mo = df_5mo.reset_index()

df = df_2mo.copy()

global daily_time_frame
daily_time_frame = False



# Calculate ATR using smoothing to match TradingView
def smoothing(prices: pd.Series, smoothing_type, smoothing_period) -> pd.Series:
        if smoothing_type == 'sma':
            return talib.SMA(prices, smoothing_period)
        elif smoothing_type == 'ema':
            return talib.EMA(prices, smoothing_period)
        elif smoothing_type == 'rma':
            return pandas_ta.rma(prices, smoothing_period)
        elif smoothing_type is None:
            return prices




def backtester(df, kernel_name, bandwidth, diff_threshold, consec, length, atr_tp, atr_sl, timeout, start_time, ma, stoch, close_on_opposite_order, CLOSE_ON_DAY_END):
    
    def run() -> pd.Series:
        atr = smoothing(talib.TRANGE(
            df['High'],
            df['Low'],
            df['Close'],
        ), "rma", length)
        return atr    

    atr_df = run()
    atr_df = pd.DataFrame(atr_df).reset_index()
    atr_df.columns = ['Datetime', 'ATR']
    
    df = df.merge(atr_df, how='inner', left_index=True, right_index=True)
    
    # Calculate stop loss and take profit
    df['take_profit'] = atr_tp * df['ATR']
    if atr_sl != 'none':
        df['stop_loss'] = atr_sl * df['ATR']
    else:
        df['stop_loss'] = 0
    
    
    '''
    def kernel(diff, bandwidth):
        return (1 / (2 * bandwidth)) * math.exp(-abs(diff / bandwidth))
    
    df = tqqq.copy()
    bandwidth = 14
    
    # New 'mean' column initialized with NaN values
    df['mean'] = np.nan
    
    for i in range(bandwidth, len(df)):  # start from 'bandwidth' to ensure enough data
        sum_ = 0
        sumw = 0
        
        for j in range(0, bandwidth):  # start from 1 because Pine's `source[i]` starts from current bar and then goes backwards
            k = (j-1)**2 / bandwidth**2  # j-1 to mimic the 0-based index in the Pine loop
            weight = kernel(k, 1)
            
            # 'i-j' gives the relative past bar in correspondence with Pine's indexing
            sum_ += df.at[i - j, 'Close'] * weight  
            sumw += weight
    
        # Compute and store the mean value for the current row outside the inner loop
        df.at[i, 'mean'] = sum_ / sumw if sumw != 0 else np.nan
    
    df['Signal'] = np.where(df['mean'] > df['mean'].shift(1), 'bullish', 'bearish')
    '''    
    

    def gaussian(diff, bandwidth):
        return np.exp(-np.square(diff / bandwidth) / 2) / np.sqrt(2 * np.pi)
    
    def logistic(diff, bandwidth):
        return 1 / (np.exp(diff / bandwidth) + 2 + np.exp(-diff / bandwidth))
    
    def cosine(diff, bandwidth):
        return np.where(np.abs(diff / bandwidth) <= 1, (np.pi / 4) * np.cos((np.pi / 2) * (diff / bandwidth)), 0.0)
    
    def laplace(diff, bandwidth):
        return (1 / (2 * bandwidth)) * np.exp(-np.abs(diff / bandwidth))
    
    def exponential(diff, bandwidth):
        return (1 / bandwidth) * np.exp(-np.abs(diff / bandwidth))
    
    def silverman(diff, bandwidth):
        return np.where(np.abs(diff / bandwidth) <= 0.5, 0.5 * np.exp(-(diff / bandwidth) / 2) * np.sin((diff / bandwidth) / 2 + np.pi / 4), 0.0)
    
    def cauchy(diff, bandwidth):
        return 1 / (np.pi * bandwidth * (1 + np.square(diff / bandwidth)))
    
    def loglogistic(diff, bandwidth):
        return 1 / np.power(1 + np.abs(diff / bandwidth), 2)
    
    def morters(diff, bandwidth):
        return np.where(np.abs(diff / bandwidth) <= np.pi, (1 + np.cos(diff / bandwidth)) / (2 * np.pi * bandwidth), 0.0)


    kernels = {
        "gaussian": gaussian,
        "logistic": logistic,
        "cosine": cosine,
        "laplace": laplace,
        "exponential": exponential,
        "silverman": silverman,
        "cauchy": cauchy,
        "loglogistic": loglogistic,
        "morters": morters,
    }
    
    def kernel(diff, bandwidth, kernel_name):
        if kernel_name in kernels:
            return kernels[kernel_name](diff, bandwidth)
        else:
            raise ValueError(f"Unknown kernel: {kernel_name}")
            
    
    # Convert 'Close' column to a numpy array for faster access
    closes = df['Close'].values
    
    # Initialize an empty numpy array for 'mean'
    means = np.empty(len(closes))
    means[:] = np.nan
    
    # Iterate over the main loop with numpy-based inner loop calculations
    for i in range(bandwidth, len(closes)):
        # Calculate weights using the inner loop logic
        diffs = (np.arange(bandwidth) - 1)**2 / bandwidth**2
        weights = kernel(diffs, 1, kernel_name)
        
        # Select the range of close values based on i and bandwidth
        selected_closes = closes[i - bandwidth + 1:i + 1]
        
        # Calculate the weighted sum and the sum of weights
        sum_ = np.dot(selected_closes, weights[::-1])  # Note: We reverse weights due to the nature of indexing in the original logic
        sumw = np.sum(weights)
        
        # Calculate the mean
        means[i] = sum_ / sumw if sumw != 0 else np.nan
    
    # Update the DataFrame with results
    df['mean'] = means
    df['older_mean'] = np.roll(df['mean'], 1)
    #df['Signal'] = np.where(df['mean'] > np.roll(df['mean'], 1), 'bullish', 'bearish')

    df['Diff'] = df['mean'] - df['older_mean']

    df['Order_Temp'] = ''
    df.loc[df['Diff'] / df['mean'] >= diff_threshold, 'Order_Temp'] = 'long'
    df.loc[df['Diff'] / df['mean'] <= -diff_threshold, 'Order_Temp'] = 'short'
    
    # Shift the orders up by 1 to simulate live trading
    df['Order_Temp'] = df['Order_Temp'].shift(1)
    
    
    # Apply requirement of x number of consecutive orders    
    if consec >= 2:
        # Create a mask for consecutive 'long' or 'short' values
        def consecutive_mask(ser):
            return ser.groupby((ser != ser.shift()).cumsum()).cumcount().add(1)
        
        df['Consecutive_Long'] = consecutive_mask(df['Order_Temp'] == 'long').where(df['Order_Temp'] == 'long')
        df['Consecutive_Short'] = consecutive_mask(df['Order_Temp'] == 'short').where(df['Order_Temp'] == 'short')
        
        # Create a new column based on the consecutive requirement
        df['Order'] = None
        df.loc[df['Consecutive_Long'] >= consec, 'Order'] = 'long'
        df.loc[df['Consecutive_Short'] >= consec, 'Order'] = 'short'
        
        # Drop helper columns
        df.drop(['Consecutive_Long', 'Consecutive_Short'], axis=1, inplace=True)
    else:
        df['Order'] = df['Order_Temp']

    '''
    # Final order logic
    df['Order_TMP'] = df['Order']
    df.loc[df['Order_TMP'] == 'short', 'Order'] = 'long'
    df.loc[df['Order_TMP'] == 'long', 'Order'] = 'short'
    '''

    # Derive the stochastic
    if stoch != 'none':
        pct_k = int(stoch.split('-')[0])
        pct_d = int(stoch.split('-')[1])
            
        def calculate_smi(df, a, b):
            ll = df['Low'].rolling(window=a).min()
            hh = df['High'].rolling(window=a).max()
            diff = hh - ll
            rdiff = df['Close'] - (hh + ll) / 2
            avgrel = pd.Series(rdiff).ewm(span=b).mean().ewm(span=b).mean()
            avgdiff = pd.Series(diff).ewm(span=b).mean().ewm(span=b).mean()
            SMI = np.where(avgdiff != 0, (avgrel / (avgdiff / 2) * 100), 0)
            SMIsignal = pd.Series(SMI).ewm(span=b).mean()
            return hh, ll, rdiff, avgrel, avgdiff, SMI, SMIsignal
        
        # Assuming you have a pandas DataFrame with a 'close' column
        # Replace df['close'] with the actual column in your DataFrame
        df['hh'], df['ll'], df['rdiff'], df['avgrel'], df['avgdiff'], df['SMI'], df['SMIsignal'] = calculate_smi(df, pct_k, pct_d)
    
        # Final order logic
        #df['Order_TMP'] = df['Order']
        df.loc[df['SMIsignal'] > df['SMI'], 'Stoch_Signal'] = 'short'
        df.loc[df['SMIsignal'] <= df['SMI'], 'Stoch_Signal'] = 'long'
    else:
        df['Stoch_Signal'] = ''
    
    


    # Derive the MA cross
    if ma != 'none':
        # Define the periods for slow and fast MAs
        slow_period = int(ma.split('-')[0])
        fast_period = int(ma.split('-')[1])
        
        # Calculate the slow and fast MAs
        df['Slow_MA'] = df['Close'].rolling(window=slow_period).mean()
        df['Fast_MA'] = df['Close'].rolling(window=fast_period).mean()
        
        # Add a column for the cross type
        # Default value is 'None' (no cross)
        df['Cross'] = 'None'
        
        # Determine if a golden cross or death cross occurs
        for i in range(slow_period, len(df)):
            if df.at[i, 'Fast_MA'] > df.at[i, 'Slow_MA'] and df.at[i - 1, 'Fast_MA'] <= df.at[i - 1, 'Slow_MA']:
                df.at[i, 'Cross'] = 'Golden Cross'
            elif df.at[i, 'Fast_MA'] < df.at[i, 'Slow_MA'] and df.at[i - 1, 'Fast_MA'] >= df.at[i - 1, 'Slow_MA']:
                df.at[i, 'Cross'] = 'Death Cross'
    else:
        df['Slow_MA'] = ''
        df['Fast_MA'] = ''
        df['Cross'] = ''
    

    df['net'] = 0.0
    df['Close Order'] = 0.0
    df['working'] = 0
    start_time = pd.to_datetime('09:00:00').time()
    end_time = pd.to_datetime('16:00:00').time()
    
    # Close order logic
    for i in range(len(df)):
        price = df.at[i, 'Close']
        tp = df.at[i, 'take_profit']
        sl = df.at[i, 'stop_loss']
        order = df.at[i, 'Order']
        working = df.at[i, 'working']
        
        if daily_time_frame == True:
        
            # if order == 'long' and working == 0:
            #     j = 1
            #     while (i + j) < len(df):
            #         # First check if close at day open
            #         if price + tp <= df.at[i + j, 'Open']:
            #             df.at[i + j, 'Close Order'] = df.at[i + j, 'Open']
            #             df.at[i, 'net'] = df.at[i + j, 'Open'] - price
            #             break
            #         elif price + tp <= df.at[i + j, 'High']:
            #             df.at[i + j, 'Close Order'] = price + tp
            #             df.at[i, 'net'] = tp
            #             break
            #         elif close_on_opposite_order == True and df.at[i + j, 'Order'] == 'short':
            #             df.at[i + j, 'Close Order'] = df.at[i + j, 'Close']
            #             df.at[i, 'net'] = df.at[i + j, 'Close'] - price
            #             break 
            #         elif j >= timeout:
            #             df.at[i + j, 'Close Order'] = df.at[i + j, 'Close']
            #             df.at[i, 'net'] = df.at[i + j, 'Close'] - price
            #             break
            #         elif j >= timeout:
            #             df.at[i + j, 'Close Order'] = df.at[i + j, 'Close']
            #             df.at[i, 'net'] = df.at[i + j, 'Close'] - price
            #             break
            #         # Optional stop loss (hour close)
            #         elif atr_sl != 'none' and price - sl >=  df.at[i + j, 'Close']:
            #             df.at[i + j, 'Close Order'] = df.at[i + j, 'Close']
            #             df.at[i, 'net'] = df.at[i + j, 'Close'] - price
            #             break
            #         # MA Close
            #         elif ma != 'none' and df.at[i + j, 'Cross'] == 'Death Cross':
            #             df.at[i + j, 'Close Order'] = df.at[i + j, 'Close']
            #             df.at[i, 'net'] = df.at[i + j, 'Close'] - price
            #             break
            #         # Stoch Close
            #         elif stoch != 'none' and df.at[i + j, 'Stoch_Signal'] == 'short':
            #             df.at[i + j, 'Close Order'] = df.at[i + j, 'Close']
            #             df.at[i, 'net'] = df.at[i + j, 'Close'] - price
            #             break
                    
            #         df.at[i + j, 'working'] = 1
            #         j += 1
            if order == 'short' and working == 0:
                j = 1
                while (i + j) < len(df):
                    # First check if close at day open
                    if price - tp >= df.at[i + j, 'Open']:
                        df.at[i + j, 'Close Order'] = df.at[i + j, 'Open']
                        df.at[i, 'net'] = price - df.at[i + j, 'Open']
                        break
                    elif price - tp >= df.at[i + j, 'Low']:
                        df.at[i + j, 'Close Order'] = price - tp
                        df.at[i, 'net'] = tp
                        break
                    elif CLOSE_ON_DAY_END == True and df.at[i+j, 'Datetime_x'].time() == end_time:
                        df.at[i + j-1, 'Close Order'] = df.at[i + j-1, 'Close']
                        df.at[i, 'net'] = price - df.at[i + j-1, 'Close']
                        df.at[i + j, 'working'] = 1
                        break
                    elif close_on_opposite_order == True and df.at[i + j, 'Order'] == 'long':
                        df.at[i + j, 'Close Order'] = df.at[i + j, 'Close']
                        df.at[i, 'net'] = price - df.at[i + j, 'Close']
                        break 
                    elif j >= timeout:
                        df.at[i + j, 'Close Order'] = df.at[i + j, 'Close']
                        df.at[i, 'net'] = price - df.at[i + j, 'Close']
                        break
                    # Optional stop loss (hour close)
                    elif atr_sl != 'none' and price + sl <= df.at[i + j, 'Close']:
                        df.at[i + j, 'Close Order'] = df.at[i + j, 'Close']
                        df.at[i, 'net'] = price - df.at[i + j, 'Close']
                        break
                    # MA Close
                    elif ma != 'none' and df.at[i + j, 'Cross'] == 'Golden Cross':
                        df.at[i + j, 'Close Order'] = df.at[i + j, 'Close']
                        df.at[i, 'net'] = price - df.at[i + j, 'Close']
                        break
                    # Stoch Close
                    elif stoch != 'none' and df.at[i + j, 'Stoch_Signal'] == 'long':
                        df.at[i + j, 'Close Order'] = df.at[i + j, 'Close']
                        df.at[i, 'net'] = price - df.at[i + j, 'Close']
                        break
                    
                    df.at[i + j, 'working'] = 1
                    j += 1
                    
        if daily_time_frame == False:
            
            # if order == 'long' and working == 0 and (df.at[i, 'Datetime_x'].time() >= start_time) and (df.at[i, 'Datetime_x'].time() <= end_time):
            #     j = 1
            #     while (i + j) < len(df):
            #         if (df.at[i+j, 'Datetime_x'].time() >= start_time) and (df.at[i+j, 'Datetime_x'].time() <= end_time):
            #             # This case handles trades open at start time.
            #             if price + tp <= df.at[i + j, 'High'] and df.at[i+j, 'Datetime_x'].time() == start_time:
            #                 if price + tp <= df.at[i + j, 'Open']:
            #                     df.at[i + j, 'Close Order'] = df.at[i + j, 'Open']
            #                     df.at[i, 'net'] = df.at[i + j, 'Open'] - price
            #                 else:
            #                     df.at[i + j, 'Close Order'] = price + tp
            #                     df.at[i, 'net'] = tp
            #                 break
            #             elif price + tp <= df.at[i + j, 'High'] and df.at[i+j, 'Datetime_x'].time() != start_time:
            #                 df.at[i + j, 'Close Order'] = price + tp
            #                 df.at[i, 'net'] = tp
            #                 break
            #             elif CLOSE_ON_DAY_END == True and df.at[i+j, 'Datetime_x'].time() == end_time:
            #                 df.at[i + j-1, 'Close Order'] = df.at[i + j-1, 'Close']
            #                 df.at[i, 'net'] = df.at[i + j-1, 'Close'] - price
            #                 df.at[i + j, 'working'] = 1
            #                 break
            #             elif close_on_opposite_order == True and df.at[i + j, 'Order'] == 'short':
            #                 df.at[i + j, 'Close Order'] = df.at[i + j, 'Close']
            #                 df.at[i, 'net'] = df.at[i + j, 'Close'] - price
            #                 break 
            #             elif j >= timeout:
            #                 df.at[i + j, 'Close Order'] = df.at[i + j, 'Close']
            #                 df.at[i, 'net'] = df.at[i + j, 'Close'] - price
            #                 break
            #             elif j >= timeout:
            #                 df.at[i + j, 'Close Order'] = df.at[i + j, 'Close']
            #                 df.at[i, 'net'] = df.at[i + j, 'Close'] - price
            #                 break
            #             # Optional stop loss (hour close)
            #             elif atr_sl != 'none' and price - sl >=  df.at[i + j, 'Close']:
            #                 df.at[i + j, 'Close Order'] = df.at[i + j, 'Close']
            #                 df.at[i, 'net'] = df.at[i + j, 'Close'] - price
            #                 break
            #             # MA Close
            #             elif ma != 'none' and df.at[i + j, 'Cross'] == 'Death Cross':
            #                 df.at[i + j, 'Close Order'] = df.at[i + j, 'Close']
            #                 df.at[i, 'net'] = df.at[i + j, 'Close'] - price
            #                 break
            #             # Stoch Close
            #             elif stoch != 'none' and df.at[i + j, 'Stoch_Signal'] == 'short':
            #                 df.at[i + j, 'Close Order'] = df.at[i + j, 'Close']
            #                 df.at[i, 'net'] = df.at[i + j, 'Close'] - price
            #                 break
                    
            #         df.at[i + j, 'working'] = 1
            #         j += 1
            if order == 'short' and working == 0 and (df.at[i, 'Datetime_x'].time() >= start_time) and (df.at[i, 'Datetime_x'].time() <= end_time):
                j = 1
                while (i + j) < len(df):
                    if (df.at[i+j, 'Datetime_x'].time() >= start_time) and (df.at[i+j, 'Datetime_x'].time() <= end_time):
                        # This case handles trades open at start time.
                        if price - tp >= df.at[i + j, 'Low'] and df.at[i+j, 'Datetime_x'].time() == start_time:
                            if price - tp >= df.at[i + j, 'Open']:
                                df.at[i + j, 'Close Order'] = df.at[i + j, 'Open']
                                df.at[i, 'net'] = price - df.at[i + j, 'Open']
                            else:
                                df.at[i + j, 'Close Order'] = price - tp
                                df.at[i, 'net'] = tp
                            break
                        elif price - tp >= df.at[i + j, 'Low'] and df.at[i+j, 'Datetime_x'].time() != start_time:
                            df.at[i + j, 'Close Order'] = price - tp
                            df.at[i, 'net'] = tp
                            break
                        elif CLOSE_ON_DAY_END == True and df.at[i+j, 'Datetime_x'].time() == end_time:
                            df.at[i + j-1, 'Close Order'] = df.at[i + j-1, 'Close']
                            df.at[i, 'net'] = price - df.at[i + j-1, 'Close']
                            df.at[i + j, 'working'] = 1
                            break
                        elif close_on_opposite_order == True and df.at[i + j, 'Order'] == 'long':
                            df.at[i + j, 'Close Order'] = df.at[i + j, 'Close']
                            df.at[i, 'net'] = price - df.at[i + j, 'Close']
                            break 
                        elif j >= timeout:
                            df.at[i + j, 'Close Order'] = df.at[i + j, 'Close']
                            df.at[i, 'net'] = price - df.at[i + j, 'Close']
                            break
                        # Optional stop loss (hour close)
                        elif atr_sl != 'none' and price + sl <= df.at[i + j, 'Close']:
                            df.at[i + j, 'Close Order'] = df.at[i + j, 'Close']
                            df.at[i, 'net'] = price - df.at[i + j, 'Close']
                            break
                        # MA Close
                        elif ma != 'none' and df.at[i + j, 'Cross'] == 'Golden Cross':
                            df.at[i + j, 'Close Order'] = df.at[i + j, 'Close']
                            df.at[i, 'net'] = price - df.at[i + j, 'Close']
                            break
                        # Stoch Close
                        elif stoch != 'none' and df.at[i + j, 'Stoch_Signal'] == 'long':
                            df.at[i + j, 'Close Order'] = df.at[i + j, 'Close']
                            df.at[i, 'net'] = price - df.at[i + j, 'Close']
                            break
                    
                    df.at[i + j, 'working'] = 1
                    j += 1
                    
    # ~~~ Final Aggregations ~~~ #
    df_final = df[['Datetime_x','Date','Open','High','Low','Close','ATR','take_profit','Order','Close Order','working','net','Slow_MA','Fast_MA','Cross','Stoch_Signal']]
    df_final['net_percent'] = df_final['net'] / df_final['Close']
    
    mask = df_final['net_percent'] != 0
    
    # Apply the logic using the boolean mask
    df_final.loc[mask, 'net_percent_slippage_fwd'] = df_final.loc[mask, 'net_percent'] - SLIPPAGE_PCT
    df_final.loc[mask, 'net_percent_slippage_rev'] = df_final.loc[mask, 'net_percent'] + SLIPPAGE_PCT
    df_final['net_percent_slippage_fwd'].fillna(0, inplace=True)
    df_final['net_percent_slippage_rev'].fillna(0, inplace=True)

    
    df_final['cum_pct'] = 1000 * (1 + df_final['net_percent_slippage_fwd']).cumprod()
    cum_pct = (df_final['cum_pct'][len(df)-1] - 1000) / 1000
    df_final['cum_pct_reverse'] = 1000 * (1 + -1*df_final['net_percent_slippage_rev']).cumprod()
    cum_pct_reverse = (df_final['cum_pct_reverse'][len(df)-1] - 1000) / 1000

    grouped_data = df_final.groupby('Date')['net_percent'].sum().reset_index()

    consecutive_positives = (grouped_data['net_percent'] > 0).astype(int).diff().ne(0).cumsum()
    best_streak = grouped_data[grouped_data['net_percent'] > 0].groupby(consecutive_positives)['net_percent'].transform('size').max()

    consecutive_negatives = (grouped_data['net_percent'] < 0).astype(int).diff().ne(0).cumsum()
    worst_streak = grouped_data[grouped_data['net_percent'] < 0].groupby(consecutive_positives)['net_percent'].transform('size').max()
    
    no_of_trades = len(df_final[df_final['net_percent'] != 0.0])
    try:
        avg_trade = df_final['net_percent'].sum() / no_of_trades
        win_percentage = len(df_final[df_final['net_percent'] > 0]) / no_of_trades
    except:
        print('Division by 0')
        avg_trade = 0
        win_percentage = 0
    '''
    Returning: Dataframe used (length), High_period, Low_period, Stdev_period, length, ATR_TP, timeout, WMA, close on opposite order, CLOSE_ON_DAY_END,
               cum percent, cum percent reverse, best trade, worst trade, best day, worst day, 
               best streak, worst streak, number of trades, average trade, win percentage
    '''
    return [cum_pct, cum_pct_reverse, 
            len(df), kernel_name, bandwidth, diff_threshold, consec, length, atr_tp, atr_sl, timeout, start_time, ma, stoch, close_on_opposite_order, CLOSE_ON_DAY_END, 
            df_final['net_percent'].max(), df_final['net_percent'].min(), grouped_data['net_percent'].max(), grouped_data['net_percent'].min(),
            best_streak, worst_streak, no_of_trades, avg_trade, win_percentage]
    


# DF to be filled
final_results = pd.DataFrame(columns=['CumPct', 'CumPctReverse',
                                      'Timeframe', 'Kernel Name', 'Bandwidth', 'Diff Threshold', 'Consec', 'Length', 'ATR_TP', 'ATR_SL', 'Timeout', 'Start Time', 'MA', 'Stoch', 'CloseOnOppOrder', 'CloseOnDayEnd',
                                      'Best Trade', 'Worst Trade', 'Best Day', 'Worst Day',
                                      'Best Streak', 'Worst Streak', 'Number of Trades', 'Avg Trade', 'Win %'])
    
    
# Testing grid
timeframes = [df_full]
close_on_opp_list = [True,False]
close_on_day_end_list = [False]
atr_tp_list = [.5, 1.0, 3.0, 4.0]
atr_sl_list = ['none', 3.0, 5.0, 8.0]
timeout_list = [3, 5, 10, 20, 40, 60]
length_list = [9]
bandwidth_list = [3, 5, 10, 20, 30, 50, 75, 100]
diff_threshold_list = [0]
consec_list = [1, 2]
start_time_list = ['09:30:00']
kernel_name_list = ["gaussian", "logistic", "cosine", "laplace", "silverman", "cauchy", "loglogistic"]
#ma_list = ['none', '50-20','100-50','10-5','30-10']
ma_list = ['none']
#stoch_list = ['none', '5-3', '10-5', '14-3', '20-8', '8-3', '18-7', '15-6']
stoch_list = ['none']



'''
df = df_full.copy()
#df = tqqq2.copy()
kernel_name = "silverman"
bandwidth = 11
diff_threshold = 0
consec = 2
length = 100
atr_tp = 1
atr_sl = 'none'
timeout= 60
start_time = '09:00:00'
close_on_opposite_order = False
CLOSE_ON_DAY_END = False
ma = 'none'
stoch = 'none'
'''



USE_ITERTOOLS = True
TOP_X = 200
TOP_X_SOURCE = 'mkr_timedStop_2023_3mo_5min.csv'


# Either loop through the list using itertools, OR import the top X resuts from a prior timeframe and only run on those.
if USE_ITERTOOLS == True:
    print('Using IterTools')
        
    CSV_NAME = 'mkr_timedStop_2015_market_1hour_short.csv'
    final_results.to_csv(CSV_NAME)
    
    import time 
    time.sleep(2)
    
    import itertools
    import concurrent.futures
    from concurrent.futures import ProcessPoolExecutor
    
    num_cores = os.cpu_count()

    # Generate all combinations of parameters
    combinations = list(itertools.product(timeframes, kernel_name_list, bandwidth_list, diff_threshold_list, consec_list, length_list, atr_tp_list, atr_sl_list, timeout_list, start_time_list, ma_list, stoch_list, close_on_opp_list, close_on_day_end_list))
    
    # Split combinations into quarters
    quarter_length = len(combinations) // 18
    quarters = [combinations[i:i + quarter_length] for i in range(0, len(combinations), quarter_length)]

    def process_combinations(combinations_subset):
        results_subset = []
        for combination in combinations_subset:
            df, kernel_name, bandwidth, diff_threshold, consec, length, atr_tp, atr_sl, timeout, start_time, ma, stoch, close_on, day_end = combination
            print(str(datetime.datetime.now()), len(df), kernel_name, bandwidth, diff_threshold, consec, length, atr_tp, atr_sl, timeout, start_time, ma, stoch, close_on, day_end)
            result = backtester(df, kernel_name, bandwidth, diff_threshold, consec, length, atr_tp, atr_sl, timeout, start_time, ma, stoch, close_on, day_end)
            
            results_subset.append(result)
        return results_subset

    if __name__ == '__main__':
        with ProcessPoolExecutor(max_workers=18) as executor:
            results = list(executor.map(process_combinations, quarters))
            
        # Flatten the results and append them to the final_results DataFrame
        for subset in results:
            for result in subset:
                # Sets the thresholds of the CumPct and CumPctReverse to write to output
                #if subset[0][0] >= 10 or subset[0][1] >= 10:
                final_results.loc[len(final_results)] = result

        final_results['Key'] = final_results['Kernel Name'].astype(str) + '-' + final_results['Bandwidth'].astype(str) + '-' + final_results['Diff Threshold'].astype(str) + '-' + final_results['Consec'].astype(str) + '-' + final_results['Length'].astype(str) + '-' + final_results['ATR_TP'].astype(str) + '-' + final_results['ATR_SL'].astype(str) + '-' + final_results['Timeout'].astype(str) + '-' + final_results['Start Time'].astype(str) + '-' + final_results['MA'].astype(str) + '-' + final_results['Stoch'].astype(str) + '-' + final_results['CloseOnOppOrder'].astype(str) + '-' + final_results['CloseOnDayEnd'].astype(str)
        final_results.to_csv(CSV_NAME)

    
else:
    print('Not using IterTools')
    
    top = pd.read_csv(TOP_X_SOURCE)
    top['best_result'] = np.maximum(top['CumPct'], top['CumPctReverse'])
    top.sort_values(by='best_result', inplace=True, ascending=False)
    top = top.reset_index(drop=True)
    top = top.head(TOP_X)
    df = timeframes[0]
    
    for row in top.iterrows():
        print(str(datetime.datetime.now()), len(df), row[1]['Kernel Name'], row[1]['Bandwidth'], row[1]['Diff Threshold'], row[1]['Consec'], row[1]['Length'], row[1]['ATR_TP'], row[1]['Timeout'], row[1]['Start Time'], row[1]['CloseOnOppOrder'], row[1]['CloseOnDayEnd'])
        result = backtester(df, row[1]['Kernel Name'], row[1]['Bandwidth'], row[1]['Diff Threshold'], row[1]['Consec'], row[1]['Length'], row[1]['ATR_TP'], row[1]['Timeout'], row[1]['Start Time'], row[1]['CloseOnOppOrder'], row[1]['CloseOnDayEnd'])
        final_results.loc[len(final_results)] = result
    
    final_results['Key'] = final_results['Kernel Name'].astype(str) + '-' + final_results['Bandwidth'].astype(str) + '-' + final_results['Diff Threshold'].astype(str) + '-' + final_results['Consec'].astype(str) + '-' + final_results['Length'].astype(str) + '-' + final_results['ATR_TP'].astype(str) + '-' + final_results['ATR_SL'].astype(str) + '-' + final_results['Timeout'].astype(str) + '-' + final_results['Start Time'].astype(str) + '-' + final_results['CloseOnOppOrder'].astype(str) + '-' + final_results['CloseOnDayEnd'].astype(str)
    final_results.to_csv('mkr_timedStop_2023_5min.csv')
    




'''
# To extract data for the long term testing:
test = df_final[['Date', 'net', 'net_percent', 'cum_pct', 'cum_pct_reverse']]
test = test.drop_duplicates()
test.to_csv('mkr long term performance.csv')

test['year'] = test['Date'].astype(str).str[:4]
result = test.groupby('year')['net_percent'].sum()
result = result.reset_index()
print(result)


###########################################################################
# To get day-over-day trades only
###########################################################################
last = df_final.copy()
last['Order'] = last['Order'].replace('', np.nan)
last['Order'] = last['Order'].ffill()

mask = (last['Datetime_x'].dt.time == pd.to_datetime('09:30:00').time()) | (last['Datetime_x'].dt.time == pd.to_datetime('16:00:00').time())
filtered_df = last[mask]

filtered_df['Next_Open'] = filtered_df['Open'].shift(-1)

filtered_df['EOD Trade'] = filtered_df.apply(lambda row: row['Next_Open'] - row['Close'] if row['Datetime_x'].time() == pd.to_datetime('16:00:00').time() and row['working'] == 1 and row['Order'] == 'long' else None, axis=1)
filtered_df['EOD Trade'] = filtered_df.apply(lambda row: row['Close'] - row['Next_Open'] if row['Datetime_x'].time() == pd.to_datetime('16:00:00').time() and row['working'] == 1 and row['Order'] == 'short' else row['EOD Trade'], axis=1)

filtered_df['EOD Trade Pct'] = filtered_df['EOD Trade'] / filtered_df['Close']


filtered_df['cum_pct_eod'] = 1000 * (1 + filtered_df['EOD Trade Pct']).cumprod()
filtered_df['cum_pct_reverse_eod'] = 1000 * (1 + -1*filtered_df['EOD Trade Pct']).cumprod()
'''