import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def get_stock_data(ticker, start=None, end=None, hourly=False, log_returns = True):
    end_date = datetime.today() if end is None else datetime.strptime(end, "%Y-%m-%d")
    
    if hourly:
        default_start = end_date - timedelta(days=60)
        start_date = default_start if start is None else datetime.strptime(start, '%Y-%m-%d')
        
        if (end_date - start_date).days > 730:
            start_date = end_date - timedelta(days=730)
            print(f"start date adjusted to {start_date.strftime('%Y-%m-%d')}")
        
    else:       
        default_start = end_date - timedelta(days=365)
        start_date = default_start if start is None else datetime.strptime(start, '%Y-%m-%d')
        
    df = yf.download(
        tickers=ticker,
        start=start_date,
        end=end_date,
        interval='60m' if hourly else '1d',
        prepost= True if hourly else False
    )
    
    df.columns = df.columns.get_level_values(0)
    if log_returns:
        df["logReturns"] = np.log(df.Close / df.Close.shift(1))
        df = df.dropna()
        
    df = df.reset_index()
    columns = ['Datetime' if hourly else 'Date', 'Open', 'High', 'Low', 'Close', 'Volume']
    if log_returns : columns = columns  + ['logReturns']
    df = df[columns]
    df.columns.name = None
    
    # if hourly:
    #     df = df.between_time('09:30', '16:00')
    
    return df

def windowed_dfs(df, columns, target, train_window):
    target_row = df[target].iloc[train_window:].values

    full_windowed = np.array([])
    for col in columns:
        df_col = df[col]
        windowed_col = np.array([df_col.iloc[i:i+train_window].values for i in range(len(df_col) - train_window)])
        if len(full_windowed) == 0:
            full_windowed = windowed_col
        else:
            full_windowed = np.concatenate((full_windowed, windowed_col), axis=1)
    column_names = [f"{col.lower()}-{i}" for col in columns for i in range(train_window, 0, -1) ]
    windowed_df= pd.DataFrame(full_windowed, columns=column_names)
    
    return windowed_df, target_row

def custom_windowed_dfs(df, column_data, target):
    max_horizon = max(column_data.values())
    target_row = df[target].iloc[max_horizon:].values

    full_windowed = np.array([])
    for col in column_data:
        df_col = df[col]
        windowed_col = np.array([df_col.iloc[i-column_data[col]:i].values for i in range(max_horizon, len(df_col))])
        
        if len(full_windowed) == 0:
            full_windowed = windowed_col
        else:
            full_windowed = np.concatenate((full_windowed, windowed_col), axis=1)

    column_names = [f"{col.lower()}-{i}" for col in column_data for i in range(column_data[col], 0, -1) ]
    windowed_df= pd.DataFrame(full_windowed, columns=column_names)
    
    return windowed_df, target_row