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

def windowed_dfs(df, size):
    df = df[["Date", "Close"]]
    
    labels = np.array([f'{i}-days-before' for i in range(size-1, 0, -1)])
    prices = np.array([df.Close.iloc[i:i+size-1] for i in range(len(df)-size)])
    X = pd.DataFrame(prices, columns=labels)
    
    y = pd.DataFrame(df.iloc[size:]).reset_index()
    y = y[['Date', 'Close']]
    
    return X,y