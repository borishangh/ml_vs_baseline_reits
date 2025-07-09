import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def get_stock_data(ticker, start=None, end=None):
    end = datetime.today() if end is None else datetime.strptime(end, "%Y-%m-%d")
    start = end - timedelta(days=365) if start is None else datetime.strptime(start, '%Y-%m-%d')
    
    df = yf.download(ticker, start=start.strftime('%Y-%m-%d'), end=end.strftime('%Y-%m-%d'))
    
    df.columns = df.columns.get_level_values(0)
    df["Returns"] = np.log(df.Close / df.Close.shift(1))
    df = df.dropna()
    
    df = df.reset_index()
    df = df[['Date', 'Open', 'High', 'Low', 'Close', 'Volume', 'Returns']]
    df.columns.name = None
    
    return df

def windowed_dfs(df, size):
    df = df[["Date", "Close"]]
    
    labels = np.array([f'{i}-days-before' for i in range(size-1, 0, -1)])
    prices = np.array([df.Close.iloc[i:i+size-1] for i in range(len(df)-size)])
    X = pd.DataFrame(prices, columns=labels)
    
    y = pd.DataFrame(df.iloc[size:]).reset_index()
    y = y[['Date', 'Close']]
    
    return X,y