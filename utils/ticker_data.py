import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta

def get_stock_data(ticker, start=None, end=datetime.today()):
    if start is None:
        start = end - timedelta(days=365)
    df = yf.download(ticker, start=start.strftime('%Y-%m-%d'), end=end.strftime('%Y-%m-%d'))
    
    df.columns = df.columns.get_level_values(0)
    df = df.reset_index()
    df = df[['Date', 'Open', 'High', 'Low', 'Close', 'Volume']]
    df.columns.name = None

    return df