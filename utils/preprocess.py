import ta
import numpy as np
import pandas as pd

def add_indicators(df):
    if 'Date' in df:
        df_indicators = df.set_index("Date")
        df_indicators["DayOfWeek"] = pd.to_datetime(df_indicators.index).dayofweek
    elif 'Datetime' in df:
        df_indicators = df.set_index("Datetime")
        df_indicators["DayOfWeek"] = pd.to_datetime(df_indicators.index).dayofweek
        df_indicators['Hour'] = pd.to_datetime(df_indicators.index).hour
        df_indicators['Hour_sin'] = np.sin(2 * np.pi * df_indicators['Hour']/24)
        df_indicators['Hour_cos'] = np.cos(2 * np.pi * df_indicators['Hour']/24)
    else:
        df_indicators = df.copy()
        
    df_indicators["RSI5"] = ta.momentum.rsi(df_indicators['Close'], window=5)
    df_indicators["RSI10"] = ta.momentum.rsi(df_indicators['Close'], window=10)
    df_indicators['SMA_5'] = ta.trend.sma_indicator(df_indicators['Close'], window=5)
    df_indicators['SMA_10'] = ta.trend.sma_indicator(df_indicators['Close'], window=10)
    df_indicators['EMA_10'] = ta.trend.ema_indicator(df_indicators['Close'], window=10)

    df_indicators['MACD'] = ta.trend.macd_diff(df_indicators['Close'])
    df_indicators['BB_upper'], df_indicators['BB_middle'], df_indicators['BB_lower'] = ta.volatility.bollinger_hband(df_indicators['Close']), ta.volatility.bollinger_mavg(df_indicators['Close']), ta.volatility.bollinger_lband(df_indicators['Close'])
    # df_indicators['ATR_10'] = ta.volatility.average_true_range(df_indicators['High'], df_indicators['Low'], df_indicators['Close'], window=10)
    
    df_indicators = df_indicators.dropna()
    
    return df_indicators

def train_test_split(X, y, dates, test_date='2025-01-01', window_size=14):
    if isinstance(test_date, str):
        test_date = pd.to_datetime(test_date)
    test_start_pos = np.where(dates > test_date)[0][0]
    train_end_pos = test_start_pos - window_size
    
    X_train, y_train = X[:train_end_pos], y[:train_end_pos]
    X_test, y_test = X[train_end_pos + window_size:], y[train_end_pos + window_size:]
    
    test_dates = dates[train_end_pos + window_size:]
    
    return X_train, y_train, X_test, y_test, test_dates