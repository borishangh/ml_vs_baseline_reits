import ta
import numpy as np
import pandas as pd

def prepare_data(df, horizon=1):
    if 'Datetime' not in df:
        return ValueError('need hourly data')
    
    df = df.set_index('Datetime')
    
    df['Return'] = df['Close'].pct_change()
    df['Volatility_24h'] = df['Return'].rolling(24).std()
    
    df['RSI_14'] = ta.momentum.rsi(df['Close'], window=14)
    df['MACD'] = ta.trend.macd_diff(df['Close'])
    df['BB_upper'], df['BB_middle'], df['BB_lower'] = ta.volatility.bollinger_hband(df['Close']), ta.volatility.bollinger_mavg(df['Close']), ta.volatility.bollinger_lband(df['Close'])
    df['ATR_14'] = ta.volatility.average_true_range(df['High'], df['Low'], df['Close'], window=14)
    
    df['DayOfWeek'] = pd.to_datetime(df.index).dayofweek
    df['Hour'] = pd.to_datetime(df.index).hour
    df['Hour_sin'] = np.sin(2 * np.pi * df['Hour']/24)
    df['Hour_cos'] = np.cos(2 * np.pi * df['Hour']/24)
    
    for lag in [1, 4, 12, 24, 48]:
        df[f'Return_lag_{lag}'] = df['Return'].shift(lag)
        df[f'Volume_lag_{lag}'] = df['Volume'].shift(lag)
    
    df['Target'] = df['Close'].shift(-horizon)
    
    df = df.dropna()
    
    features = df.drop(columns=['Target'])
    target = df['Target']
    
    return features, target