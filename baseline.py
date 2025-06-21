import yfinance as yf
import pandas as pd
import numpy as np
import os
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline

def fetch_stock_data(ticker, start_date="2023-01-01"):
    df = yf.download(ticker, start=start_date)
    df.columns = ['Close', 'High', 'Low', 'Open', 'Volume']
    df = df.reset_index()
    df.dropna(inplace=True)
    df = df[['Date', 'Close']]
    df.set_index('Date', inplace=True)
    return df

def create_windowed_data(data, window_size=7):
    X, y = [], []
    for i in range(window_size, len(data)):
        window = data.iloc[i-window_size:i].values
        target = data.iloc[i]
        X.append(window)
        y.append(target)
    
    X = np.array(X)
    y = np.array(y)
    return X, y, data.index[window_size:]

def generate_baseline_predictions(X, window_size=7):
    y_pred_last_day = X[:, -1]
    
    linear_preds = []
    for window in X:
        model = LinearRegression()
        x = np.arange(window_size).reshape(-1, 1)
        model.fit(x, window)
        linear_preds.append(model.predict([[window_size]]).item())
    
    y_pred_mean = X.mean(axis=1)
    
    quadratic_preds = []
    for window in X:
        model = make_pipeline(PolynomialFeatures(degree=2), LinearRegression())
        x = np.arange(window_size).reshape(-1, 1)
        model.fit(x, window)
        quadratic_preds.append(model.predict([[window_size]]).item())
    
    return {
        'last_day': y_pred_last_day,
        'linear': np.array(linear_preds),
        'mean': y_pred_mean,
        'quadratic': np.array(quadratic_preds)
    }

def save_predictions_to_csv(predictions, dates, actual, ticker, base_dir="baseline_data"):
    for model_type in ['last_price', 'linear', 'mean', 'quadratic']:
        os.makedirs(f"{base_dir}/{model_type}", exist_ok=True)
    
    pd.DataFrame({
        'Date': dates,
        'Actual': actual,
        'Predicted': predictions['last_day']
    }).to_csv(f"{base_dir}/last_price/{ticker}_predictions.csv", index=False)
    
    pd.DataFrame({
        'Date': dates,
        'Actual': actual,
        'Predicted': predictions['linear']
    }).to_csv(f"{base_dir}/linear/{ticker}_predictions.csv", index=False)
    
    pd.DataFrame({
        'Date': dates,
        'Actual': actual,
        'Predicted': predictions['mean']
    }).to_csv(f"{base_dir}/mean/{ticker}_predictions.csv", index=False)
    
    pd.DataFrame({
        'Date': dates,
        'Actual': actual,
        'Predicted': predictions['quadratic']
    }).to_csv(f"{base_dir}/quadratic/{ticker}_predictions.csv", index=False)

def make_baseline_csv(ticker_list, start_date="2023-01-01"):
    for ticker in ticker_list:
        print(f"on {ticker}")
        try:
            df = fetch_stock_data(ticker, start_date)
            X, y, dates = create_windowed_data(df['Close'])
            predictions = generate_baseline_predictions(X)
            
            save_predictions_to_csv(predictions, dates, y, ticker)
            print(f"processed {ticker}")
        except Exception as e:
            print(f"error on {ticker}: {str(e)}")

ticker_list = [
    "DLF.NS", "LODHA.NS", "GODREJPROP.NS", "OBEROIRLTY.NS", 
    "PRESTIGE.NS", "EMBASSY.BO", "MINDSPACE.BO", "NXST.BO", "BIRET.BO"
]

make_baseline_csv(ticker_list)