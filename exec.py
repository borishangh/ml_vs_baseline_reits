import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import os
from utils.models import (
    build_lstm_model,
    build_ridge_model,
    build_xgboost_model,
    build_rf_model,
    compile_model,
    train_model,
    forecast_future,
)


def prepare_data(
    ticker, start_date="2015-01-01", end_date="2025-03-01", window_size=60
):
    df = yf.download(ticker, start=start_date, end=end_date)
    df.columns = ["Close", "High", "Low", "Open", "Volume"]
    df = df.reset_index()
    df.dropna(inplace=True)

    df = df[["Date", "Close"]]
    df.set_index("Date", inplace=True)

    scaler = MinMaxScaler(feature_range=(0, 1))
    df["Close"] = scaler.fit_transform(df[["Close"]])

    return df, scaler, window_size


def create_sequences(data, window_size):
    features, targets = [], []
    for i in range(len(data) - window_size):
        features.append(data[i : i + window_size])
        targets.append(data[i + window_size])
    return np.array(features), np.array(targets)


def train_test_split(X, y, dates, test_date="2023-01-01", window_size=60):
    if isinstance(test_date, str):
        test_date = pd.to_datetime(test_date)
    test_start_pos = np.where(dates > test_date)[0][0]
    train_end_pos = test_start_pos - window_size

    X_train, y_train = X[:train_end_pos], y[:train_end_pos]
    X_test, y_test = X[train_end_pos + window_size :], y[train_end_pos + window_size :]

    test_dates = dates[train_end_pos + window_size + window_size :]

    return X_train, y_train, X_test, y_test, test_dates


def save_results(
    ticker, model_type, test_dates, y_test, y_pred, future_dates, future_pred
):
    """Save results to files"""
    os.makedirs(f"charts/{model_type}", exist_ok=True)
    os.makedirs(f"data/{model_type}", exist_ok=True)

    plt.figure(figsize=(14, 7))
    plt.plot(test_dates, y_test, label="Actual Price", color="blue")
    plt.plot(test_dates, y_pred, label="Predicted Price", color="red")
    plt.plot(
        future_dates,
        future_pred,
        label="Forecasted Price",
        color="green",
        linestyle="dashed",
    )
    plt.xlabel("Date")
    plt.ylabel(f"{ticker} Stock Price")
    plt.title(f"{model_type.upper()} Stock Price Forecast")
    plt.legend()
    plt.savefig(f"charts/{model_type}/{ticker.split('.')[0]}_predict.png")
    plt.close()

    pd.DataFrame(
        {"Date": test_dates, "Actual": y_test.flatten(), "Predicted": y_pred.flatten()}
    ).to_csv(f"data/{model_type}/{ticker.split('.')[0]}_predict.csv", index=False)

    pd.DataFrame({"Date": future_dates, "Forecasted": future_pred.flatten()}).to_csv(
        f"data/{model_type}/{ticker.split('.')[0]}_forecast.csv", index=False
    )


def execute(ticker, model_type="lstm", window_size=60):
    print(f"Processing {ticker} with {model_type} model (window={window_size})...")

    # 1. Prepare data
    df, scaler, window_size = prepare_data(ticker, window_size=window_size)
    X, y = create_sequences(df["Close"].values, window_size)
    X_train, y_train, X_test, y_test, test_dates = train_test_split(
        X, y, df.index, window_size=window_size
    )

    # 2. Build and train model
    if model_type == "lstm":
        model = build_lstm_model((window_size, 1))
        model = compile_model(model, model_type)
        X_train_reshaped = X_train.reshape((X_train.shape[0], window_size, 1))
        X_test_reshaped = X_test.reshape((X_test.shape[0], window_size, 1))
    elif model_type == "rf":
        model = build_rf_model()
        X_train_reshaped = X_train.reshape(X_train.shape[0], -1)
        X_test_reshaped = X_test.reshape(X_test.shape[0], -1)
    elif model_type == "ridge":
        model = build_ridge_model()
        X_train_reshaped = X_train.reshape(X_train.shape[0], -1)
        X_test_reshaped = X_test.reshape(X_test.shape[0], -1)
    elif model_type == "xgboost":
        model = build_xgboost_model()
        X_train_reshaped = X_train.reshape(X_train.shape[0], -1)
        X_test_reshaped = X_test.reshape(X_test.shape[0], -1)

    model, history = train_model(model, X_train_reshaped, y_train, model_type)

    # 3. Evaluate and forecast
    y_pred = model.predict(X_test_reshaped)
    y_pred = scaler.inverse_transform(y_pred.reshape(-1, 1))
    y_test = scaler.inverse_transform(y_test.reshape(-1, 1))

    # 4. Generate forecasts
    last_window = df["Close"].values[-window_size:].reshape(-1, 1)
    future_pred = forecast_future(
        model, last_window, scaler, model_type, window_size=window_size
    )
    future_dates = pd.date_range(start=df.index[-1] + pd.Timedelta(days=1), periods=30)

    # 5. Save results
    save_results(
        ticker, model_type, test_dates, y_test, y_pred, future_dates, future_pred
    )
    print(f"Completed {ticker} with {model_type}\n")


if __name__ == "__main__":
    window_size = 60
    real_estate_stocks = [
        # "DLF.NS",
        "LODHA.NS",
        "GODREJPROP.NS",
        "OBEROIRLTY.NS",
        "PRESTIGE.NS",
        "PHOENIXLTD.NS",
        "BRIGADE.NS",
        "SOBHA.NS",
        "MAHLIFE.NS",
    ]

    reits = ["EMBASSY.BO", "MINDSPACE.BO", "NXST.BO", "BIRET.BO"]

    stocks = real_estate_stocks + reits
    # Process all stocks with all models
    for ticker in stocks:
        for model_type in ["rf", "ridge", "xgboost", "lstm"]:
            execute(ticker, model_type, window_size=window_size)
