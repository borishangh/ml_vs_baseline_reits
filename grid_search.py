import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import make_scorer, mean_absolute_error
import multiprocessing
import warnings
warnings.filterwarnings('ignore')

STOCKS = ["DLF.NS", "LODHA.NS", "GODREJPROP.NS"]
WINDOW_SIZE = 60
TEST_DATE = '2023-01-01'
N_JOBS = multiprocessing.cpu_count() - 1

def prepare_data(ticker, start_date='2015-01-01', end_date='2025-03-01'):
    df = yf.download(ticker, start=start_date, end=end_date)
    df = df[['Close']].reset_index()
    df.dropna(inplace=True)
    return df

def create_sequences(data, window_size):
    X, y = [], []
    for i in range(len(data) - window_size):
        X.append(data[i:i + window_size])
        y.append(data[i + window_size])
    return np.array(X), np.array(y)

def time_series_train_test_split(X, y, dates, test_date, window_size):
    if isinstance(test_date, str):
        test_date = pd.to_datetime(test_date)
    test_start_pos = np.where(dates > test_date)[0][0]
    train_end_pos = test_start_pos - window_size
    
    X_train = X[:train_end_pos]
    y_train = y[:train_end_pos]
    X_test = X[train_end_pos + window_size:]
    y_test = y[train_end_pos + window_size:]
    
    return X_train, y_train, X_test, y_test

def tune_model(model, param_grid, X_train, y_train):
    tscv = TimeSeriesSplit(n_splits=5)
    scorer = make_scorer(mean_absolute_error, greater_is_better=False)
    
    if len(X_train.shape) == 3:
        X_train_reshaped = X_train.reshape(X_train.shape[0], -1)
    else:
        X_train_reshaped = X_train
    
    grid_search = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        cv=tscv,
        scoring=scorer,
        verbose=2,
        n_jobs=N_JOBS,
        return_train_score=True
    )
    
    grid_search.fit(X_train_reshaped, y_train)
    return grid_search

def main():
    results = []
    
    for ticker in STOCKS:
        print(f"\n{'='*50}")
        print(f"Tuning models for {ticker}")
        print(f"{'='*50}")
        
        df = prepare_data(ticker)
        X, y = create_sequences(df['Close'].values, WINDOW_SIZE)
        dates = df['Date'][WINDOW_SIZE:].values 

        X_train, y_train, X_test, y_test = time_series_train_test_split(
            X, y, dates, TEST_DATE, WINDOW_SIZE
        )
        
        scaler = MinMaxScaler()
        X_train_scaled = scaler.fit_transform(
            X_train.reshape(-1, 1)
        ).reshape(X_train.shape)
        X_test_scaled = scaler.transform(
            X_test.reshape(-1, 1)
        ).reshape(X_test.shape)
        
        print("\nTuning Random Forest...")
        rf_param_grid = {
            'n_estimators': [100, 200, 300],
            'max_depth': [10, 20, 30, None],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'max_features': ['sqrt', 'log2']
        }
        
        rf_grid = tune_model(
            RandomForestRegressor(random_state=42, n_jobs=N_JOBS),
            rf_param_grid,
            X_train_scaled,
            y_train
        )
        
        xgb_param_grid = {
            'n_estimators': [100, 200, 300],
            'max_depth': [3, 6, 9],
            'learning_rate': [0.01, 0.1, 0.2],
            'subsample': [0.6, 0.8, 1.0],
            'colsample_bytree': [0.6, 0.8, 1.0],
            'gamma': [0, 0.1, 0.2]
        }
        
        xgb_grid = tune_model(
            XGBRegressor(random_state=42, n_jobs=N_JOBS),
            xgb_param_grid,
            X_train_scaled,
            y_train
        )
        
        results.append({
            'ticker': ticker,
            'rf_best_params': rf_grid.best_params_,
            'rf_best_score': -rf_grid.best_score_,
            'xgb_best_params': xgb_grid.best_params_,
            'xgb_best_score': -xgb_grid.best_score_
        })
        
        print(f"\n{ticker} Results:")
        print("Random Forest Best Params:", rf_grid.best_params_)
        print("Random Forest Best MAE:", -rf_grid.best_score_)
        print("XGBoost Best Params:", xgb_grid.best_params_)
        print("XGBoost Best MAE:", -xgb_grid.best_score_)
    
    results_df = pd.DataFrame(results)
    results_df.to_csv('hptuning_results.csv', index=False)
    
    print("\nFinal Summary:")
    print(results_df)

if __name__ == "__main__":
    main()