import tensorflow as tf
import tensorflow.keras.ops as ops
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import (LSTM, Dense, Dropout, Input, 
                                    MultiHeadAttention, LayerNormalization,
                                    Conv1D, GlobalAveragePooling1D, Reshape, Layer)
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from xgboost import XGBRegressor
import numpy as np


def build_lstm_model(input_shape):
    model = Sequential([
        LSTM(128, activation='tanh', return_sequences=True, input_shape=input_shape),
        Dropout(0.3),
        LSTM(64, activation='tanh', return_sequences=True),
        Dropout(0.3),
        LSTM(32, activation='tanh'),
        Dropout(0.3),
        Dense(16, activation='relu'),
        Dense(1)
    ])
    return model

def build_rf_model():
    return RandomForestRegressor(
        n_estimators=200,
        max_depth=20,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1
    )

def build_ridge_model():
    return Ridge(
        alpha=1.0,
        random_state=42
    )

def build_xgboost_model():
    return XGBRegressor(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1
    )

def build_rf_tuned():
    return RandomForestRegressor(
        n_estimators=200,
        max_depth=20,
        min_samples_split=2,
        min_samples_leaf=4,
        max_features='sqrt',
        random_state=42,
        n_jobs=-1
    )

def build_xgboost_tuned():
    return XGBRegressor(
        n_estimators=100,
        max_depth=3,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=1.0,
        gamma=0.1,
        random_state=42,
        n_jobs=-1
    )

def compile_model(model, model_type='lstm', learning_rate=0.001):
    if model_type in ['lstm', 'transformer']:
        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        model.compile(optimizer=optimizer, 
                     loss=tf.keras.losses.Huber(), 
                     metrics=['mae'])
    return model

def train_model(model, X_train, y_train, model_type='lstm'):
    if model_type in ['lstm', 'transformer']:
        early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=5,
            restore_best_weights=True)
        history = model.fit(
            X_train, y_train,
            epochs=50,
            batch_size=64,
            callbacks=[early_stopping],
            verbose=1)
        return model, history
    else:
        X_reshaped = X_train.reshape(X_train.shape[0], -1)
        model.fit(X_reshaped, y_train)
        return model, None

def forecast_future(model, last_window, scaler, model_type='lstm', horizon=30, window_size=60):
    forecasts = []
    current_window = last_window.copy()
    
    for _ in range(horizon):
        if model_type in ['lstm', 'transformer']:
            model_input = current_window.reshape(1, window_size, 1)
            next_pred = model.predict(model_input, verbose=0)[0, 0]
        else:
            model_input = current_window.reshape(1, -1)
            next_pred = model.predict(model_input)[0]
        
        forecasts.append(next_pred)
        current_window = np.vstack([current_window[1:], [[next_pred]]])
    
    return scaler.inverse_transform(np.array(forecasts).reshape(-1, 1))