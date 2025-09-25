from typing import Dict, List
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from xgboost import XGBRegressor
import tensorflow as tf
from joblib import Parallel, delayed
import joblib
import os
import logging

def create_features(df: pd.DataFrame, config: Dict) -> pd.DataFrame:
    """
    Engineer features: lags, indicators.

    Args:
        df (pd.DataFrame): OHLCV data.
        config (Dict): Config for periods.

    Returns:
        pd.DataFrame: Features.
    """
    # Lags
    for lag in config['features']['lags']:
        df[f'lag_{lag}'] = df['Close'].shift(lag)

    # RSI
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=config['features']['indicators']['rsi_period']).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=config['features']['indicators']['rsi_period']).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))

    # MACD
    ema_fast = df['Close'].ewm(span=config['features']['indicators']['macd_fast'], adjust=False).mean()
    ema_slow = df['Close'].ewm(span=config['features']['indicators']['macd_slow'], adjust=False).mean()
    df['MACD'] = ema_fast - ema_slow
    df['MACD_signal'] = df['MACD'].ewm(span=config['features']['indicators']['macd_signal'], adjust=False).mean()

    # SMA
    df['SMA'] = df['Close'].rolling(window=config['features']['indicators']['sma_period']).mean()

    # Volatility
    df['Volatility'] = df['Close'].pct_change().rolling(20).std()

    # Drop NaNs
    df.dropna(inplace=True)
    return df

def train_model(stock: str, df: pd.DataFrame, config: Dict):
    """
    Train ML model for a stock.

    Args:
        stock (str): Ticker.
        df (pd.DataFrame): Data.
        config (Dict): Model config.

    Returns:
        model: Trained model.
    """
    features_df = create_features(df.copy(), config)
    X = features_df.drop(['Open', 'High', 'Low', 'Close', 'Volume'], axis=1, errors='ignore')  # Features
    y = features_df['Close'].pct_change().shift(-1)  # Next % change as target
    features_df.dropna(inplace=True)

    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=config['train_test_split'], shuffle=False)

    if config['model_type'] == 'RandomForest':
        model = RandomForestRegressor(n_estimators=100, random_state=42)
    elif config['model_type'] == 'GradientBoosting':
        model = XGBRegressor(n_estimators=100, random_state=42)
    elif config['model_type'] == 'LSTM':
        X_train = np.reshape(X_train.values, (X_train.shape[0], X_train.shape[1], 1))
        model = tf.keras.Sequential([
            tf.keras.layers.LSTM(50, input_shape=(X_train.shape[1], 1)),
            tf.keras.layers.Dense(1)
        ])
        model.compile(optimizer='adam', loss='mse')
        model.fit(X_train, y_train, epochs=10, batch_size=32, verbose=0)
        # For LSTM, test reshape similarly
        X_test = np.reshape(X_test.values, (X_test.shape[0], X_test.shape[1], 1))
    else:
        raise ValueError("Unsupported model type.")

    if config['model_type'] != 'LSTM':
        model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    logging.info(f"Model for {stock}: MSE = {mse}")

    # Save model
    os.makedirs('models', exist_ok=True)
    joblib.dump(model, f'models/{stock}_{config["model_type"]}.pkl') if config['model_type'] != 'LSTM' else model.save(f'models/{stock}_LSTM.h5')

    return model

def train_models(data: Dict[str, pd.DataFrame], stocks: List[str], config: Dict) -> Dict[str, object]:
    """
    Train models in parallel.

    Args:
        data (Dict): Historical data.
        stocks (List[str]): Stocks to train.
        config (Dict): Config.

    Returns:
        Dict[str, object]: Models per stock.
    """
    models = Parallel(n_jobs=-1)(delayed(train_model)(stock, data[stock], config) for stock in stocks)
    return dict(zip(stocks, models))