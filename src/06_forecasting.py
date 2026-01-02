"""
Forecasting Module
Implements LSTM models for BTC price forecasting.
"""

import numpy as np
import pandas as pd
import math
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras import layers
from tensorflow.keras.layers import Dense, LSTM, Dropout
from sklearn.preprocessing import MinMaxScaler
from typing import Tuple


def prepare_forecasting_data(data: pd.DataFrame, lookback: int = 60,
                            train_ratio: float = 0.8) -> Tuple[np.ndarray, np.ndarray, 
                                                                np.ndarray, np.ndarray, int]:
    """
    Prepare data for LSTM forecasting model.
    
    Args:
        data: DataFrame with 'close' prices
        lookback: Number of previous timesteps to use as variables
        train_ratio: Ratio of data to use for training
        
    Returns:
        Tuple of (X_train, y_train, X_test, y_test, train_idx)
    """
    close_array = data.values
    train_len = math.ceil(len(close_array) * train_ratio)
    
    # Training data
    train_data = close_array[:train_len]
    X_train, y_train = _create_sequences_for_forecast(train_data, lookback)
    
    # Testing data
    test_data = close_array[train_len - lookback:]
    X_test, y_test = _create_sequences_for_forecast(test_data, lookback)
    
    print(f"Forecasting data prepared:")
    print(f"  X_train: {X_train.shape}, y_train: {y_train.shape}")
    print(f"  X_test: {X_test.shape}, y_test: {y_test.shape}")
    
    return X_train, y_train, X_test, y_test, train_len


def _create_sequences_for_forecast(data: np.ndarray, lookback: int) -> Tuple[np.ndarray, np.ndarray]:
    """Helper function to create sequences for forecasting."""
    X, y = [], []
    for i in range(lookback, len(data)):
        X.append(data[i - lookback:i, 0])
        y.append(data[i, 0])
    return np.array(X), np.array(y)


def build_lstm_model(timesteps: int, lstm1_units: int = 512, 
                    lstm2_units: int = 256) -> Sequential:
    """
    Build standard LSTM model for price forecasting.
    
    Args:
        timesteps: Number of timesteps in input
        lstm1_units: Number of units in first LSTM layer
        lstm2_units: Number of units in second LSTM layer
        
    Returns:
        Compiled Keras Sequential model
    """
    model = Sequential()
    model.add(LSTM(units=lstm1_units, return_sequences=True, 
                   activation='relu', input_shape=(timesteps, 1)))
    model.add(LSTM(units=lstm2_units, activation='relu', return_sequences=False))
    model.add(Dense(units=1))
    
    model.compile(optimizer='Adam', loss='mean_squared_error', metrics=['mae'])
    
    print("LSTM Model built successfully")
    model.summary()
    
    return model


def build_optimized_lstm_model(timesteps: int, lstm1_units: int = 50,
                               lstm2_units: int = 100, dense_units: int = 50) -> Sequential:
    """
    Build optimized LSTM model with fewer parameters.
    
    Args:
        timesteps: Number of timesteps in input
        lstm1_units: Number of units in first LSTM layer
        lstm2_units: Number of units in second LSTM layer
        dense_units: Number of units in dense layer
        
    Returns:
        Compiled Keras Sequential model
    """
    model = Sequential()
    model.add(LSTM(lstm1_units, return_sequences=True, input_shape=(timesteps, 1)))
    model.add(LSTM(lstm2_units, return_sequences=False))
    model.add(Dense(dense_units))
    model.add(Dense(1))
    
    model.compile(optimizer='Adam', loss='mean_squared_error', metrics=['mae'])
    
    print("Optimized LSTM Model built successfully")
    model.summary()
    
    return model


def train_model(model: Sequential, X_train: np.ndarray, y_train: np.ndarray,
               epochs: int = 3, batch_size: int = 100) -> object:
    """
    Train LSTM model.
    
    Args:
        model: Compiled Keras model
        X_train: Training input data
        y_train: Training target data
        epochs: Number of epochs
        batch_size: Batch size
        
    Returns:
        Training history object
    """
    history = model.fit(
        X_train, y_train,
        epochs=epochs,
        batch_size=batch_size,
        verbose=1
    )
    return history


def forecast(model: Sequential, X_test: np.ndarray, scaler: MinMaxScaler) -> np.ndarray:
    """
    Make predictions and inverse transform to original scale.
    
    Args:
        model: Trained Keras model
        X_test: Test input data
        scaler: Fitted MinMaxScaler
        
    Returns:
        Predictions in original scale
    """
    predictions = model.predict(X_test, verbose=0)
    predictions = scaler.inverse_transform(predictions)
    return predictions


def save_model(model: Sequential, path: str) -> None:
    """Save trained model."""
    model.save(path)
    print(f"Model saved to {path}")


def load_forecasting_model(path: str) -> Sequential:
    """Load trained model."""
    model = load_model(path)
    print(f"Model loaded from {path}")
    return model
