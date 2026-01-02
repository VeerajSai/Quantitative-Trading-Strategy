"""
Data Scaling and Sequence Generation Module
Handles normalization and creation of sequences for LSTM models.
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import joblib
from typing import Tuple


def fit_scaler(data: pd.DataFrame, column: str = 'volume') -> MinMaxScaler:
    """
    Fit MinMaxScaler on a specific column.
    
    Args:
        data: DataFrame with data to fit
        column: Column name to fit scaler on
        
    Returns:
        Fitted MinMaxScaler object
    """
    scaler = MinMaxScaler().fit(data[[column]])
    return scaler


def scale_data(data: pd.DataFrame, scaler: MinMaxScaler, 
               column: str = 'volume') -> pd.DataFrame:
    """
    Apply fitted scaler to data.
    
    Args:
        data: DataFrame with data to scale
        scaler: Fitted MinMaxScaler
        column: Column to scale
        
    Returns:
        DataFrame with scaled column
    """
    data = data.copy()
    data[column] = scaler.transform(data[[column]])
    return data


def save_scaler(scaler: MinMaxScaler, path: str) -> None:
    """Save scaler to disk."""
    joblib.dump(scaler, path)
    print(f"Scaler saved to {path}")


def load_scaler(path: str) -> MinMaxScaler:
    """Load scaler from disk."""
    scaler = joblib.load(path)
    print(f"Scaler loaded from {path}")
    return scaler


def create_sequences(data: pd.DataFrame, lookback: int = 24) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create sequences for LSTM models.
    
    Converts a DataFrame into X (input sequences) and y (target values).
    
    Args:
        data: DataFrame with time series data
        lookback: Number of timesteps to look back
        
    Returns:
        Tuple of (X_sequences, y_targets) as numpy arrays
    """
    data_x = []
    data_y = []
    
    for i in range(len(data) - int(lookback)):
        x_floats = np.array(data.iloc[i:i+lookback])
        y_floats = np.array(data.iloc[i+lookback])
        data_x.append(x_floats)
        data_y.append(y_floats)
    
    X = np.array(data_x)
    y = np.array(data_y)
    
    print(f"Sequences created: X shape {X.shape}, y shape {y.shape}")
    
    return X, y


def inverse_transform_predictions(predictions: np.ndarray, scaler: MinMaxScaler) -> np.ndarray:
    """
    Inverse transform scaled predictions back to original scale.
    
    Args:
        predictions: Scaled predictions
        scaler: Fitted MinMaxScaler
        
    Returns:
        Unscaled predictions
    """
    return scaler.inverse_transform(predictions)
