"""
Preprocessing Module
Handles data cleaning, timestamp conversion, and feature engineering.
"""

import numpy as np
import pandas as pd
from typing import Tuple


def convert_timestamps(data: pd.DataFrame) -> pd.DataFrame:
    """
    Convert Unix millisecond timestamps to datetime.
    
    Args:
        data: DataFrame with 'open_time' and 'close_time' columns in milliseconds
        
    Returns:
        DataFrame with converted datetime columns
    """
    data = data.copy()
    data['open_time'] = pd.to_datetime(data['open_time'], unit='ms')
    data['close_time'] = pd.to_datetime(data['close_time'], unit='ms')
    print("Timestamps converted successfully")
    return data


def engineer_features(data: pd.DataFrame, ma_window: int = 7, 
                      volatility_window: int = 7) -> pd.DataFrame:
    """
    Create engineered features for trading analysis.
    
    Args:
        data: Input DataFrame
        ma_window: Window for moving average calculation
        volatility_window: Window for volatility calculation
        
    Returns:
        DataFrame with engineered features
    """
    data = data.copy()
    
    # Return calculation (percentage change)
    data['Return'] = data['close'].pct_change()
    
    # Moving average
    data['MA7'] = data['close'].rolling(window=ma_window).mean()
    
    # Volatility (standard deviation of returns)
    data['Volatility'] = data['Return'].rolling(window=volatility_window).std()
    
    # Candlestick patterns
    data['Upper_Shadow'] = data['high'] - np.maximum(data['close'], data['open'])
    data['Lower_Shadow'] = np.minimum(data['close'], data['open']) - data['low']
    
    # High-to-low ratio (handle inf/nan values)
    data['high2low'] = (data['high'] / data['low']).replace([np.inf, -np.inf, np.nan], 0.)
    
    print("Features engineered successfully")
    return data


def clean_data(data: pd.DataFrame, columns_to_drop: list = None) -> pd.DataFrame:
    """
    Remove rows with NaN values and unnecessary columns.
    
    Args:
        data: Input DataFrame
        columns_to_drop: List of columns to remove
        
    Returns:
        Cleaned DataFrame
    """
    data = data.copy()
    
    # Drop specified columns if they exist
    if columns_to_drop:
        data = data.drop(columns=[col for col in columns_to_drop if col in data.columns])
    
    # Drop rows with NaN values
    initial_rows = len(data)
    data = data.dropna()
    dropped_rows = initial_rows - len(data)
    
    print(f"Data cleaning completed. Dropped {dropped_rows} rows with NaN values.")
    print(f"Remaining rows: {len(data)}")
    
    return data


def split_train_test(data: pd.DataFrame, train_ratio: float = 0.8) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split data into training and testing sets (time-based, no shuffle).
    
    Args:
        data: Input DataFrame
        train_ratio: Fraction of data to use for training
        
    Returns:
        Tuple of (train_df, test_df)
    """
    split_idx = int(len(data) * train_ratio)
    train = data.iloc[split_idx:]
    test = data.iloc[:split_idx]
    
    print(f"\nData split:")
    print(f"  Train shape: {train.shape}")
    print(f"  Test shape: {test.shape}")
    
    return train, test
