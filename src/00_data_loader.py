"""
Data Loading Module
Handles loading and initial inspection of BTC trading data.
"""

import pandas as pd
import os
from pathlib import Path


def load_data(file_path: str) -> pd.DataFrame:
    """
    Load BTC USDT data from CSV file.
    
    Args:
        file_path: Path to the CSV file
        
    Returns:
        DataFrame with raw BTC data
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Data file not found at {file_path}")
    
    data = pd.read_csv(file_path)
    print(f"Data loaded successfully. Shape: {data.shape}")
    return data


def inspect_data(data: pd.DataFrame) -> None:
    """
    Display basic information about the dataset.
    
    Args:
        data: DataFrame to inspect
    """
    print("\n=== Data Shape ===")
    print(f"Rows: {data.shape[0]}, Columns: {data.shape[1]}")
    
    print("\n=== First 5 Rows ===")
    print(data.head())
    
    print("\n=== Data Info ===")
    print(data.info())
    
    print("\n=== Data Statistics ===")
    print(data.describe())
    
    print("\n=== Missing Values ===")
    print(data.isnull().sum())
