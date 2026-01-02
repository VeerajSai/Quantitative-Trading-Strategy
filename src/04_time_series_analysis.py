"""
Time Series Analysis Module
Handles stationarity tests and time series diagnostics.
"""

import pandas as pd
import numpy as np
from statsmodels.tsa.stattools import adfuller


def check_stationarity(series: pd.Series, series_name: str = "Series") -> dict:
    """
    Perform Augmented Dickey-Fuller test for stationarity.
    
    Args:
        series: Time series to test
        series_name: Name of the series for reporting
        
    Returns:
        Dictionary with test results
    """
    result = adfuller(series.dropna().values)
    
    print(f"\n{'='*60}")
    print(f"Stationarity Test: {series_name}")
    print(f"{'='*60}")
    print(f'ADF Statistic: {result[0]:.6f}')
    print(f'p-value: {result[1]:.6f}')
    print('Critical Values:')
    for key, value in result[4].items():
        print(f'  {key}: {value:.3f}')
    
    # Determine stationarity
    is_stationary = (result[1] <= 0.05) and (result[4]['5%'] > result[0])
    
    if is_stationary:
        print("\n✓ Series is STATIONARY")
    else:
        print("\n✗ Series is NON-STATIONARY")
    
    return {
        'adf_statistic': result[0],
        'p_value': result[1],
        'critical_values': result[4],
        'is_stationary': is_stationary
    }


def test_series_stationarity_levels(data: pd.DataFrame, column: str = 'close') -> None:
    """
    Test stationarity at different levels (original, first difference, second difference).
    
    Args:
        data: DataFrame with time series
        column: Column to test
    """
    print("\n" + "="*60)
    print("TIME SERIES STATIONARITY ANALYSIS")
    print("="*60)
    
    # Original series
    check_stationarity(data[column], f"{column} (Original)")
    
    # First difference
    first_diff = data[column].diff()
    check_stationarity(first_diff, f"{column} (First Difference)")
    
    # Second difference
    second_diff = data[column].diff().diff()
    check_stationarity(second_diff, f"{column} (Second Difference)")
