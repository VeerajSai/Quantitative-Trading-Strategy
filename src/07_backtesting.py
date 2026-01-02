"""
Strategy Backtesting Module
Implements trading strategy using machine learning predictions.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import joblib
from typing import Tuple


def prepare_strategy_data(data: pd.DataFrame, feature_columns: list) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Prepare data for strategy backtesting.
    
    Args:
        data: Full dataset with features and returns
        feature_columns: List of feature column names
        
    Returns:
        Tuple of (train_data, test_data) with Target column
    """
    data = data.copy()
    
    # Create target: 1 if return is positive, 0 if negative
    data['Target'] = (data['Return'] > 0).astype(int)
    
    # Split data (70% train, 30% test, time-based, no shuffle)
    split_idx = int(len(data) * 0.7)
    train = data.iloc[:split_idx]
    test = data.iloc[split_idx:]
    
    print(f"Strategy data prepared:")
    print(f"  Train set: {train.shape[0]} samples")
    print(f"  Test set: {test.shape[0]} samples")
    print(f"  Features: {len(feature_columns)}")
    
    return train, test


def build_random_forest_classifier(n_estimators: int = 100, random_state: int = 0) -> RandomForestClassifier:
    """
    Build Random Forest classifier for trading signal generation.
    
    Args:
        n_estimators: Number of trees in forest
        random_state: Random seed for reproducibility
        
    Returns:
        Initialized RandomForestClassifier
    """
    clf = RandomForestClassifier(n_estimators=n_estimators, random_state=random_state)
    return clf


def train_strategy_model(clf: RandomForestClassifier, X_train: pd.DataFrame,
                        y_train: pd.Series) -> None:
    """
    Train Random Forest classifier on training data.
    
    Args:
        clf: RandomForestClassifier instance
        X_train: Training features
        y_train: Training targets
    """
    clf.fit(X_train, y_train)
    print("Strategy model trained successfully")


def generate_trading_signals(clf: RandomForestClassifier, X_test: pd.DataFrame) -> np.ndarray:
    """
    Generate trading signals (buy=1, no trade=0).
    
    Args:
        clf: Trained RandomForestClassifier
        X_test: Test features
        
    Returns:
        Array of trading signals
    """
    signals = clf.predict(X_test)
    return signals


def backtest_strategy(test_data: pd.DataFrame, predictions: np.ndarray) -> pd.DataFrame:
    """
    Backtest strategy by comparing market returns with strategy returns.
    
    Args:
        test_data: Test data with actual returns
        predictions: Trading signal predictions
        
    Returns:
        DataFrame with backtest results
    """
    results = test_data.copy()
    results['Predicted'] = predictions
    
    # Calculate strategy returns (only trade when prediction = 1)
    results['Strategy Return'] = results['Return'] * results['Predicted']
    
    # Calculate cumulative returns
    results['Cumulative Market Returns'] = np.cumsum(results['Return'])
    results['Cumulative Strategy Returns'] = np.cumsum(results['Strategy Return'])
    
    return results


def calculate_strategy_metrics(results: pd.DataFrame) -> dict:
    """
    Calculate key performance metrics for the strategy.
    
    Args:
        results: DataFrame with strategy results
        
    Returns:
        Dictionary with performance metrics
    """
    market_return = results['Return'].sum()
    strategy_return = results['Strategy Return'].sum()
    
    market_volatility = results['Return'].std()
    strategy_volatility = results['Strategy Return'].std()
    
    # Sharpe ratio (assuming 0 risk-free rate)
    market_sharpe = (market_return / market_volatility) if market_volatility != 0 else 0
    strategy_sharpe = (strategy_return / strategy_volatility) if strategy_volatility != 0 else 0
    
    # Win rate
    positive_strategy_returns = (results['Strategy Return'] > 0).sum()
    win_rate = positive_strategy_returns / len(results)
    
    metrics = {
        'Total Market Return': market_return,
        'Total Strategy Return': strategy_return,
        'Market Volatility': market_volatility,
        'Strategy Volatility': strategy_volatility,
        'Market Sharpe Ratio': market_sharpe,
        'Strategy Sharpe Ratio': strategy_sharpe,
        'Strategy Win Rate': win_rate,
        'Return Outperformance': strategy_return - market_return
    }
    
    return metrics


def save_strategy_model(clf: RandomForestClassifier, path: str) -> None:
    """Save trained strategy model."""
    joblib.dump(clf, path)
    print(f"Strategy model saved to {path}")


def load_strategy_model(path: str) -> RandomForestClassifier:
    """Load trained strategy model."""
    clf = joblib.load(path)
    print(f"Strategy model loaded from {path}")
    return clf


def print_strategy_report(metrics: dict) -> None:
    """Print formatted strategy performance report."""
    print("\n" + "="*60)
    print("STRATEGY BACKTESTING REPORT")
    print("="*60)
    print(f"Total Market Return:       {metrics['Total Market Return']:.6f}")
    print(f"Total Strategy Return:     {metrics['Total Strategy Return']:.6f}")
    print(f"Return Outperformance:     {metrics['Return Outperformance']:.6f}")
    print(f"\nMarket Volatility:         {metrics['Market Volatility']:.6f}")
    print(f"Strategy Volatility:       {metrics['Strategy Volatility']:.6f}")
    print(f"\nMarket Sharpe Ratio:       {metrics['Market Sharpe Ratio']:.4f}")
    print(f"Strategy Sharpe Ratio:     {metrics['Strategy Sharpe Ratio']:.4f}")
    print(f"\nStrategy Win Rate:         {metrics['Strategy Win Rate']:.2%}")
    print("="*60)
