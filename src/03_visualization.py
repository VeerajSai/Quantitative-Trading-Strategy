"""
Visualization Module
Handles all plotting and visualization tasks for EDA and results.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import warnings

warnings.filterwarnings('ignore')

# Set consistent style
plt.style.use({
    "axes.grid": True,
    "axes.grid.axis": "y",
    "axes.edgecolor": "#474d56",
    "axes.facecolor": "#161a1e",
    "grid.color": "#2c2e31",
    "axes.titlecolor": "red",
    "figure.facecolor": "#161a1e",
    "figure.titlesize": "x-large",
    "figure.titleweight": "semibold",
    "lines.linewidth": 1.0,
    "lines.linestyle": "-",
    "axes.labelcolor": "white",
    "xtick.color": "white",
    "ytick.color": "white",
    "legend.edgecolor": "white",
    "legend.facecolor": "#161a1e",
    "legend.fontsize": "medium",
    "legend.shadow": False,
    "text.color": "white"
})


def plot_candlestick(data: pd.DataFrame, label: str = "BTC") -> go.Figure:
    """
    Create interactive candlestick chart using Plotly.
    
    Args:
        data: DataFrame with OHLC data
        label: Label for the chart
        
    Returns:
        Plotly Figure object
    """
    candlestick = go.Figure(data=[go.Candlestick(
        x=data.index,
        open=data['open'],
        high=data['high'],
        low=data['low'],
        close=data['close']
    )])

    candlestick.update_layout(
        paper_bgcolor="black",
        plot_bgcolor="black",
        margin_l=10, margin_b=0, margin_r=0, margin_t=0,
        grid_columns=1, grid_rows=1,
        xaxis=dict(title='Time', rangeslider=dict(visible=True)),
        title=f'{label} Candlestick Chart',
        yaxis=dict(title='Price in USD', ticksuffix='$')
    )
    candlestick.update_xaxes(showline=True, linewidth=2, linecolor='black', gridcolor='black')
    candlestick.update_yaxes(showline=True, linewidth=2, linecolor='black', gridcolor='black')

    return candlestick


def plot_close_price(data: pd.DataFrame, title: str = "BTC Close Price History") -> None:
    """Plot close price history."""
    plt.figure(figsize=(12, 6))
    plt.title(title, fontsize=16, fontweight='semibold')
    plt.xlabel('Date', fontsize=14, fontweight='bold')
    plt.ylabel('Close Price USD ($)', fontsize=14, fontweight='bold')
    plt.plot(data['close'])
    plt.tight_layout()
    plt.show()


def plot_close_with_ma(data: pd.DataFrame, ma_column: str = 'MA7') -> None:
    """Plot close price with moving average."""
    plt.figure(figsize=(12, 6))
    plt.title('BTC Price History', fontsize=16, fontweight='semibold')
    plt.xlabel('Date', fontsize=14, fontweight='bold')
    plt.ylabel('Price USD ($)', fontsize=14, fontweight='bold')
    plt.plot(data['close'], label='Close Price')
    plt.plot(data[ma_column], label=f'{ma_column} Moving Average')
    plt.legend(loc='upper right', bbox_to_anchor=(1.0, 1.0))
    plt.tight_layout()
    plt.show()


def plot_returns(data: pd.DataFrame) -> None:
    """Plot returns over time."""
    plt.figure(figsize=(12, 6))
    plt.title('Return per Minute', fontsize=16, fontweight='semibold')
    plt.xlabel('Date', fontsize=14, fontweight='bold')
    plt.ylabel('Percentage Change (%)', fontsize=14, fontweight='bold')
    plt.plot(data['Return'], linestyle='--', marker='o')
    plt.tight_layout()
    plt.show()


def plot_returns_distribution(data: pd.DataFrame, xlim: Tuple[float, float] = (-0.02, 0.02)) -> None:
    """Plot histogram of returns distribution."""
    plt.figure(figsize=(12, 6))
    plt.title('Returns Distribution', fontsize=16, fontweight='semibold')
    plt.xlabel('Rate of Return', fontsize=14, fontweight='bold')
    plt.ylabel('Count', fontsize=14, fontweight='bold')
    plt.hist(data['Return'], bins=100)
    plt.xlim(xlim)
    plt.tight_layout()
    plt.show()


def plot_volume(data: pd.DataFrame) -> None:
    """Plot trading volume over time."""
    plt.figure(figsize=(12, 6))
    plt.title('BTC Volume History', fontsize=16, fontweight='semibold')
    plt.xlabel('Date', fontsize=14, fontweight='bold')
    plt.ylabel('Volume', fontsize=14, fontweight='bold')
    plt.plot(data['volume'])
    plt.tight_layout()
    plt.show()


def plot_correlation_heatmap(data: pd.DataFrame) -> None:
    """Plot correlation matrix as heatmap."""
    corr_matrix = data.corr()
    plt.figure(figsize=(16, 14))
    sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap='coolwarm', 
                square=True, annot_kws={"size": 10})
    plt.title('Feature Correlation Matrix', fontsize=16, fontweight='semibold')
    plt.tight_layout()
    plt.show()


def plot_model_history(history, title: str = "Model Training History") -> None:
    """
    Plot training and validation loss from model history.
    
    Args:
        history: Keras training history object
        title: Title for the plot
    """
    plt.figure(figsize=(12, 6))
    plt.title(title, fontsize=16, fontweight='semibold')
    plt.xlabel('Epoch', fontsize=14, fontweight='bold')
    plt.ylabel('Loss', fontsize=14, fontweight='bold')
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.legend(loc='upper right')
    plt.tight_layout()
    plt.show()


def plot_lstm_forecast(train_data: pd.DataFrame, test_data: pd.DataFrame, 
                       predictions: np.ndarray, title: str = "LSTM Forecasting") -> None:
    """
    Plot LSTM forecast results.
    
    Args:
        train_data: Training data close prices
        test_data: Test data close prices with predictions
        predictions: Model predictions
        title: Title for the plot
    """
    plt.figure(figsize=(14, 7))
    plt.title(title, fontsize=16, fontweight='semibold')
    plt.xlabel('Time', fontsize=14, fontweight='bold')
    plt.ylabel('BTC Price', fontsize=14, fontweight='bold')
    plt.plot(train_data, label='Train')
    plt.plot(test_data, label='Actual')
    plt.plot(predictions, label='Predictions', linestyle='--')
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_anomalies(normal_data: pd.DataFrame, anomaly_data: pd.DataFrame, 
                   normal_values: np.ndarray, anomaly_values: np.ndarray,
                   x_label: str = "Time", y_label: str = "Volume") -> go.Figure:
    """
    Plot anomalies detected by autoencoder.
    
    Args:
        normal_data: DataFrame with normal data indices
        anomaly_data: DataFrame with anomaly indices
        normal_values: Scaled normal values
        anomaly_values: Scaled anomaly values
        x_label: X-axis label
        y_label: Y-axis label
        
    Returns:
        Plotly Figure object
    """
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=normal_data.index, y=normal_values,
        mode='lines', name='Normal Data'
    ))
    fig.add_trace(go.Scatter(
        x=anomaly_data.index, y=anomaly_values,
        mode='markers', name='Anomaly',
        marker=dict(size=8, color='red')
    ))
    fig.update_layout(
        showlegend=True,
        title="Detected Anomalies",
        xaxis_title=x_label,
        yaxis_title=y_label,
        hovermode='x unified'
    )
    return fig


def plot_strategy_returns(market_returns: pd.Series, strategy_returns: pd.Series) -> None:
    """
    Plot cumulative market returns vs strategy returns.
    
    Args:
        market_returns: Cumulative market returns
        strategy_returns: Cumulative strategy returns
    """
    plt.figure(figsize=(14, 7))
    plt.title('Cumulative Market Returns vs Strategy Returns', fontsize=16, fontweight='semibold')
    plt.xlabel('Time', fontsize=14, fontweight='bold')
    plt.ylabel('Cumulative Returns', fontsize=14, fontweight='bold')
    plt.plot(market_returns, color='r', label='Market Returns', linewidth=2)
    plt.plot(strategy_returns, color='g', label='Strategy Returns', linewidth=2)
    plt.legend(loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
