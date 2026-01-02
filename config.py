"""
Configuration file for BTC Trading Strategy project.
Centralized settings for data paths, model parameters, and feature names.
"""

import os
from pathlib import Path

# Project paths
PROJECT_ROOT = Path(__file__).parent
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"

# Data paths (update with your actual data location)
DATA_FILE = DATA_DIR / "BTCUSDT-1m-2023-11.csv"

# Model paths
SCALER_PATH = MODELS_DIR / "scaler.gz"
ANOMALY_MODEL_PATH = MODELS_DIR / "model_anomaly.keras"
FORECASTING_MODEL_PATH = MODELS_DIR / "model_forecasting.keras"
RANDOM_FOREST_MODEL_PATH = MODELS_DIR / "random_forest_classifier.pkl"

# Feature engineering parameters
LOOKBACK_PERIOD = 60  # For LSTM sequence generation
MA_WINDOW = 7  # Moving average window
VOLATILITY_WINDOW = 7  # Volatility calculation window

# Data split parameters
TRAIN_TEST_SPLIT_RATIO = 0.8
TRAIN_VALIDATION_SPLIT_RATIO = 0.1
ANOMALY_DETECTION_TRAIN_RATIO = 0.8
ANOMALY_DETECTION_SEQUENCE_LOOKBACK = 24

# Model parameters
LSTM_AUTOENCODER_EPOCHS = 50
LSTM_AUTOENCODER_BATCH_SIZE = 128
LSTM_FORECASTING_EPOCHS = 3
LSTM_FORECASTING_BATCH_SIZE = 100
LSTM_OPTIMIZED_EPOCHS = 3
LSTM_OPTIMIZED_BATCH_SIZE = 10

# Anomaly detection threshold
ANOMALY_THRESHOLD = 0.02

# Random Forest parameters
RANDOM_FOREST_RANDOM_STATE = 0

# Feature columns for modeling
FEATURE_COLUMNS = [
    'open', 'high', 'low', 'close', 'volume', 'quote_volume', 'count',
    'taker_buy_volume', 'taker_buy_quote_volume', 'Return', 'MA7',
    'Volatility', 'Upper_Shadow', 'Lower_Shadow', 'high2low'
]

# Output settings
SAVE_PLOTS = True
PLOT_DPI = 100
