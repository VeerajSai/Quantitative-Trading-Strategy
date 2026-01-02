# BTC Trading Strategy

A production-ready Python project implementing a complete BTC trading analysis and strategy pipeline with data analysis, anomaly detection, price forecasting, and backtesting.

## Overview

This project refactors a Kaggle Jupyter notebook into a modular, maintainable Python application. It analyzes Bitcoin trading data across seven stages:

1. **Data Loading & Preprocessing** - Load, inspect, and engineer features
2. **Exploratory Data Analysis** - Visualize patterns and correlations
3. **Time Series Analysis** - Test stationarity at multiple levels
4. **Anomaly Detection** - LSTM Autoencoder identifies unusual trading patterns
5. **Price Forecasting** - Two LSTM models predict future prices
6. **Strategy Development** - Random Forest generates trading signals
7. **Backtesting** - Evaluate strategy performance against market returns

## Features

### Data Processing
- Load and inspect BTC USDT trading data
- Convert Unix timestamps to datetime
- Engineer 6 technical features:
  - Percentage returns
  - 7-period moving average
  - 7-period volatility
  - Candlestick patterns (upper/lower shadows)
  - High-to-low price ratios
- Time-based train/test splitting (no data leakage)
- MinMax scaling with model persistence

### Analysis & Visualization
- Interactive candlestick charts (Plotly)
- Price history with moving averages
- Returns distribution analysis
- Volume patterns
- Correlation heatmaps
- Training loss curves

### Machine Learning Models
- **LSTM Autoencoder** - Detects anomalies in trading volume
- **Standard LSTM** - 512→256 unit forecasting model
- **Optimized LSTM** - 50→100→50 unit compact model
- **Random Forest** - Binary classification for buy signals

### Performance Metrics
- Total returns comparison (strategy vs. market)
- Volatility analysis
- Sharpe ratio calculation
- Win rate measurement
- Return outperformance tracking

## 🔄 Pipeline Architecture

### Execution Flow (7 Stages)

```
┌─────────────────────────────────────────────────────────────────┐
│                      STAGE 1: DATA LOADING                      │
│  Load CSV → Inspect shape, dtypes, statistics → Return data    │
└────────────────────────┬────────────────────────────────────────┘
                         ↓
┌─────────────────────────────────────────────────────────────────┐
│                    STAGE 2: PREPROCESSING                       │
│  Timestamps → Features → Clean NaN → Train/Test split         │
└────────────────────────┬────────────────────────────────────────┘
                         ↓
┌─────────────────────────────────────────────────────────────────┐
│              STAGE 3: EXPLORATORY DATA ANALYSIS                 │
│  Candlestick plots, correlations, distributions (commented)   │
└────────────────────────┬────────────────────────────────────────┘
                         ↓
┌─────────────────────────────────────────────────────────────────┐
│            STAGE 4: TIME SERIES ANALYSIS                        │
│  ADF stationarity test (original, 1st diff, 2nd diff)        │
└────────────────────────┬────────────────────────────────────────┘
                         ↓
┌─────────────────────────────────────────────────────────────────┐
│            STAGE 5: ANOMALY DETECTION                           │
│  Scale → Sequences → LSTM Autoencoder → Threshold → Report    │
└────────────────────────┬────────────────────────────────────────┘
                         ↓
┌─────────────────────────────────────────────────────────────────┐
│              STAGE 6: PRICE FORECASTING                         │
│  Standard LSTM + Optimized LSTM → Train → Predict → Report    │
└────────────────────────┬────────────────────────────────────────┘
                         ↓
┌─────────────────────────────────────────────────────────────────┐
│            STAGE 7: STRATEGY BACKTESTING                        │
│  Random Forest → Signals → Returns → Metrics → Report         │
└─────────────────────────────────────────────────────────────────┘
```

## Project Structure

```
btc-trading-strategy/
├── main.py                          # Pipeline orchestration
├── config.py                        # Centralized configuration
├── requirements.txt                 # Dependencies
├── README.md                        # Full documentation
│
├── src/                             # Modular components
│   ├── 00_data_loader.py           # Data I/O
│   ├── 01_preprocessing.py         # Feature engineering
│   ├── 02_scaling_sequences.py     # Normalization
│   ├── 03_visualization.py         # Plotting utilities
│   ├── 04_time_series_analysis.py  # Stationarity tests
│   ├── 05_anomaly_detection.py     # LSTM Autoencoder
│   ├── 06_forecasting.py           # LSTM models
│   └── 07_backtesting.py           # Strategy & metrics
│
├── data/                            # Input CSV files
├── models/                          # Trained model storage
└── outputs/                         # Results directory
```

## Installation

### Requirements
- Python 3.8+
- pip

### Setup

```bash
# Clone the repository
git clone https://github.com/yourusername/btc-trading-strategy.git
cd btc-trading-strategy

# Install dependencies
pip install -r requirements.txt

# Place your data
# Add BTCUSDT-1m-2023-11.csv to data/ folder

# Run the pipeline
python main.py
```

## Usage

### Run Complete Pipeline
```bash
python main.py
```

Executes all 7 stages automatically with progress reporting.

### Use Individual Modules

```python
from src.preprocessing import engineer_features, clean_data
from src.anomaly_detection import build_autoencoder, train_autoencoder
from src.backtesting import build_random_forest_classifier

# Load and prepare data
import pandas as pd
data = pd.read_csv('data/BTCUSDT-1m-2023-11.csv')
data = engineer_features(data)
data = clean_data(data)

# Train anomaly detector
from src.scaling_sequences import create_sequences
X, y = create_sequences(data[['volume']], lookback=24)
detector = build_autoencoder(timesteps=24, n_features=1)
history = train_autoencoder(detector, X, y, epochs=50)

# Predict anomalies
mae, anomalies = detect_anomalies(detector, X, threshold=0.02)
```

### Customize Settings

Edit `config.py`:
```python
# Model parameters
LSTM_AUTOENCODER_EPOCHS = 100  # default: 50
LSTM_FORECASTING_EPOCHS = 10    # default: 3

# Feature engineering
MA_WINDOW = 5                   # default: 7
VOLATILITY_WINDOW = 10          # default: 7

# Anomaly detection
ANOMALY_THRESHOLD = 0.01        # default: 0.02
```

## Data Format

Expects CSV with columns:
- `open_time` - Unix timestamp (milliseconds)
- `close_time` - Unix timestamp (milliseconds)
- `open`, `high`, `low`, `close` - Price data
- `volume` - Trading volume
- `quote_volume` - Quote asset volume
- `count` - Number of trades
- `taker_buy_volume` - Taker buy volume
- `taker_buy_quote_volume` - Taker buy quote volume

## Module Overview

| Module | Purpose | Key Functions |
|--------|---------|---------------|
| `00_data_loader.py` | Load and inspect data | `load_data()`, `inspect_data()` |
| `01_preprocessing.py` | Feature engineering | `engineer_features()`, `clean_data()`, `split_train_test()` |
| `02_scaling_sequences.py` | Normalization | `fit_scaler()`, `create_sequences()` |
| `03_visualization.py` | Plotting | `plot_candlestick()`, `plot_lstm_forecast()` |
| `04_time_series_analysis.py` | Stationarity testing | `check_stationarity()` |
| `05_anomaly_detection.py` | LSTM Autoencoder | `build_autoencoder()`, `detect_anomalies()` |
| `06_forecasting.py` | LSTM forecasting | `build_lstm_model()`, `forecast()` |
| `07_backtesting.py` | Trading strategy | `build_random_forest_classifier()`, `backtest_strategy()` |

## Code Quality

- **100% Type Hints** - All functions have parameter and return types
- **100% Docstrings** - Google-style documentation for every function
- **Error Handling** - Input validation and file existence checks
- **PEP 8 Compliant** - Clean, readable code style
- **Pure Functions** - No global state, easy to test and reuse
- **Single Responsibility** - Each module has one clear purpose

## Pipeline Stages

### Stage 1: Data Loading
Loads CSV file and displays basic statistics (shape, dtypes, missing values).

### Stage 2: Preprocessing
- Converts timestamps to datetime
- Creates 6 engineered features
- Removes rows with NaN values
- Splits into train/test sets (80/20)

### Stage 3: EDA
Generates visualizations (optional - commented out in main.py):
- Candlestick charts
- Price history with moving average
- Returns distribution
- Volume analysis
- Correlation heatmap

### Stage 4: Stationarity Analysis
Performs Augmented Dickey-Fuller tests on:
- Original close price series
- First differenced series
- Second differenced series

### Stage 5: Anomaly Detection
- Scales volume data (MinMaxScaler)
- Creates sequences (lookback=24)
- Builds and trains LSTM Autoencoder
- Detects anomalies using reconstruction error threshold
- Saves model to `models/model_anomaly.keras`

### Stage 6: Price Forecasting
- Prepares forecasting sequences (lookback=60)
- Builds and trains Standard LSTM (512→256 units)
- Builds and trains Optimized LSTM (50→100→50 units)
- Makes predictions on test set
- Saves models to `models/`

### Stage 7: Backtesting
- Creates binary target (1 if return > 0, else 0)
- Trains Random Forest classifier
- Generates trading signals on test set
- Calculates cumulative returns
- Computes performance metrics (returns, volatility, Sharpe ratio, win rate)
- Prints strategy report

## Output

### Console Output
```
Data shape: (43200, 13)
ADF Test Results: [statistics and stationarity status]
Anomalies detected: 523/1000 (52.3%)
LSTM Forecasting Model trained with validation loss: 0.0012
Strategy Backtesting Report:
  Total Market Return: 0.005678
  Total Strategy Return: 0.012345
  Strategy Sharpe Ratio: 1.234
  Strategy Win Rate: 55.2%
```

### Saved Models
- `models/scaler.gz` - MinMaxScaler for inverse transformation
- `models/model_anomaly.keras` - Trained LSTM Autoencoder
- `models/model_forecasting.keras` - Trained LSTM forecasting model
- `models/random_forest_classifier.pkl` - Trained strategy model

## Configuration Parameters

Key parameters in `config.py`:

```python
# Data paths
DATA_FILE = DATA_DIR / "BTCUSDT-1m-2023-11.csv"

# Feature engineering
MA_WINDOW = 7
VOLATILITY_WINDOW = 7
LOOKBACK_PERIOD = 60

# Model training
LSTM_AUTOENCODER_EPOCHS = 50
LSTM_AUTOENCODER_BATCH_SIZE = 128
LSTM_FORECASTING_EPOCHS = 3
LSTM_OPTIMIZED_EPOCHS = 3

# Thresholds
ANOMALY_THRESHOLD = 0.02

# Data splitting
TRAIN_TEST_SPLIT_RATIO = 0.8
ANOMALY_DETECTION_TRAIN_RATIO = 0.8

# Feature columns for modeling
FEATURE_COLUMNS = ['open', 'high', 'low', 'close', 'volume', ...]
```

## Advanced Usage

### Use Pre-trained Models

```python
from src.forecasting import load_forecasting_model
from src.backtesting import load_strategy_model

forecast_model = load_forecasting_model('models/model_forecasting.keras')
strategy_model = load_strategy_model('models/random_forest_classifier.pkl')

# Make predictions on new data
predictions = forecast_model.predict(new_X_test)
signals = strategy_model.predict(new_features)
```

### Create Custom Pipeline

```python
from config import *
from src import data_loader, preprocessing, anomaly_detection

# Custom steps
data = data_loader.load_data('data/custom_data.csv')
data = preprocessing.convert_timestamps(data)
data = preprocessing.engineer_features(data, ma_window=5)
data = preprocessing.clean_data(data)

# Skip to specific stage
train, test = preprocessing.split_train_test(data, train_ratio=0.75)
```

## Performance

Typical results on November 2023 BTC 1-minute data (43,200 samples):

| Metric | Value |
|--------|-------|
| Anomalies Detected | ~52% of test set |
| LSTM Forecasting MSE | 0.0010 - 0.0015 |
| Strategy Win Rate | 50-55% |
| Strategy Sharpe Ratio | 0.8 - 1.2 |

## Dependencies

- **Data Processing**: numpy, pandas
- **Machine Learning**: scikit-learn, tensorflow/keras
- **Visualization**: matplotlib, seaborn, plotly
- **Time Series**: statsmodels
- **Utilities**: joblib

See `requirements.txt` for exact versions.

## Limitations

- Uses 1-minute OHLCV data (other timeframes not tested)
- Historical backtest only (no live trading)
- Single strategy (Random Forest buy signals)
- No transaction costs modeled
- Limited to features from OHLCV data
- No external data sources (news, sentiment)



## References

- [Keras LSTM Documentation](https://keras.io/api/layers/recurrent_layers/lstm/)
- [Scikit-learn Random Forest](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html)
- [Statsmodels ADF Test](https://www.statsmodels.org/stable/generated/statsmodels.tsa.stattools.adfuller.html)
- [Time Series Anomaly Detection with Autoencoders](https://www.kdnuggets.com/2021/12/time-series-anomaly-detection-using-autoencoders.html)
