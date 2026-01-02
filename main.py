"""
Main Pipeline Orchestration
Runs the complete BTC trading strategy workflow end-to-end.
"""

import sys
import os
from pathlib import Path

# Add project to path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
from config import *
from src import (
    data_loader,
    preprocessing,
    scaling_sequences,
    visualization,
    time_series_analysis,
    anomaly_detection,
    forecasting,
    backtesting
)


def create_directories():
    """Create necessary directories if they don't exist."""
    for directory in [DATA_DIR, MODELS_DIR, OUTPUTS_DIR]:
        directory.mkdir(parents=True, exist_ok=True)
    print("✓ Directories created/verified")


def stage_1_data_loading():
    """Stage 1: Load and inspect raw data."""
    print("\n" + "="*70)
    print("STAGE 1: DATA LOADING AND INSPECTION")
    print("="*70)
    
    # Load data
    data = data_loader.load_data(str(DATA_FILE))
    data_loader.inspect_data(data)
    
    return data


def stage_2_preprocessing(data):
    """Stage 2: Data preprocessing and feature engineering."""
    print("\n" + "="*70)
    print("STAGE 2: DATA PREPROCESSING")
    print("="*70)
    
    # Convert timestamps
    data = preprocessing.convert_timestamps(data)
    
    # Engineer features
    data = preprocessing.engineer_features(
        data,
        ma_window=MA_WINDOW,
        volatility_window=VOLATILITY_WINDOW
    )
    
    # Clean data
    data = preprocessing.clean_data(data, columns_to_drop=['close_time', 'ignore'])
    
    return data


def stage_3_eda(data):
    """Stage 3: Exploratory Data Analysis."""
    print("\n" + "="*70)
    print("STAGE 3: EXPLORATORY DATA ANALYSIS")
    print("="*70)
    print("\nNote: Visualization functions are available but not auto-displayed.")
    print("To view plots, uncomment visualization.plot_* calls in main.py")
    
    # Create candlestick data subset for visualization
    subset = data.iloc[:1000]  # Use first 1000 rows for candlestick
    
    # Available visualizations (uncomment to display):
    # visualization.plot_candlestick(subset)
    # visualization.plot_close_price(data)
    # visualization.plot_close_with_ma(data)
    # visualization.plot_returns(data)
    # visualization.plot_returns_distribution(data)
    # visualization.plot_volume(data)
    # visualization.plot_correlation_heatmap(data)
    
    print("✓ Data exploration completed")
    print(f"  - Correlation matrix shape: {data.corr().shape}")


def stage_4_stationarity_analysis(data):
    """Stage 4: Time series stationarity analysis."""
    print("\n" + "="*70)
    print("STAGE 4: TIME SERIES STATIONARITY ANALYSIS")
    print("="*70)
    
    time_series_analysis.test_series_stationarity_levels(data, column='close')


def stage_5_anomaly_detection(data):
    """Stage 5: Train anomaly detection model."""
    print("\n" + "="*70)
    print("STAGE 5: ANOMALY DETECTION (LSTM AUTOENCODER)")
    print("="*70)
    
    # Prepare data for anomaly detection
    train_ad, test_ad = preprocessing.split_train_test(
        data, 
        train_ratio=ANOMALY_DETECTION_TRAIN_RATIO
    )
    
    # Scale volume data
    scaler_ad = scaling_sequences.fit_scaler(train_ad, column='volume')
    train_ad_scaled = scaling_sequences.scale_data(train_ad, scaler_ad)
    test_ad_scaled = scaling_sequences.scale_data(test_ad, scaler_ad)
    
    # Create sequences
    X_train_ad, y_train_ad = scaling_sequences.create_sequences(
        train_ad_scaled[['volume']],
        lookback=ANOMALY_DETECTION_SEQUENCE_LOOKBACK
    )
    X_test_ad, y_test_ad = scaling_sequences.create_sequences(
        test_ad_scaled[['volume']],
        lookback=ANOMALY_DETECTION_SEQUENCE_LOOKBACK
    )
    
    # Get dimensions
    tsteps = X_train_ad.shape[1]
    nfeatures = X_train_ad.shape[2]
    
    # Build and train autoencoder
    detector = anomaly_detection.build_autoencoder(tsteps, nfeatures)
    history_ad = anomaly_detection.train_autoencoder(
        detector,
        X_train_ad, y_train_ad,
        epochs=LSTM_AUTOENCODER_EPOCHS,
        batch_size=LSTM_AUTOENCODER_BATCH_SIZE,
        model_path=str(ANOMALY_MODEL_PATH)
    )
    
    # Detect anomalies
    mae_scores, anomaly_flags = anomaly_detection.detect_anomalies(
        detector,
        X_test_ad,
        threshold=ANOMALY_THRESHOLD
    )
    
    # Create results dataframe
    anomaly_results = anomaly_detection.create_anomaly_dataframe(
        test_ad,
        test_ad_scaled,
        mae_scores,
        anomaly_flags,
        threshold=ANOMALY_THRESHOLD
    )
    
    # Visualization (uncomment to display)
    # anomaly_df = anomaly_results
    # anomalies_only = anomaly_df[anomaly_df['anomaly'] == True]
    # visualization.plot_anomalies(
    #     anomaly_df, anomalies_only,
    #     test_ad_scaled[tsteps:]['volume'].values,
    #     anomalies_only['volume'].values
    # )
    
    return history_ad


def stage_6_forecasting(data):
    """Stage 6: Train LSTM forecasting models."""
    print("\n" + "="*70)
    print("STAGE 6: PRICE FORECASTING (LSTM MODELS)")
    print("="*70)
    
    # Prepare forecasting data
    close_df = data.filter(['close'])
    
    # Standard LSTM Model
    print("\n--- Training Standard LSTM Model ---")
    X_train_f, y_train_f, X_test_f, y_test_f, train_len = forecasting.prepare_forecasting_data(
        close_df,
        lookback=LOOKBACK_PERIOD,
        train_ratio=TRAIN_TEST_SPLIT_RATIO
    )
    
    # Reshape for LSTM
    X_train_f = np.reshape(X_train_f, (X_train_f.shape[0], X_train_f.shape[1], 1))
    X_test_f = np.reshape(X_test_f, (X_test_f.shape[0], X_test_f.shape[1], 1))
    
    # Build and train standard model
    model = forecasting.build_lstm_model(X_train_f.shape[1])
    history_model = forecasting.train_model(
        model,
        X_train_f, y_train_f,
        epochs=LSTM_FORECASTING_EPOCHS,
        batch_size=LSTM_FORECASTING_BATCH_SIZE
    )
    
    # Prepare scaler for inverse transform
    scaler_forecast = scaling_sequences.fit_scaler(close_df, column='close')
    
    # Make predictions
    predictions = forecasting.forecast(model, X_test_f, scaler_forecast)
    
    # Visualization (uncomment to display)
    # visualization.plot_model_history(history_model, "LSTM Model Training History")
    
    print("\n--- Training Optimized LSTM Model ---")
    optimized_model = forecasting.build_optimized_lstm_model(X_train_f.shape[1])
    history_optimized = forecasting.train_model(
        optimized_model,
        X_train_f, y_train_f,
        epochs=LSTM_OPTIMIZED_EPOCHS,
        batch_size=LSTM_OPTIMIZED_BATCH_SIZE
    )
    
    o_predictions = forecasting.forecast(optimized_model, X_test_f, scaler_forecast)
    
    # Save models
    forecasting.save_model(model, str(FORECASTING_MODEL_PATH))
    
    return history_model, predictions, history_optimized, o_predictions, train_len


def stage_7_backtesting(data):
    """Stage 7: Strategy backtesting with Random Forest."""
    print("\n" + "="*70)
    print("STAGE 7: STRATEGY BACKTESTING")
    print("="*70)
    
    # Prepare strategy data
    train_data, test_data = backtesting.prepare_strategy_data(data, FEATURE_COLUMNS)
    
    # Build and train classifier
    clf = backtesting.build_random_forest_classifier(random_state=RANDOM_FOREST_RANDOM_STATE)
    backtesting.train_strategy_model(clf, train_data[FEATURE_COLUMNS], train_data['Target'])
    
    # Generate signals
    signals = backtesting.generate_trading_signals(clf, test_data[FEATURE_COLUMNS])
    
    # Backtest strategy
    backtest_results = backtesting.backtest_strategy(test_data, signals)
    
    # Calculate metrics
    metrics = backtesting.calculate_strategy_metrics(backtest_results)
    backtesting.print_strategy_report(metrics)
    
    # Visualization (uncomment to display)
    # visualization.plot_strategy_returns(
    #     backtest_results['Cumulative Market Returns'],
    #     backtest_results['Cumulative Strategy Returns']
    # )
    
    # Save model
    backtesting.save_strategy_model(clf, str(RANDOM_FOREST_MODEL_PATH))
    
    return backtest_results, metrics


def main():
    """Execute complete pipeline."""
    print("\n")
    print("╔" + "="*68 + "╗")
    print("║" + " "*15 + "BTC TRADING STRATEGY PIPELINE" + " "*25 + "║")
    print("╚" + "="*68 + "╝")
    
    # Setup
    create_directories()
    
    # Stage 1: Data Loading
    data = stage_1_data_loading()
    
    # Stage 2: Preprocessing
    data = stage_2_preprocessing(data)
    
    # Stage 3: EDA
    stage_3_eda(data)
    
    # Stage 4: Stationarity Analysis
    stage_4_stationarity_analysis(data)
    
    # Stage 5: Anomaly Detection
    stage_5_anomaly_detection(data)
    
    # Stage 6: Forecasting
    stage_6_forecasting(data)
    
    # Stage 7: Backtesting
    stage_7_backtesting(data)
    
    print("\n" + "="*70)
    print("✓ PIPELINE EXECUTION COMPLETED SUCCESSFULLY")
    print("="*70)
    print(f"\nOutputs saved to: {OUTPUTS_DIR}")
    print(f"Models saved to: {MODELS_DIR}")


if __name__ == "__main__":
    main()
