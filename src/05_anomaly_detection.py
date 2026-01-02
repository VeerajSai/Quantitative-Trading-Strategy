"""
Anomaly Detection Module
Implements LSTM Autoencoder for detecting anomalies in trading data.
"""

import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras import layers
from tensorflow.keras.callbacks import ModelCheckpoint
from typing import Tuple


def build_autoencoder(timesteps: int, n_features: int, 
                     encoder_units: int = 128) -> Sequential:
    """
    Build LSTM Autoencoder model for anomaly detection.
    
    Args:
        timesteps: Number of timesteps in input sequences
        n_features: Number of features in each timestep
        encoder_units: Number of units in encoder LSTM layer
        
    Returns:
        Compiled Keras Sequential model
    """
    model = Sequential()
    
    # Encoder
    model.add(layers.LSTM(encoder_units, input_shape=(timesteps, n_features), 
                         dropout=0.2))
    model.add(layers.Dropout(rate=0.5))
    
    # Repeat vector to match input length
    model.add(layers.RepeatVector(timesteps))
    
    # Decoder
    model.add(layers.LSTM(encoder_units, return_sequences=True, dropout=0.2))
    model.add(layers.Dropout(rate=0.5))
    model.add(layers.TimeDistributed(layers.Dense(n_features)))
    
    model.compile(loss='mae', optimizer='adam')
    
    print("Autoencoder model built successfully")
    model.summary()
    
    return model


def train_autoencoder(model: Sequential, X_train: np.ndarray, y_train: np.ndarray,
                     epochs: int = 50, batch_size: int = 128,
                     validation_split: float = 0.1,
                     model_path: str = "model_anomaly.keras") -> object:
    """
    Train LSTM Autoencoder.
    
    Args:
        model: Compiled Keras model
        X_train: Training input data
        y_train: Training target data
        epochs: Number of training epochs
        batch_size: Batch size
        validation_split: Validation split ratio
        model_path: Path to save best model
        
    Returns:
        Training history object
    """
    checkpoint = ModelCheckpoint(
        filepath=model_path,
        monitor='val_loss',
        verbose=1,
        save_best_only=True,
        mode='auto',
        save_freq='epoch'
    )
    
    history = model.fit(
        X_train, y_train,
        epochs=epochs,
        batch_size=batch_size,
        verbose=1,
        validation_split=validation_split,
        callbacks=[checkpoint],
        shuffle=False
    )
    
    print(f"Model saved to {model_path}")
    return history


def detect_anomalies(model: Sequential, X_test: np.ndarray, 
                    threshold: float = 0.02) -> Tuple[np.ndarray, np.ndarray]:
    """
    Detect anomalies using trained autoencoder.
    
    Args:
        model: Trained autoencoder model
        X_test: Test data
        threshold: MAE threshold for anomaly detection
        
    Returns:
        Tuple of (reconstruction_errors, anomaly_flags)
    """
    X_pred = model.predict(X_test, verbose=0)
    mae = np.mean(np.abs(X_pred - X_test), axis=1)
    anomalies = mae > threshold
    
    num_anomalies = np.sum(anomalies)
    print(f"\nAnomalies detected: {num_anomalies}/{len(mae)} ({100*num_anomalies/len(mae):.2f}%)")
    
    return mae, anomalies


def create_anomaly_dataframe(test_data: pd.DataFrame, close_data: pd.DataFrame,
                            mae_scores: np.ndarray, anomaly_flags: np.ndarray,
                            threshold: float = 0.02) -> pd.DataFrame:
    """
    Create comprehensive anomaly detection results DataFrame.
    
    Args:
        test_data: Original test data
        close_data: Close price data (scaled)
        mae_scores: Reconstruction error scores
        anomaly_flags: Boolean anomaly indicators
        threshold: Threshold value
        
    Returns:
        DataFrame with anomaly results
    """
    timesteps = len(close_data) - len(mae_scores)
    
    results_df = pd.DataFrame({
        'loss': mae_scores,
        'threshold': threshold,
        'anomaly': anomaly_flags
    }, index=close_data.index[timesteps:])
    
    results_df['close'] = test_data['close'].iloc[timesteps:].values
    
    return results_df
