"""
Cryptocurrency Price Prediction using LSTM Neural Networks
Author: Advanced Crypto Predictor
Date: 2025

This script uses Long Short-Term Memory (LSTM) neural networks to predict cryptocurrency prices.
LSTM is a type of recurrent neural network (RNN) that is particularly good at learning from sequences,
making it ideal for time series forecasting like cryptocurrency price prediction.
"""

import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

try:
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
except ImportError:
    print("TensorFlow not installed. Installing required packages...")
    import os
    os.system('pip install tensorflow')
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau


class CryptoPricePredictor:
    """
    Advanced LSTM-based cryptocurrency price predictor.
    
    This class implements a sophisticated neural network architecture for predicting
    cryptocurrency prices using historical data.
    """
    
    def __init__(self, crypto_symbol='BTC-USD', prediction_days=60, future_days=7):
        """
        Initialize the predictor.
        
        Args:
            crypto_symbol (str): Cryptocurrency symbol (e.g., 'BTC-USD', 'ETH-USD')
            prediction_days (int): Number of historical days to use for prediction
            future_days (int): Number of days to predict into the future
        """
        self.crypto_symbol = crypto_symbol
        self.prediction_days = prediction_days
        self.future_days = future_days
        self.model = None
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.data = None
        self.history = None
        
    def fetch_data(self, start_date=None, end_date=None):
        """
        Fetch cryptocurrency data from Yahoo Finance.
        
        Args:
            start_date (str): Start date in 'YYYY-MM-DD' format
            end_date (str): End date in 'YYYY-MM-DD' format
        """
        if start_date is None:
            start_date = (datetime.now() - timedelta(days=365*3)).strftime('%Y-%m-%d')
        if end_date is None:
            end_date = datetime.now().strftime('%Y-%m-%d')
            
        print(f"Fetching data for {self.crypto_symbol} from {start_date} to {end_date}...")
        self.data = yf.download(self.crypto_symbol, start=start_date, end=end_date)
        
        if self.data.empty:
            raise ValueError(f"No data found for {self.crypto_symbol}")
            
        print(f"Successfully fetched {len(self.data)} days of data")
        return self.data
    
    def prepare_data(self, test_size=0.2):
        """
        Prepare and split data for training.
        
        Args:
            test_size (float): Proportion of data to use for testing
            
        Returns:
            tuple: (X_train, y_train, X_test, y_test)
        """
        if self.data is None:
            raise ValueError("No data available. Call fetch_data() first.")
        
        # Use closing price for prediction
        prices = self.data['Close'].values.reshape(-1, 1)
        
        # Scale the data
        scaled_data = self.scaler.fit_transform(prices)
        
        # Create sequences
        X, y = [], []
        for i in range(self.prediction_days, len(scaled_data)):
            X.append(scaled_data[i-self.prediction_days:i, 0])
            y.append(scaled_data[i, 0])
        
        X, y = np.array(X), np.array(y)
        
        # Reshape for LSTM [samples, time steps, features]
        X = np.reshape(X, (X.shape[0], X.shape[1], 1))
        
        # Split data
        split_idx = int(len(X) * (1 - test_size))
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        
        print(f"Training samples: {len(X_train)}, Testing samples: {len(X_test)}")
        return X_train, y_train, X_test, y_test
    
    def build_model(self, lstm_units=[128, 64, 32], dropout_rate=0.2):
        """
        Build advanced LSTM neural network architecture.
        
        Args:
            lstm_units (list): List of units for each LSTM layer
            dropout_rate (float): Dropout rate to prevent overfitting
        """
        self.model = Sequential()
        
        # First LSTM layer with return sequences
        self.model.add(LSTM(units=lstm_units[0], return_sequences=True, 
                           input_shape=(self.prediction_days, 1)))
        self.model.add(Dropout(dropout_rate))
        
        # Middle LSTM layers
        for units in lstm_units[1:-1]:
            self.model.add(LSTM(units=units, return_sequences=True))
            self.model.add(Dropout(dropout_rate))
        
        # Last LSTM layer
        self.model.add(LSTM(units=lstm_units[-1], return_sequences=False))
        self.model.add(Dropout(dropout_rate))
        
        # Dense layers
        self.model.add(Dense(units=25, activation='relu'))
        self.model.add(Dense(units=1))
        
        # Compile model
        self.model.compile(optimizer=Adam(learning_rate=0.001), 
                          loss='mean_squared_error',
                          metrics=['mae'])
        
        print("\nModel Architecture:")
        self.model.summary()
        
    def train(self, X_train, y_train, X_test, y_test, epochs=100, batch_size=32):
        """
        Train the LSTM model.
        
        Args:
            X_train, y_train: Training data
            X_test, y_test: Testing data
            epochs (int): Number of training epochs
            batch_size (int): Batch size for training
        """
        if self.model is None:
            self.build_model()
        
        # Callbacks
        early_stopping = EarlyStopping(monitor='val_loss', patience=15, 
                                       restore_best_weights=True)
        reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, 
                                     patience=5, min_lr=0.00001)
        
        print(f"\nTraining model for {epochs} epochs...")
        self.history = self.model.fit(
            X_train, y_train,
            validation_data=(X_test, y_test),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=[early_stopping, reduce_lr],
            verbose=1
        )
        
        print("\nTraining completed!")
        
    def predict(self, X_test):
        """
        Make predictions on test data.
        
        Args:
            X_test: Test data
            
        Returns:
            numpy.array: Predictions
        """
        predictions = self.model.predict(X_test)
        # Inverse transform to get actual prices
        predictions = self.scaler.inverse_transform(predictions)
        return predictions
    
    def predict_future(self, days=None):
        """
        Predict future prices.
        
        Args:
            days (int): Number of days to predict (uses future_days if None)
            
        Returns:
            list: Predicted prices for future days
        """
        if days is None:
            days = self.future_days
            
        # Get last prediction_days of data
        last_days = self.data['Close'].values[-self.prediction_days:]
        last_days_scaled = self.scaler.transform(last_days.reshape(-1, 1))
        
        # Predict future prices
        future_predictions = []
        current_batch = last_days_scaled.reshape(1, self.prediction_days, 1)
        
        for _ in range(days):
            # Predict next day
            next_pred = self.model.predict(current_batch, verbose=0)[0]
            future_predictions.append(next_pred)
            
            # Update batch with new prediction
            current_batch = np.append(current_batch[:, 1:, :], 
                                     [[next_pred]], axis=1)
        
        # Inverse transform
        future_predictions = self.scaler.inverse_transform(
            np.array(future_predictions).reshape(-1, 1)
        )
        
        return future_predictions.flatten()
    
    def evaluate(self, y_true, y_pred):
        """
        Evaluate model performance.
        
        Args:
            y_true: Actual values
            y_pred: Predicted values
            
        Returns:
            dict: Dictionary containing evaluation metrics
        """
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        
        # Calculate accuracy as percentage within tolerance
        tolerance = 0.05  # 5% tolerance
        within_tolerance = np.abs((y_true - y_pred) / y_true) <= tolerance
        accuracy = np.mean(within_tolerance) * 100
        
        metrics = {
            'MSE': mse,
            'RMSE': rmse,
            'MAE': mae,
            'R2': r2,
            'Accuracy (±5%)': accuracy
        }
        
        print("\n=== Model Performance Metrics ===")
        for metric, value in metrics.items():
            if metric == 'Accuracy (±5%)':
                print(f"{metric}: {value:.2f}%")
            else:
                print(f"{metric}: {value:.4f}")
        
        return metrics
    
    def plot_training_history(self):
        """Plot training history."""
        if self.history is None:
            print("No training history available. Train the model first.")
            return
        
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))
        
        # Loss plot
        axes[0].plot(self.history.history['loss'], label='Training Loss')
        axes[0].plot(self.history.history['val_loss'], label='Validation Loss')
        axes[0].set_title('Model Loss During Training')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].legend()
        axes[0].grid(True)
        
        # MAE plot
        axes[1].plot(self.history.history['mae'], label='Training MAE')
        axes[1].plot(self.history.history['val_mae'], label='Validation MAE')
        axes[1].set_title('Mean Absolute Error During Training')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('MAE')
        axes[1].legend()
        axes[1].grid(True)
        
        plt.tight_layout()
        plt.savefig('training_history.png', dpi=300, bbox_inches='tight')
        print("Training history plot saved as 'training_history.png'")
        plt.show()
    
    def plot_predictions(self, y_true, y_pred, future_predictions=None):
        """
        Plot actual vs predicted prices.
        
        Args:
            y_true: Actual prices
            y_pred: Predicted prices
            future_predictions: Future price predictions (optional)
        """
        plt.figure(figsize=(16, 8))
        
        # Create date range for test data
        test_dates = self.data.index[-len(y_true):]
        
        # Plot actual vs predicted
        plt.plot(test_dates, y_true, color='blue', label='Actual Price', linewidth=2)
        plt.plot(test_dates, y_pred, color='red', label='Predicted Price', 
                linewidth=2, linestyle='--')
        
        # Plot future predictions
        if future_predictions is not None:
            last_date = self.data.index[-1]
            future_dates = pd.date_range(start=last_date + timedelta(days=1), 
                                        periods=len(future_predictions))
            plt.plot(future_dates, future_predictions, color='green', 
                    label=f'Future Predictions ({len(future_predictions)} days)', 
                    linewidth=2, marker='o')
        
        plt.title(f'{self.crypto_symbol} Price Prediction using LSTM Neural Network', 
                 fontsize=16, fontweight='bold')
        plt.xlabel('Date', fontsize=12)
        plt.ylabel('Price (USD)', fontsize=12)
        plt.legend(fontsize=11)
        plt.grid(True, alpha=0.3)
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig('price_predictions.png', dpi=300, bbox_inches='tight')
        print("Price predictions plot saved as 'price_predictions.png'")
        plt.show()
    
    def plot_detailed_analysis(self, y_true, y_pred):
        """
        Create detailed analysis plots.
        
        Args:
            y_true: Actual prices
            y_pred: Predicted prices
        """
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # 1. Actual vs Predicted scatter
        axes[0, 0].scatter(y_true, y_pred, alpha=0.5)
        axes[0, 0].plot([y_true.min(), y_true.max()], 
                       [y_true.min(), y_true.max()], 
                       'r--', lw=2)
        axes[0, 0].set_xlabel('Actual Price')
        axes[0, 0].set_ylabel('Predicted Price')
        axes[0, 0].set_title('Actual vs Predicted Prices')
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. Residuals plot
        residuals = y_true - y_pred
        axes[0, 1].scatter(range(len(residuals)), residuals, alpha=0.5)
        axes[0, 1].axhline(y=0, color='r', linestyle='--')
        axes[0, 1].set_xlabel('Sample Index')
        axes[0, 1].set_ylabel('Residuals')
        axes[0, 1].set_title('Residuals Plot')
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. Residuals distribution
        axes[1, 0].hist(residuals, bins=50, edgecolor='black', alpha=0.7)
        axes[1, 0].set_xlabel('Residuals')
        axes[1, 0].set_ylabel('Frequency')
        axes[1, 0].set_title('Distribution of Residuals')
        axes[1, 0].grid(True, alpha=0.3)
        
        # 4. Percentage error distribution
        percentage_error = ((y_true - y_pred) / y_true) * 100
        axes[1, 1].hist(percentage_error, bins=50, edgecolor='black', alpha=0.7)
        axes[1, 1].axvline(x=0, color='r', linestyle='--')
        axes[1, 1].set_xlabel('Percentage Error (%)')
        axes[1, 1].set_ylabel('Frequency')
        axes[1, 1].set_title('Distribution of Percentage Error')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('detailed_analysis.png', dpi=300, bbox_inches='tight')
        print("Detailed analysis plot saved as 'detailed_analysis.png'")
        plt.show()


def main():
    """
    Main function to demonstrate the cryptocurrency price predictor.
    """
    print("=" * 80)
    print("ADVANCED CRYPTOCURRENCY PRICE PREDICTION USING LSTM NEURAL NETWORKS")
    print("=" * 80)
    print()
    
    # Configuration
    CRYPTO_SYMBOL = 'BTC-USD'  # Bitcoin - you can change to 'ETH-USD', 'ADA-USD', etc.
    PREDICTION_DAYS = 60  # Use 60 days of historical data
    FUTURE_DAYS = 7  # Predict 7 days into the future
    EPOCHS = 50  # Training epochs (can be increased for better performance)
    
    print(f"Configuration:")
    print(f"  - Cryptocurrency: {CRYPTO_SYMBOL}")
    print(f"  - Historical window: {PREDICTION_DAYS} days")
    print(f"  - Future prediction: {FUTURE_DAYS} days")
    print(f"  - Training epochs: {EPOCHS}")
    print()
    
    # Initialize predictor
    predictor = CryptoPricePredictor(
        crypto_symbol=CRYPTO_SYMBOL,
        prediction_days=PREDICTION_DAYS,
        future_days=FUTURE_DAYS
    )
    
    # Fetch data (3 years of historical data)
    try:
        predictor.fetch_data()
    except Exception as e:
        print(f"Error fetching data: {e}")
        return
    
    # Prepare data
    X_train, y_train, X_test, y_test = predictor.prepare_data(test_size=0.2)
    
    # Build and train model
    predictor.build_model(lstm_units=[128, 64, 32], dropout_rate=0.2)
    predictor.train(X_train, y_train, X_test, y_test, epochs=EPOCHS, batch_size=32)
    
    # Make predictions on test data
    predictions = predictor.predict(X_test)
    y_test_actual = predictor.scaler.inverse_transform(y_test.reshape(-1, 1))
    
    # Evaluate model
    metrics = predictor.evaluate(y_test_actual.flatten(), predictions.flatten())
    
    # Predict future prices
    print(f"\nPredicting future {FUTURE_DAYS} days...")
    future_predictions = predictor.predict_future(days=FUTURE_DAYS)
    
    print(f"\nFuture Price Predictions for {CRYPTO_SYMBOL}:")
    print("-" * 50)
    last_date = predictor.data.index[-1]
    last_price = predictor.data['Close'].values[-1]
    print(f"Current Price (Last known): ${last_price:.2f}")
    print()
    
    for i, price in enumerate(future_predictions, 1):
        future_date = last_date + timedelta(days=i)
        change = ((price - last_price) / last_price) * 100
        print(f"Day {i} ({future_date.strftime('%Y-%m-%d')}): ${price:.2f} "
              f"({change:+.2f}% from current)")
    
    print()
    print("=" * 50)
    print("HOW WELL DOES THE MODEL WORK?")
    print("=" * 50)
    print()
    print("The LSTM neural network model's performance can be evaluated as follows:")
    print()
    print(f"1. R² Score: {metrics['R2']:.4f}")
    print("   - Measures how well predictions match actual values")
    print("   - Range: 0 to 1 (higher is better)")
    if metrics['R2'] > 0.9:
        print("   - Rating: EXCELLENT - Model captures price patterns very well")
    elif metrics['R2'] > 0.7:
        print("   - Rating: GOOD - Model has strong predictive capability")
    elif metrics['R2'] > 0.5:
        print("   - Rating: MODERATE - Model shows reasonable predictions")
    else:
        print("   - Rating: NEEDS IMPROVEMENT - Consider more data or tuning")
    print()
    
    print(f"2. RMSE (Root Mean Squared Error): ${metrics['RMSE']:.2f}")
    print("   - Average prediction error in dollars")
    print("   - Lower values indicate better accuracy")
    print()
    
    print(f"3. Accuracy (±5% tolerance): {metrics['Accuracy (±5%)']:.2f}%")
    print("   - Percentage of predictions within 5% of actual price")
    print("   - Shows practical prediction reliability")
    print()
    
    print("LIMITATIONS AND CONSIDERATIONS:")
    print("-" * 50)
    print("• Cryptocurrency markets are highly volatile and unpredictable")
    print("• External factors (news, regulations, market sentiment) aren't captured")
    print("• Past performance doesn't guarantee future results")
    print("• Model works best for short-term predictions (1-7 days)")
    print("• Longer predictions have higher uncertainty")
    print()
    
    print("RECOMMENDATIONS:")
    print("-" * 50)
    print("• Use predictions as one input among many for decision-making")
    print("• Combine with fundamental analysis and market research")
    print("• Monitor model performance and retrain with new data regularly")
    print("• Consider ensemble methods combining multiple models")
    print()
    
    # Generate visualizations
    print("Generating visualizations...")
    predictor.plot_training_history()
    predictor.plot_predictions(y_test_actual.flatten(), predictions.flatten(), 
                              future_predictions)
    predictor.plot_detailed_analysis(y_test_actual.flatten(), predictions.flatten())
    
    print()
    print("=" * 80)
    print("Analysis complete! Check the generated PNG files for visualizations.")
    print("=" * 80)


if __name__ == "__main__":
    main()
