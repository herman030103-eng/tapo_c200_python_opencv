# Configuration file for Cryptocurrency Price Predictor

# Cryptocurrency settings
CRYPTO_SYMBOLS = [
    'BTC-USD',   # Bitcoin
    'ETH-USD',   # Ethereum
    'ADA-USD',   # Cardano
    'SOL-USD',   # Solana
    'DOGE-USD',  # Dogecoin
]

# Model hyperparameters
MODEL_CONFIG = {
    'prediction_days': 60,      # Number of historical days to use
    'future_days': 7,           # Number of days to predict ahead
    'lstm_units': [128, 64, 32], # LSTM layer sizes
    'dropout_rate': 0.2,        # Dropout for regularization
    'learning_rate': 0.001,     # Adam optimizer learning rate
    'batch_size': 32,           # Training batch size
    'epochs': 50,               # Maximum training epochs
    'test_size': 0.2,           # Train/test split ratio
}

# Data settings
DATA_CONFIG = {
    'years_of_history': 3,      # Years of historical data to fetch
    'min_data_points': 100,     # Minimum data points required
}

# Visualization settings
PLOT_CONFIG = {
    'figure_size': (16, 8),     # Default figure size
    'dpi': 300,                 # Resolution for saved images
    'style': 'seaborn',         # Plot style
}

# Evaluation settings
EVAL_CONFIG = {
    'tolerance': 0.05,          # Accuracy tolerance (5%)
}
