"""
Quick Start Example for Cryptocurrency Price Prediction
This is a simplified version for quick testing and demonstration.
"""

from crypto_price_predictor import CryptoPricePredictor
from datetime import datetime

def quick_demo():
    """Quick demonstration of the crypto predictor."""
    
    print("=" * 70)
    print("QUICK CRYPTO PRICE PREDICTION DEMO")
    print("=" * 70)
    print()
    
    # Create predictor for Bitcoin
    print("Initializing Bitcoin (BTC-USD) predictor...")
    predictor = CryptoPricePredictor(
        crypto_symbol='BTC-USD',
        prediction_days=30,  # Use 30 days for faster demo
        future_days=3  # Predict 3 days ahead
    )
    
    # Fetch last year of data for faster demo
    print("Fetching 1 year of historical data...")
    start_date = (datetime.now() - pd.Timedelta(days=365)).strftime('%Y-%m-%d')
    predictor.fetch_data(start_date=start_date)
    
    # Prepare data
    print("Preparing data...")
    X_train, y_train, X_test, y_test = predictor.prepare_data(test_size=0.15)
    
    # Build and train with fewer epochs for demo
    print("Building and training model (this may take a few minutes)...")
    predictor.build_model(lstm_units=[64, 32], dropout_rate=0.2)
    predictor.train(X_train, y_train, X_test, y_test, epochs=20, batch_size=32)
    
    # Make predictions
    print("\nMaking predictions...")
    predictions = predictor.predict(X_test)
    y_test_actual = predictor.scaler.inverse_transform(y_test.reshape(-1, 1))
    
    # Evaluate
    metrics = predictor.evaluate(y_test_actual.flatten(), predictions.flatten())
    
    # Future predictions
    print("\nPredicting next 3 days...")
    future_predictions = predictor.predict_future()
    
    last_price = predictor.data['Close'].values[-1]
    print(f"\nCurrent BTC Price: ${last_price:.2f}")
    print("\nFuture Predictions:")
    for i, price in enumerate(future_predictions, 1):
        change = ((price - last_price) / last_price) * 100
        print(f"  Day {i}: ${price:.2f} ({change:+.2f}%)")
    
    # Create visualizations
    print("\nGenerating visualizations...")
    predictor.plot_predictions(y_test_actual.flatten(), predictions.flatten(), 
                              future_predictions)
    
    print("\n" + "=" * 70)
    print("DEMO COMPLETE!")
    print("Check 'price_predictions.png' for visualization.")
    print("=" * 70)


def compare_cryptocurrencies():
    """Compare predictions for multiple cryptocurrencies."""
    
    import pandas as pd
    
    cryptos = ['BTC-USD', 'ETH-USD', 'ADA-USD']
    results = []
    
    print("=" * 70)
    print("COMPARING MULTIPLE CRYPTOCURRENCIES")
    print("=" * 70)
    print()
    
    for crypto in cryptos:
        print(f"\nProcessing {crypto}...")
        try:
            predictor = CryptoPricePredictor(crypto, prediction_days=30, future_days=7)
            
            # Fetch and prepare data
            start_date = (datetime.now() - pd.Timedelta(days=365)).strftime('%Y-%m-%d')
            predictor.fetch_data(start_date=start_date)
            X_train, y_train, X_test, y_test = predictor.prepare_data(test_size=0.2)
            
            # Train
            predictor.build_model(lstm_units=[64, 32])
            predictor.train(X_train, y_train, X_test, y_test, epochs=20, batch_size=32)
            
            # Predict
            predictions = predictor.predict(X_test)
            y_test_actual = predictor.scaler.inverse_transform(y_test.reshape(-1, 1))
            
            # Evaluate
            metrics = predictor.evaluate(y_test_actual.flatten(), predictions.flatten())
            
            # Future prediction
            future = predictor.predict_future()[6]  # 7-day prediction
            current = predictor.data['Close'].values[-1]
            change_pct = ((future - current) / current) * 100
            
            results.append({
                'Crypto': crypto,
                'Current Price': f'${current:.2f}',
                '7-Day Prediction': f'${future:.2f}',
                'Expected Change': f'{change_pct:+.2f}%',
                'R² Score': f'{metrics["R2"]:.4f}',
                'Accuracy': f'{metrics["Accuracy (±5%)"]:.2f}%'
            })
            
        except Exception as e:
            print(f"Error processing {crypto}: {e}")
    
    # Display results
    print("\n" + "=" * 70)
    print("COMPARISON RESULTS")
    print("=" * 70)
    df = pd.DataFrame(results)
    print(df.to_string(index=False))
    print("=" * 70)


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == 'compare':
        compare_cryptocurrencies()
    else:
        quick_demo()
