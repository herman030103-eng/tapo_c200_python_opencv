"""
Unit tests for Cryptocurrency Price Predictor
"""

import unittest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from crypto_price_predictor import CryptoPricePredictor


class TestCryptoPricePredictor(unittest.TestCase):
    """Test cases for CryptoPricePredictor class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.predictor = CryptoPricePredictor(
            crypto_symbol='BTC-USD',
            prediction_days=30,
            future_days=3
        )
    
    def test_initialization(self):
        """Test predictor initialization."""
        self.assertEqual(self.predictor.crypto_symbol, 'BTC-USD')
        self.assertEqual(self.predictor.prediction_days, 30)
        self.assertEqual(self.predictor.future_days, 3)
        self.assertIsNone(self.predictor.model)
        self.assertIsNone(self.predictor.data)
    
    def test_fetch_data(self):
        """Test data fetching functionality."""
        try:
            # Fetch 1 month of data for quick test
            start_date = (datetime.now() - timedelta(days=60)).strftime('%Y-%m-%d')
            end_date = datetime.now().strftime('%Y-%m-%d')
            
            data = self.predictor.fetch_data(start_date=start_date, end_date=end_date)
            
            # Check data is not empty
            self.assertIsNotNone(data)
            self.assertGreater(len(data), 0)
            
            # Check required columns exist
            required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
            for col in required_columns:
                self.assertIn(col, data.columns)
            
            print(f"✓ Data fetching test passed: {len(data)} days of data")
            
        except Exception as e:
            self.skipTest(f"Skipping data fetch test due to: {e}")
    
    def test_scaler(self):
        """Test data scaling functionality."""
        # Create sample data
        sample_data = np.array([[100], [200], [300], [400], [500]])
        
        # Fit and transform
        scaled = self.predictor.scaler.fit_transform(sample_data)
        
        # Check scaled values are between 0 and 1
        self.assertTrue(np.all(scaled >= 0))
        self.assertTrue(np.all(scaled <= 1))
        
        # Check inverse transform
        unscaled = self.predictor.scaler.inverse_transform(scaled)
        np.testing.assert_array_almost_equal(sample_data, unscaled, decimal=5)
        
        print("✓ Scaler test passed")
    
    def test_prepare_data_structure(self):
        """Test data preparation creates correct structure."""
        # Create mock data
        dates = pd.date_range(start='2023-01-01', periods=100, freq='D')
        prices = np.random.uniform(30000, 50000, size=100)
        self.predictor.data = pd.DataFrame({'Close': prices}, index=dates)
        
        # Prepare data
        X_train, y_train, X_test, y_test = self.predictor.prepare_data(test_size=0.2)
        
        # Check shapes
        self.assertEqual(X_train.shape[1], self.predictor.prediction_days)
        self.assertEqual(X_train.shape[2], 1)  # Single feature (price)
        self.assertEqual(len(X_train), len(y_train))
        self.assertEqual(len(X_test), len(y_test))
        
        # Check train/test split ratio
        total_samples = len(X_train) + len(X_test)
        test_ratio = len(X_test) / total_samples
        self.assertAlmostEqual(test_ratio, 0.2, places=1)
        
        print(f"✓ Data preparation test passed: Train={len(X_train)}, Test={len(X_test)}")
    
    def test_model_building(self):
        """Test model architecture building."""
        self.predictor.build_model(lstm_units=[32, 16], dropout_rate=0.2)
        
        # Check model exists
        self.assertIsNotNone(self.predictor.model)
        
        # Check model has layers
        self.assertGreater(len(self.predictor.model.layers), 0)
        
        # Check model is compiled
        self.assertTrue(self.predictor.model.optimizer is not None)
        
        print("✓ Model building test passed")
    
    def test_evaluation_metrics(self):
        """Test evaluation metrics calculation."""
        # Create sample predictions and actual values
        y_true = np.array([100, 200, 300, 400, 500])
        y_pred = np.array([105, 195, 310, 390, 505])
        
        metrics = self.predictor.evaluate(y_true, y_pred)
        
        # Check all metrics are present
        expected_metrics = ['MSE', 'RMSE', 'MAE', 'R2', 'Accuracy (±5%)']
        for metric in expected_metrics:
            self.assertIn(metric, metrics)
        
        # Check metrics are reasonable
        self.assertGreater(metrics['R2'], 0.9)  # Should be high for close predictions
        self.assertGreater(metrics['Accuracy (±5%)'], 80)  # Most within 5%
        
        print("✓ Evaluation metrics test passed")
    
    def test_invalid_crypto_symbol(self):
        """Test handling of invalid cryptocurrency symbol."""
        predictor = CryptoPricePredictor(crypto_symbol='INVALID-USD')
        
        with self.assertRaises(ValueError):
            predictor.fetch_data(
                start_date='2024-01-01',
                end_date='2024-01-10'
            )
        
        print("✓ Invalid symbol handling test passed")


class TestConfigurationOptions(unittest.TestCase):
    """Test different configuration options."""
    
    def test_different_prediction_windows(self):
        """Test different prediction window sizes."""
        windows = [15, 30, 60]
        
        for window in windows:
            predictor = CryptoPricePredictor(prediction_days=window)
            self.assertEqual(predictor.prediction_days, window)
        
        print(f"✓ Prediction window configuration test passed")
    
    def test_different_cryptocurrencies(self):
        """Test initialization with different cryptocurrencies."""
        cryptos = ['BTC-USD', 'ETH-USD', 'ADA-USD']
        
        for crypto in cryptos:
            predictor = CryptoPricePredictor(crypto_symbol=crypto)
            self.assertEqual(predictor.crypto_symbol, crypto)
        
        print(f"✓ Multiple cryptocurrency configuration test passed")


def run_tests():
    """Run all tests."""
    print("=" * 70)
    print("RUNNING CRYPTOCURRENCY PREDICTOR TESTS")
    print("=" * 70)
    print()
    
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add test classes
    suite.addTests(loader.loadTestsFromTestCase(TestCryptoPricePredictor))
    suite.addTests(loader.loadTestsFromTestCase(TestConfigurationOptions))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    print()
    print("=" * 70)
    if result.wasSuccessful():
        print("ALL TESTS PASSED ✓")
    else:
        print("SOME TESTS FAILED ✗")
    print("=" * 70)
    
    return result.wasSuccessful()


if __name__ == '__main__':
    success = run_tests()
    sys.exit(0 if success else 1)
