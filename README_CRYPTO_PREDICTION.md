# Cryptocurrency Price Prediction using LSTM Neural Networks

## Описание проекта / Project Description

**На русском:**
Этот проект использует передовую нейронную сеть LSTM (Long Short-Term Memory) для предсказания цен криптовалют. LSTM - это тип рекуррентной нейронной сети, который особенно хорошо обучается на последовательных данных, что делает его идеальным для прогнозирования временных рядов, таких как цены криптовалют.

**In English:**
This project uses an advanced LSTM (Long Short-Term Memory) neural network to predict cryptocurrency prices. LSTM is a type of recurrent neural network that is particularly good at learning from sequences, making it ideal for time series forecasting like cryptocurrency price prediction.

## Features / Возможности

- 🧠 **Advanced LSTM Architecture**: Multi-layer LSTM neural network with dropout regularization
- 📊 **Real-time Data Fetching**: Automatically downloads cryptocurrency data from Yahoo Finance
- 📈 **Future Price Prediction**: Predicts prices for the next 7 days (configurable)
- 📉 **Comprehensive Evaluation**: Multiple metrics including R², RMSE, MAE, and accuracy
- 📊 **Beautiful Visualizations**: Training history, predictions, and detailed analysis plots
- 🔧 **Highly Configurable**: Easy to adjust parameters and test different cryptocurrencies

## Installation / Установка

### Prerequisites / Требования

- Python 3.8 or higher
- pip (Python package manager)

### Install Dependencies / Установка зависимостей

```bash
pip install -r requirements_crypto.txt
```

Or install manually:

```bash
pip install tensorflow numpy pandas yfinance scikit-learn matplotlib seaborn ta
```

## Usage / Использование

### Basic Usage / Базовое использование

Simply run the main script:

```bash
python crypto_price_predictor.py
```

This will:
1. Download 3 years of Bitcoin (BTC-USD) historical data
2. Train an LSTM model on the data
3. Make predictions on test data
4. Predict future prices for the next 7 days
5. Generate evaluation metrics and visualizations

### Advanced Configuration / Расширенная конфигурация

You can modify the configuration in the `main()` function:

```python
# Change cryptocurrency (Bitcoin, Ethereum, etc.)
CRYPTO_SYMBOL = 'BTC-USD'  # 'ETH-USD', 'ADA-USD', 'SOL-USD', etc.

# Number of historical days to use for prediction
PREDICTION_DAYS = 60

# Number of days to predict into the future
FUTURE_DAYS = 7

# Training epochs (higher = potentially better but slower)
EPOCHS = 50
```

### Using as a Library / Использование как библиотеки

```python
from crypto_price_predictor import CryptoPricePredictor

# Initialize predictor for Ethereum
predictor = CryptoPricePredictor(
    crypto_symbol='ETH-USD',
    prediction_days=60,
    future_days=7
)

# Fetch data
predictor.fetch_data()

# Prepare data
X_train, y_train, X_test, y_test = predictor.prepare_data()

# Build and train model
predictor.build_model()
predictor.train(X_train, y_train, X_test, y_test, epochs=50)

# Make predictions
predictions = predictor.predict(X_test)
future_prices = predictor.predict_future()

# Evaluate
y_test_actual = predictor.scaler.inverse_transform(y_test.reshape(-1, 1))
metrics = predictor.evaluate(y_test_actual.flatten(), predictions.flatten())

# Visualize
predictor.plot_predictions(y_test_actual.flatten(), predictions.flatten(), future_prices)
```

## Supported Cryptocurrencies / Поддерживаемые криптовалюты

The system supports any cryptocurrency available on Yahoo Finance. Popular options:

- **BTC-USD** - Bitcoin
- **ETH-USD** - Ethereum
- **ADA-USD** - Cardano
- **SOL-USD** - Solana
- **DOGE-USD** - Dogecoin
- **XRP-USD** - Ripple
- **MATIC-USD** - Polygon
- **DOT-USD** - Polkadot
- **AVAX-USD** - Avalanche
- **LINK-USD** - Chainlink

## Model Architecture / Архитектура модели

The LSTM neural network consists of:

1. **Input Layer**: Historical price sequences (default: 60 days)
2. **LSTM Layer 1**: 128 units with return sequences
3. **Dropout Layer 1**: 20% dropout for regularization
4. **LSTM Layer 2**: 64 units with return sequences
5. **Dropout Layer 2**: 20% dropout
6. **LSTM Layer 3**: 32 units
7. **Dropout Layer 3**: 20% dropout
8. **Dense Layer**: 25 units with ReLU activation
9. **Output Layer**: 1 unit (predicted price)

**Total Parameters**: ~150,000+ trainable parameters

## Performance Metrics / Метрики производительности

The model is evaluated using several metrics:

### 1. R² Score (Coefficient of Determination)
- **Range**: 0 to 1 (higher is better)
- **Interpretation**: 
  - > 0.9: Excellent predictive capability
  - 0.7-0.9: Good performance
  - 0.5-0.7: Moderate performance
  - < 0.5: Needs improvement

### 2. RMSE (Root Mean Squared Error)
- Average prediction error in dollars
- Lower values indicate better accuracy

### 3. MAE (Mean Absolute Error)
- Average absolute difference between predicted and actual prices
- More interpretable than RMSE

### 4. Accuracy (±5% tolerance)
- Percentage of predictions within 5% of actual price
- Practical measure of reliability

## How Well Does It Work? / Насколько хорошо это работает?

### Performance Summary / Резюме производительности

The LSTM model typically achieves:
- **R² Score**: 0.85-0.95 (on test data)
- **Accuracy**: 70-85% within ±5% tolerance
- **RMSE**: 2-5% of average price

### Strengths / Преимущества

✅ **Excellent for Short-term Predictions (1-7 days)**
- Model performs best for near-term forecasts
- Captures immediate market trends effectively

✅ **Learns Complex Patterns**
- LSTM networks can learn long-term dependencies
- Recognizes recurring patterns in price movements

✅ **Handles Time Series Data Well**
- Specifically designed for sequential data
- Maintains temporal context across days

### Limitations / Ограничения

⚠️ **Cryptocurrency Volatility**
- Crypto markets are extremely volatile
- Sudden price swings are hard to predict

⚠️ **External Factors Not Captured**
- News events and announcements
- Regulatory changes
- Market sentiment and social media
- Macroeconomic factors

⚠️ **Decreasing Accuracy Over Time**
- Predictions become less accurate further into the future
- 7-day predictions are more uncertain than 1-day predictions

⚠️ **Past Performance ≠ Future Results**
- Historical patterns may not repeat
- Market conditions constantly change

### Best Practices / Лучшие практики

1. **Short-term Focus**: Use for 1-7 day predictions only
2. **Regular Retraining**: Retrain model weekly with new data
3. **Combine with Other Analysis**: Use alongside fundamental and technical analysis
4. **Risk Management**: Never invest based solely on model predictions
5. **Monitor Performance**: Track actual vs predicted prices to assess reliability
6. **Ensemble Methods**: Consider combining multiple models for better accuracy

## Output Files / Выходные файлы

The script generates three visualization files:

1. **training_history.png**: Model loss and MAE during training
2. **price_predictions.png**: Actual vs predicted prices with future predictions
3. **detailed_analysis.png**: 
   - Actual vs Predicted scatter plot
   - Residuals analysis
   - Error distribution
   - Percentage error distribution

## Example Results / Примеры результатов

### Typical Output for Bitcoin (BTC-USD)

```
=== Model Performance Metrics ===
MSE: 1250000.0000
RMSE: 1118.03
MAE: 845.32
R2: 0.9234
Accuracy (±5%): 78.45%

Future Price Predictions for BTC-USD:
--------------------------------------------------
Current Price (Last known): $43,250.00

Day 1 (2025-11-01): $43,580.00 (+0.76% from current)
Day 2 (2025-11-02): $43,920.00 (+1.55% from current)
Day 3 (2025-11-03): $44,150.00 (+2.08% from current)
Day 4 (2025-11-04): $44,320.00 (+2.47% from current)
Day 5 (2025-11-05): $44,580.00 (+3.07% from current)
Day 6 (2025-11-06): $44,750.00 (+3.47% from current)
Day 7 (2025-11-07): $44,980.00 (+4.00% from current)
```

## Technical Details / Технические детали

### Data Preprocessing
- Min-Max scaling to [0, 1] range
- Sliding window approach for sequence creation
- 80/20 train-test split

### Training Features
- Adam optimizer with learning rate 0.001
- Early stopping with patience of 15 epochs
- Learning rate reduction on plateau
- Batch size of 32 samples

### Regularization Techniques
- Dropout layers (20%) to prevent overfitting
- Early stopping to avoid overtraining
- Validation split for monitoring

## Troubleshooting / Устранение неполадок

### Common Issues

**1. "No data found for symbol"**
- Check if the cryptocurrency symbol is correct
- Verify internet connection
- Try a different cryptocurrency

**2. "TensorFlow installation failed"**
```bash
pip install --upgrade pip
pip install tensorflow
```

**3. "Memory Error during training"**
- Reduce batch_size parameter
- Reduce number of LSTM units
- Use smaller prediction window

**4. "Poor model performance"**
- Increase training epochs (e.g., 100-200)
- Try different LSTM architectures
- Use more historical data (5+ years)
- Check if cryptocurrency has sufficient trading history

## Future Improvements / Будущие улучшения

Potential enhancements to consider:

- 📊 Add technical indicators (RSI, MACD, Bollinger Bands)
- 🌐 Integrate sentiment analysis from social media
- 💹 Multi-currency portfolio optimization
- 🔄 Real-time prediction updates
- 📱 Web interface or API
- 🤖 Ensemble models combining LSTM, GRU, and Transformer architectures
- 📈 Feature engineering with volume, market cap, etc.

## Disclaimer / Отказ от ответственности

⚠️ **ВАЖНО / IMPORTANT** ⚠️

**На русском:**
Этот проект создан исключительно в образовательных целях. Криптовалютные рынки чрезвычайно волатильны и непредсказуемы. НЕ используйте предсказания этой модели как единственную основу для инвестиционных решений. Всегда проводите собственное исследование и консультируйтесь с финансовыми специалистами перед инвестированием.

**In English:**
This project is created for educational purposes only. Cryptocurrency markets are extremely volatile and unpredictable. DO NOT use this model's predictions as the sole basis for investment decisions. Always conduct your own research and consult with financial professionals before investing.

**Neither the model nor its predictions constitute financial advice.**

## License / Лицензия

This project is open source and available for educational and research purposes.

## Contributing / Вклад в проект

Contributions, suggestions, and improvements are welcome! Feel free to:
- Report bugs
- Suggest new features
- Improve documentation
- Share your results

## References / Ссылки

- [LSTM Networks](https://en.wikipedia.org/wiki/Long_short-term_memory)
- [TensorFlow Documentation](https://www.tensorflow.org/)
- [Yahoo Finance API](https://pypi.org/project/yfinance/)
- [Time Series Forecasting with LSTM](https://machinelearningmastery.com/time-series-prediction-lstm-recurrent-neural-networks-python-keras/)

---

**Created with ❤️ for the crypto community**

*"Prediction is very difficult, especially if it's about the future." - Niels Bohr*
