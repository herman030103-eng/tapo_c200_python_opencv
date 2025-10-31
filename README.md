# Repository Projects

This repository contains two projects:

1. **Tapo C200 Camera with OpenCV** - Original camera streaming project
2. **🆕 Cryptocurrency Price Prediction with LSTM Neural Networks** - Advanced AI-powered crypto price forecasting

---

## 🚀 NEW: Cryptocurrency Price Prediction

A sophisticated machine learning project that uses LSTM (Long Short-Term Memory) neural networks to predict cryptocurrency prices. The system can forecast Bitcoin, Ethereum, and other cryptocurrency prices for 1-7 days into the future.

**Key Features:**
- 🧠 Advanced LSTM neural network (128, 64, 32 units)
- 📊 Real-time cryptocurrency data fetching
- 📈 7-day future price predictions
- 📉 Comprehensive evaluation metrics (R², RMSE, MAE, Accuracy)
- 📊 Beautiful visualizations and analysis
- ⚡ Supports all major cryptocurrencies (BTC, ETH, ADA, SOL, DOGE, etc.)

**Quick Start:**
```bash
# Install dependencies
pip install -r requirements_crypto.txt

# Run the predictor
python crypto_price_predictor.py

# Or try the quick demo
python demo_example.py
```

**📖 Full Documentation:** [README_CRYPTO_PREDICTION.md](README_CRYPTO_PREDICTION.md)

---

## Tapo C200 TP Link Camera OpenCV Script Documentation
#### Introduction

This documentation provides a guide on how to use the Tapo C200 TP Link camera with OpenCV. The script connects to the camera using an RTSP link and captures video frames for further processing or display.

### Youtube
https://www.youtube.com/watch?v=-kcVOxRNR9M

#### Prerequisites

To use this script, ensure that you have the following:

+ Tapo C200 TP Link camera
+ Python 3 installed
+ OpenCV library installed (pip install opencv-python)

#### Usage
1. Set up camera connection details:
```
ip_address = '192.168.0.XX'  # Replace with the IP address of your camera
port = '554'                # Replace with the port number for your camera
username = 'admin'          # Replace with the username for your camera
password = 'password'       # Replace with the password for your camera
```
2. Construct the RTSP stream URLs:
```
url_640x480 = f"rtsp://{username}:{password}@{ip_address}:{port}/stream2"
url_1080p = f"rtsp://{username}:{password}@{ip_address}:{port}/stream1"
```

3. Set the desired RTSP stream URL:
```
rtsp_url = url_640x480  # Set it to either `url_640x480` or `url_1080p` based on the desired resolution
```

4. Run the script and observe the video stream

#### Example Configuration:
```
# Set up camera connection details
ip_address = '192.168.0.XX' # Replace with the IP address of your camera
port = '554'          # Replace with the port number for your camera
username = 'admin'    # Replace with the username for your camera
password = 'password' # Replace with the password for your camera

# Construct the RTSP stream URLs using variables
url_640x480 = f"rtsp://{username}:{password}@{ip_address}:{port}/stream2"
url_1080p = f"rtsp://{username}:{password}@{ip_address}:{port}/stream1"

# Set up RTSP stream URL
rtsp_url = url_640x480

````

#### Conclusion
This script provides a basic implementation to connect to a Tapo C200 TP Link camera using an RTSP link and utilize the video feed with OpenCV. Feel free to modify the configuration and adapt it to your specific use case.

Please note that this script assumes a working network connection to the Tapo C200 camera and proper configuration of the camera's IP address, port, username, and password.

#### Resources & References
This section provides resources used in the discovery phase of the script in attempt to make the video feed work for python open cv.

https://www.tp-link.com/us/support/faq/2680/

https://helpdesk.cctvdiscover.com/network/rtsp_stream.html

https://www.ispyconnect.com/

https://www.ispyconnect.com/camera/tp-link

https://programtalk.com/vs4/python/JurajNyiri/pytapo/


#### How to Setup Tapo Smart Home WiFi Camera C200
"https://www.youtube.com/watch?v=ozBOifbkqGk"
