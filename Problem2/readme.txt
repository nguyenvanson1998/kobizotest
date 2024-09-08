# Real-time Transaction Anomaly Detection System

## Overview
This project implements a real-time anomaly detection system for financial transactions. It uses machine learning techniques, specifically an autoencoder, to identify suspicious activities such as large transactions, rapid transactions, and potential fraud.

## Features
- Real-time processing of transaction data
- Detection of large transactions based on historical data
- Identification of rapid transactions from the same account
- Fraud detection using a trained autoencoder model
- Continuous updating of transaction history and detection thresholds

## Requirements
- Python 3.7+
- pandas
- numpy
- scikit-learn
- PyTorch
- joblib

You can install the required packages using:

pip install -r requirements.txt

## Project Structure

- `train_model.py`: Script for preprocessing data and training the autoencoder model
- `detector.py`: Contains the `TransactionDetector` class and anomaly detection logic
- `main.py`: Example script demonstrating how to use the system
- `autoencoder_model.pth`: Saved PyTorch model (generated after training)
- `scaler.pkl`: Saved StandardScaler object (generated after training)

## Usage

1. Train the model:
python3 autoencoder.py
This will generate `autoencoder_model.pth` and `scaler.pkl`.

2. Use the system:

```python
from detector import TransactionDetector, detect_anomaly

# Initialize the detector with your training data
detector = TransactionDetector(train_data)

# Process a new transaction
new_transaction = {
    'transaction_id': '0x1234...',
    'time_stamp': 'Aug-01-2024 10:00:00 AM UTC',
    'from': '0x000000000231c53e9dCbD5Ee410f065FBc170c29',
    'to': '0x00000000041d945c46E073F0048cEf510D148dEA',
    'value': '$1000',
    'method_called': 'transfer'
}

is_anomaly, anomaly_score = detect_anomaly(new_transaction, detector)

if is_anomaly:
    print(f"Transaction detected as anomalous with score {anomaly_score:.4f}")
else:
    print(f"Transaction appears normal with score {anomaly_score:.4f}")

Customization
You can adjust the thresholds for large and rapid transactions in the TransactionDetector class initialization:
detector = TransactionDetector(train_data, large_transaction_threshold=0.95, rapid_transaction_threshold=0.05)
Note
This system is designed for educational and demonstration purposes. In a production environment, additional security measures, error handling, and optimizations would be necessary.
License
MIT License
Contributing
Contributions, issues, and feature requests are welcome. Feel free to check issues page if you want to contribute.
Author
Sonnv