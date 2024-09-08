import numpy as np
import pandas as pd
from datetime import datetime
from tensorflow.keras.models import load_model
import joblib
def custom_date_parser(date_string):
    # Loại bỏ 'AM' hoặc 'PM' vì thời gian đã ở định dạng 24 giờ
    date_string = date_string.replace(' AM', '').replace(' PM', '')
    return datetime.strptime(date_string, '%b-%d-%Y %H:%M:%S UTC')
def preprocess_data(data):
    # convert timestamp to datetime
    # data['time_stamp'] = pd.to_datetime(data['time_stamp'], format='%b-%d-%Y %H:%M:%S UTC', errors='coerce')
    data['time_stamp'] = data['time_stamp'].apply(custom_date_parser)
    
    # caculate duration
    data = data.sort_values(['from', 'time_stamp'])
    
    data['duration'] = data.groupby('from')['time_stamp'].diff().dt.total_seconds().fillna(1000)
    
    # process 'value'
    data['value'] = data['value'].str.replace('$', '').str.replace(',', '').astype(float)
    
    # encode 'method_called'
    method_mapping = {'buy': 1, 'transfer': 2, 'swap': 3, 'printMoney': 4}
    data['method_called'] = data['method_called'].map(method_mapping)
    return data
# Load model và scaler đã train
autoencoder = load_model('autoencoder_model.h5')
scaler = joblib.load('scaler.pkl')

class TransactionDetector:
    def __init__(self, train_data, large_transaction_threshold=0.95, rapid_transaction_threshold=0.05):
        self.transaction_history = train_data
        self.large_transaction_threshold = large_transaction_threshold
        self.rapid_transaction_threshold = rapid_transaction_threshold
        
        # Sắp xếp dữ liệu theo thời gian và tính duration
        self.transaction_history = self.transaction_history.sort_values('time_stamp')
        self.transaction_history['duration'] = self.transaction_history.groupby('from')['time_stamp'].diff().dt.total_seconds().fillna(100)
        
        # Chuẩn bị các ngưỡng
        self.large_transaction_value = self.transaction_history['value'].quantile(self.large_transaction_threshold)
        self.rapid_transaction_duration = self.transaction_history['duration'].quantile(self.rapid_transaction_threshold)

    def detect_large_transaction(self, value):
        return value > self.large_transaction_value

    def detect_rapid_transaction(self, account, timestamp):
        last_transaction = self.transaction_history[self.transaction_history['from'] == account].iloc[-1]
        duration = (timestamp - last_transaction['time_stamp']).total_seconds()
        return duration < self.rapid_transaction_duration, duration

    def update_history(self, new_transaction):
        self.transaction_history = pd.concat([self.transaction_history, new_transaction], ignore_index=True)
        self.transaction_history = self.transaction_history.sort_values('time_stamp')
        
        # Cập nhật duration cho giao dịch mới
        new_duration = self.transaction_history.groupby('from')['time_stamp'].diff().dt.total_seconds().iloc[-1]
        self.transaction_history.iloc[-1, self.transaction_history.columns.get_loc('duration')] = new_duration
        
        # Cập nhật các ngưỡng
        self.large_transaction_value = self.transaction_history['value'].quantile(self.large_transaction_threshold)
        self.rapid_transaction_duration = self.transaction_history['duration'].quantile(self.rapid_transaction_threshold)

def preprocess_transaction(transaction, detector):
    # Xử lý timestamp
    timestamp = custom_date_parser(transaction['time_stamp'])
    
    # Xử lý giá trị
    value = float(transaction['value'].replace('$', '').replace(',', ''))
    
    # Mã hóa phương thức
    method_mapping = {'buy': 1, 'transfer': 2, 'swap': 3, 'printMoney': 4}
    method = method_mapping.get(transaction['method_called'], 0)
    
    # Phát hiện large transaction
    large_transaction = int(detector.detect_large_transaction(value))
    
    # Phát hiện rapid transaction và lấy duration
    rapid_transaction, duration = detector.detect_rapid_transaction(transaction['from'], timestamp)
    rapid_transaction = int(rapid_transaction)
    
    # Cập nhật lịch sử giao dịch
    new_transaction = pd.DataFrame({
        'transaction_id': [transaction['transaction_id']],
        'time_stamp': [timestamp],
        'from': [transaction['from']],
        'to': [transaction['to']],
        'value': [value],
        'method_called': [transaction['method_called']],
        'large_transaction': [large_transaction],
        'rapid_transaction': [rapid_transaction],
        'duration': [duration]
    })
    detector.update_history(new_transaction)
    
    return [value, method, large_transaction, rapid_transaction, duration]

def detect_anomaly(transaction, detector, threshold=0.1):
    # Tiền xử lý transaction
    features = preprocess_transaction(transaction, detector)
    
    # Chuẩn hóa dữ liệu
    features_scaled = scaler.transform([features])
    
    # Dự đoán
    reconstructed = autoencoder.predict(features_scaled)
    
    # Tính toán lỗi reconstruction
    mse = np.mean(np.power(features_scaled - reconstructed, 2), axis=1)
    
    # Nếu lỗi reconstruction lớn hơn ngưỡng, coi là bất thường
    return mse[0] > threshold, mse[0]

# Đọc dữ liệu train
train_data = pd.read_csv('dataset.csv')  # Giả sử dữ liệu của bạn được lưu trong file CSV

train_data = preprocess_data(train_data)
# Khởi tạo detector với dữ liệu train
detector = TransactionDetector(train_data)

# Ví dụ sử dụng
new_transaction = {
    'transaction_id': '0x123434284234',
    'time_stamp': 'Aug-01-2024 10:00:00 AM UTC',
    'from': '0x000000000231c53e9dCbD5Ee410f065FBc170c29',
    'to': '0x00000000041d945c46E073F0048cEf510D148dEA',
    'value': '$1000',
    'method_called': 'transfer'
}

is_anomaly, anomaly_score = detect_anomaly(new_transaction, detector)

if is_anomaly:
    print(f"Transaction is anomaly with score {anomaly_score:.4f}")
else:
    print(f"Transaction is detect as casual with anomaly score {anomaly_score:.4f}")