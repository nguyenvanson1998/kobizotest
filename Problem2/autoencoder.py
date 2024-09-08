import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.optimizers import Adam
from datetime import datetime

data = pd.read_csv('dataset.csv')  
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
    

    features = ['value', 'method_called', 'large_transaction', 'rapid_transaction', 'duration']
    X = data[features]
    y = data['fraud_transaction']
    
    return X, y

X,y = preprocess_data(data)

# Chia dữ liệu thành tập train và test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Chuẩn hóa dữ liệu
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Xây dựng model Autoencoder
input_dim = X_train.shape[1]
encoding_dim = 3

input_layer = Input(shape=(input_dim,))
encoder = Dense(32, activation="relu")(input_layer)
encoder = Dense(16, activation="relu")(encoder)
encoder = Dense(encoding_dim, activation="relu")(encoder)
decoder = Dense(16, activation="relu")(encoder)
decoder = Dense(32, activation="relu")(decoder)
decoder = Dense(input_dim, activation="sigmoid")(decoder)

autoencoder = Model(inputs=input_layer, outputs=decoder)
autoencoder.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')

# Train model
autoencoder.fit(X_train_scaled, X_train_scaled, 
                epochs=50, 
                batch_size=32, 
                shuffle=True, 
                validation_data=(X_test_scaled, X_test_scaled))

# Lưu model và scaler
autoencoder.save('autoencoder_model.h5')
import joblib
joblib.dump(scaler, 'scaler.pkl')

print("Model saving.")