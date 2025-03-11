# -*- coding: utf-8 -*-
"""
Created on Thu Mar  6 17:30:08 2025

@author: User
"""
import os
import time
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.metrics import mean_squared_error, mean_absolute_error, accuracy_score

os.chdir(r'D:\IITR\ML_BCSD\Codes_BCSD\Update')
def nse(observed, predicted):
    return 1 - (np.sum((observed - predicted) ** 2) / np.sum((observed - np.mean(observed)) ** 2))

def kge(observed, predicted):
    r = np.corrcoef(observed, predicted)[0, 1]
    beta = np.mean(predicted) / np.mean(observed)
    gamma = (np.std(predicted) / np.mean(predicted)) / (np.std(observed) / np.mean(observed))
    return 1 - np.sqrt((r - 1) ** 2 + (beta - 1) ** 2 + (gamma - 1) ** 2)

# Load datasets
data_class = pd.read_csv("data_class.csv")
data_reg = pd.read_csv("data_reg.csv")

# Preprocessing Classification Data
label_encoder = LabelEncoder()
data_class.iloc[:, -1] = label_encoder.fit_transform(data_class.iloc[:, -1])
scaler_class = MinMaxScaler()
data_class.iloc[:, :-1] = scaler_class.fit_transform(data_class.iloc[:, :-1])

# Preprocessing Regression Data
scaler_reg = MinMaxScaler()
data_reg.iloc[:, :-1] = scaler_reg.fit_transform(data_reg.iloc[:, :-1])

# Create sequences
def create_sequences(data, target_col, n_steps=10):
    X, y = [], []
    for i in range(len(data) - n_steps):
        X.append(data.iloc[i:i + n_steps, :-1].values)
        y.append(data.iloc[i + n_steps][target_col])
    return np.array(X), np.array(y)

X_class, y_class = create_sequences(data_class, target_col=data_class.columns[-1])
X_reg, y_reg = create_sequences(data_reg, target_col=data_reg.columns[-1])

X_class = X_class.reshape((X_class.shape[0], X_class.shape[1], X_class.shape[2]))
X_reg = X_reg.reshape((X_reg.shape[0], X_reg.shape[1], X_reg.shape[2]))

# Define models
model_class = Sequential([
    LSTM(50, activation='relu', return_sequences=True, input_shape=(X_class.shape[1], X_class.shape[2])),
    LSTM(50, activation='relu'),
    Dense(1, activation='sigmoid')
])
model_class.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

model_reg = Sequential([
    LSTM(50, activation='relu', return_sequences=True, input_shape=(X_reg.shape[1], X_reg.shape[2])),
    LSTM(50, activation='relu'),
    Dense(1)
])
model_reg.compile(optimizer='adam', loss='mse', metrics=['mse'])

# Train and evaluate Classification Model
start_time = time.time()
model_class.fit(X_class, y_class, epochs=20, batch_size=16, verbose=1)
time_class = time.time() - start_time
y_pred_class = (model_class.predict(X_class) > 0.5).astype("int32")
accuracy = accuracy_score(y_class, y_pred_class)

# Train and evaluate Regression Model
start_time = time.time()
model_reg.fit(X_reg, y_reg, epochs=20, batch_size=16, verbose=1)
time_reg = time.time() - start_time
y_pred_reg = model_reg.predict(X_reg).flatten()

mse = mean_squared_error(y_reg, y_pred_reg)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_reg, y_pred_reg)
nse_value = nse(y_reg, y_pred_reg)
kge_value = kge(y_reg, y_pred_reg)

# Save results to CSV
results = pd.DataFrame({
    "Metric": ["Accuracy", "Training Time (s)", "MSE", "RMSE", "MAE", "NSE", "KGE"],
    "Classification": [accuracy, time_class, None, None, None, None, None],
    "Regression": [None, time_reg, mse, rmse, mae, nse_value, kge_value]
})

os.chdir(r'C:\Users\User\Dropbox\IITR\BCSD')
results.to_csv("evaluation_metrics.csv", index=False)

print("Evaluation metrics saved to evaluation_metrics.csv")
