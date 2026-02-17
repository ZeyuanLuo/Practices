import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Data preparations and cleaning
df = pd.read_csv('Adidas_US_Sales_Dataset.csv', sep=';')


# cleaning function
def clean_currency(value):
    if isinstance(value, str):
        return float(value.replace('$', '').replace(' ', '').replace(',', ''))
    return value


df['invoicedate'] = pd.to_datetime(df['invoicedate'], format='%d/%m/%Y')
df['totalsales'] = df['totalsales'].apply(clean_currency)

# Group total sales by days
daily_sales = df.groupby('invoicedate')['totalsales'].sum().reset_index().sort_values('invoicedate')
sales_data = daily_sales['totalsales'].values.reshape(-1, 1)

# Use MinMax Scaler
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(sales_data)


# From last 7 days we predict
def create_sequences(data, window_size):
    X, y = [], []
    for i in range(len(data) - window_size):
        X.append(data[i: i + window_size])
        y.append(data[i + window_size])
    return np.array(X), np.array(y)


WINDOW_SIZE = 7
X, y = create_sequences(scaled_data, WINDOW_SIZE)

# Dimensional transformation：Conv1d needs (Batch, Channel, Length)
#  X : (Sample, Length, Feature=1) ->  (Sample, Channel=1, Length)
X_tensor = torch.FloatTensor(X).permute(0, 2, 1)
y_tensor = torch.FloatTensor(y)

# Divide Train Test Datasets
train_size = int(len(X_tensor) * 0.8)
X_train, X_test = X_tensor[:train_size], X_tensor[train_size:]
y_train, y_test = y_tensor[:train_size], y_tensor[train_size:]

# DataLoader
train_dataset = TensorDataset(X_train, y_train)
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)


# Define CNN models
class SalesCNN(nn.Module):
    def __init__(self):
        super(SalesCNN, self).__init__()
        # 1D Convolution layer: Input channel 1, output channel 64, convolution kernel size 2
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=64, kernel_size=2)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool1d(kernel_size=2)
        self.flatten = nn.Flatten()

        # Fully connected layer
        # Calculate the dimensions after flattening:
        # Input length 7 -> Conv(k=2) -> 6 -> Pool(k=2) -> 3
        # Number of output features = 64 × 3 = 192
        self.fc1 = nn.Linear(64 * 3, 50)
        self.fc2 = nn.Linear(50, 1)  # 回归输出

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.pool(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x


model = SalesCNN()

# Train the model
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

epochs = 300
model.train()

print("Start training...")
for epoch in range(epochs):
    epoch_loss = 0
    for batch_X, batch_y in train_loader:
        optimizer.zero_grad()  # Clear gradient
        outputs = model(batch_X)  # Forward
        loss = criterion(outputs, batch_y)  # Calculate losses
        loss.backward()  # Backward
        optimizer.step()  # Update the parameter
        epoch_loss += loss.item()

    if (epoch + 1) % 20 == 0:
        print(f"Epoch [{epoch + 1}/{epochs}], Loss: {epoch_loss / len(train_loader):.4f}")

# Test and Evaluate
model.eval()
with torch.no_grad():
    predictions = model(X_test)

    # Reverse normalisation, restoring true values
    y_test_actual = scaler.inverse_transform(y_test.numpy())
    y_pred_actual = scaler.inverse_transform(predictions.numpy())

# 1. Basic error indicator
mae = mean_absolute_error(y_test_actual, y_pred_actual)
mse = mean_squared_error(y_test_actual, y_pred_actual)
rmse = np.sqrt(mse)
r2 = r2_score(y_test_actual, y_pred_actual)

# 2. MAPE (Average absolute percentage error)
mape = np.mean(np.abs((y_test_actual - y_pred_actual) / (y_test_actual + 1e-10))) * 100

# Accuracy within Tolerance
def calculate_accuracy(y_true, y_pred, tolerance=0.10):
    # Calculate the percentage error for each point
    errors = np.abs((y_true - y_pred) / (y_true + 1e-10))
    # The number of statistical errors less than tolerance
    correct_count = np.sum(errors <= tolerance)
    return (correct_count / len(y_true)) * 100

acc_10 = calculate_accuracy(y_test_actual, y_pred_actual, tolerance=0.10) # 10%
acc_20 = calculate_accuracy(y_test_actual, y_pred_actual, tolerance=0.20) # 20%

print("-" * 30)
print("Model Performance Evaluation:")
print("-" * 30)
print(f"MAE : ${mae:,.2f}")
print(f"RMSE: ${rmse:,.2f}")
print(f"R² Score: {r2:.4f}")
print(f"MAPE: {mape:.2f}%")
print("-" * 30)
print(f"Accuracy (±10% Tolerance): {acc_10:.2f}%")
print(f"Accuracy (±20% Tolerance): {acc_20:.2f}%")
print("-" * 30)

# Results
plt.figure(figsize=(12, 6))
plt.plot(y_test_actual, label='Actual Sales', color='blue')
plt.plot(y_pred_actual, label='Predicted Sales (CNN)', color='orange', linestyle='--')
plt.title('Adidas Daily Sales Forecast using PyTorch 1D-CNN')
plt.xlabel('Days (Test Set)')
plt.ylabel('Total Sales ($)')
plt.legend()
plt.show()
