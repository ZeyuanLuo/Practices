import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

# Load dataset
df = pd.read_csv('housing.csv')

# Display basic dataset information
print("Data Shape:", df.shape)
print("\nFirst 5 rows of data:")
print(df.head())
print("\nDataset Info:")
print(df.info())
print("\nDescriptive Statistics:")
print(df.describe())

# Check for missing values
print("\nMissing Values:")
print(df.isnull().sum())


# Remove categorical column (ocean_proximity) as per requirement
df = df.drop(columns=['ocean_proximity'])


# feature list
features = ['longitude', 'latitude', 'housing_median_age', 'total_rooms',
           'total_bedrooms', 'population', 'households', 'median_income']

X = df[features]
y = df['median_house_value']

print(f"Features shape: {X.shape}")
print(f"Target variable shape: {y.shape}")
print(f"Features used: {features}")
'''
# Data Visualization
# Correlation heatmap
plt.figure(figsize=(12, 8))
correlation_matrix = df.corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0, fmt='.2f')
plt.title('Feature Correlation Heatmap')
plt.tight_layout()
plt.show()

# Target variable distribution analysis
plt.figure(figsize=(12, 4))

# Histogram of house values
plt.subplot(1, 2, 1)
plt.hist(y, bins=50, alpha=0.7, color='skyblue')
plt.xlabel('Median House Value ($)')
plt.ylabel('Frequency')
plt.title('Distribution of House Prices')

# Scatter plot: Income vs House Value
plt.subplot(1, 2, 2)
plt.scatter(df['median_income'], y, alpha=0.5)
plt.xlabel('Median Income')
plt.ylabel('Median House Value ($)')
plt.title('Income vs House Price Relationship')
plt.tight_layout()
plt.show()
'''
# Split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, shuffle=True
)

print(f"\nTraining set size: {X_train.shape}")
print(f"Test set size: {X_test.shape}")

# Handle missing values by filling with median
train_median = X_train['total_bedrooms'].median()
X_train['total_bedrooms'] = X_train['total_bedrooms'].fillna(train_median)
test_median = X_test['total_bedrooms'].median()
X_test['total_bedrooms'] = X_test['total_bedrooms'].fillna(test_median)

# Feature standardization - crucial for linear regression
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Initialize and train Linear Regression model
lr_model = LinearRegression()
lr_model.fit(X_train_scaled, y_train)

# Display model coefficients (feature importance)
feature_importance = pd.DataFrame({
    'feature': features,
    'coefficient': lr_model.coef_
}).sort_values('coefficient', key=abs, ascending=False)

print("\n=== Feature Importance (by coefficient magnitude) ===")
print(feature_importance)
print(f"Model Intercept: ${lr_model.intercept_:.2f}")

# Make predictions on test set
y_pred = lr_model.predict(X_test_scaled)

# Calculate evaluation metrics
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Display model performance
print("\n=== Model Performance Evaluation ===")
print(f"Mean Squared Error (MSE): ${mse:,.2f}")
print(f"Root Mean Squared Error (RMSE): ${rmse:,.2f}")
print(f"Mean Absolute Error (MAE): ${mae:,.2f}")
print(f"R-squared (R²): {r2:.4f}")
print(f"Explained Variance: {r2*100:.2f}%")


# Residual Analysis - Critical for model diagnostics
residuals = y_test - y_pred

plt.figure(figsize=(15, 5))

# Plot 1: Residuals vs Predicted values
plt.subplot(1, 3, 1)
plt.scatter(y_pred, residuals, alpha=0.6, color='blue')
plt.axhline(y=0, color='red', linestyle='--', linewidth=2)
plt.xlabel('Predicted House Value ($)')
plt.ylabel('Residuals ($)')
plt.title('Residuals vs Predicted Values')
plt.grid(True, alpha=0.3)

# Plot 2: Distribution of residuals
plt.subplot(1, 3, 2)
plt.hist(residuals, bins=50, alpha=0.7, color='lightcoral', edgecolor='black')
plt.xlabel('Residuals ($)')
plt.ylabel('Frequency')
plt.title('Distribution of Residuals')
plt.grid(True, alpha=0.3)

# Plot 3: Actual vs Predicted values
plt.subplot(1, 3, 3)
plt.scatter(y_test, y_pred, alpha=0.6, color='green')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()],
         'red', linestyle='--', linewidth=2)
plt.xlabel('Actual House Value ($)')
plt.ylabel('Predicted House Value ($)')
plt.title('Actual vs Predicted Values')
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# Additional Performance Analysis
print("\n=== Additional Analysis ===")

# Calculate percentage errors
percentage_errors = np.abs((y_test - y_pred) / y_test) * 100
mean_percentage_error = np.mean(percentage_errors)
median_percentage_error = np.median(percentage_errors)

print(f"Mean Absolute Percentage Error: {mean_percentage_error:.2f}%")
print(f"Median Absolute Percentage Error: {median_percentage_error:.2f}%")

# Check residual statistics
print(f"\nResidual Statistics:")
print(f"Mean of residuals: ${residuals.mean():.2f}")
print(f"Standard deviation of residuals: ${residuals.std():.2f}")
