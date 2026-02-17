import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import lightgbm as lgb
import matplotlib.pyplot as plt
import seaborn as sns


# Load the dataset
def load_data(file_path):
    """
    Load housing data from CSV file

    Parameters:
    file_path (str): Path to the CSV file

    Returns:
    pd.DataFrame: Loaded dataset
    """
    print("Loading data...")
    data = pd.read_csv(file_path)
    print(f"Dataset shape: {data.shape}")
    print(f"Columns: {data.columns.tolist()}")
    return data


# Exploratory Data Analysis
def explore_data(data):
    """
    Perform exploratory data analysis on the dataset

    Parameters:
    data (pd.DataFrame): Input dataset
    """
    print("\n=== Exploratory Data Analysis ===")

    # Basic information
    print("\nDataset Info:")
    print(data.info())

    # Statistical summary
    print("\nStatistical Summary:")
    print(data.describe())

    # Check for missing values
    print("\nMissing Values:")
    print(data.isnull().sum())

    # Check target variable distribution
    plt.figure(figsize=(15, 10))

    plt.subplot(2, 3, 1)
    data['median_house_value'].hist(bins=50)
    plt.title('Distribution of Median House Value')
    plt.xlabel('House Value')
    plt.ylabel('Frequency')

    plt.subplot(2, 3, 2)
    data['median_income'].hist(bins=50)
    plt.title('Distribution of Median Income')
    plt.xlabel('Income')
    plt.ylabel('Frequency')

    plt.subplot(2, 3, 3)
    data['ocean_proximity'].value_counts().plot(kind='bar')
    plt.title('Ocean Proximity Distribution')
    plt.xlabel('Ocean Proximity')
    plt.ylabel('Count')

    plt.subplot(2, 3, 4)
    plt.scatter(data['median_income'], data['median_house_value'], alpha=0.5)
    plt.title('Income vs House Value')
    plt.xlabel('Median Income')
    plt.ylabel('Median House Value')

    plt.subplot(2, 3, 5)
    data['housing_median_age'].hist(bins=30)
    plt.title('Distribution of Housing Median Age')
    plt.xlabel('Housing Age')
    plt.ylabel('Frequency')

    plt.tight_layout()
    plt.show()


# Preprocess the data
def preprocess_data(data):
    """
    Preprocess the housing data for training

    Parameters:
    data (pd.DataFrame): Raw dataset

    Returns:
    tuple: Processed features, target variable, and preprocessing objects
    """
    print("\nPreprocessing data...")

    # Create a copy to avoid modifying original data
    df = data.copy()

    # Handle missing values in total_bedrooms
    if df['total_bedrooms'].isnull().any():
        median_bedrooms = df['total_bedrooms'].median()
        df['total_bedrooms'].fillna(median_bedrooms, inplace=True)
        print(f"Filled missing values in total_bedrooms with median: {median_bedrooms}")

    # Feature engineering
    df['rooms_per_household'] = df['total_rooms'] / df['households']
    df['bedrooms_per_room'] = df['total_bedrooms'] / df['total_rooms']
    df['population_per_household'] = df['population'] / df['households']

    # Encode categorical variable
    label_encoder = LabelEncoder()
    df['ocean_proximity_encoded'] = label_encoder.fit_transform(df['ocean_proximity'])

    # Define features and target
    feature_columns = [
        'longitude', 'latitude', 'housing_median_age', 'total_rooms',
        'total_bedrooms', 'population', 'households', 'median_income',
        'rooms_per_household', 'bedrooms_per_room', 'population_per_household',
        'ocean_proximity_encoded'
    ]

    X = df[feature_columns]
    y = df['median_house_value']

    print(f"Features shape: {X.shape}")
    print(f"Target shape: {y.shape}")

    return X, y, label_encoder


# Train LightGBM model
def train_lightgbm(X, y, test_size=0.2, random_state=42):
    """
    Train LightGBM regression model

    Parameters:
    X (pd.DataFrame): Feature matrix
    y (pd.Series): Target variable
    test_size (float): Proportion of test set
    random_state (int): Random seed for reproducibility

    Returns:
    tuple: Trained model, test features, test target, and scaler
    """
    print("\nTraining LightGBM model...")

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    # Scale the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Create LightGBM dataset
    train_data = lgb.Dataset(X_train_scaled, label=y_train)
    test_data = lgb.Dataset(X_test_scaled, label=y_test, reference=train_data)

    # Define parameters for LightGBM
    params = {
        'objective': 'regression',
        'metric': 'rmse',
        'boosting_type': 'gbdt',
        'num_leaves': 31,
        'learning_rate': 0.05,
        'feature_fraction': 0.9,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'verbose': 0,
        'random_state': random_state
    }

    # Train the model
    model = lgb.train(
        params,
        train_data,
        num_boost_round=1000,
        valid_sets=[test_data],
        callbacks=[
            lgb.early_stopping(stopping_rounds=50, verbose=True),
            lgb.log_evaluation(50)
        ]
    )

    print(f"Model training completed. Best iteration: {model.best_iteration}")

    return model, X_test_scaled, y_test, scaler


# Evaluate the model
def evaluate_model(model, X_test, y_test):
    """
    Evaluate the trained LightGBM model

    Parameters:
    model: Trained LightGBM model
    X_test: Test features
    y_test: True target values for test set
    """
    print("\n=== Model Evaluation ===")

    # Make predictions
    y_pred = model.predict(X_test)

    # Calculate metrics
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f"Mean Squared Error (MSE): {mse:,.2f}")
    print(f"Root Mean Squared Error (RMSE): {rmse:,.2f}")
    print(f"Mean Absolute Error (MAE): {mae:,.2f}")
    print(f"R² Score: {r2:.4f}")

    # Plot actual vs predicted values
    plt.figure(figsize=(15, 5))

    plt.subplot(1, 3, 1)
    plt.scatter(y_test, y_pred, alpha=0.5)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    plt.xlabel('Actual Values')
    plt.ylabel('Predicted Values')
    plt.title('Actual vs Predicted House Values')

    plt.subplot(1, 3, 2)
    residuals = y_test - y_pred
    plt.scatter(y_pred, residuals, alpha=0.5)
    plt.axhline(y=0, color='r', linestyle='--')
    plt.xlabel('Predicted Values')
    plt.ylabel('Residuals')
    plt.title('Residual Plot')

    plt.subplot(1, 3, 3)
    # Feature importance
    feature_importance = pd.DataFrame({
        'feature': [f'feature_{i}' for i in range(len(model.feature_importance()))],
        'importance': model.feature_importance()
    })
    feature_importance = feature_importance.sort_values('importance', ascending=True)
    plt.barh(feature_importance['feature'], feature_importance['importance'])
    plt.xlabel('Feature Importance')
    plt.title('LightGBM Feature Importance')

    plt.tight_layout()
    plt.show()

    return y_pred, rmse, r2



# Main execution function
def main():
    """
    Main function to run the complete LightGBM pipeline
    """
    # Load your data (replace with your actual file path)
    file_path = "housing.csv"  # Update this path
    data = load_data(file_path)

    # Explore the data
    explore_data(data)

    # Preprocess data
    X, y, label_encoder = preprocess_data(data)

    # Train LightGBM model
    model, X_test, y_test, scaler = train_lightgbm(X, y)

    # Evaluate model
    y_pred, rmse, r2 = evaluate_model(model, X_test, y_test)

    print("\n=== Model Training Complete ===")
    print(f"Final RMSE: {rmse:,.2f}")
    print(f"Final R² Score: {r2:.4f}")


if __name__ == "__main__":
    main()