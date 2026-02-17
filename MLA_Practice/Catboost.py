# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
from catboost import CatBoostRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt


# Load dataset
def load_data(file_path):
    """
    Load housing data from CSV file
    """
    data = pd.read_csv(file_path)
    print(f"Dataset shape: {data.shape}")
    print("\nFirst few rows:")
    print(data.head())
    print("\nData info:")
    print(data.info())
    return data


# Preprocess data
def preprocess_data(data):
    """
    Preprocess the housing data for training
    """
    # Handle missing values
    data = data.dropna()

    # Separate features and target
    X = data.drop('median_house_value', axis=1)
    y = data['median_house_value']

    # Identify categorical features for CatBoost
    categorical_features = ['ocean_proximity']

    # Convert categorical column to string type (CatBoost requirement)
    for col in categorical_features:
        X[col] = X[col].astype(str)

    return X, y, categorical_features


# Train CatBoost model
def train_catboost(X, y, categorical_features):
    """
    Train CatBoost regression model
    """
    # Split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Initialize CatBoost Regressor
    model = CatBoostRegressor(
        iterations=1000,
        learning_rate=0.1,
        depth=6,
        cat_features=categorical_features,
        random_seed=42,
        verbose=100,
        loss_function='RMSE'
    )

    # Train the model
    model.fit(
        X_train, y_train,
        eval_set=(X_test, y_test),
        use_best_model=True,
        plot=False
    )

    return model, X_test, y_test


# Evaluate model
def evaluate_model(model, X_test, y_test):
    """
    Evaluate the trained model
    """
    # Make predictions
    y_pred = model.predict(X_test)

    # Calculate metrics
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)

    print(f"\nModel Performance:")
    print(f"RMSE: {rmse:.2f}")
    print(f"R² Score: {r2:.4f}")

    # Plot feature importance
    feature_importance = model.get_feature_importance()
    feature_names = X_test.columns

    plt.figure(figsize=(10, 6))
    plt.barh(feature_names, feature_importance)
    plt.xlabel('Feature Importance')
    plt.title('CatBoost Feature Importance')
    plt.tight_layout()
    plt.show()

    return rmse, r2


# Main execution
def main():
    """
    Main function to run the housing price prediction
    """
    # Load data
    print("Loading housing data...")
    data = load_data('housing.csv')  # Replace with your CSV file path

    # Preprocess data
    print("\nPreprocessing data...")
    X, y, categorical_features = preprocess_data(data)

    # Train model
    print("\nTraining CatBoost model...")
    model, X_test, y_test = train_catboost(X, y, categorical_features)

    # Evaluate model
    print("\nEvaluating model...")
    rmse, r2 = evaluate_model(model, X_test, y_test)

    print('rmse =', rmse)
    print('r2 =', r2)

    # Save model (optional)
    model.save_model('catboost_housing_model.cbm')
    print("\nModel saved as 'catboost_housing_model.cbm'")


if __name__ == "__main__":
    main()