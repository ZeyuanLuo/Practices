# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt
import seaborn as sns
import joblib


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
    print("\nMissing values:")
    print(data.isnull().sum())
    return data


# Preprocess data
def preprocess_data(data):
    """
    Preprocess the housing data for Random Forest training
    """
    # Create a copy to avoid modifying original data
    df = data.copy()

    # Handle missing values
    print("Handling missing values...")
    numeric_columns = df.select_dtypes(include=[np.number]).columns
    categorical_columns = df.select_dtypes(include=['object']).columns

    # Impute numerical missing values with median
    imputer = SimpleImputer(strategy='median')
    df[numeric_columns] = imputer.fit_transform(df[numeric_columns])

    # Handle categorical variable (ocean_proximity)
    print("Encoding categorical variables...")
    label_encoder = LabelEncoder()
    df['ocean_proximity_encoded'] = label_encoder.fit_transform(df['ocean_proximity'])

    # Separate features and target
    feature_columns = ['longitude', 'latitude', 'housing_median_age', 'total_rooms',
                       'total_bedrooms', 'population', 'households', 'median_income',
                       'ocean_proximity_encoded']

    X = df[feature_columns]
    y = df['median_house_value']

    return X, y, label_encoder


# Train Random Forest model
def train_random_forest(X, y, use_grid_search=False):
    """
    Train Random Forest regression model
    """
    # Split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, shuffle=True
    )

    if use_grid_search:
        print("Performing Grid Search for hyperparameter tuning...")
        # Define parameter grid for GridSearch
        param_grid = {
            'n_estimators': [100, 200, 300],
            'max_depth': [10, 20, None],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'max_features': ['auto', 'sqrt']
        }

        rf = RandomForestRegressor(random_state=42, n_jobs=-1)
        grid_search = GridSearchCV(
            estimator=rf,
            param_grid=param_grid,
            cv=5,
            scoring='neg_mean_squared_error',
            n_jobs=-1,
            verbose=1
        )

        grid_search.fit(X_train, y_train)
        best_model = grid_search.best_estimator_
        print(f"Best parameters: {grid_search.best_params_}")
        print(f"Best cross-validation score: {-grid_search.best_score_:.2f}")

    else:
        print("Training Random Forest with default parameters...")
        # Use default parameters for faster training
        best_model = RandomForestRegressor(
            n_estimators=200,
            max_depth=20,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1,
            verbose=1
        )
        best_model.fit(X_train, y_train)

    return best_model, X_train, X_test, y_train, y_test


# Evaluate model
def evaluate_model(model, X_test, y_test, feature_names):
    """
    Evaluate the trained Random Forest model
    """
    # Make predictions
    y_pred = model.predict(X_test)

    # Calculate metrics
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f"\n=== Model Performance ===")
    print(f"RMSE: {rmse:,.2f}")
    print(f"MAE: {mae:,.2f}")
    print(f"R² Score: {r2:.4f}")

    # Plot feature importance
    feature_importance = model.feature_importances_
    feature_importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': feature_importance
    }).sort_values('importance', ascending=True)

    plt.figure(figsize=(10, 8))
    plt.barh(feature_importance_df['feature'], feature_importance_df['importance'])
    plt.xlabel('Feature Importance')
    plt.title('Random Forest Feature Importance')
    plt.tight_layout()
    plt.show()

    # Plot actual vs predicted values
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, y_pred, alpha=0.5)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    plt.xlabel('Actual Prices')
    plt.ylabel('Predicted Prices')
    plt.title('Actual vs Predicted House Prices')
    plt.tight_layout()
    plt.show()

    # Residual plot
    residuals = y_test - y_pred
    plt.figure(figsize=(10, 6))
    plt.scatter(y_pred, residuals, alpha=0.5)
    plt.axhline(y=0, color='r', linestyle='--')
    plt.xlabel('Predicted Prices')
    plt.ylabel('Residuals')
    plt.title('Residual Plot')
    plt.tight_layout()
    plt.show()

    return rmse, r2, mae


# Make predictions on new data
def predict_new_data(model, new_data, label_encoder):
    """
    Make predictions on new data
    """
    # Preprocess new data in the same way as training data
    if 'ocean_proximity' in new_data.columns:
        new_data['ocean_proximity_encoded'] = label_encoder.transform(new_data['ocean_proximity'])

    feature_columns = ['longitude', 'latitude', 'housing_median_age', 'total_rooms',
                       'total_bedrooms', 'population', 'households', 'median_income',
                       'ocean_proximity_encoded']

    X_new = new_data[feature_columns]
    predictions = model.predict(X_new)

    return predictions


# Main execution
def main():
    """
    Main function to run the housing price prediction with Random Forest
    """
    try:
        # Load data
        print("Loading housing data...")
        data = load_data('housing.csv')  # Replace with your CSV file path

        # Preprocess data
        print("\nPreprocessing data...")
        X, y, label_encoder = preprocess_data(data)

        # Train model
        print("\nTraining Random Forest model...")
        model, X_train, X_test, y_train, y_test = train_random_forest(
            X, y, use_grid_search=False  # Set to True for hyperparameter tuning
        )

        # Evaluate model
        print("\nEvaluating model...")
        rmse, r2, mae = evaluate_model(model, X_test, y_test, X.columns.tolist())

        # Save model and preprocessing objects
        print("\nSaving model...")
        joblib.dump(model, 'random_forest_housing_model.pkl')
        joblib.dump(label_encoder, 'label_encoder.pkl')
        print("Model saved as 'random_forest_housing_model.pkl'")
        print("Label encoder saved as 'label_encoder.pkl'")

        # Print model insights
        print(f"\n=== Model Insights ===")
        print(f"Number of features: {X.shape[1]}")
        print(f"Training samples: {X_train.shape[0]}")
        print(f"Test samples: {X_test.shape[0]}")
        print(f"Model can explain {r2 * 100:.1f}% of price variance")

    except Exception as e:
        print(f"Error occurred: {e}")


if __name__ == "__main__":
    main()