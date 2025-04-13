import pandas as pd
import numpy as np
import pickle
import os
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, mean_squared_error, r2_score
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.inspection import permutation_importance  # Changed from sklearn.feature_selection

# Create models directory if it doesn't exist
os.makedirs('models', exist_ok=True)

print("Training Agricultural Models using scikit-learn...")

# Load datasets
print("\n1. Loading datasets...")
try:
    crop_recommendation = pd.read_csv('datasets/Crop_Recommendation.csv')
    print(f"   - Crop_Recommendation.csv loaded: {crop_recommendation.shape[0]} rows, {crop_recommendation.shape[1]} columns")
except Exception as e:
    print(f"   - Error loading Crop_Recommendation.csv: {e}")
    crop_recommendation = None
    
try:
    crop_data = pd.read_csv('datasets/crop_data.csv')
    print(f"   - crop_data.csv loaded: {crop_data.shape[0]} rows, {crop_data.shape[1]} columns")
except Exception as e:
    print(f"   - Error loading crop_data.csv: {e}")
    crop_data = None

# MODEL 1: Scikit-learn Crop Recommendation Model
print("\n2. Training scikit-learn Crop Recommendation Model...")
if crop_recommendation is not None:
    # Preprocessing
    crop_recommendation.dropna(inplace=True)
    
    # Prepare features and target
    X = crop_recommendation.drop('Crop', axis=1)
    y = crop_recommendation['Crop']
    
    # Encode the target (crop names)
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_encoded, test_size=0.2, random_state=42)
    
    # Build scikit-learn model
    print("   Building Random Forest for crop recommendation...")
    
    # Define model parameters
    param_grid = {
        'n_estimators': [100],
        'max_depth': [None],
        'min_samples_split': [2],
        'min_samples_leaf': [1]
    }
    
    # Create and train the model with cross-validation
    model = GridSearchCV(
        RandomForestClassifier(random_state=42),
        param_grid=param_grid,
        cv=3,
        n_jobs=-1,
        verbose=1
    )
    
    # Train model
    print("\n   Training scikit-learn model for crop recommendation...")
    model.fit(X_train, y_train)
    
    # Get best model
    best_model = model.best_estimator_
    
    # Evaluate model
    print("\n   Evaluating crop recommendation model...")
    accuracy = best_model.score(X_test, y_test)
    print(f"   Test accuracy: {accuracy:.4f}")
    
    # Get predictions for classification report
    y_pred = best_model.predict(X_test)
    
    # Convert numeric predictions back to crop names
    y_test_crop_names = label_encoder.inverse_transform(y_test)
    y_pred_crop_names = label_encoder.inverse_transform(y_pred)
    
    # Print classification report
    print("\n   Classification Report:")
    print(classification_report(y_test_crop_names, y_pred_crop_names))
    
    # Save the model, encoder and scaler
    print("\n   Saving scikit-learn crop recommendation model and artifacts...")
    
    # Save the model
    with open('models/crop_recommendation_sklearn_model.pkl', 'wb') as f:
        pickle.dump(best_model, f)
    
    # Save the scaler
    with open('models/crop_recommendation_scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)
    
    # Save the label encoder
    with open('models/crop_label_encoder.pkl', 'wb') as f:
        pickle.dump(label_encoder, f)
    
    # Save list of crop classes
    with open('models/crop_unique_values.pkl', 'wb') as f:
        pickle.dump(list(label_encoder.classes_), f)
    
    print("   Crop Recommendation Model saved successfully!")
else:
    print("   Skipping Crop Recommendation Model due to missing dataset.")

# MODEL 2: Scikit-learn Yield Prediction Model
print("\n3. Training scikit-learn Rice Yield Prediction Model...")
if crop_data is not None:
    # Preprocessing
    crop_data.dropna(subset=['RICE YIELD (Kg per ha)'], inplace=True)
    
    # Define target and features
    target = 'RICE YIELD (Kg per ha)'
    # Select relevant features (excluding other yield columns)
    numerical_features = [col for col in crop_data.columns if col != target and 'YIELD' not in col 
                        and crop_data[col].dtype != 'object']
    categorical_features = [col for col in crop_data.columns if col != target and 'YIELD' not in col 
                          and crop_data[col].dtype == 'object']
    
    # Prepare target variable
    y = crop_data[target].values
    
    # Handle numerical features
    X_numerical = crop_data[numerical_features].copy()
    
    # Handle categorical features
    categorical_encoders = {}
    X_categorical_encoded = pd.DataFrame()
    
    for col in categorical_features:
        le = LabelEncoder()
        encoded = le.fit_transform(crop_data[col].astype(str))
        X_categorical_encoded[col] = encoded
        categorical_encoders[col] = le
    
    # Scale numerical features
    numerical_scaler = StandardScaler()
    X_numerical_scaled = numerical_scaler.fit_transform(X_numerical)
    
    # Create DataFrames for scaled data
    X_numerical_df = pd.DataFrame(
        X_numerical_scaled, 
        columns=numerical_features,
        index=crop_data.index
    )
    
    # Combine features
    X_combined = pd.concat([X_numerical_df, X_categorical_encoded], axis=1)
    
    # Split the dataset
    X_train, X_test, y_train, y_test = train_test_split(X_combined, y, test_size=0.2, random_state=42)
    
    # Build scikit-learn model for regression
    print("   Building Random Forest for rice yield prediction...")
    
    # Define model parameters
    param_grid = {
        'n_estimators': [100],
        'max_depth': [None],
        'min_samples_split': [2],
        'min_samples_leaf': [1]
    }
    
    # Create and train the model with cross-validation
    yield_model = GridSearchCV(
        RandomForestRegressor(random_state=42),
        param_grid=param_grid,
        cv=3,
        n_jobs=-1,
        verbose=1
    )
    
    # Train model
    print("\n   Training scikit-learn model for rice yield prediction...")
    yield_model.fit(X_train, y_train)
    
    # Get best model
    best_yield_model = yield_model.best_estimator_
    
    # Evaluate model
    print("\n   Evaluating rice yield prediction model...")
    y_pred = best_yield_model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)
    
    print(f"   Mean Squared Error: {mse:.2f}")
    print(f"   Root Mean Squared Error: {rmse:.2f}")
    print(f"   R² Score: {r2:.4f}")
    
    # Feature importance analysis using scikit-learn's permutation importance
    print("\n   Calculating feature importances...")
    result = permutation_importance(
        best_yield_model, X_test, y_test, 
        n_repeats=10, 
        random_state=42
    )
    
    # Convert to dictionary format
    feature_importance = {}
    for i, col in enumerate(X_test.columns):
        feature_importance[col] = result.importances_mean[i]
    
    # Sort features by importance
    feature_importance = {k: v for k, v in sorted(
        feature_importance.items(), 
        key=lambda item: item[1], 
        reverse=True
    )}
    
    # Print top features
    print("\n   Top 10 most important features for predicting rice yield:")
    for i, (feature, importance) in enumerate(list(feature_importance.items())[:10]):
        print(f"   {i+1}. {feature}: {importance:.4f}")
    
    # Save the models and artifacts
    print("\n   Saving scikit-learn yield prediction model and artifacts...")
    
    # Save the model
    with open('models/rice_yield_sklearn_model.pkl', 'wb') as f:
        pickle.dump(best_yield_model, f)
    
    # Save feature lists
    with open('models/yield_numerical_features.pkl', 'wb') as f:
        pickle.dump(numerical_features, f)
    
    with open('models/yield_categorical_features.pkl', 'wb') as f:
        pickle.dump(categorical_features, f)
    
    # Save scalers and encoders
    with open('models/yield_numerical_scaler.pkl', 'wb') as f:
        pickle.dump(numerical_scaler, f)
    
    with open('models/yield_categorical_encoders.pkl', 'wb') as f:
        pickle.dump(categorical_encoders, f)
    
    # Save feature importance for later use in explanations
    with open('models/yield_feature_importance.pkl', 'wb') as f:
        pickle.dump(feature_importance, f)
    
    print("   Rice Yield Prediction Model saved successfully!")
else:
    print("   Skipping Rice Yield Prediction Model due to missing dataset.")

# Save metadata about the datasets for question answering
print("\n4. Saving dataset metadata for question answering...")
metadata = {
    "crop_recommendation": {
        "available": crop_recommendation is not None,
        "columns": crop_recommendation.columns.tolist() if crop_recommendation is not None else [],
        "unique_crops": crop_recommendation['Crop'].unique().tolist() if crop_recommendation is not None else [],
        "model_type": "sklearn_random_forest"
    },
    "crop_data": {
        "available": crop_data is not None,
        "columns": crop_data.columns.tolist() if crop_data is not None else [],
        "states": crop_data['State Name'].unique().tolist() if crop_data is not None else [],
        "years": sorted(crop_data['Year'].unique().tolist()) if crop_data is not None else [],
        "model_type": "sklearn_random_forest"
    }
}

with open('models/dataset_metadata.pkl', 'wb') as f:
    pickle.dump(metadata, f)
print("   Dataset metadata saved for question-answering capabilities!")

print("\nTraining complete! All scikit-learn models and metadata saved to the 'models' directory.")