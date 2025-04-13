import pandas as pd
import numpy as np
import pickle
import os
from flask import Flask, request, jsonify
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import re
from flask_cors import CORS

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

print("Starting agricultural models API server...")

# Download NLTK resources
try:
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
except:
    print("NLTK resources might not be available. Basic functionality will still work.")

# Load models and metadata
def load_models():
    models = {}
    
    # Load dataset metadata
    try:
        with open('models/dataset_metadata.pkl', 'rb') as f:
            models['metadata'] = pickle.load(f)
    except:
        print("Dataset metadata not found. Please run train.py first.")
        models['metadata'] = None
    
    # Load crop recommendation model and artifacts
    try:
        # Load scikit-learn model
        with open('models/crop_recommendation_sklearn_model.pkl', 'rb') as f:
            models['crop_model'] = pickle.load(f)
        
        with open('models/crop_recommendation_scaler.pkl', 'rb') as f:
            models['crop_scaler'] = pickle.load(f)
        with open('models/crop_label_encoder.pkl', 'rb') as f:
            models['crop_label_encoder'] = pickle.load(f)
        with open('models/crop_unique_values.pkl', 'rb') as f:
            models['crop_list'] = pickle.load(f)
            
        print("Crop recommendation scikit-learn model loaded successfully.")
    except Exception as e:
        print(f"Crop recommendation model not found or incomplete: {str(e)}")
        models['crop_model'] = None
    
    # Load yield prediction model
    try:
        # Load scikit-learn yield model
        with open('models/rice_yield_sklearn_model.pkl', 'rb') as f:
            models['yield_model'] = pickle.load(f)
        
        # Load feature lists
        with open('models/yield_numerical_features.pkl', 'rb') as f:
            models['yield_numerical_features'] = pickle.load(f)
        
        with open('models/yield_categorical_features.pkl', 'rb') as f:
            models['yield_categorical_features'] = pickle.load(f)
        
        # Load scalers and encoders
        with open('models/yield_numerical_scaler.pkl', 'rb') as f:
            models['yield_numerical_scaler'] = pickle.load(f)
        
        with open('models/yield_categorical_encoders.pkl', 'rb') as f:
            models['yield_categorical_encoders'] = pickle.load(f)
        
        # Load feature importance
        with open('models/yield_feature_importance.pkl', 'rb') as f:
            models['yield_importance'] = pickle.load(f)
            
        print("Rice yield scikit-learn model loaded successfully.")
    except Exception as e:
        print(f"Yield prediction model not found or incomplete: {str(e)}")
        models['yield_model'] = None
    
    return models

# Load the models at startup
models = load_models()

# Helper functions for question-answering
def extract_keywords(question):
    """Extract keywords from a question"""
    # Tokenize and convert to lowercase
    tokens = word_tokenize(question.lower())
    
    # Remove stopwords and punctuation
    stop_words = set(stopwords.words('english'))
    keywords = [word for word in tokens if word.isalnum() and word not in stop_words]
    
    return keywords

def classify_question_type(question, keywords):
    """Determine what type of agricultural question is being asked"""
    # Convert to lowercase for easier matching
    q = question.lower()
    
    # Question type classification
    if any(word in q for word in ['recommend', 'suitable', 'best', 'grow', 'plant', 'cultivate']):
        if any(word in q for word in ['crop', 'plant']):
            return 'crop_recommendation'
    
    if any(word in q for word in ['yield', 'produce', 'production', 'harvest']):
        if any(word in q for word in ['rice', 'predict', 'estimate', 'forecast']):
            return 'yield_prediction'
        
    if any(word in q for word in ['state', 'district', 'region', 'area']):
        return 'regional_info'
        
    if any(word in q for word in ['fertilizer', 'nitrogen', 'phosphorus', 'potassium', 'npk']):
        return 'soil_info'
    
    # Default to general info if nothing specific is detected
    return 'general_info'

def extract_soil_params(question):
    """Extract soil parameters from a question for crop recommendation"""
    params = {}
    
    # Regular expressions to find soil parameter values
    patterns = {
        'Nitrogen': r'nitrogen[:\s]+(\d+\.?\d*)|n[:\s]+(\d+\.?\d*)',
        'Phosphorus': r'phosphorus[:\s]+(\d+\.?\d*)|p[:\s]+(\d+\.?\d*)',
        'Potassium': r'potassium[:\s]+(\d+\.?\d*)|k[:\s]+(\d+\.?\d*)',
        'Temperature': r'temperature[:\s]+(\d+\.?\d*)',
        'Humidity': r'humidity[:\s]+(\d+\.?\d*)',
        'pH_Value': r'ph[:\s]+(\d+\.?\d*)',
        'Rainfall': r'rainfall[:\s]+(\d+\.?\d*)'
    }
    
    for param, pattern in patterns.items():
        match = re.search(pattern, question.lower())
        if match:
            # Extract the first non-None group
            value = next((g for g in match.groups() if g is not None), None)
            if value:
                params[param] = float(value)
    
    return params

def extract_region_info(question):
    """Extract state and district information from the question"""
    region_info = {}
    
    # Extract state information if available
    if models['metadata'] and models['metadata']['crop_data']['available']:
        states = models['metadata']['crop_data']['states']
        for state in states:
            if state.lower() in question.lower():
                region_info['state'] = state
                break
    
    # We could also extract district info here if needed
    
    return region_info

def answer_crop_recommendation(question):
    """Generate an answer for crop recommendation questions using scikit-learn model"""
    if models['crop_model'] is None:
        return {
            "success": False,
            "message": "Crop recommendation model is not available."
        }
    
    # Extract soil parameters
    params = extract_soil_params(question)
    
    # If we have all required parameters, make a prediction
    required_params = ['Nitrogen', 'Phosphorus', 'Potassium', 
                     'Temperature', 'Humidity', 'pH_Value', 'Rainfall']
    
    if all(param in params for param in required_params):
        # Prepare input for model prediction
        X = np.array([[
            params['Nitrogen'], 
            params['Phosphorus'], 
            params['Potassium'],
            params['Temperature'], 
            params['Humidity'], 
            params['pH_Value'], 
            params['Rainfall']
        ]])
        
        # Scale the input
        X_scaled = models['crop_scaler'].transform(X)
        
        # Predict with scikit-learn model
        y_pred = models['crop_model'].predict_proba(X_scaled)[0]
        
        # Get indices of top predictions
        top_indices = np.argsort(y_pred)[-5:][::-1]
        
        # Map indices to crop names using label encoder
        top_crops = models['crop_label_encoder'].inverse_transform(top_indices)
        top_probs = y_pred[top_indices]
        
        # Return structured data
        recommendations = [
            {"crop": name, "probability": float(prob)} 
            for name, prob in zip(top_crops, top_probs)
        ]
        
        return {
            "success": True,
            "recommendations": recommendations,
            "params": params
        }
    else:
        # If we're missing parameters, return the missing ones
        missing = [param for param in required_params if param not in params]
        return {
            "success": False,
            "message": "Missing required parameters",
            "missing_params": missing,
            "provided_params": params
        }

def answer_yield_prediction(question):
    """Generate an answer for rice yield prediction questions using scikit-learn model"""
    if models['yield_model'] is None:
        return {
            "success": False,
            "message": "Yield prediction model is not available."
        }
    
    # Extract region information
    region_info = extract_region_info(question)
    
    if 'state' in region_info:
        # Get important features for rice yield from the saved feature importance
        top_features = dict(list(models['yield_importance'].items())[:5])
        
        # Return information about rice yield in the region
        return {
            "success": True,
            "state": region_info['state'],
            "important_factors": {
                feature: {"importance": float(importance)}
                for feature, importance in top_features.items()
            }
        }
    else:
        return {
            "success": False,
            "message": "Region information required",
            "available_states": models['metadata']['crop_data']['states'] if models['metadata'] else []
        }

def get_general_info():
    """Return general information about the agricultural models"""
    info = {
        "success": True,
        "models": {
            "crop_recommendation": models['crop_model'] is not None,
            "yield_prediction": models['yield_model'] is not None
        }
    }
    
    if models['metadata'] and models['metadata']['crop_recommendation']['available']:
        info["available_crops"] = models['metadata']['crop_recommendation']['unique_crops']
        
    if models['metadata'] and models['metadata']['crop_data']['available']:
        info["available_states"] = models['metadata']['crop_data']['states']
        info["available_years"] = models['metadata']['crop_data']['years']
        
    return info

# API Endpoints

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy",
        "models_loaded": {
            "crop_recommendation": models['crop_model'] is not None,
            "yield_prediction": models['yield_model'] is not None,
            "metadata": models['metadata'] is not None
        }
    })

@app.route('/api/info', methods=['GET'])
def get_api_info():
    """Get information about the available models and data"""
    return jsonify(get_general_info())

@app.route('/api/ask', methods=['POST'])
def ask_question():
    """Process a natural language question about agriculture"""
    data = request.get_json()
    if not data or 'question' not in data:
        return jsonify({"success": False, "message": "No question provided"}), 400
    
    question = data['question']
    
    # Extract keywords and classify the question
    keywords = extract_keywords(question)
    question_type = classify_question_type(question, keywords)
    
    # Generate answer based on question type
    if question_type == 'crop_recommendation':
        answer = answer_crop_recommendation(question)
    elif question_type == 'yield_prediction':
        answer = answer_yield_prediction(question)
    elif question_type == 'regional_info':
        # For now, handle regional info questions as yield prediction
        answer = answer_yield_prediction(question)
    elif question_type == 'soil_info':
        # For now, handle soil info questions as crop recommendation
        answer = answer_crop_recommendation(question)
    else:
        # General information
        answer = get_general_info()
    
    # Include the question type in the response
    answer["question_type"] = question_type
    
    return jsonify(answer)

@app.route('/api/recommend-crop', methods=['POST'])
def recommend_crop():
    """Direct crop recommendation based on soil parameters using scikit-learn model"""
    data = request.get_json()
    if not data:
        return jsonify({"success": False, "message": "No data provided"}), 400
    
    required_params = ['Nitrogen', 'Phosphorus', 'Potassium', 
                     'Temperature', 'Humidity', 'pH_Value', 'Rainfall']
    
    # Check if all required parameters are provided
    if not all(param in data for param in required_params):
        missing = [param for param in required_params if param not in data]
        return jsonify({
            "success": False,
            "message": "Missing required parameters",
            "missing_params": missing
        }), 400
    
    if models['crop_model'] is None:
        return jsonify({
            "success": False,
            "message": "Crop recommendation model is not available."
        }), 503
    
    # Prepare input for model prediction
    try:
        X = np.array([[
            float(data['Nitrogen']), 
            float(data['Phosphorus']), 
            float(data['Potassium']),
            float(data['Temperature']), 
            float(data['Humidity']), 
            float(data['pH_Value']), 
            float(data['Rainfall'])
        ]])
        
        # Scale the input
        X_scaled = models['crop_scaler'].transform(X)
        
        # Predict with scikit-learn model
        y_pred_prob = models['crop_model'].predict_proba(X_scaled)[0]
        
        # Get indices of top predictions
        top_indices = np.argsort(y_pred_prob)[-5:][::-1]
        
        # Map indices to crop names using label encoder
        top_crops = models['crop_label_encoder'].inverse_transform(top_indices)
        top_probs = y_pred_prob[top_indices]
        
        # Return structured data
        recommendations = [
            {"crop": name, "probability": float(prob)} 
            for name, prob in zip(top_crops, top_probs)
        ]
        
        return jsonify({
            "success": True,
            "recommendations": recommendations,
            "params": {param: float(data[param]) for param in required_params}
        })
    except Exception as e:
        return jsonify({
            "success": False,
            "message": f"Error processing request: {str(e)}"
        }), 500

@app.route('/api/predict-yield', methods=['POST'])
def predict_yield():
    """Direct yield prediction based on region using scikit-learn model"""
    data = request.get_json()
    if not data or 'state' not in data:
        return jsonify({
            "success": False, 
            "message": "State information required",
            "available_states": models['metadata']['crop_data']['states'] if models['metadata'] else []
        }), 400
    
    if models['yield_model'] is None:
        return jsonify({
            "success": False,
            "message": "Yield prediction model is not available."
        }), 503
    
    state = data['state']
    
    # Check if the state is valid
    if models['metadata'] and state not in models['metadata']['crop_data']['states']:
        return jsonify({
            "success": False,
            "message": f"State '{state}' not found in the dataset",
            "available_states": models['metadata']['crop_data']['states']
        }), 400
    
    # Get important features for rice yield
    top_features = dict(list(models['yield_importance'].items())[:5])
    
    # Return information about rice yield in the region
    return jsonify({
        "success": True,
        "state": state,
        "important_factors": {
            feature: {"importance": float(importance)}
            for feature, importance in top_features.items()
        }
    })

@app.route('/api/combined-analysis', methods=['POST'])
def combined_analysis():
    """Process multiple agricultural queries in a single request"""
    data = request.get_json()
    if not data:
        return jsonify({"success": False, "message": "No data provided"}), 400
    
    response = {"success": True, "results": {}}
    
    # Process crop recommendation if soil data is provided
    if "soil_data" in data:
        soil_data = data["soil_data"]
        required_params = ['Nitrogen', 'Phosphorus', 'Potassium', 
                         'Temperature', 'Humidity', 'pH_Value', 'Rainfall']
        
        if all(param in soil_data for param in required_params) and models['crop_model'] is not None:
            try:
                X = np.array([[
                    float(soil_data['Nitrogen']), 
                    float(soil_data['Phosphorus']), 
                    float(soil_data['Potassium']),
                    float(soil_data['Temperature']), 
                    float(soil_data['Humidity']), 
                    float(soil_data['pH_Value']), 
                    float(soil_data['Rainfall'])
                ]])
                
                X_scaled = models['crop_scaler'].transform(X)
                y_pred_prob = models['crop_model'].predict_proba(X_scaled)[0]
                
                # Get indices of top predictions
                top_indices = np.argsort(y_pred_prob)[-5:][::-1]
                
                # Map indices to crop names
                top_crops = models['crop_label_encoder'].inverse_transform(top_indices)
                top_probs = y_pred_prob[top_indices]
                
                recommendations = [
                    {"crop": name, "probability": float(prob)} 
                    for name, prob in zip(top_crops, top_probs)
                ]
                
                response["results"]["crop_recommendation"] = {
                    "success": True,
                    "recommendations": recommendations,
                    "params": {param: float(soil_data[param]) for param in required_params}
                }
            except Exception as e:
                response["results"]["crop_recommendation"] = {
                    "success": False,
                    "message": f"Error processing crop recommendation: {str(e)}"
                }
        else:
            missing = [param for param in required_params if param not in soil_data]
            response["results"]["crop_recommendation"] = {
                "success": False,
                "message": "Missing required soil parameters",
                "missing_params": missing if missing else []
            }
    
    # Process yield prediction if region data is provided
    if "region_data" in data:
        region_data = data["region_data"]
        if "state" in region_data and models['yield_model'] is not None:
            state = region_data["state"]
            
            # Check if the state is valid
            if models['metadata'] and state in models['metadata']['crop_data']['states']:
                # Get important features for rice yield
                top_features = dict(list(models['yield_importance'].items())[:5])
                
                response["results"]["yield_prediction"] = {
                    "success": True,
                    "state": state,
                    "important_factors": {
                        feature: {"importance": float(importance)}
                        for feature, importance in top_features.items()
                    }
                }
            else:
                response["results"]["yield_prediction"] = {
                    "success": False,
                    "message": f"State '{state}' not found in the dataset",
                    "available_states": models['metadata']['crop_data']['states'] if models['metadata'] else []
                }
        else:
            response["results"]["yield_prediction"] = {
                "success": False,
                "message": "State information required for yield prediction"
            }
    
    # Process natural language questions if provided
    if "questions" in data and isinstance(data["questions"], list):
        questions = data["questions"]
        qa_results = []
        
        for question in questions:
            keywords = extract_keywords(question)
            question_type = classify_question_type(question, keywords)
            
            # Generate answer based on question type
            if question_type == 'crop_recommendation':
                answer = answer_crop_recommendation(question)
            elif question_type in ['yield_prediction', 'regional_info']:
                answer = answer_yield_prediction(question)
            elif question_type == 'soil_info':
                answer = answer_crop_recommendation(question)
            else:
                answer = get_general_info()
            
            # Add question type
            answer["question_type"] = question_type
            answer["question"] = question
            
            qa_results.append(answer)
        
        response["results"]["qa_responses"] = qa_results
    
    return jsonify(response)

@app.route('/api/model-info', methods=['GET'])
def get_model_info():
    """Get detailed information about the models"""
    if models['crop_model'] is None or models['yield_model'] is None:
        return jsonify({
            "success": False,
            "message": "One or more models are not available."
        }), 503
    
    # Get crop model information
    crop_model = models['crop_model']
    crop_model_info = {
        "type": "scikit-learn RandomForestClassifier",
        "n_estimators": crop_model.n_estimators,
        "max_depth": crop_model.max_depth if crop_model.max_depth is not None else "None (unlimited)",
        "min_samples_split": crop_model.min_samples_split,
        "min_samples_leaf": crop_model.min_samples_leaf,
        "feature_importances": {
            f"feature_{i}": float(importance)
            for i, importance in enumerate(crop_model.feature_importances_)
        }
    }
    
    # Get yield model information
    yield_model = models['yield_model']
    yield_model_info = {
        "type": "scikit-learn RandomForestRegressor",
        "n_estimators": yield_model.n_estimators,
        "max_depth": yield_model.max_depth if yield_model.max_depth is not None else "None (unlimited)",
        "min_samples_split": yield_model.min_samples_split,
        "min_samples_leaf": yield_model.min_samples_leaf,
        "feature_importances": {
            f"feature_{i}": float(importance)
            for i, importance in enumerate(yield_model.feature_importances_)
        }
    }
    
    return jsonify({
        "success": True,
        "crop_recommendation_model": crop_model_info,
        "yield_prediction_model": yield_model_info,
        "model_framework": "scikit-learn"
    })

if __name__ == '__main__':
    # Check if we have any models
    if models['metadata'] is None:
        print("No models found. Please run train.py first to train the models.")
        exit(1)
    
    # Run the Flask app
    print("\n=== Agricultural API Backend ===")
    print("Available endpoints:")
    print("  - GET  /api/health      : Check API health and model status")
    print("  - GET  /api/info        : Get information about available models and data")
    print("  - POST /api/ask         : Process natural language questions")
    print("  - POST /api/recommend-crop : Direct crop recommendation endpoint")
    print("  - POST /api/predict-yield : Rice yield prediction endpoint")
    print("  - POST /api/combined-analysis : Combined analysis endpoint")
    
    app.run(host='0.0.0.0', port=5010, debug=True)