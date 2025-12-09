"""
ML Model Predictor for IAD Risk Assessment
This module loads the trained ANN model and provides prediction functionality
"""

import os
import pickle
import json
import numpy as np
import pandas as pd
from typing import Dict, Any, Optional

# Model paths
MODEL_DIR = os.path.join(os.path.dirname(__file__), 'ml_model')
MODEL_PATH = os.path.join(MODEL_DIR, 'best_model_ann.pkl')
SCALER_PATH = os.path.join(MODEL_DIR, 'scaler.pkl')
LABEL_ENCODER_PATH = os.path.join(MODEL_DIR, 'label_encoder.pkl')
METADATA_PATH = os.path.join(MODEL_DIR, 'model_metadata.json')

# Global variables for loaded model components
_model = None
_scaler = None
_label_encoder = None
_model_metadata = None


def load_model_components():
    """Load model, scaler, label encoder, and metadata"""
    global _model, _scaler, _label_encoder, _model_metadata
    
    if _model is not None:
        return _model, _scaler, _label_encoder, _model_metadata
    
    try:
        # Load model
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")
        
        # Try joblib first (preferred for scikit-learn)
        try:
            import joblib
            _model = joblib.load(MODEL_PATH)
        except ImportError:
            # Fallback to pickle
            with open(MODEL_PATH, 'rb') as f:
                _model = pickle.load(f)
        
        # Load scaler
        if os.path.exists(SCALER_PATH):
            with open(SCALER_PATH, 'rb') as f:
                _scaler = pickle.load(f)
        else:
            raise FileNotFoundError(f"Scaler file not found: {SCALER_PATH}")
        
        # Load label encoder
        if os.path.exists(LABEL_ENCODER_PATH):
            with open(LABEL_ENCODER_PATH, 'rb') as f:
                _label_encoder = pickle.load(f)
        else:
            raise FileNotFoundError(f"Label encoder file not found: {LABEL_ENCODER_PATH}")
        
        # Load metadata
        if os.path.exists(METADATA_PATH):
            with open(METADATA_PATH, 'r') as f:
                _model_metadata = json.load(f)
        else:
            _model_metadata = {
                'model_type': 'ANN (MLPClassifier)',
                'input_features': 22,
                'output_classes': 4
            }
        
        return _model, _scaler, _label_encoder, _model_metadata
    
    except Exception as e:
        raise Exception(f"Error loading model components: {str(e)}")


def format_questionnaire_data(form_data: Dict[str, Any]) -> pd.DataFrame:
    """
    Format questionnaire form data into the format expected by the ML model
    
    Args:
        form_data: Dictionary containing form data with keys like 'q1', 'q2', etc.
    
    Returns:
        DataFrame with columns Q1-Q22
    """
    # Extract Q1-Q22 values
    features = {}
    for i in range(1, 23):
        key = f'q{i}'
        if key in form_data:
            try:
                features[f'Q{i}'] = int(form_data[key])
            except (ValueError, TypeError):
                features[f'Q{i}'] = 0
        else:
            features[f'Q{i}'] = 0
    
    # Create DataFrame with single row
    df = pd.DataFrame([features])
    
    # Ensure columns are in correct order (Q1, Q2, ..., Q22)
    column_order = [f'Q{i}' for i in range(1, 23)]
    df = df[column_order]
    
    return df


def predict_risk_level(formatted_data: pd.DataFrame) -> Dict[str, Any]:
    """
    Predict risk level using the trained ML model
    
    Args:
        formatted_data: DataFrame with Q1-Q22 columns
    
    Returns:
        Dictionary containing prediction results
    """
    try:
        # Load model components
        model, scaler, label_encoder, metadata = load_model_components()
        
        # Extract features (Q1-Q22)
        features = formatted_data.values
        
        # Scale features
        features_scaled = scaler.transform(features)
        
        # Make prediction
        prediction = model.predict(features_scaled)[0]
        probabilities = model.predict_proba(features_scaled)[0]
        
        # Get predicted class name
        predicted_class_idx = int(prediction)
        predicted_risk_level = label_encoder.inverse_transform([predicted_class_idx])[0]
        
        # Get confidence (probability of predicted class)
        confidence = float(probabilities[predicted_class_idx])
        
        # Get all class probabilities
        class_probabilities = {}
        for idx, class_name in enumerate(label_encoder.classes_):
            class_probabilities[class_name] = float(probabilities[idx])
        
        return {
            'success': True,
            'predicted_risk_level': predicted_risk_level,
            'confidence': confidence,
            'class_probabilities': class_probabilities,
            'model_info': {
                'model_name': metadata.get('model_type', 'ANN (MLPClassifier)'),
                'model_path': MODEL_PATH,
                'input_features': metadata.get('input_features', 22),
                'output_classes': metadata.get('output_classes', 4)
            }
        }
    
    except Exception as e:
        return {
            'success': False,
            'error': str(e),
            'predicted_risk_level': None,
            'confidence': 0.0
        }


def get_predictor():
    """Get predictor object with model information (for backward compatibility)"""
    class Predictor:
        def __init__(self):
            self.model, self.scaler, self.label_encoder, self.metadata = load_model_components()
        
        def get_model_info(self):
            """Get model information"""
            return {
                'model_type': self.metadata.get('model_type', 'ANN (MLPClassifier)'),
                'input_features': self.metadata.get('input_features', 22),
                'output_classes': self.metadata.get('output_classes', 4),
                'class_names': self.label_encoder.classes_.tolist() if hasattr(self.label_encoder, 'classes_') else []
            }
        
        def get_feature_importance(self):
            """Get feature importance (if available)"""
            # MLPClassifier doesn't have feature_importances_ like tree models
            # Return equal importance for all features as placeholder
            features = [f'Q{i}' for i in range(1, 23)]
            return {feat: 1.0 / len(features) for feat in features}
    
    return Predictor()


# For testing
if __name__ == "__main__":
    # Test with sample data
    test_data = {f'q{i}': 2 for i in range(1, 23)}
    formatted = format_questionnaire_data(test_data)
    result = predict_risk_level(formatted)
    print("Test Prediction Result:")
    print(json.dumps(result, indent=2))

