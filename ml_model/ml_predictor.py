"""
Machine Learning Prediction Utility
This module provides functions to load the trained model and make predictions.
"""

import joblib
import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple

class MLPredictor:
    def __init__(self, model_path='ml_model/artifacts/best_model.pkl', 
                 encoder_path='ml_model/artifacts/label_encoder.pkl', 
                 scaler_path='ml_model/artifacts/scaler.pkl',
                 feature_columns_path='ml_model/artifacts/feature_columns.pkl',
                 model_info_path='ml_model/artifacts/model_info.pkl'):
        """Initialize the ML predictor with saved model components"""
        self.model = None
        self.label_encoder = None
        self.scaler = None
        self.feature_columns = None
        self.model_info = None
        
        try:
            # Load all components
            self.model = joblib.load(model_path)
            self.label_encoder = joblib.load(encoder_path)
            self.scaler = joblib.load(scaler_path)
            self.feature_columns = joblib.load(feature_columns_path)
            self.model_info = joblib.load(model_info_path)
            
            print(f"ML Predictor initialized successfully!")
            print(f"Model: {self.model_info['best_model_name']}")
            print(f"Accuracy: {self.model_info['accuracy']:.4f}")
            print(f"Features: {len(self.feature_columns)}")
            print(f"Classes: {self.model_info['target_classes']}")
            
        except Exception as e:
            print(f"Error loading ML components: {str(e)}")
            raise e
    
    def preprocess_input(self, input_data: Dict[str, Any]) -> np.ndarray:
        """Preprocess input data for prediction"""
        try:
            # Ensure all expected feature columns are present
            processed_data = {}
            for feature in self.feature_columns:
                if feature in input_data:
                    processed_data[feature] = input_data[feature]
                else:
                    processed_data[feature] = 0  # Default value for missing features
            
            # Create DataFrame from processed data
            df = pd.DataFrame([processed_data])
            
            # Select only the features used in training
            feature_data = df[self.feature_columns].copy()
            
            # Handle missing values
            feature_data = feature_data.fillna(0)
            
            # Convert to numpy array
            features = feature_data.values
            
            return features
            
        except Exception as e:
            print(f"Error preprocessing input: {str(e)}")
            raise e
    
    def predict(self, input_data: Dict[str, Any]) -> Tuple[str, float, Dict[str, float]]:
        """
        Make prediction on input data
        
        Args:
            input_data: Dictionary containing questionnaire responses
            
        Returns:
            Tuple of (predicted_class, confidence, class_probabilities)
        """
        try:
            # Preprocess input
            features = self.preprocess_input(input_data)
            
            # Determine if model needs scaled features
            model_name = self.model_info['best_model_name']
            needs_scaling = model_name in ['Logistic Regression', 'SVM', 'K-Nearest Neighbors']
            
            if needs_scaling:
                features = self.scaler.transform(features)
            
            # Make prediction
            prediction = self.model.predict(features)[0]
            predicted_class = self.label_encoder.inverse_transform([prediction])[0]
            
            # Get prediction probabilities
            if hasattr(self.model, 'predict_proba'):
                probabilities = self.model.predict_proba(features)[0]
                confidence = np.max(probabilities)
                
                # Create class probability dictionary
                class_probabilities = {
                    self.label_encoder.classes_[i]: float(prob) 
                    for i, prob in enumerate(probabilities)
                }
            else:
                confidence = 1.0  # Fallback for models without probability
                class_probabilities = {predicted_class: 1.0}
            
            return predicted_class, confidence, class_probabilities
            
        except Exception as e:
            print(f"Error making prediction: {str(e)}")
            raise e
    
    def get_feature_importance(self) -> Dict[str, float]:
        """Get feature importance if available"""
        try:
            if hasattr(self.model, 'feature_importances_'):
                importances = self.model.feature_importances_
                feature_importance = {
                    feature: float(importance) 
                    for feature, importance in zip(self.feature_columns, importances)
                }
                return feature_importance
            else:
                return {}
        except Exception as e:
            print(f"Error getting feature importance: {str(e)}")
            return {}
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information"""
        return self.model_info.copy()

# Global predictor instance
_predictor = None

def get_predictor() -> MLPredictor:
    """Get or create global predictor instance"""
    global _predictor
    if _predictor is None:
        _predictor = MLPredictor()
    return _predictor

def predict_risk_level(input_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Convenience function to predict risk level from questionnaire data
    
    Args:
        input_data: Dictionary containing questionnaire responses
        
    Returns:
        Dictionary with prediction results
    """
    try:
        predictor = get_predictor()
        predicted_class, confidence, class_probabilities = predictor.predict(input_data)
        
        return {
            'success': True,
            'predicted_risk_level': predicted_class,
            'confidence': confidence,
            'class_probabilities': class_probabilities,
            'model_info': {
                'model_name': predictor.model_info['best_model_name'],
                'accuracy': predictor.model_info['accuracy']
            }
        }
        
    except Exception as e:
        return {
            'success': False,
            'error': str(e),
            'predicted_risk_level': 'Unknown',
            'confidence': 0.0
        }

def format_questionnaire_data(form_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Format form data to match the expected input format for ML model
    
    Args:
        form_data: Raw form data from Flask request
        
    Returns:
        Formatted data for ML prediction
    """
    try:
        # Define the scoring mapping (same as in app.py)
        scoring_map = {
            'Q1': 0,   # Age - no scoring
            'Q2': {'male': 0, 'female': 1, 'other': 2},  # Gender
            'Q3': {'8th': 0, '9th': 1, '10th': 2, 'other': 3},  # Education
            'Q4': 0,   # Occupation - no scoring
            'Q5': {'nuclear': 0, 'joint': 1, 'other': 2},  # Family type
            'Q6': {'low': 0, 'middle': 1, 'high': 2},  # SES
            'Q7': {'less_2': 1, '2_4': 2, '4_6': 3, 'more_6': 4},  # Online hours
            'Q8': {'social_media': 3, 'gaming': 4, 'streaming': 2, 'education': 1, 'other': 2},  # Primary activity
            'Q9': {'never': 1, 'rarely': 2, 'sometimes': 3, 'often': 4, 'always': 5},  # Overuse
            'Q10': {'yes': 4, 'no': 1},  # Neglect responsibilities
            'Q11': {'never': 1, 'rarely': 2, 'sometimes': 3, 'often': 4, 'always': 5},  # Restlessness
            'Q12': {'yes': 4, 'no': 1},  # Failed attempts
            'Q13': {'yes': 4, 'no': 1},  # Hiding usage
            'Q14': {'yes': 4, 'no': 1},  # Academic impact
            'Q15': {'yes': 4, 'no': 1},  # Social preferences
            'Q16': {'excellent': 1, 'very_good': 2, 'good': 3, 'fair': 4, 'poor': 5},  # Mental health
            'Q17': {'yes': 5, 'no': 1},  # Suicidal thoughts
            'Q18': {'yes': 5, 'no': 1},  # Suicide attempts
            'Q19': {'yes': 4, 'no': 1},  # Professional diagnosis
            'Q20': {'yes': 3, 'no': 1},  # Family complaints
            'Q21': {'yes': 4, 'no': 1},  # Relationship impact
            'Q22': {'yes': 4, 'no': 1}   # Social isolation
        }
        
        # Convert form data to the format expected by the model
        # Only include features that the model expects (exclude Q1 and Q4)
        formatted_data = {}
        
        # Map form field names to model feature names (excluding Q1 and Q4)
        for i in range(1, 23):  # Q1 to Q22
            q_key = f'q{i}'
            question_key = f'Q{i}'
            
            # Skip Q1 and Q4 as they are excluded from the model
            if question_key in ['Q1', 'Q4']:
                continue
                
            if q_key in form_data:
                value = form_data[q_key]
                
                # Convert to appropriate numeric score
                if question_key in scoring_map:
                    if isinstance(scoring_map[question_key], dict):
                        # Look up value in mapping
                        if value in scoring_map[question_key]:
                            formatted_data[question_key] = scoring_map[question_key][value]
                        else:
                            # Default to 0 if value not found
                            formatted_data[question_key] = 0
                    else:
                        # Direct score value
                        formatted_data[question_key] = scoring_map[question_key]
                else:
                    # Try to convert to int if it's a number
                    try:
                        formatted_data[question_key] = int(value)
                    except ValueError:
                        # If not a number, default to 0
                        formatted_data[question_key] = 0
            else:
                formatted_data[question_key] = 0
        
        return formatted_data
        
    except Exception as e:
        print(f"Error formatting questionnaire data: {str(e)}")
        raise e

if __name__ == "__main__":
    # Test the predictor
    print("Testing ML Predictor...")
    
    try:
        predictor = MLPredictor()
        
        # Test with sample data (all numeric values)
        sample_data = {
            'Q1': 25, 'Q2': 1, 'Q3': 2, 'Q4': 0, 'Q5': 0, 'Q6': 1,
            'Q7': 2, 'Q8': 3, 'Q9': 2, 'Q10': 1, 'Q11': 1, 'Q12': 1,
            'Q13': 1, 'Q14': 1, 'Q15': 1, 'Q16': 2, 'Q17': 1, 'Q18': 1,
            'Q19': 1, 'Q20': 1, 'Q21': 1, 'Q22': 1
        }
        
        prediction, confidence, probabilities = predictor.predict(sample_data)
        
        print(f"\nTest Prediction Results:")
        print(f"Predicted Risk Level: {prediction}")
        print(f"Confidence: {confidence:.4f}")
        print(f"Class Probabilities: {probabilities}")
        
        # Test feature importance
        importance = predictor.get_feature_importance()
        if importance:
            print(f"\nTop 5 Most Important Features:")
            sorted_features = sorted(importance.items(), key=lambda x: x[1], reverse=True)
            for feature, imp in sorted_features[:5]:
                print(f"  {feature}: {imp:.4f}")
        
    except Exception as e:
        print(f"Test failed: {str(e)}")
        print("Make sure to run ml_training.py first to create the model files.")
