"""
Machine Learning Model Training for Digital Media & Mental Health Assessment
This script trains multiple ML models on questionnaire data and evaluates their performance.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix, 
                           f1_score, roc_auc_score, roc_curve, precision_recall_curve)
import joblib
import warnings
warnings.filterwarnings('ignore')

class MLModelTrainer:
    def __init__(self, csv_file='questionnaire_data.csv'):
        """Initialize the ML trainer with data file"""
        self.csv_file = csv_file
        self.models = {}
        self.results = {}
        self.best_model = None
        self.best_model_name = None
        self.label_encoder = LabelEncoder()
        self.scaler = StandardScaler()
        
    def load_and_preprocess_data(self):
        """Load and preprocess the questionnaire data"""
        print("Loading and preprocessing data...")
        
        # Load data
        self.df = pd.read_csv(self.csv_file)
        print(f"Loaded {len(self.df)} records with {len(self.df.columns)} columns")
        
        # Display basic info
        print(f"\nDataset Info:")
        print(f"Shape: {self.df.shape}")
        print(f"Risk Level Distribution:")
        print(self.df['risk_level'].value_counts())
        
        # Exclude specified columns
        exclude_columns = ['Q1', 'Q4', 'total_score', 'record_id']
        feature_columns = [col for col in self.df.columns if col not in exclude_columns and col != 'risk_level']
        
        print(f"\nExcluded columns: {exclude_columns}")
        print(f"Feature columns ({len(feature_columns)}): {feature_columns}")
        
        # Prepare features and target
        self.X = self.df[feature_columns].copy()
        self.y = self.df['risk_level'].copy()
        
        # Handle missing values
        print(f"\nMissing values in features: {self.X.isnull().sum().sum()}")
        if self.X.isnull().sum().sum() > 0:
            self.X = self.X.fillna(self.X.median())
            print("Filled missing values with median")
        
        # Encode categorical target variable
        self.y_encoded = self.label_encoder.fit_transform(self.y)
        print(f"\nTarget classes: {self.label_encoder.classes_}")
        
        # Check if any features need encoding (should be all numeric based on our data generator)
        print(f"Feature data types:\n{self.X.dtypes.value_counts()}")
        
        # Split the data
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y_encoded, test_size=0.2, random_state=42, stratify=self.y_encoded
        )
        
        # Scale features
        self.X_train_scaled = self.scaler.fit_transform(self.X_train)
        self.X_test_scaled = self.scaler.transform(self.X_test)
        
        print(f"\nData split:")
        print(f"Training set: {self.X_train.shape[0]} samples")
        print(f"Test set: {self.X_test.shape[0]} samples")
        
        return self.X_train, self.X_test, self.y_train, self.y_test
    
    def train_models(self):
        """Train multiple ML models"""
        print("\nTraining ML models...")
        
        # Define models
        self.models = {
            'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
            'Random Forest': RandomForestClassifier(random_state=42, n_estimators=100),
            'Gradient Boosting': GradientBoostingClassifier(random_state=42),
            'SVM': SVC(random_state=42, probability=True),
            'K-Nearest Neighbors': KNeighborsClassifier(n_neighbors=5)
        }
        
        # Train each model
        for name, model in self.models.items():
            print(f"\nTraining {name}...")
            
            # Use scaled data for models that benefit from it
            if name in ['Logistic Regression', 'SVM', 'K-Nearest Neighbors']:
                model.fit(self.X_train_scaled, self.y_train)
                y_pred = model.predict(self.X_test_scaled)
                y_pred_proba = model.predict_proba(self.X_test_scaled) if hasattr(model, 'predict_proba') else None
            else:
                model.fit(self.X_train, self.y_train)
                y_pred = model.predict(self.X_test)
                y_pred_proba = model.predict_proba(self.X_test) if hasattr(model, 'predict_proba') else None
            
            # Calculate metrics
            accuracy = accuracy_score(self.y_test, y_pred)
            f1 = f1_score(self.y_test, y_pred, average='weighted')
            
            # Store results
            self.results[name] = {
                'model': model,
                'accuracy': accuracy,
                'f1_score': f1,
                'predictions': y_pred,
                'probabilities': y_pred_proba
            }
            
            print(f"  Accuracy: {accuracy:.4f}")
            print(f"  F1-Score: {f1:.4f}")
        
        # Find best model
        best_accuracy = max(self.results.items(), key=lambda x: x[1]['accuracy'])
        self.best_model_name = best_accuracy[0]
        self.best_model = best_accuracy[1]['model']
        
        print(f"\nBest Model: {self.best_model_name}")
        print(f"Best Accuracy: {best_accuracy[1]['accuracy']:.4f}")
        
        return self.results
    
    def create_visualizations(self):
        """Create comprehensive visualizations for model comparison"""
        print("\nCreating visualizations...")
        
        # Set style
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
        
        # 1. Model Performance Comparison
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Model Performance Comparison', fontsize=16, fontweight='bold')
        
        # Accuracy comparison
        models = list(self.results.keys())
        accuracies = [self.results[model]['accuracy'] for model in models]
        f1_scores = [self.results[model]['f1_score'] for model in models]
        
        axes[0, 0].bar(models, accuracies, color='skyblue', alpha=0.7)
        axes[0, 0].set_title('Accuracy Comparison')
        axes[0, 0].set_ylabel('Accuracy')
        axes[0, 0].tick_params(axis='x', rotation=45)
        axes[0, 0].grid(True, alpha=0.3)
        
        # F1-Score comparison
        axes[0, 1].bar(models, f1_scores, color='lightcoral', alpha=0.7)
        axes[0, 1].set_title('F1-Score Comparison')
        axes[0, 1].set_ylabel('F1-Score')
        axes[0, 1].tick_params(axis='x', rotation=45)
        axes[0, 1].grid(True, alpha=0.3)
        
        # Combined metrics
        x = np.arange(len(models))
        width = 0.35
        axes[1, 0].bar(x - width/2, accuracies, width, label='Accuracy', alpha=0.7)
        axes[1, 0].bar(x + width/2, f1_scores, width, label='F1-Score', alpha=0.7)
        axes[1, 0].set_title('Accuracy vs F1-Score')
        axes[1, 0].set_ylabel('Score')
        axes[1, 0].set_xticks(x)
        axes[1, 0].set_xticklabels(models, rotation=45)
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # Best model highlight
        best_idx = models.index(self.best_model_name)
        axes[1, 1].bar(models, accuracies, color=['gold' if i == best_idx else 'lightgray' for i in range(len(models))])
        axes[1, 1].set_title(f'Best Model: {self.best_model_name}')
        axes[1, 1].set_ylabel('Accuracy')
        axes[1, 1].tick_params(axis='x', rotation=45)
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save to visualizations directory
        import os
        visualizations_dir = os.path.join('ml_model', 'visualizations')
        os.makedirs(visualizations_dir, exist_ok=True)
        
        plt.savefig(os.path.join(visualizations_dir, 'model_performance_comparison.png'), dpi=300, bbox_inches='tight')
        plt.show()
        
        # 2. Confusion Matrices
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Confusion Matrices for All Models', fontsize=16, fontweight='bold')
        axes = axes.flatten()
        
        for i, (name, result) in enumerate(self.results.items()):
            cm = confusion_matrix(self.y_test, result['predictions'])
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[i],
                       xticklabels=self.label_encoder.classes_,
                       yticklabels=self.label_encoder.classes_)
            axes[i].set_title(f'{name}\nAccuracy: {result["accuracy"]:.3f}')
            axes[i].set_xlabel('Predicted')
            axes[i].set_ylabel('Actual')
        
        # Remove empty subplot
        axes[-1].remove()
        plt.tight_layout()
        plt.savefig(os.path.join(visualizations_dir, 'confusion_matrices.png'), dpi=300, bbox_inches='tight')
        plt.show()
        
        # 3. ROC Curves (for binary classification or multi-class)
        plt.figure(figsize=(12, 8))
        
        for name, result in self.results.items():
            if result['probabilities'] is not None:
                # For multi-class, use macro average
                try:
                    auc = roc_auc_score(self.y_test, result['probabilities'], multi_class='ovr', average='macro')
                    fpr, tpr, _ = roc_curve(self.y_test, result['probabilities'][:, 1], pos_label=1)
                    plt.plot(fpr, tpr, label=f'{name} (AUC = {auc:.3f})', linewidth=2)
                except:
                    # Fallback for multi-class
                    auc = roc_auc_score(self.y_test, result['probabilities'], multi_class='ovr', average='macro')
                    plt.plot([0, 1], [0, 1], 'k--', alpha=0.5)
                    plt.text(0.5, 0.1, f'{name}: AUC = {auc:.3f}', fontsize=10)
        
        plt.plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Random')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curves Comparison')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(visualizations_dir, 'roc_curves.png'), dpi=300, bbox_inches='tight')
        plt.show()
        
        # 4. Feature Importance (for tree-based models)
        tree_models = ['Random Forest', 'Gradient Boosting']
        if any(model in self.results for model in tree_models):
            fig, axes = plt.subplots(1, len([m for m in tree_models if m in self.results]), figsize=(15, 6))
            if len([m for m in tree_models if m in self.results]) == 1:
                axes = [axes]
            
            fig.suptitle('Feature Importance Analysis', fontsize=16, fontweight='bold')
            
            for i, model_name in enumerate([m for m in tree_models if m in self.results]):
                if hasattr(self.results[model_name]['model'], 'feature_importances_'):
                    importances = self.results[model_name]['model'].feature_importances_
                    feature_names = self.X.columns
                    
                    # Sort features by importance
                    indices = np.argsort(importances)[::-1][:10]  # Top 10 features
                    
                    axes[i].bar(range(len(indices)), importances[indices])
                    axes[i].set_title(f'{model_name} - Top 10 Features')
                    axes[i].set_xlabel('Features')
                    axes[i].set_ylabel('Importance')
                    axes[i].set_xticks(range(len(indices)))
                    axes[i].set_xticklabels([feature_names[j] for j in indices], rotation=45, ha='right')
                    axes[i].grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(os.path.join(visualizations_dir, 'feature_importance.png'), dpi=300, bbox_inches='tight')
            plt.show()
        
        print("All visualizations saved successfully!")
    
    def save_best_model(self):
        """Save the best model and preprocessing objects"""
        print(f"\nSaving best model: {self.best_model_name}")
        
        # Create organized folder structure
        import os
        model_dir = 'ml_model'
        artifacts_dir = os.path.join(model_dir, 'artifacts')
        reports_dir = os.path.join(model_dir, 'reports')
        visualizations_dir = os.path.join(model_dir, 'visualizations')
        
        # Create directories if they don't exist
        os.makedirs(artifacts_dir, exist_ok=True)
        os.makedirs(reports_dir, exist_ok=True)
        os.makedirs(visualizations_dir, exist_ok=True)
        
        # Save model artifacts
        joblib.dump(self.best_model, os.path.join(artifacts_dir, 'best_model.pkl'))
        joblib.dump(self.label_encoder, os.path.join(artifacts_dir, 'label_encoder.pkl'))
        joblib.dump(self.scaler, os.path.join(artifacts_dir, 'scaler.pkl'))
        joblib.dump(self.X.columns.tolist(), os.path.join(artifacts_dir, 'feature_columns.pkl'))
        
        # Save model metadata
        model_info = {
            'best_model_name': self.best_model_name,
            'accuracy': self.results[self.best_model_name]['accuracy'],
            'f1_score': self.results[self.best_model_name]['f1_score'],
            'feature_columns': self.X.columns.tolist(),
            'target_classes': self.label_encoder.classes_.tolist()
        }
        joblib.dump(model_info, os.path.join(artifacts_dir, 'model_info.pkl'))
        
        print("Model and preprocessing objects saved successfully!")
        print(f"Artifacts saved to: {artifacts_dir}/")
        return model_info
    
    def generate_report(self):
        """Generate a comprehensive model evaluation report"""
        print("\nGenerating evaluation report...")
        
        report = f"""
# Machine Learning Model Evaluation Report

## Dataset Information
- **Total Records**: {len(self.df)}
- **Features**: {len(self.X.columns)}
- **Target Classes**: {', '.join(self.label_encoder.classes_)}
- **Training Set**: {len(self.X_train)} samples
- **Test Set**: {len(self.X_test)} samples

## Model Performance Summary

| Model | Accuracy | F1-Score |
|-------|----------|----------|
"""
        
        for name, result in self.results.items():
            report += f"| {name} | {result['accuracy']:.4f} | {result['f1_score']:.4f} |\n"
        
        report += f"""
## Best Model
- **Model**: {self.best_model_name}
- **Accuracy**: {self.results[self.best_model_name]['accuracy']:.4f}
- **F1-Score**: {self.results[self.best_model_name]['f1_score']:.4f}

## Detailed Classification Report for Best Model
"""
        
        # Add detailed classification report
        best_predictions = self.results[self.best_model_name]['predictions']
        report += classification_report(self.y_test, best_predictions, 
                                      target_names=self.label_encoder.classes_)
        
        # Save report to reports directory
        import os
        reports_dir = os.path.join('ml_model', 'reports')
        os.makedirs(reports_dir, exist_ok=True)
        
        report_path = os.path.join(reports_dir, 'model_evaluation_report.txt')
        with open(report_path, 'w') as f:
            f.write(report)
        
        print(f"Evaluation report saved as '{report_path}'")
        return report

def main():
    """Main execution function"""
    print("Starting Machine Learning Model Training")
    print("="*60)
    
    # Initialize trainer
    trainer = MLModelTrainer()
    
    # Load and preprocess data
    trainer.load_and_preprocess_data()
    
    # Train models
    trainer.train_models()
    
    # Create visualizations
    trainer.create_visualizations()
    
    # Save best model
    model_info = trainer.save_best_model()
    
    # Generate report
    trainer.generate_report()
    
    print("\n" + "="*60)
    print("Model training completed successfully!")
    print(f"Best model: {model_info['best_model_name']}")
    print(f"Best accuracy: {model_info['accuracy']:.4f}")
    print("\nFiles created:")
    print("ml_model/artifacts/")
    print("  - best_model.pkl (trained model)")
    print("  - label_encoder.pkl (target encoder)")
    print("  - scaler.pkl (feature scaler)")
    print("  - feature_columns.pkl (feature names)")
    print("  - model_info.pkl (model metadata)")
    print("ml_model/reports/")
    print("  - model_evaluation_report.txt (detailed report)")
    print("ml_model/visualizations/")
    print("  - model_performance_comparison.png")
    print("  - confusion_matrices.png")
    print("  - roc_curves.png")
    print("  - feature_importance.png")

if __name__ == "__main__":
    main()
