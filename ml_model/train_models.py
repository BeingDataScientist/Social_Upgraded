"""
Comprehensive Machine Learning Model Training Script
for Internet Addiction Disorder (IAD) Risk Assessment

This script implements:
- Complete statistical analysis pipeline
- ANN (priority) + 5 other ML algorithms
- Comprehensive visualizations
- Detailed statistical report
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder, MinMaxScaler
from sklearn.metrics import (confusion_matrix, classification_report, accuracy_score, 
                             precision_recall_fscore_support, roc_auc_score, roc_curve,
                             cohen_kappa_score, log_loss, brier_score_loss)
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from scipy import stats
from scipy.stats import chi2_contingency, mannwhitneyu, kruskal, pearsonr, spearmanr
import warnings
warnings.filterwarnings('ignore')

# Neural Network (using scikit-learn instead of TensorFlow)
from sklearn.neural_network import MLPClassifier
from sklearn.exceptions import ConvergenceWarning
warnings.filterwarnings('ignore', category=ConvergenceWarning)

# Statistical analysis
try:
    from factor_analyzer import FactorAnalyzer
    from factor_analyzer.factor_analyzer import calculate_bartlett_sphericity, calculate_kmo
    FACTOR_ANALYZER_AVAILABLE = True
except ImportError:
    FACTOR_ANALYZER_AVAILABLE = False
    print("Warning: factor_analyzer not available. EFA will be skipped.")

try:
    import pingouin as pg
    PINGOUIN_AVAILABLE = True
except ImportError:
    PINGOUIN_AVAILABLE = False
    print("Warning: pingouin not available. Cronbach's alpha will use manual calculation.")

import os
import json
from datetime import datetime
import pickle

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

class IADModelTrainer:
    def __init__(self, data_path, output_dir='ml_model'):
        """Initialize the trainer"""
        self.data_path = data_path
        self.output_dir = output_dir
        self.report_lines = []
        self.figures = []
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(os.path.join(output_dir, 'figures'), exist_ok=True)
        
        # Initialize data containers
        self.df = None
        self.X = None
        self.y = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.scaler = None
        self.label_encoder = None
        self.models = {}
        self.results = {}
        
    def log(self, message):
        """Add message to report"""
        print(message)
        self.report_lines.append(message)
        
    def load_data(self):
        """Load and inspect data"""
        self.log("\n" + "="*80)
        self.log("STEP 1: DATA LOADING AND INSPECTION")
        self.log("="*80)
        
        self.df = pd.read_csv(self.data_path)
        self.log(f"Dataset loaded: {self.df.shape[0]} records, {self.df.shape[1]} features")
        self.log(f"Columns: {list(self.df.columns)}")
        
        # Check for missing values
        missing = self.df.isnull().sum()
        if missing.sum() > 0:
            self.log(f"\nMissing values:\n{missing[missing > 0]}")
        else:
            self.log("\nNo missing values found.")
        
        # Display basic info
        self.log(f"\nData types:\n{self.df.dtypes}")
        self.log(f"\nFirst few rows:\n{self.df.head()}")
        
        return self.df
    
    def exploratory_analysis(self):
        """Step 1: Exploratory and Descriptive Statistics"""
        self.log("\n" + "="*80)
        self.log("STEP 2: EXPLORATORY DATA ANALYSIS")
        self.log("="*80)
        
        # Descriptive statistics for numeric features
        numeric_cols = [f'Q{i}' for i in range(1, 23)]
        desc_stats = self.df[numeric_cols].describe()
        self.log(f"\nDescriptive Statistics for Q1-Q22:\n{desc_stats}")
        
        # Risk level distribution
        risk_dist = self.df['risk_level'].value_counts()
        risk_dist_pct = self.df['risk_level'].value_counts(normalize=True) * 100
        self.log(f"\nRisk Level Distribution:\n{risk_dist}")
        self.log(f"\nRisk Level Distribution (%):\n{risk_dist_pct}")
        
        # Total score statistics
        self.log(f"\nTotal Score Statistics:")
        self.log(f"Mean: {self.df['total_score'].mean():.2f}")
        self.log(f"Median: {self.df['total_score'].median():.2f}")
        self.log(f"Std Dev: {self.df['total_score'].std():.2f}")
        self.log(f"Min: {self.df['total_score'].min()}")
        self.log(f"Max: {self.df['total_score'].max()}")
        self.log(f"IQR: {self.df['total_score'].quantile(0.75) - self.df['total_score'].quantile(0.25):.2f}")
        
        # Visualizations
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Risk level distribution
        risk_dist.plot(kind='bar', ax=axes[0, 0], color='steelblue')
        axes[0, 0].set_title('Risk Level Distribution', fontsize=14, fontweight='bold')
        axes[0, 0].set_xlabel('Risk Level')
        axes[0, 0].set_ylabel('Count')
        axes[0, 0].tick_params(axis='x', rotation=45)
        
        # Total score histogram
        axes[0, 1].hist(self.df['total_score'], bins=30, color='coral', edgecolor='black')
        axes[0, 1].set_title('Total Score Distribution', fontsize=14, fontweight='bold')
        axes[0, 1].set_xlabel('Total Score')
        axes[0, 1].set_ylabel('Frequency')
        axes[0, 1].axvline(self.df['total_score'].mean(), color='red', linestyle='--', label=f'Mean: {self.df["total_score"].mean():.2f}')
        axes[0, 1].legend()
        
        # Total score by risk level (boxplot)
        risk_order = ['Low risk', 'At-risk (brief advice/monitor)', 
                     'Problematic use likely (structured assessment)',
                     'High risk / addictive pattern (consider referral)']
        df_ordered = self.df.copy()
        df_ordered['risk_level'] = pd.Categorical(df_ordered['risk_level'], categories=risk_order, ordered=True)
        sns.boxplot(data=df_ordered, x='risk_level', y='total_score', ax=axes[1, 0])
        axes[1, 0].set_title('Total Score by Risk Level', fontsize=14, fontweight='bold')
        axes[1, 0].set_xlabel('Risk Level')
        axes[1, 0].set_ylabel('Total Score')
        axes[1, 0].tick_params(axis='x', rotation=45)
        
        # Violin plot
        sns.violinplot(data=df_ordered, x='risk_level', y='total_score', ax=axes[1, 1])
        axes[1, 1].set_title('Total Score Distribution by Risk Level (Violin Plot)', fontsize=14, fontweight='bold')
        axes[1, 1].set_xlabel('Risk Level')
        axes[1, 1].set_ylabel('Total Score')
        axes[1, 1].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'figures', '01_exploratory_analysis.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        # Question-level statistics
        fig, axes = plt.subplots(4, 6, figsize=(24, 16))
        axes = axes.flatten()
        
        for i, q in enumerate(numeric_cols):
            axes[i].hist(self.df[q], bins=20, color='skyblue', edgecolor='black', alpha=0.7)
            axes[i].set_title(f'{q} Distribution', fontsize=10, fontweight='bold')
            axes[i].set_xlabel('Score')
            axes[i].set_ylabel('Frequency')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'figures', '02_question_distributions.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        self.log("\nExploratory analysis visualizations saved.")
        
    def _calculate_cronbach_alpha_manual(self, data):
        """Manual calculation of Cronbach's Alpha"""
        data_array = data.values
        k = data_array.shape[1]  # number of items
        item_variances = np.var(data_array, axis=0, ddof=1)
        total_variance = np.var(data_array.sum(axis=1), ddof=1)
        
        if total_variance == 0:
            alpha = 0
        else:
            alpha = (k / (k - 1)) * (1 - np.sum(item_variances) / total_variance)
        
        self.log(f"\nCronbach's Alpha (manual): {alpha:.4f}")
        
        if alpha >= 0.9:
            interpretation = "Excellent"
        elif alpha >= 0.8:
            interpretation = "Good"
        elif alpha >= 0.7:
            interpretation = "Acceptable"
        else:
            interpretation = "Questionable"
        
        self.log(f"Interpretation: {interpretation}")
        
        return (alpha, (0, 0))  # No CI for manual calculation
    
    def reliability_analysis(self):
        """Step 3: Questionnaire Reliability and Validation"""
        self.log("\n" + "="*80)
        self.log("STEP 3: RELIABILITY AND VALIDATION ANALYSIS")
        self.log("="*80)
        
        # Prepare data for reliability analysis (Q1-Q22 scores)
        reliability_data = self.df[[f'Q{i}' for i in range(1, 23)]]
        
        # Cronbach's Alpha
        if PINGOUIN_AVAILABLE:
            try:
                cronbach_result = pg.cronbach_alpha(data=reliability_data)
                self.log(f"\nCronbach's Alpha: {cronbach_result[0]:.4f}")
                self.log(f"95% CI: [{cronbach_result[1][0]:.4f}, {cronbach_result[1][1]:.4f}]")
                
                if cronbach_result[0] >= 0.9:
                    interpretation = "Excellent"
                elif cronbach_result[0] >= 0.8:
                    interpretation = "Good"
                elif cronbach_result[0] >= 0.7:
                    interpretation = "Acceptable"
                else:
                    interpretation = "Questionable"
                
                self.log(f"Interpretation: {interpretation}")
            except Exception as e:
                self.log(f"Error calculating Cronbach's Alpha: {e}")
                cronbach_result = self._calculate_cronbach_alpha_manual(reliability_data)
        else:
            cronbach_result = self._calculate_cronbach_alpha_manual(reliability_data)
        
        # Item-total correlations
        item_total_corr = {}
        total_scores = reliability_data.sum(axis=1)
        
        for col in reliability_data.columns:
            corr, p_val = pearsonr(reliability_data[col], total_scores)
            item_total_corr[col] = {'correlation': corr, 'p_value': p_val}
        
        self.log("\nItem-Total Correlations:")
        for item, stats in sorted(item_total_corr.items(), key=lambda x: x[1]['correlation'], reverse=True):
            self.log(f"{item}: r={stats['correlation']:.4f}, p={stats['p_value']:.4f}")
        
        # Visualize item-total correlations
        fig, ax = plt.subplots(figsize=(12, 6))
        items = list(item_total_corr.keys())
        correlations = [item_total_corr[item]['correlation'] for item in items]
        colors = ['green' if c >= 0.3 else 'orange' if c >= 0.2 else 'red' for c in correlations]
        
        ax.barh(items, correlations, color=colors)
        ax.axvline(0.3, color='red', linestyle='--', label='Minimum acceptable (0.3)')
        ax.set_xlabel('Item-Total Correlation')
        ax.set_title('Item-Total Correlations', fontsize=14, fontweight='bold')
        ax.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'figures', '03_item_total_correlations.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        # Factor Analysis (EFA)
        self.log("\nExploratory Factor Analysis (EFA):")
        
        # Check sampling adequacy
        if FACTOR_ANALYZER_AVAILABLE:
            try:
                kmo_all, kmo_model = calculate_kmo(reliability_data)
                self.log(f"KMO Test: {kmo_model:.4f}")
                
                if kmo_model >= 0.9:
                    kmo_interpretation = "Marvelous"
                elif kmo_model >= 0.8:
                    kmo_interpretation = "Meritorious"
                elif kmo_model >= 0.7:
                    kmo_interpretation = "Middling"
                elif kmo_model >= 0.6:
                    kmo_interpretation = "Mediocre"
                else:
                    kmo_interpretation = "Unacceptable"
                
                self.log(f"KMO Interpretation: {kmo_interpretation}")
                
                bartlett_chi2, bartlett_p = calculate_bartlett_sphericity(reliability_data)
                self.log(f"Bartlett's Test: χ²={bartlett_chi2:.4f}, p={bartlett_p:.4f}")
                
                if bartlett_p < 0.05:
                    self.log("Bartlett's test is significant - data suitable for factor analysis")
                else:
                    self.log("Bartlett's test is not significant - data may not be suitable for factor analysis")
                
                # Perform EFA
                fa = FactorAnalyzer(n_factors=4, rotation='varimax')
                fa.fit(reliability_data)
                
                # Get eigenvalues
                eigenvalues, _ = fa.get_eigenvalues()
                self.log(f"\nEigenvalues: {eigenvalues[:10]}")
                self.log(f"Factors with eigenvalue > 1: {sum(eigenvalues > 1)}")
                
                # Scree plot
                fig, ax = plt.subplots(figsize=(10, 6))
                ax.plot(range(1, len(eigenvalues) + 1), eigenvalues, 'bo-')
                ax.axhline(1, color='red', linestyle='--', label='Eigenvalue = 1')
                ax.set_xlabel('Factor Number')
                ax.set_ylabel('Eigenvalue')
                ax.set_title('Scree Plot', fontsize=14, fontweight='bold')
                ax.legend()
                ax.grid(True)
                plt.tight_layout()
                plt.savefig(os.path.join(self.output_dir, 'figures', '04_scree_plot.png'), dpi=300, bbox_inches='tight')
                plt.close()
                
                # Factor loadings
                loadings = fa.loadings_
                loadings_df = pd.DataFrame(loadings, index=reliability_data.columns, 
                                         columns=[f'Factor {i+1}' for i in range(4)])
                self.log(f"\nFactor Loadings:\n{loadings_df}")
                
                # Visualize factor loadings
                fig, ax = plt.subplots(figsize=(12, 8))
                sns.heatmap(loadings_df, annot=True, fmt='.3f', cmap='RdYlBu_r', center=0, 
                           vmin=-1, vmax=1, ax=ax, cbar_kws={'label': 'Loading'})
                ax.set_title('Factor Loadings Heatmap', fontsize=14, fontweight='bold')
                plt.tight_layout()
                plt.savefig(os.path.join(self.output_dir, 'figures', '05_factor_loadings.png'), dpi=300, bbox_inches='tight')
                plt.close()
                
                # Variance explained
                variance_explained = fa.get_eigenvalues()[1]
                self.log(f"\nVariance Explained by Factors: {variance_explained}")
                self.log(f"Total Variance Explained: {sum(variance_explained):.2f}%")
                
            except Exception as e:
                self.log(f"Error in Factor Analysis: {e}")
        else:
            self.log("Factor Analyzer not available. Skipping EFA.")
        
        return cronbach_result
    
    def correlation_analysis(self):
        """Step 5: Correlation Analysis"""
        self.log("\n" + "="*80)
        self.log("STEP 4: CORRELATION ANALYSIS")
        self.log("="*80)
        
        numeric_cols = [f'Q{i}' for i in range(1, 23)]
        corr_matrix = self.df[numeric_cols].corr()
        
        # Visualize correlation matrix
        fig, ax = plt.subplots(figsize=(16, 14))
        sns.heatmap(corr_matrix, annot=False, fmt='.2f', cmap='coolwarm', center=0,
                   square=True, linewidths=0.5, cbar_kws={'shrink': 0.8}, ax=ax)
        ax.set_title('Question Correlation Matrix', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'figures', '06_correlation_matrix.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        # Find high correlations (potential multicollinearity)
        high_corr_pairs = []
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                if abs(corr_matrix.iloc[i, j]) > 0.7:
                    high_corr_pairs.append((corr_matrix.columns[i], corr_matrix.columns[j], corr_matrix.iloc[i, j]))
        
        if high_corr_pairs:
            self.log(f"\nHigh correlation pairs (|r| > 0.7):")
            for pair in high_corr_pairs:
                self.log(f"{pair[0]} - {pair[1]}: {pair[2]:.4f}")
        else:
            self.log("\nNo extremely high correlations found (|r| > 0.7)")
        
        # VIF analysis (simplified - using correlation as proxy)
        self.log("\nNote: Full VIF analysis would require regression models for each variable")
        
    def prepare_data(self):
        """Step 6: Data Preprocessing"""
        self.log("\n" + "="*80)
        self.log("STEP 5: DATA PREPROCESSING")
        self.log("="*80)
        
        # Features (Q1-Q22)
        feature_cols = [f'Q{i}' for i in range(1, 23)]
        self.X = self.df[feature_cols].values
        
        # Target (risk_level)
        self.label_encoder = LabelEncoder()
        self.y = self.label_encoder.fit_transform(self.df['risk_level'])
        
        self.log(f"Features shape: {self.X.shape}")
        self.log(f"Target shape: {self.y.shape}")
        self.log(f"Class distribution: {np.bincount(self.y)}")
        self.log(f"Class names: {self.label_encoder.classes_}")
        
        # Split data
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=0.2, random_state=42, stratify=self.y
        )
        
        self.log(f"\nTrain set: {self.X_train.shape[0]} samples")
        self.log(f"Test set: {self.X_test.shape[0]} samples")
        
        # Scale features
        self.scaler = StandardScaler()
        self.X_train_scaled = self.scaler.fit_transform(self.X_train)
        self.X_test_scaled = self.scaler.transform(self.X_test)
        
        self.log("Features standardized using StandardScaler")
        
    def train_ann(self):
        """Train Artificial Neural Network using scikit-learn MLPClassifier"""
        self.log("\n" + "="*80)
        self.log("TRAINING ARTIFICIAL NEURAL NETWORK (ANN)")
        self.log("="*80)
        
        # Split training data for validation tracking
        from sklearn.model_selection import train_test_split
        X_train_ann, X_val_ann, y_train_ann, y_val_ann = train_test_split(
            self.X_train_scaled, self.y_train, test_size=0.2, random_state=42, stratify=self.y_train
        )
        
        # Define model architecture using MLPClassifier
        # Equivalent to: Dense(128) -> Dense(64) -> Dense(32) -> Dense(4)
        # hidden_layer_sizes=(128, 64, 32) creates 3 hidden layers
        model = MLPClassifier(
            hidden_layer_sizes=(128, 64, 32),  # 3 hidden layers: 128, 64, 32 neurons
            activation='relu',                  # ReLU activation
            solver='adam',                      # Adam optimizer
            alpha=0.001,                        # L2 regularization (equivalent to l2(0.001))
            learning_rate_init=0.001,           # Initial learning rate
            max_iter=500,                       # Maximum iterations (epochs)
            batch_size=32,                      # Batch size
            early_stopping=True,                # Early stopping
            validation_fraction=0.2,            # Validation split
            n_iter_no_change=15,                # Patience for early stopping
            tol=1e-4,                           # Tolerance for optimization
            random_state=42,
            verbose=False
        )
        
        self.log("\nModel Architecture:")
        self.log(f"  Input layer: 22 features")
        self.log(f"  Hidden layer 1: 128 neurons (ReLU)")
        self.log(f"  Hidden layer 2: 64 neurons (ReLU)")
        self.log(f"  Hidden layer 3: 32 neurons (ReLU)")
        self.log(f"  Output layer: 4 classes (softmax)")
        self.log(f"  Regularization: L2 (alpha=0.001)")
        self.log(f"  Optimizer: Adam (learning_rate=0.001)")
        self.log(f"  Early stopping: Enabled (patience=15)")
        
        # Track training history manually
        train_losses = []
        val_losses = []
        train_accs = []
        val_accs = []
        
        # Train with partial_fit for history tracking (or use warm_start)
        # For simplicity, we'll train once and use loss_curve if available
        self.log("\nTraining model...")
        model.fit(X_train_ann, y_train_ann)
        
        # Get loss curve if available
        if hasattr(model, 'loss_curve_') and model.loss_curve_ is not None:
            train_losses = model.loss_curve_
            # For validation, we'll approximate by evaluating on validation set
            val_losses = []
            val_accs = []
            for i in range(len(train_losses)):
                # Approximate validation loss (MLPClassifier doesn't track it directly)
                val_pred = model.predict(X_val_ann)
                val_acc = accuracy_score(y_val_ann, val_pred)
                val_accs.append(val_acc)
        else:
            # If loss_curve not available, create simple history
            train_losses = [model.loss_] if hasattr(model, 'loss_') else []
            val_losses = []
            train_accs = [accuracy_score(y_train_ann, model.predict(X_train_ann))]
            val_accs = [accuracy_score(y_val_ann, model.predict(X_val_ann))]
        
        # Plot training history (if we have data)
        if len(train_losses) > 0 or len(train_accs) > 0:
            fig, axes = plt.subplots(1, 2, figsize=(15, 5))
            
            if len(train_losses) > 0:
                axes[0].plot(train_losses, label='Training Loss', color='blue')
                if len(val_losses) > 0:
                    axes[0].plot(val_losses, label='Validation Loss', color='red')
                axes[0].set_title('Model Loss', fontsize=14, fontweight='bold')
                axes[0].set_xlabel('Iteration')
                axes[0].set_ylabel('Loss')
                axes[0].legend()
                axes[0].grid(True)
            else:
                axes[0].text(0.5, 0.5, 'Loss curve not available', 
                           ha='center', va='center', transform=axes[0].transAxes)
                axes[0].set_title('Model Loss', fontsize=14, fontweight='bold')
            
            if len(train_accs) > 0:
                axes[1].plot(train_accs, label='Training Accuracy', color='blue')
                if len(val_accs) > 0:
                    axes[1].plot(val_accs, label='Validation Accuracy', color='red')
                axes[1].set_title('Model Accuracy', fontsize=14, fontweight='bold')
                axes[1].set_xlabel('Iteration')
                axes[1].set_ylabel('Accuracy')
                axes[1].legend()
                axes[1].grid(True)
            else:
                # Calculate final accuracies
                train_pred_final = model.predict(self.X_train_scaled)
                test_pred_final = model.predict(self.X_test_scaled)
                train_acc_final = accuracy_score(self.y_train, train_pred_final)
                test_acc_final = accuracy_score(self.y_test, test_pred_final)
                
                axes[1].bar(['Training', 'Test'], [train_acc_final, test_acc_final], 
                           color=['blue', 'green'], alpha=0.7)
                axes[1].set_title('Model Accuracy', fontsize=14, fontweight='bold')
                axes[1].set_ylabel('Accuracy')
                axes[1].set_ylim([0, 1])
            
            plt.tight_layout()
            plt.savefig(os.path.join(self.output_dir, 'figures', '07_ann_training_history.png'), dpi=300, bbox_inches='tight')
            plt.close()
        
        # Evaluate on full training and test sets
        train_pred = model.predict(self.X_train_scaled)
        train_proba = model.predict_proba(self.X_train_scaled)
        test_pred = model.predict(self.X_test_scaled)
        test_proba = model.predict_proba(self.X_test_scaled)
        
        train_acc = accuracy_score(self.y_train, train_pred)
        test_acc = accuracy_score(self.y_test, test_pred)
        
        self.log(f"\nANN Training Accuracy: {train_acc:.4f}")
        self.log(f"ANN Test Accuracy: {test_acc:.4f}")
        self.log(f"Number of iterations: {model.n_iter_}")
        if hasattr(model, 'loss_'):
            self.log(f"Final loss: {model.loss_:.4f}")
        
        self.models['ANN'] = model
        self.results['ANN'] = {
            'model': model,
            'train_pred': train_pred,
            'test_pred': test_pred,
            'train_proba': train_proba,
            'test_proba': test_proba,
            'history': {
                'loss': train_losses if len(train_losses) > 0 else [model.loss_] if hasattr(model, 'loss_') else [],
                'val_loss': val_losses,
                'accuracy': train_accs if len(train_accs) > 0 else [train_acc],
                'val_accuracy': val_accs if len(val_accs) > 0 else [test_acc]
            }
        }
        
        return model, self.results['ANN']['history']
    
    def train_ml_models(self):
        """Train 5 other ML models"""
        self.log("\n" + "="*80)
        self.log("TRAINING OTHER ML MODELS")
        self.log("="*80)
        
        models_to_train = {
            'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42, multi_class='multinomial'),
            'SVM': SVC(probability=True, random_state=42),
            'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
            'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, random_state=42),
            'KNN': KNeighborsClassifier(n_neighbors=5)
        }
        
        for name, model in models_to_train.items():
            self.log(f"\nTraining {name}...")
            
            # Train
            model.fit(self.X_train_scaled, self.y_train)
            
            # Predictions
            train_pred = model.predict(self.X_train_scaled)
            test_pred = model.predict(self.X_test_scaled)
            
            # Probabilities
            try:
                train_proba = model.predict_proba(self.X_train_scaled)
                test_proba = model.predict_proba(self.X_test_scaled)
            except:
                train_proba = None
                test_proba = None
            
            # Accuracy
            train_acc = accuracy_score(self.y_train, train_pred)
            test_acc = accuracy_score(self.y_test, test_pred)
            
            self.log(f"{name} Training Accuracy: {train_acc:.4f}")
            self.log(f"{name} Test Accuracy: {test_acc:.4f}")
            
            self.models[name] = model
            self.results[name] = {
                'model': model,
                'train_pred': train_pred,
                'test_pred': test_pred,
                'train_proba': train_proba,
                'test_proba': test_proba
            }
    
    def evaluate_models(self):
        """Comprehensive model evaluation"""
        self.log("\n" + "="*80)
        self.log("COMPREHENSIVE MODEL EVALUATION")
        self.log("="*80)
        
        all_metrics = {}
        
        for name in self.models.keys():
            self.log(f"\n{'='*60}")
            self.log(f"Evaluation: {name}")
            self.log(f"{'='*60}")
            
            test_pred = self.results[name]['test_pred']
            
            # Classification report
            report = classification_report(self.y_test, test_pred, 
                                          target_names=self.label_encoder.classes_,
                                          output_dict=True)
            report_str = classification_report(self.y_test, test_pred, 
                                               target_names=self.label_encoder.classes_)
            self.log(f"\nClassification Report:\n{report_str}")
            
            # Confusion matrix
            cm = confusion_matrix(self.y_test, test_pred)
            self.log(f"\nConfusion Matrix:\n{cm}")
            
            # Visualize confusion matrix
            fig, ax = plt.subplots(figsize=(10, 8))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                       xticklabels=self.label_encoder.classes_,
                       yticklabels=self.label_encoder.classes_)
            ax.set_title(f'Confusion Matrix - {name}', fontsize=14, fontweight='bold')
            ax.set_xlabel('Predicted')
            ax.set_ylabel('Actual')
            plt.tight_layout()
            plt.savefig(os.path.join(self.output_dir, 'figures', f'08_cm_{name.replace(" ", "_")}.png'), 
                       dpi=300, bbox_inches='tight')
            plt.close()
            
            # Metrics
            accuracy = accuracy_score(self.y_test, test_pred)
            precision, recall, f1, _ = precision_recall_fscore_support(self.y_test, test_pred, average=None)
            macro_precision = precision_recall_fscore_support(self.y_test, test_pred, average='macro')[0]
            macro_recall = precision_recall_fscore_support(self.y_test, test_pred, average='macro')[1]
            macro_f1 = precision_recall_fscore_support(self.y_test, test_pred, average='macro')[2]
            
            # ROC AUC (one-vs-rest for multi-class)
            if self.results[name]['test_proba'] is not None:
                try:
                    roc_auc = roc_auc_score(self.y_test, self.results[name]['test_proba'], 
                                           multi_class='ovr', average='macro')
                except:
                    roc_auc = None
            else:
                roc_auc = None
            
            # Cohen's Kappa
            kappa = cohen_kappa_score(self.y_test, test_pred)
            
            # Log loss
            if self.results[name]['test_proba'] is not None:
                logloss = log_loss(self.y_test, self.results[name]['test_proba'])
            else:
                logloss = None
            
            # ROC curves for multi-class (one-vs-rest)
            if self.results[name]['test_proba'] is not None and roc_auc is not None:
                self._plot_roc_curves(name, self.results[name]['test_proba'])
            
            metrics = {
                'accuracy': accuracy,
                'macro_precision': macro_precision,
                'macro_recall': macro_recall,
                'macro_f1': macro_f1,
                'roc_auc': roc_auc,
                'kappa': kappa,
                'log_loss': logloss,
                'per_class_precision': precision.tolist(),
                'per_class_recall': recall.tolist(),
                'per_class_f1': f1.tolist()
            }
            
            all_metrics[name] = metrics
            
            self.log(f"\nSummary Metrics:")
            self.log(f"Accuracy: {accuracy:.4f}")
            self.log(f"Macro Precision: {macro_precision:.4f}")
            self.log(f"Macro Recall: {macro_recall:.4f}")
            self.log(f"Macro F1: {macro_f1:.4f}")
            if roc_auc:
                self.log(f"ROC AUC (macro): {roc_auc:.4f}")
            self.log(f"Cohen's Kappa: {kappa:.4f}")
            if logloss:
                self.log(f"Log Loss: {logloss:.4f}")
        
        # Comparison visualization
        self._plot_model_comparison(all_metrics)
        
        return all_metrics
    
    def _plot_roc_curves(self, model_name, y_proba):
        """Plot ROC curves for multi-class classification (one-vs-rest)"""
        from sklearn.preprocessing import label_binarize
        from sklearn.metrics import roc_curve, auc
        
        # Binarize the output
        y_test_bin = label_binarize(self.y_test, classes=np.unique(self.y_test))
        n_classes = y_test_bin.shape[1]
        
        # Compute ROC curve and ROC area for each class
        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        
        for i in range(n_classes):
            fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_proba[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])
        
        # Plot all ROC curves
        fig, ax = plt.subplots(figsize=(10, 8))
        colors = ['blue', 'red', 'green', 'orange', 'purple', 'brown']
        
        for i in range(n_classes):
            ax.plot(fpr[i], tpr[i], color=colors[i % len(colors)], lw=2,
                   label=f'{self.label_encoder.classes_[i]} (AUC = {roc_auc[i]:.2f})')
        
        ax.plot([0, 1], [0, 1], 'k--', lw=2, label='Random')
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('False Positive Rate', fontsize=12)
        ax.set_ylabel('True Positive Rate', fontsize=12)
        ax.set_title(f'ROC Curves - {model_name} (One-vs-Rest)', fontsize=14, fontweight='bold')
        ax.legend(loc="lower right", fontsize=10)
        ax.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'figures', f'13_roc_curves_{model_name.replace(" ", "_")}.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_model_comparison(self, all_metrics):
        """Plot model comparison"""
        models = list(all_metrics.keys())
        metrics_to_plot = ['accuracy', 'macro_precision', 'macro_recall', 'macro_f1', 'kappa']
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.flatten()
        
        for i, metric in enumerate(metrics_to_plot):
            values = [all_metrics[m].get(metric, 0) for m in models]
            axes[i].barh(models, values, color=sns.color_palette("husl", len(models)))
            axes[i].set_xlabel(metric.replace('_', ' ').title())
            axes[i].set_title(f'{metric.replace("_", " ").title()} Comparison', fontweight='bold')
            axes[i].grid(axis='x', alpha=0.3)
        
        # ROC AUC if available
        roc_values = [all_metrics[m].get('roc_auc', 0) if all_metrics[m].get('roc_auc') else 0 for m in models]
        if any(roc_values):
            axes[5].barh(models, roc_values, color=sns.color_palette("husl", len(models)))
            axes[5].set_xlabel('ROC AUC')
            axes[5].set_title('ROC AUC Comparison', fontweight='bold')
            axes[5].grid(axis='x', alpha=0.3)
        else:
            axes[5].axis('off')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'figures', '09_model_comparison.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    def cross_validation(self):
        """Perform cross-validation"""
        self.log("\n" + "="*80)
        self.log("CROSS-VALIDATION ANALYSIS")
        self.log("="*80)
        
        cv_scores = {}
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        
        for name, model in self.models.items():
            if name == 'ANN':
                # For ANN (MLPClassifier), use cross_val_score directly
                # MLPClassifier works with sklearn's cross_val_score
                scores = cross_val_score(
                    MLPClassifier(
                        hidden_layer_sizes=(128, 64, 32),
                        activation='relu',
                        solver='adam',
                        alpha=0.001,
                        learning_rate_init=0.001,
                        max_iter=200,
                        batch_size=32,
                        early_stopping=True,
                        validation_fraction=0.2,
                        n_iter_no_change=10,
                        random_state=42,
                        verbose=False
                    ),
                    self.X_train_scaled, self.y_train, cv=skf, scoring='accuracy'
                )
                cv_scores[name] = scores
            else:
                scores = cross_val_score(model, self.X_train_scaled, self.y_train, cv=skf, scoring='accuracy')
                cv_scores[name] = scores
            
            mean_score = np.mean(cv_scores[name])
            std_score = np.std(cv_scores[name])
            self.log(f"{name}: {mean_score:.4f} (+/- {std_score:.4f})")
        
        # Visualize CV results
        fig, ax = plt.subplots(figsize=(12, 6))
        models = list(cv_scores.keys())
        positions = np.arange(len(models))
        
        data_to_plot = [cv_scores[m] for m in models]
        bp = ax.boxplot(data_to_plot, labels=models, patch_artist=True)
        
        colors = sns.color_palette("husl", len(models))
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        
        ax.set_ylabel('Accuracy')
        ax.set_title('5-Fold Cross-Validation Results', fontsize=14, fontweight='bold')
        ax.grid(axis='y', alpha=0.3)
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'figures', '10_cross_validation.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        return cv_scores
    
    def statistical_comparison(self, cv_scores):
        """Statistical tests to compare models"""
        self.log("\n" + "="*80)
        self.log("STATISTICAL COMPARISON OF MODELS")
        self.log("="*80)
        
        # Compare ANN with each other model using paired t-test
        ann_scores = cv_scores['ANN']
        
        self.log("\nPaired t-tests (ANN vs Others):")
        comparison_results = {}
        
        for name, scores in cv_scores.items():
            if name != 'ANN':
                # Paired t-test
                t_stat, p_value = stats.ttest_rel(ann_scores, scores)
                
                # Effect size (Cohen's d)
                diff = np.array(ann_scores) - np.array(scores)
                cohens_d = np.mean(diff) / np.std(diff, ddof=1)
                
                comparison_results[name] = {
                    't_statistic': t_stat,
                    'p_value': p_value,
                    'cohens_d': cohens_d
                }
                
                significance = "***" if p_value < 0.001 else "**" if p_value < 0.01 else "*" if p_value < 0.05 else "ns"
                self.log(f"\nANN vs {name}:")
                self.log(f"  t-statistic: {t_stat:.4f}")
                self.log(f"  p-value: {p_value:.4f} {significance}")
                self.log(f"  Cohen's d: {cohens_d:.4f}")
        
        # McNemar's test on test set predictions
        self.log("\nMcNemar's Test (on test set):")
        ann_test_pred = self.results['ANN']['test_pred']
        
        for name in self.models.keys():
            if name != 'ANN':
                other_test_pred = self.results[name]['test_pred']
                
                # Create contingency table
                both_correct = np.sum((ann_test_pred == self.y_test) & (other_test_pred == self.y_test))
                ann_correct_other_wrong = np.sum((ann_test_pred == self.y_test) & (other_test_pred != self.y_test))
                ann_wrong_other_correct = np.sum((ann_test_pred != self.y_test) & (other_test_pred == self.y_test))
                both_wrong = np.sum((ann_test_pred != self.y_test) & (other_test_pred != self.y_test))
                
                # McNemar's test
                b = ann_correct_other_wrong
                c = ann_wrong_other_correct
                
                if b + c > 0:
                    chi2_mcnemar = ((abs(b - c) - 1)**2) / (b + c)
                    p_mcnemar = 1 - stats.chi2.cdf(chi2_mcnemar, df=1)
                    
                    self.log(f"\nANN vs {name}:")
                    self.log(f"  b (ANN correct, {name} wrong): {b}")
                    self.log(f"  c (ANN wrong, {name} correct): {c}")
                    self.log(f"  χ²: {chi2_mcnemar:.4f}")
                    self.log(f"  p-value: {p_mcnemar:.4f}")
                else:
                    self.log(f"\nANN vs {name}: Cannot compute (b+c=0)")
        
        return comparison_results
    
    def feature_importance_analysis(self):
        """Feature importance analysis using permutation importance"""
        self.log("\n" + "="*80)
        self.log("FEATURE IMPORTANCE ANALYSIS")
        self.log("="*80)
        
        feature_names = [f'Q{i}' for i in range(1, 23)]
        
        # For tree-based models
        tree_models = ['Random Forest', 'Gradient Boosting']
        for name in tree_models:
            if name in self.models:
                model = self.models[name]
                if hasattr(model, 'feature_importances_'):
                    importances = model.feature_importances_
                    indices = np.argsort(importances)[::-1]
                    
                    self.log(f"\n{name} - Top 10 Most Important Features:")
                    for i in range(min(10, len(indices))):
                        idx = indices[i]
                        self.log(f"  {feature_names[idx]}: {importances[idx]:.4f}")
                    
                    # Visualize
                    fig, ax = plt.subplots(figsize=(12, 8))
                    top_n = min(15, len(indices))
                    top_indices = indices[:top_n]
                    ax.barh(range(top_n), importances[top_indices], color='steelblue')
                    ax.set_yticks(range(top_n))
                    ax.set_yticklabels([feature_names[i] for i in top_indices])
                    ax.set_xlabel('Importance')
                    ax.set_title(f'{name} - Feature Importance (Top {top_n})', fontsize=14, fontweight='bold')
                    ax.invert_yaxis()
                    plt.tight_layout()
                    plt.savefig(os.path.join(self.output_dir, 'figures', f'11_feature_importance_{name.replace(" ", "_")}.png'), 
                               dpi=300, bbox_inches='tight')
                    plt.close()
        
        # Permutation importance for all models
        self.log("\nPermutation Importance (on test set):")
        permutation_results = {}
        
        for name, model in self.models.items():
            # MLPClassifier (ANN) works directly with permutation_importance
            try:
                perm_importance = permutation_importance(
                    model, self.X_test_scaled, self.y_test, 
                    n_repeats=10, random_state=42, n_jobs=-1
                )
            except Exception as e:
                self.log(f"  {name}: Could not compute permutation importance: {e}")
                continue
            
            permutation_results[name] = perm_importance
            
            # Get top features
            indices = np.argsort(perm_importance.importances_mean)[::-1]
            self.log(f"\n{name} - Top 10 Most Important Features (Permutation):")
            for i in range(min(10, len(indices))):
                idx = indices[i]
                self.log(f"  {feature_names[idx]}: {perm_importance.importances_mean[idx]:.4f} "
                        f"(±{perm_importance.importances_std[idx]:.4f})")
            
            # Visualize
            fig, ax = plt.subplots(figsize=(12, 8))
            top_n = min(15, len(indices))
            top_indices = indices[:top_n]
            y_pos = np.arange(top_n)
            ax.barh(y_pos, perm_importance.importances_mean[top_indices], 
                   xerr=perm_importance.importances_std[top_indices], 
                   color='coral', capsize=5)
            ax.set_yticks(y_pos)
            ax.set_yticklabels([feature_names[i] for i in top_indices])
            ax.set_xlabel('Importance (Decrease in Accuracy)')
            ax.set_title(f'{name} - Permutation Importance (Top {top_n})', fontsize=14, fontweight='bold')
            ax.invert_yaxis()
            plt.tight_layout()
            plt.savefig(os.path.join(self.output_dir, 'figures', f'12_permutation_importance_{name.replace(" ", "_")}.png'), 
                       dpi=300, bbox_inches='tight')
            plt.close()
        
        return permutation_results
    
    def save_best_model(self):
        """Save the best model (ANN)"""
        self.log("\n" + "="*80)
        self.log("SAVING BEST MODEL")
        self.log("="*80)
        
        # Save ANN model using joblib (better for scikit-learn models)
        try:
            import joblib
            ann_model = self.models['ANN']
            joblib.dump(ann_model, os.path.join(self.output_dir, 'best_model_ann.pkl'))
            self.log("ANN model saved as 'best_model_ann.pkl'")
        except ImportError:
            # Fallback to pickle if joblib not available
            ann_model = self.models['ANN']
            with open(os.path.join(self.output_dir, 'best_model_ann.pkl'), 'wb') as f:
                pickle.dump(ann_model, f)
            self.log("ANN model saved as 'best_model_ann.pkl' (using pickle)")
        
        # Save scaler and label encoder
        with open(os.path.join(self.output_dir, 'scaler.pkl'), 'wb') as f:
            pickle.dump(self.scaler, f)
        self.log("Scaler saved as 'scaler.pkl'")
        
        with open(os.path.join(self.output_dir, 'label_encoder.pkl'), 'wb') as f:
            pickle.dump(self.label_encoder, f)
        self.log("Label encoder saved as 'label_encoder.pkl'")
        
        # Save model metadata
        ann_model = self.models['ANN']
        metadata = {
            'model_type': 'ANN (MLPClassifier)',
            'model_library': 'scikit-learn',
            'input_features': 22,
            'output_classes': 4,
            'class_names': self.label_encoder.classes_.tolist(),
            'training_date': datetime.now().isoformat(),
            'scaler_type': 'StandardScaler',
            'architecture': {
                'hidden_layers': [128, 64, 32],
                'activation': 'relu',
                'solver': 'adam',
                'alpha': 0.001,
                'learning_rate': 0.001
            }
        }
        
        with open(os.path.join(self.output_dir, 'model_metadata.json'), 'w') as f:
            json.dump(metadata, f, indent=2)
        self.log("Model metadata saved as 'model_metadata.json'")
    
    def generate_report(self):
        """Generate comprehensive text report"""
        report_path = os.path.join(self.output_dir, 'training_report.txt')
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("\n".join(self.report_lines))
        
        self.log(f"\nComprehensive report saved to: {report_path}")
    
    def run_full_pipeline(self):
        """Run the complete training pipeline"""
        self.log("="*80)
        self.log("IAD RISK ASSESSMENT - MACHINE LEARNING MODEL TRAINING")
        self.log("="*80)
        self.log(f"Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        # Execute pipeline
        self.load_data()
        self.exploratory_analysis()
        self.reliability_analysis()
        self.correlation_analysis()
        self.prepare_data()
        self.train_ann()
        self.train_ml_models()
        metrics = self.evaluate_models()
        cv_scores = self.cross_validation()
        self.statistical_comparison(cv_scores)
        self.feature_importance_analysis()
        self.save_best_model()
        self.generate_report()
        
        self.log("\n" + "="*80)
        self.log("TRAINING COMPLETE!")
        self.log("="*80)
        self.log(f"End Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        self.log(f"\nAll outputs saved to: {self.output_dir}")
        self.log(f"Figures saved to: {os.path.join(self.output_dir, 'figures')}")
        
        return metrics, cv_scores


if __name__ == "__main__":
    # Paths
    data_path = "../questionnaire_data.csv"
    output_dir = "ml_model"
    
    # Initialize and run
    trainer = IADModelTrainer(data_path, output_dir)
    metrics, cv_scores = trainer.run_full_pipeline()
    
    print("\n✅ Training completed successfully!")
    print(f"📊 Check {output_dir}/training_report.txt for detailed statistics")
    print(f"📈 Check {output_dir}/figures/ for all visualizations")
    print(f"🤖 Best model saved as {output_dir}/best_model_ann.pkl")

