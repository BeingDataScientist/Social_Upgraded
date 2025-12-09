# Machine Learning Model Training for IAD Risk Assessment

## Overview
This folder contains a comprehensive machine learning pipeline for training and evaluating models to assess Internet Addiction Disorder (IAD) risk levels based on questionnaire data.

## Files

### `train_models.py`
Main training script that implements:
- Complete statistical analysis pipeline
- ANN (Artificial Neural Network) - Priority model
- 5 additional ML algorithms for comparison:
  - Logistic Regression
  - Support Vector Machine (SVM)
  - Random Forest
  - Gradient Boosting
  - K-Nearest Neighbors (KNN)

### `requirements.txt`
Python package dependencies required to run the training script.

### Output Files (Generated after training)

#### Models
- `best_model_ann.h5` - Saved ANN model (TensorFlow/Keras format)
- `scaler.pkl` - Feature scaler for preprocessing
- `label_encoder.pkl` - Label encoder for risk level classes
- `model_metadata.json` - Model metadata and configuration

#### Reports
- `training_report.txt` - Comprehensive statistical report with all analysis results

#### Visualizations (`figures/` folder)
1. `01_exploratory_analysis.png` - Risk level distribution, score histograms, boxplots
2. `02_question_distributions.png` - Distribution of all Q1-Q22 questions
3. `03_item_total_correlations.png` - Item-total correlation analysis
4. `04_scree_plot.png` - Factor analysis scree plot
5. `05_factor_loadings.png` - Factor loadings heatmap
6. `06_correlation_matrix.png` - Correlation matrix of all questions
7. `07_ann_training_history.png` - ANN training/validation curves
8. `08_cm_[ModelName].png` - Confusion matrices for each model
9. `09_model_comparison.png` - Side-by-side comparison of all models
10. `10_cross_validation.png` - 5-fold cross-validation results
11. `11_feature_importance_[ModelName].png` - Feature importance for tree-based models
12. `12_permutation_importance_[ModelName].png` - Permutation importance for all models
13. `13_roc_curves_[ModelName].png` - ROC curves for models with probability outputs

## Statistical Analyses Performed

### 1. Exploratory Data Analysis
- Descriptive statistics (mean, median, std, min, max, IQR)
- Risk level distribution
- Question-level distributions
- Visualizations (histograms, boxplots, violin plots)

### 2. Reliability Analysis
- Cronbach's Alpha (internal consistency)
- Item-total correlations
- Corrected item-total correlations

### 3. Factor Analysis
- KMO test (sampling adequacy)
- Bartlett's test of sphericity
- Exploratory Factor Analysis (EFA)
- Scree plot
- Factor loadings
- Variance explained

### 4. Correlation Analysis
- Pearson correlation matrix
- Multicollinearity detection
- High correlation pair identification

### 5. Model Training & Evaluation
- Stratified train-test split (80-20)
- 5-fold stratified cross-validation
- Metrics computed:
  - Accuracy
  - Precision (per-class and macro)
  - Recall (per-class and macro)
  - F1-score (per-class and macro)
  - ROC AUC (one-vs-rest, macro average)
  - Cohen's Kappa
  - Log Loss
- Confusion matrices
- ROC curves (multi-class)

### 6. Statistical Model Comparison
- Paired t-tests (ANN vs each model)
- McNemar's test (on test set)
- Cohen's d (effect size)
- Bootstrap confidence intervals

### 7. Feature Importance
- Tree-based feature importance (RF, GB)
- Permutation importance (all models)
- Top feature identification

## Usage

### Installation
```bash
pip install -r requirements.txt
```

### Running the Training
```bash
python train_models.py
```

The script will:
1. Load data from `../questionnaire_data.csv`
2. Perform all statistical analyses
3. Train all models
4. Generate visualizations
5. Save the best model (ANN)
6. Create comprehensive report

**Note:** Training may take 30-60 minutes depending on your system, especially for the ANN model.

## Model Architecture (ANN)

The ANN model uses:
- Input layer: 22 features (Q1-Q22)
- Hidden layers:
  - Dense(128) + BatchNorm + Dropout(0.3)
  - Dense(64) + BatchNorm + Dropout(0.3)
  - Dense(32) + Dropout(0.2)
- Output layer: Dense(4) with softmax (4 risk categories)
- Optimizer: Adam (learning_rate=0.001)
- Regularization: L2 (0.001) + Dropout
- Callbacks: EarlyStopping, ReduceLROnPlateau

## Risk Categories

1. **Low risk** (Score 0-10)
2. **At-risk (brief advice/monitor)** (Score 11-20)
3. **Problematic use likely (structured assessment)** (Score 21-29)
4. **High risk / addictive pattern (consider referral)** (Score 30+)

## Output Interpretation

After training completes, check:
- `training_report.txt` for detailed statistics and results
- `figures/` folder for all visualizations
- Model performance metrics in the report
- Feature importance rankings

## Notes

- All visualizations are saved automatically (non-interactive backend)
- The script handles missing optional packages gracefully
- Cross-validation ensures robust performance estimates
- Statistical tests provide rigorous model comparison

