# Digital Media & Mental Health Assessment

This project provides a web-based questionnaire for assessing digital media use and mental health, integrated with a machine learning model for risk prediction.

## 🚀 Features

- **Interactive Questionnaire**: A multi-section HTML form for user input.
- **Traditional Scoring**: Calculates a total score and determines a risk level based on predefined thresholds.
- **Machine Learning Risk Prediction**: Utilizes trained ML models to predict risk levels based on questionnaire responses.
- **Balanced Data Generation**: A multi-threaded Python script to generate synthetic datasets with an equal distribution across risk categories for robust model training.
- **Model Training Pipeline**: Trains and evaluates multiple common ML algorithms (Logistic Regression, Random Forest, Gradient Boosting, SVM, K-Nearest Neighbors).
- **Model Evaluation & Visualization**: Generates performance metrics, comparison charts, confusion matrices, ROC curves, and feature importance plots.
- **Best Model Persistence**: Automatically saves the best performing model and its components for deployment.
- **Flask Integration**: Seamlessly integrates the ML model into the Flask web application for real-time predictions.
- **ML Model Information Page**: A dedicated endpoint (`/ml-info`) to display details about the deployed ML model, including its performance metrics and feature importance.
- **Organized Project Structure**: A clean and professional folder structure for easy navigation and maintenance.

## 📁 Project Structure

```
Social_Upgraded/
├── app.py                                    # Main Flask application
├── data_gen.py                              # Data generation script
├── questionnaire_data.csv                   # Generated dataset (output of data_gen.py)
├── requirements.txt                         # All Python dependencies
├── README.md                                # This file
├── docs/                                    # Documentation folder
│   └── MasterQuestion.pdf                  # Original questionnaire PDF
├── ml_model/                               # Machine Learning models and scripts
│   ├── ml_training.py                      # Script to train and save ML models
│   ├── ml_predictor.py                     # Utility for loading and using ML models
│   ├── artifacts/                          # Model artifacts and components
│   │   ├── best_model.pkl                  # Saved best performing ML model
│   │   ├── label_encoder.pkl               # Saved LabelEncoder for target variable
│   │   ├── scaler.pkl                      # Saved StandardScaler for numerical features
│   │   ├── feature_columns.pkl             # List of feature columns used by the model
│   │   └── model_info.pkl                  # Metadata about the best model (accuracy, F1, etc.)
│   ├── reports/                            # Model evaluation reports
│   │   └── model_evaluation_report.txt     # Detailed report from model training
│   └── visualizations/                     # Model performance visualizations
│       ├── model_performance_comparison.png # Visualization of model performance
│       ├── confusion_matrices.png          # Visualization of confusion matrices
│       ├── roc_curves.png                  # Visualization of ROC curves
│       └── feature_importance.png          # Visualization of feature importance
├── static/                                 # Static web assets (CSS, JS, images)
│   ├── css/                               # Stylesheets
│   └── js/                                # JavaScript files
└── templates/                              # HTML templates
    └── questionnaire.html                  # Main questionnaire form
```

## ⚙️ Setup and Installation

1.  **Clone the repository (if applicable):**
    ```bash
    git clone <repository_url>
    cd Social_Upgraded
    ```

2.  **Create a virtual environment (recommended):**
    ```bash
    python -m venv venv
    .\venv\Scripts\activate   # On Windows
    source venv/bin/activate  # On macOS/Linux
    ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

## 🏃 How to Run

### 1. Generate Synthetic Data (Optional, but recommended for ML training)

First, generate a dataset that the ML models can be trained on. This script creates `questionnaire_data.csv`.

```bash
python data_gen.py
```
This will generate 500 records, equally distributed across the four risk categories, with live progress updates.

### 2. Train Machine Learning Models

Next, train the ML models using the generated data. This script will save the best model and related artifacts to the `ml_model/` directory and generate evaluation plots.

```bash
python ml_model/ml_training.py
```

### 3. Run the Flask Web Application

Finally, start the Flask server to access the questionnaire and ML predictions.

```bash
python app.py
```

Open your web browser and navigate to:
-   **Questionnaire**: `http://localhost:5000`
-   **ML Model Information**: `http://localhost:5000/ml-info`

## 📝 Notes

-   The `app.py` will use the traditional scoring method and, if `ml_model/artifacts/best_model.pkl` exists, it will also provide an ML-powered risk prediction.
-   Ensure `questionnaire_data.csv` is present in the root directory before running `ml_model/ml_training.py`.
-   The `static/css` and `static/js` folders are currently empty but are included for future web asset organization.

## ⚖️ License

This project is for research and educational purposes.
