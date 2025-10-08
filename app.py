from flask import Flask, render_template, request, jsonify, session, redirect, url_for, send_from_directory
import json
import os
import pandas as pd
import numpy as np
from ml_model.ml_predictor import predict_risk_level, format_questionnaire_data, get_predictor

app = Flask(__name__)
app.secret_key = 'your-secret-key-here'  # Change this in production

@app.route('/')
def index():
    """Serve the main questionnaire page"""
    return render_template('questionnaire.html')

@app.route('/submit', methods=['POST'])
def submit_questionnaire():
    """Handle form submission and display results"""
    try:
        # Get form data
        form_data = request.form.to_dict()
        
        # Create the Python Flask terminal style output
        python_output = {}
        for i in range(1, 23):  # Questions 1-22
            question_key = f'q{i}'
            if question_key in form_data:
                value = form_data[question_key]
                # Convert to appropriate type
                try:
                    # Try to convert to int if it's a number
                    python_output[f'Q{i}'] = int(value)
                except ValueError:
                    # If not a number, keep as string
                    python_output[f'Q{i}'] = value
            else:
                python_output[f'Q{i}'] = 0
        
        # Print to Flask terminal
        print("\n" + "="*50)
        print("QUESTIONNAIRE SUBMISSION RECEIVED")
        print("="*50)
        print(f"Python Flask Terminal Output:")
        print(python_output)
        print("="*50)
        
        # Calculate total score (legacy method)
        total_score = calculate_total_score(python_output)
        risk_level_legacy = determine_risk_level(total_score)
        
        # ML Model Prediction
        ml_prediction_result = None
        if os.path.exists('ml_model/artifacts/best_model.pkl'):
            try:
                # Format data for ML model
                formatted_data = format_questionnaire_data(form_data)
                
                # Get ML prediction
                ml_prediction_result = predict_risk_level(formatted_data)
                
                print(f"🤖 ML Model Prediction:")
                print(f"   Risk Level: {ml_prediction_result.get('predicted_risk_level', 'Unknown')}")
                print(f"   Confidence: {ml_prediction_result.get('confidence', 0):.4f}")
                print(f"   Model: {ml_prediction_result.get('model_info', {}).get('model_name', 'Unknown')}")
                
            except Exception as ml_error:
                print(f"⚠️  ML Prediction Error: {str(ml_error)}")
                ml_prediction_result = {
                    'success': False,
                    'error': str(ml_error)
                }
        else:
            print("⚠️  ML model not found. Run ml_model/ml_training.py first to train the model.")
        
        # Store results in session for display on results page
        session['results'] = {
            'python_output': python_output,
            'total_score': total_score,
            'max_score': 68,
            'risk_level_legacy': risk_level_legacy,
            'ml_prediction': ml_prediction_result
        }
        
        # Return JSON response for AJAX
        response_data = {
            'success': True,
            'redirect_url': url_for('show_results')
        }
        
        return jsonify(response_data)
        
    except Exception as e:
        print(f"Error processing form: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

def calculate_total_score(data):
    """Calculate total score from questionnaire data"""
    # Define scoring for each question
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
    
    total = 0
    for i in range(1, 23):
        question_key = f'Q{i}'
        if question_key in data:
            value = data[question_key]
            if question_key in scoring_map:
                if isinstance(scoring_map[question_key], dict):
                    # Look up value in mapping
                    if value in scoring_map[question_key]:
                        total += scoring_map[question_key][value]
                else:
                    # Direct score value
                    total += scoring_map[question_key]
    
    return total

def determine_risk_level(score):
    """Determine risk level based on total score"""
    if score >= 30:
        return "High risk / addictive pattern (consider referral)"
    elif score >= 21:
        return "Problematic use likely (structured assessment)"
    elif score >= 11:
        return "At-risk (brief advice/monitor)"
    else:
        return "Low risk"

def analyze_user_position(user_score):
    """Analyze user's position relative to the dataset"""
    try:
        # Load the dataset
        df = pd.read_csv('questionnaire_data.csv')
        
        # Get score statistics
        scores = df['total_score'].values
        mean_score = np.mean(scores)
        median_score = np.median(scores)
        std_score = np.std(scores)
        min_score = np.min(scores)
        max_score = np.max(scores)
        
        # Calculate user's percentile
        percentile = (np.sum(scores <= user_score) / len(scores)) * 100
        
        # Get risk level distribution
        risk_distribution = df['risk_level'].value_counts().to_dict()
        
        # Calculate how many people have similar scores (±5 points)
        similar_scores = np.sum(np.abs(scores - user_score) <= 5)
        similar_percentage = (similar_scores / len(scores)) * 100
        
        return {
            'user_score': user_score,
            'mean_score': mean_score,
            'median_score': median_score,
            'std_score': std_score,
            'min_score': min_score,
            'max_score': max_score,
            'percentile': percentile,
            'risk_distribution': risk_distribution,
            'similar_scores': similar_scores,
            'similar_percentage': similar_percentage,
            'total_records': len(scores)
        }
    except Exception as e:
        print(f"Error analyzing user position: {str(e)}")
        return None

@app.route('/results')
def show_results():
    """Display results page"""
    # Get results from session
    results = session.get('results')
    
    if not results:
        return redirect(url_for('index'))
    
    # Extract data from session
    python_output = results.get('python_output', {})
    total_score = results.get('total_score', 0)
    max_score = results.get('max_score', 68)
    risk_level_legacy = results.get('risk_level_legacy', 'Unknown')
    ml_prediction = results.get('ml_prediction', {})
    
    # Analyze user's position in the dataset
    dataset_analysis = analyze_user_position(total_score)
    
    # Calculate risk level colors and percentages first
    risk_colors = {
        "Low risk": "#28a745",
        "At-risk (brief advice/monitor)": "#ffc107", 
        "Problematic use likely (structured assessment)": "#fd7e14",
        "High risk / addictive pattern (consider referral)": "#dc3545"
    }
    
    legacy_color = risk_colors.get(risk_level_legacy, "#6c757d")
    ml_color = "#007bff"
    
    # Create Python output display
    python_output_str = str(python_output).replace("'", '"')
    
    # Create ML prediction display
    ml_display = ""
    if ml_prediction and ml_prediction.get('success', False):
        predicted_risk = ml_prediction.get('predicted_risk_level', 'Unknown')
        confidence = ml_prediction.get('confidence', 0)
        model_name = ml_prediction.get('model_info', {}).get('model_name', 'Unknown')
        
        ml_display = f"""
        <div class="confidence-display">
            <p><strong>Predicted Risk Level:</strong></p>
            <div class="risk-level" style="background: {ml_color}; margin: 10px 0;">{predicted_risk}</div>
            <p><strong>Confidence:</strong> {confidence:.1%}</p>
            <div class="confidence-bar">
                <div class="confidence-fill" style="width: {confidence*100}%;">
                    <div class="confidence-text">{confidence:.1%}</div>
                </div>
            </div>
            <p><strong>Model Used:</strong> {model_name}</p>
        </div>
        """
    elif ml_prediction and not ml_prediction.get('success', False):
        error_msg = ml_prediction.get('error', 'Unknown error')
        ml_display = f"""
        <div class="ml-results">
            <h3>🤖 Machine Learning Prediction</h3>
            <div class="error-card">
                <p class="error">ML Prediction Error: {error_msg}</p>
            </div>
        </div>
        """
    else:
        ml_display = """
        <div class="ml-results">
            <h3>🤖 Machine Learning Prediction</h3>
            <div class="info-card">
                <p>ML model not available. Please run the training script first.</p>
            </div>
        </div>
        """
    
    # Calculate score percentage
    score_percentage = (total_score / max_score) * 100
    
    # Get ML confidence percentage
    ml_confidence = 0
    if ml_prediction and ml_prediction.get('success', False):
        ml_confidence = ml_prediction.get('confidence', 0) * 100
    
    return f"""
    <html>
    <head>
        <title>Assessment Results - Digital Media & Mental Health</title>
        <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
        <style>
            * {{ margin: 0; padding: 0; box-sizing: border-box; }}
            body {{ 
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; 
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                min-height: 100vh;
                padding: 20px;
            }}
            .container {{ 
                max-width: 1200px; 
                margin: 0 auto; 
                background: white; 
                border-radius: 20px; 
                box-shadow: 0 20px 40px rgba(0,0,0,0.1);
                overflow: hidden;
            }}
            .header {{ 
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white; 
                padding: 40px; 
                text-align: center;
            }}
            .header h1 {{ font-size: 2.5em; margin-bottom: 10px; }}
            .header p {{ font-size: 1.2em; opacity: 0.9; }}
            .content {{ padding: 40px; }}
            .results-grid {{ 
                display: grid; 
                grid-template-columns: 1fr 1fr; 
                gap: 30px; 
                margin-bottom: 40px;
            }}
            .result-card {{ 
                background: #f8f9fa; 
                padding: 30px; 
                border-radius: 15px; 
                border-left: 5px solid;
                box-shadow: 0 5px 15px rgba(0,0,0,0.08);
                transition: transform 0.3s ease;
            }}
            .result-card:hover {{ transform: translateY(-5px); }}
            .traditional {{ border-left-color: {legacy_color}; }}
            .ml-prediction {{ border-left-color: {ml_color}; }}
            .card-title {{ 
                font-size: 1.5em; 
                margin-bottom: 20px; 
                color: #333;
                display: flex;
                align-items: center;
                gap: 10px;
            }}
            .score-display {{ 
                background: white; 
                padding: 20px; 
                border-radius: 10px; 
                margin: 20px 0;
                box-shadow: 0 2px 10px rgba(0,0,0,0.05);
            }}
            .score-bar {{ 
                background: #e9ecef; 
                height: 25px; 
                border-radius: 15px; 
                margin: 15px 0;
                overflow: hidden;
                position: relative;
            }}
            .score-fill {{ 
                height: 100%; 
                border-radius: 15px; 
                transition: width 1s ease;
                position: relative;
            }}
            .score-text {{ 
                position: absolute; 
                top: 50%; 
                left: 50%; 
                transform: translate(-50%, -50%);
                color: white;
                font-weight: bold;
                text-shadow: 1px 1px 2px rgba(0,0,0,0.5);
            }}
            .risk-level {{ 
                font-size: 1.3em; 
                font-weight: bold; 
                padding: 10px 20px;
                border-radius: 25px;
                display: inline-block;
                color: white;
                text-shadow: 1px 1px 2px rgba(0,0,0,0.3);
            }}
            .confidence-display {{
                background: white;
                padding: 20px;
                border-radius: 10px;
                margin: 20px 0;
                box-shadow: 0 2px 10px rgba(0,0,0,0.05);
            }}
            .confidence-bar {{
                background: #e9ecef;
                height: 25px;
                border-radius: 15px;
                margin: 15px 0;
                overflow: hidden;
                position: relative;
            }}
            .confidence-fill {{
                height: 100%;
                background: linear-gradient(90deg, #28a745, #20c997);
                border-radius: 15px;
                transition: width 1s ease;
                position: relative;
            }}
            .confidence-text {{
                position: absolute;
                top: 50%;
                left: 50%;
                transform: translate(-50%, -50%);
                color: white;
                font-weight: bold;
                text-shadow: 1px 1px 2px rgba(0,0,0,0.5);
            }}
            .charts-section {{
                background: #f8f9fa;
                padding: 30px;
                border-radius: 15px;
                margin: 30px 0;
            }}
            .chart-container {{
                display: grid;
                grid-template-columns: 1fr 1fr;
                gap: 30px;
                margin-top: 30px;
            }}
            .chart-box {{
                background: white;
                padding: 20px;
                border-radius: 10px;
                box-shadow: 0 2px 10px rgba(0,0,0,0.05);
            }}
            .terminal {{ 
                background: #1e1e1e; 
                color: #00ff00; 
                padding: 25px; 
                border-radius: 10px; 
                font-family: 'Courier New', monospace; 
                margin: 30px 0;
                box-shadow: 0 5px 15px rgba(0,0,0,0.2);
            }}
            .terminal h4 {{ color: #00ff00; margin-bottom: 15px; }}
            .terminal pre {{ white-space: pre-wrap; word-wrap: break-word; }}
            .action-buttons {{
                display: flex;
                gap: 20px;
                justify-content: center;
                margin: 40px 0;
            }}
            .btn {{
                padding: 15px 30px;
                border: none;
                border-radius: 25px;
                font-size: 1.1em;
                font-weight: bold;
                cursor: pointer;
                transition: all 0.3s ease;
                text-decoration: none;
                display: inline-block;
                text-align: center;
            }}
            .btn-primary {{
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
            }}
            .btn-secondary {{
                background: linear-gradient(135deg, #6c757d 0%, #495057 100%);
                color: white;
            }}
            .btn:hover {{
                transform: translateY(-2px);
                box-shadow: 0 5px 15px rgba(0,0,0,0.2);
            }}
            .ml-info {{ 
                background: #e3f2fd; 
                padding: 20px; 
                border-radius: 10px; 
                margin: 30px 0;
                text-align: center;
            }}
            .ml-info a {{
                color: #1976d2;
                text-decoration: none;
                font-weight: bold;
            }}
            .ml-info a:hover {{
                text-decoration: underline;
            }}
            .error {{ color: #dc3545; font-weight: bold; }}
            .success {{ color: #28a745; font-weight: bold; }}
            .info-card {{ background: #d1ecf1; padding: 15px; border-radius: 8px; margin: 10px 0; }}
            .error-card {{ background: #f8d7da; padding: 15px; border-radius: 8px; margin: 10px 0; }}
            .prediction-card {{ background: #d4edda; padding: 15px; border-radius: 8px; margin: 10px 0; }}
            @media (max-width: 768px) {{
                .results-grid, .chart-container {{ grid-template-columns: 1fr; }}
                .action-buttons {{ flex-direction: column; align-items: center; }}
                .header h1 {{ font-size: 2em; }}
                .content {{ padding: 20px; }}
            }}
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1>🎯 Assessment Results</h1>
                <p>Digital Media & Mental Health Risk Analysis</p>
            </div>
            
            <div class="content">
                <!-- AI Prediction Only -->
                <div class="result-card ml-prediction" style="max-width: 600px; margin: 0 auto;">
                    <div class="card-title">
                        🤖 AI Risk Assessment
                    </div>
                    {ml_display}
                </div>
                
                <!-- Charts Section -->
                <div class="charts-section">
                    <h3 style="text-align: center; margin-bottom: 20px; color: #333;">📈 AI Analysis & Dataset Comparison</h3>
                    <div class="chart-container" style="grid-template-columns: 1fr 1fr; gap: 30px; margin-bottom: 30px;">
                        <div class="chart-box">
                            <h4 style="text-align: center; margin-bottom: 15px;">🎯 Risk Level Distribution</h4>
                            <canvas id="riskChart" width="400" height="300"></canvas>
                        </div>
                        <div class="chart-box">
                            <h4 style="text-align: center; margin-bottom: 15px;">🤖 AI Confidence Analysis</h4>
                            <canvas id="confidenceChart" width="400" height="300"></canvas>
                        </div>
                    </div>
                    <div class="chart-container" style="grid-template-columns: 1fr 1fr; gap: 30px;">
                        <div class="chart-box">
                            <h4 style="text-align: center; margin-bottom: 15px;">📊 Score Distribution</h4>
                            <canvas id="scoreChart" width="400" height="300"></canvas>
                        </div>
                        <div class="chart-box">
                            <h4 style="text-align: center; margin-bottom: 15px;">📈 Risk Trend Analysis</h4>
                            <canvas id="trendChart" width="400" height="300"></canvas>
                        </div>
                    </div>
                </div>
                
                <!-- Action Buttons -->
                <div class="action-buttons">
                    <a href="/" class="btn btn-primary">🔄 Take New Assessment</a>
                    <button onclick="goBack()" class="btn btn-secondary">← Back to Previous Choices</button>
                </div>
                
                
                <!-- Additional Info -->
                <div class="ml-info">
                    <h3>📋 Additional Information</h3>
                    <p><a href="/ml-info">View ML Model Information & Performance</a></p>
                </div>
            </div>
        </div>
        
        <script>
            // Chart 1: Risk Level Distribution (Doughnut Chart)
            const riskCtx = document.getElementById('riskChart').getContext('2d');
            {f'''
            const riskDistribution = {dataset_analysis['risk_distribution']};
            const riskLabels = Object.keys(riskDistribution);
            const riskData = Object.values(riskDistribution);
            const riskColors = ['#28a745', '#ffc107', '#fd7e14', '#dc3545'];
            
            new Chart(riskCtx, {{
                type: 'doughnut',
                data: {{
                    labels: riskLabels,
                    datasets: [{{
                        data: riskData,
                        backgroundColor: riskColors.slice(0, riskLabels.length),
                        borderWidth: 3,
                        borderColor: '#fff'
                    }}]
                }},
                options: {{
                    responsive: true,
                    animation: {{
                        animateRotate: true,
                        animateScale: true,
                        duration: 2000
                    }},
                    plugins: {{
                        legend: {{
                            position: 'bottom',
                            labels: {{
                                usePointStyle: true,
                                padding: 20
                            }}
                        }}
                    }}
                }}
            }});
            ''' if dataset_analysis else '''
            const riskLevels = ['Low Risk', 'At-Risk', 'Problematic', 'High Risk'];
            const riskColors = ['#28a745', '#ffc107', '#fd7e14', '#dc3545'];
            
            new Chart(riskCtx, {{
                type: 'doughnut',
                data: {{
                    labels: riskLevels,
                    datasets: [{{
                        data: [25, 25, 25, 25],
                        backgroundColor: riskColors,
                        borderWidth: 3,
                        borderColor: '#fff'
                    }}]
                }},
                options: {{
                    responsive: true,
                    animation: {{
                        animateRotate: true,
                        animateScale: true,
                        duration: 2000
                    }},
                    plugins: {{
                        legend: {{
                            position: 'bottom',
                            labels: {{
                                usePointStyle: true,
                                padding: 20
                            }}
                        }}
                    }}
                }}
            }});
            '''}
            
            // Chart 2: AI Confidence Analysis (Polar Area Chart)
            const confidenceCtx = document.getElementById('confidenceChart').getContext('2d');
            {f'''
            const confidence = {ml_prediction.get('confidence', 0) if ml_prediction and ml_prediction.get('success', False) else 0};
            const confidenceData = [confidence * 100, (1 - confidence) * 100];
            
            new Chart(confidenceCtx, {{
                type: 'polarArea',
                data: {{
                    labels: ['AI Confidence', 'Uncertainty'],
                    datasets: [{{
                        data: confidenceData,
                        backgroundColor: ['#28a745', '#e9ecef'],
                        borderWidth: 2,
                        borderColor: '#fff'
                    }}]
                }},
                options: {{
                    responsive: true,
                    animation: {{
                        animateRotate: true,
                        animateScale: true,
                        duration: 2000
                    }},
                    plugins: {{
                        legend: {{
                            position: 'bottom',
                            labels: {{
                                usePointStyle: true,
                                padding: 20
                            }}
                        }}
                    }}
                }}
            }});
            ''' if ml_prediction and ml_prediction.get('success', False) else '''
            new Chart(confidenceCtx, {{
                type: 'polarArea',
                data: {{
                    labels: ['No AI Data'],
                    datasets: [{{
                        data: [100],
                        backgroundColor: ['#6c757d'],
                        borderWidth: 2,
                        borderColor: '#fff'
                    }}]
                }},
                options: {{
                    responsive: true,
                    animation: {{
                        animateRotate: true,
                        animateScale: true,
                        duration: 2000
                    }},
                    plugins: {{
                        legend: {{
                            position: 'bottom',
                            labels: {{
                                usePointStyle: true,
                                padding: 20
                            }}
                        }}
                    }}
                }}
            }});
            '''}
            
            // Chart 3: Score Distribution (Bar Chart with Animation)
            const scoreCtx = document.getElementById('scoreChart').getContext('2d');
            {f'''
            new Chart(scoreCtx, {{
                type: 'bar',
                data: {{
                    labels: ['Your Score', 'Dataset Average', 'Dataset Min', 'Dataset Max'],
                    datasets: [{{
                        label: 'Score Comparison',
                        data: [{total_score}, {dataset_analysis['mean_score']:.1f}, {dataset_analysis['min_score']}, {dataset_analysis['max_score']}],
                        backgroundColor: ['{legacy_color}', '#007bff', '#6c757d', '#6c757d'],
                        borderWidth: 2,
                        borderColor: '#fff',
                        borderRadius: 8,
                        borderSkipped: false
                    }}]
                }},
                options: {{
                    responsive: true,
                    animation: {{
                        duration: 2000,
                        easing: 'easeInOutQuart'
                    }},
                    plugins: {{
                        legend: {{
                            display: false
                        }}
                    }},
                    scales: {{
                        y: {{
                            beginAtZero: true,
                            title: {{
                                display: true,
                                text: 'Score',
                                font: {{
                                    size: 14,
                                    weight: 'bold'
                                }}
                            }},
                            grid: {{
                                color: 'rgba(0,0,0,0.1)'
                            }}
                        }},
                        x: {{
                            grid: {{
                                display: false
                            }}
                        }}
                    }}
                }}
            }});
            ''' if dataset_analysis else '''
            new Chart(scoreCtx, {{
                type: 'bar',
                data: {{
                    labels: ['Your Score', 'Max Possible'],
                    datasets: [{{
                        label: 'Score',
                        data: [{total_score}, {max_score}],
                        backgroundColor: ['{legacy_color}', '#e9ecef'],
                        borderWidth: 2,
                        borderColor: '#fff',
                        borderRadius: 8,
                        borderSkipped: false
                    }}]
                }},
                options: {{
                    responsive: true,
                    animation: {{
                        duration: 2000,
                        easing: 'easeInOutQuart'
                    }},
                    plugins: {{
                        legend: {{
                            display: false
                        }}
                    }},
                    scales: {{
                        y: {{
                            beginAtZero: true,
                            title: {{
                                display: true,
                                text: 'Score',
                                font: {{
                                    size: 14,
                                    weight: 'bold'
                                }}
                            }},
                            grid: {{
                                color: 'rgba(0,0,0,0.1)'
                            }}
                        }},
                        x: {{
                            grid: {{
                                display: false
                            }}
                        }}
                    }}
                }}
            }});
            '''}
            
            // Chart 4: Risk Trend Analysis (Line Chart with Animation)
            const trendCtx = document.getElementById('trendChart').getContext('2d');
            {f'''
            const riskTrendData = {{
                labels: ['Low Risk', 'At-Risk', 'Problematic', 'High Risk'],
                datasets: [{{
                    label: 'Risk Distribution',
                    data: Object.values({dataset_analysis['risk_distribution']}),
                    borderColor: '#007bff',
                    backgroundColor: 'rgba(0, 123, 255, 0.1)',
                    borderWidth: 3,
                    fill: true,
                    tension: 0.4,
                    pointBackgroundColor: '#007bff',
                    pointBorderColor: '#fff',
                    pointBorderWidth: 2,
                    pointRadius: 6
                }}]
            }};
            
            new Chart(trendCtx, {{
                type: 'line',
                data: riskTrendData,
                options: {{
                    responsive: true,
                    animation: {{
                        duration: 2000,
                        easing: 'easeInOutQuart'
                    }},
                    plugins: {{
                        legend: {{
                            display: false
                        }}
                    }},
                    scales: {{
                        y: {{
                            beginAtZero: true,
                            title: {{
                                display: true,
                                text: 'Number of People',
                                font: {{
                                    size: 14,
                                    weight: 'bold'
                                }}
                            }},
                            grid: {{
                                color: 'rgba(0,0,0,0.1)'
                            }}
                        }},
                        x: {{
                            title: {{
                                display: true,
                                text: 'Risk Levels',
                                font: {{
                                    size: 14,
                                    weight: 'bold'
                                }}
                            }},
                            grid: {{
                                display: false
                            }}
                        }}
                    }}
                }}
            }});
            ''' if dataset_analysis else '''
            new Chart(trendCtx, {{
                type: 'line',
                data: {{
                    labels: ['Low Risk', 'At-Risk', 'Problematic', 'High Risk'],
                    datasets: [{{
                        label: 'Risk Distribution',
                        data: [25, 25, 25, 25],
                        borderColor: '#007bff',
                        backgroundColor: 'rgba(0, 123, 255, 0.1)',
                        borderWidth: 3,
                        fill: true,
                        tension: 0.4,
                        pointBackgroundColor: '#007bff',
                        pointBorderColor: '#fff',
                        pointBorderWidth: 2,
                        pointRadius: 6
                    }}]
                }},
                options: {{
                    responsive: true,
                    animation: {{
                        duration: 2000,
                        easing: 'easeInOutQuart'
                    }},
                    plugins: {{
                        legend: {{
                            display: false
                        }}
                    }},
                    scales: {{
                        y: {{
                            beginAtZero: true,
                            title: {{
                                display: true,
                                text: 'Number of People',
                                font: {{
                                    size: 14,
                                    weight: 'bold'
                                }}
                            }},
                            grid: {{
                                color: 'rgba(0,0,0,0.1)'
                            }}
                        }},
                        x: {{
                            title: {{
                                display: true,
                                text: 'Risk Levels',
                                font: {{
                                    size: 14,
                                    weight: 'bold'
                                }}
                            }},
                            grid: {{
                                display: false
                            }}
                        }}
                    }}
                }}
            }});
            '''}
            
            // Go back function
            function goBack() {{
                if (window.history.length > 1) {{
                    window.history.back();
                }} else {{
                    window.location.href = '/';
                }}
            }}
            
            // Animate progress bars on load
            window.addEventListener('load', function() {{
                setTimeout(function() {{
                    document.querySelectorAll('.score-fill, .confidence-fill').forEach(function(bar) {{
                        bar.style.transition = 'width 2s ease';
                    }});
                }}, 500);
            }});
        </script>
    </body>
    </html>
    """

@app.route('/ml-info')
def ml_info():
    """Display ML model information from artifacts, reports, and visualizations"""
    try:
        if os.path.exists('ml_model/artifacts/best_model.pkl'):
            # Load model information from artifacts
            predictor = get_predictor()
            model_info = predictor.get_model_info()
            feature_importance = predictor.get_feature_importance()
            
            # Read evaluation report
            report_content = ""
            if os.path.exists('ml_model/reports/model_evaluation_report.txt'):
                with open('ml_model/reports/model_evaluation_report.txt', 'r', encoding='utf-8') as f:
                    report_content = f.read()
            
            # Check available visualizations
            visualizations = []
            viz_dir = 'ml_model/visualizations'
            if os.path.exists(viz_dir):
                viz_files = [f for f in os.listdir(viz_dir) if f.endswith('.png')]
                visualizations = viz_files
            
            # Create feature importance HTML
            importance_html = ""
            if feature_importance:
                sorted_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
                importance_html = "<h4>🔍 Top 10 Most Important Features:</h4><ul>"
                for feature, importance in sorted_features[:10]:
                    importance_html += f"<li><strong>{feature}</strong>: {importance:.4f}</li>"
                importance_html += "</ul>"
            
            # Create visualizations HTML
            viz_html = ""
            if visualizations:
                viz_html = "<h4>📊 Available Visualizations:</h4><div style='display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 15px; margin: 20px 0;'>"
                for viz in visualizations:
                    viz_path = f"ml_model/visualizations/{viz}"
                    viz_name = viz.replace('.png', '').replace('_', ' ').title()
                    viz_html += f"""
                    <div style='background: #f8f9fa; padding: 15px; border-radius: 8px; text-align: center;'>
                        <h5>{viz_name}</h5>
                        <img src='/{viz_path}' style='max-width: 100%; height: auto; border-radius: 5px;' alt='{viz_name}'>
                    </div>
                    """
                viz_html += "</div>"
            
            return f"""
            <html>
            <head>
                <title>ML Model Information & Performance</title>
                <style>
                    * {{ margin: 0; padding: 0; box-sizing: border-box; }}
                    body {{ 
                        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; 
                        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                        min-height: 100vh;
                        padding: 20px;
                    }}
                    .container {{ 
                        max-width: 1200px; 
                        margin: 0 auto; 
                        background: white; 
                        border-radius: 20px; 
                        box-shadow: 0 20px 40px rgba(0,0,0,0.1);
                        overflow: hidden;
                    }}
                    .header {{ 
                        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                        color: white; 
                        padding: 40px; 
                        text-align: center;
                    }}
                    .header h1 {{ font-size: 2.5em; margin-bottom: 10px; }}
                    .content {{ padding: 40px; }}
                    .model-card {{ 
                        background: #e8f5e8; 
                        padding: 25px; 
                        border-radius: 15px; 
                        margin: 20px 0;
                        border-left: 5px solid #28a745;
                        box-shadow: 0 5px 15px rgba(0,0,0,0.08);
                    }}
                    .feature-list {{ 
                        background: #f0f8ff; 
                        padding: 20px; 
                        border-radius: 15px; 
                        margin: 20px 0;
                        border-left: 5px solid #007bff;
                        box-shadow: 0 5px 15px rgba(0,0,0,0.08);
                    }}
                    .report-section {{
                        background: #fff3cd;
                        padding: 20px;
                        border-radius: 15px;
                        margin: 20px 0;
                        border-left: 5px solid #ffc107;
                        box-shadow: 0 5px 15px rgba(0,0,0,0.08);
                    }}
                    .viz-section {{
                        background: #d1ecf1;
                        padding: 20px;
                        border-radius: 15px;
                        margin: 20px 0;
                        border-left: 5px solid #17a2b8;
                        box-shadow: 0 5px 15px rgba(0,0,0,0.08);
                    }}
                    .success {{ color: #28a745; font-weight: bold; }}
                    .error {{ color: #dc3545; font-weight: bold; }}
                    .btn {{
                        display: inline-block;
                        padding: 12px 25px;
                        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                        color: white;
                        text-decoration: none;
                        border-radius: 25px;
                        font-weight: bold;
                        margin: 10px 5px;
                        transition: transform 0.3s ease;
                    }}
                    .btn:hover {{
                        transform: translateY(-2px);
                        box-shadow: 0 5px 15px rgba(0,0,0,0.2);
                    }}
                    .metric-grid {{
                        display: grid;
                        grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
                        gap: 15px;
                        margin: 20px 0;
                    }}
                    .metric-card {{
                        background: white;
                        padding: 20px;
                        border-radius: 10px;
                        text-align: center;
                        box-shadow: 0 2px 10px rgba(0,0,0,0.05);
                    }}
                    .metric-value {{
                        font-size: 2em;
                        font-weight: bold;
                        color: #007bff;
                    }}
                    pre {{
                        background: #f8f9fa;
                        padding: 15px;
                        border-radius: 8px;
                        overflow-x: auto;
                        white-space: pre-wrap;
                        word-wrap: break-word;
                    }}
                    @media (max-width: 768px) {{
                        .header h1 {{ font-size: 2em; }}
                        .content {{ padding: 20px; }}
                        .metric-grid {{ grid-template-columns: 1fr; }}
                    }}
                </style>
            </head>
            <body>
                <div class="container">
                    <div class="header">
                        <h1>🤖 ML Model Information & Performance</h1>
                        <p>Comprehensive Analysis from Training Artifacts</p>
                    </div>
                    
                    <div class="content">
                        <div class="model-card">
                            <h2>📊 Model Performance Metrics</h2>
                            <div class="metric-grid">
                                <div class="metric-card">
                                    <div class="metric-value">{model_info['accuracy']:.4f}</div>
                                    <div>Accuracy</div>
                                </div>
                                <div class="metric-card">
                                    <div class="metric-value">{model_info['f1_score']:.4f}</div>
                                    <div>F1-Score</div>
                                </div>
                                <div class="metric-card">
                                    <div class="metric-value">{len(model_info['feature_columns'])}</div>
                                    <div>Features</div>
                                </div>
                                <div class="metric-card">
                                    <div class="metric-value">{len(model_info['target_classes'])}</div>
                                    <div>Classes</div>
                                </div>
                            </div>
                            <p><strong>Model Name:</strong> {model_info['best_model_name']}</p>
                            <p><strong>Target Classes:</strong> {', '.join(model_info['target_classes'])}</p>
                        </div>
                        
                        <div class="feature-list">
                            <h3>📋 Features Used in Model</h3>
                            <p>{', '.join(model_info['feature_columns'])}</p>
                        </div>
                        
                        <div class="feature-list">
                            {importance_html}
                        </div>
                        
                        {f'''
                        <div class="report-section">
                            <h3>📄 Model Evaluation Report</h3>
                            <pre>{report_content}</pre>
                        </div>
                        ''' if report_content else ''}
                        
                        {viz_html if viz_html else ''}
                        
                        <div style="text-align: center; margin: 40px 0;">
                            <a href="/" class="btn">← Back to Questionnaire</a>
                            <a href="/results" class="btn">View Results</a>
                        </div>
                    </div>
                </div>
            </body>
            </html>
            """
        else:
            return """
            <html>
            <head>
                <title>ML Model Information</title>
                <style>
                    body { 
                        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; 
                        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                        min-height: 100vh;
                        padding: 20px;
                    }
                    .container { 
                        max-width: 800px; 
                        margin: 0 auto; 
                        background: white; 
                        border-radius: 20px; 
                        box-shadow: 0 20px 40px rgba(0,0,0,0.1);
                        padding: 40px;
                        text-align: center;
                    }
                    .error { color: #dc3545; font-weight: bold; font-size: 1.2em; }
                    .btn {
                        display: inline-block;
                        padding: 12px 25px;
                        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                        color: white;
                        text-decoration: none;
                        border-radius: 25px;
                        font-weight: bold;
                        margin: 10px 5px;
                    }
                </style>
            </head>
            <body>
                <div class="container">
                    <h1>🤖 Machine Learning Model Information</h1>
                    <p class="error">ML model not found!</p>
                    <p>Please run <code>python ml_model/ml_training.py</code> first to train the model.</p>
                    <a href="/" class="btn">← Back to Questionnaire</a>
                </div>
            </body>
            </html>
            """
    except Exception as e:
        return f"""
        <html>
        <head>
            <title>ML Model Information</title>
            <style>
                body {{ 
                    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; 
                    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                    min-height: 100vh;
                    padding: 20px;
                }}
                .container {{ 
                    max-width: 800px; 
                    margin: 0 auto; 
                    background: white; 
                    border-radius: 20px; 
                    box-shadow: 0 20px 40px rgba(0,0,0,0.1);
                    padding: 40px;
                    text-align: center;
                }}
                .error {{ color: #dc3545; font-weight: bold; font-size: 1.2em; }}
                .btn {{
                    display: inline-block;
                    padding: 12px 25px;
                    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                    color: white;
                    text-decoration: none;
                    border-radius: 25px;
                    font-weight: bold;
                    margin: 10px 5px;
                }}
            </style>
        </head>
        <body>
            <div class="container">
                <h1>🤖 Machine Learning Model Information</h1>
                <p class="error">Error loading ML model: {str(e)}</p>
                <a href="/" class="btn">← Back to Questionnaire</a>
            </div>
        </body>
        </html>
        """

@app.route('/ml_model/visualizations/<filename>')
def serve_visualization(filename):
    """Serve visualization images from the ml_model/visualizations directory"""
    return send_from_directory('ml_model/visualizations', filename)

if __name__ == '__main__':
    print("Starting Flask application...")
    print("Open your browser and go to: http://localhost:5000")
    print("Submit the questionnaire to see the Python Flask terminal output!")
    app.run(debug=True, host='0.0.0.0', port=5000)
