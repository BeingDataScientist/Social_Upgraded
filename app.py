from flask import Flask, render_template, request, jsonify, session, redirect, url_for, send_from_directory, flash
import json
import os
import pandas as pd
import numpy as np
from datetime import datetime, date
from ml_model.ml_predictor import predict_risk_level, format_questionnaire_data, get_predictor
from openai_analyzer import format_questionnaire_for_openai, analyze_with_openai
from database import db, Doctor, Patient, PatientLog, init_db
from auth import hash_password, verify_password, login_required, get_current_doctor

app = Flask(__name__)
app.secret_key = 'your-secret-key-here-change-in-production'  # Change this in production
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///health_assessment.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

# Initialize database
init_db(app)

@app.route('/')
def index():
    """Redirect to login if not logged in, otherwise dashboard"""
    if 'doctor_id' in session:
        return redirect(url_for('dashboard'))
    return redirect(url_for('login'))

@app.route('/login', methods=['GET', 'POST'])
def login():
    """Login page"""
    if request.method == 'POST':
        email = request.form.get('email')
        password = request.form.get('password')
        
        doctor = Doctor.query.filter_by(email=email).first()
        
        if doctor and verify_password(doctor.password, password):
            session['doctor_id'] = doctor.id
            session['doctor_name'] = doctor.name
            session['doctor_profession'] = doctor.profession
            flash('Login successful!', 'success')
            return redirect(url_for('dashboard'))
        else:
            flash('Invalid email or password', 'error')
    
    return render_template('login.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    """Registration page"""
    if request.method == 'POST':
        email = request.form.get('email')
        password = request.form.get('password')
        name = request.form.get('name')
        dob = request.form.get('dob')
        profession = request.form.get('profession')
        address = request.form.get('address')
        
        # Check if email already exists
        if Doctor.query.filter_by(email=email).first():
            flash('Email already registered', 'error')
            return render_template('register.html')
        
        # Create new doctor
        try:
            dob_date = datetime.strptime(dob, '%Y-%m-%d').date()
            new_doctor = Doctor(
                email=email,
                password=hash_password(password),
                name=name,
                dob=dob_date,
                profession=profession,
                address=address
            )
            db.session.add(new_doctor)
            db.session.commit()
            flash('Registration successful! Please login.', 'success')
            return redirect(url_for('login'))
        except Exception as e:
            db.session.rollback()
            flash(f'Registration failed: {str(e)}', 'error')
    
    return render_template('register.html')

@app.route('/logout')
def logout():
    """Logout"""
    session.clear()
    flash('Logged out successfully', 'success')
    return redirect(url_for('login'))

@app.route('/dashboard')
@login_required
def dashboard():
    """Doctor dashboard"""
    doctor = Doctor.query.get(session['doctor_id'])
    return render_template('dashboard.html', 
                         doctor_name=doctor.name,
                         doctor_profession=doctor.profession)

@app.route('/add-patient', methods=['GET', 'POST'])
@login_required
def add_patient():
    """Add new patient"""
    if request.method == 'POST':
        name = request.form.get('name')
        age = int(request.form.get('age'))
        gender = request.form.get('gender')
        previous_history = request.form.get('previous_history', '')
        
        new_patient = Patient(
            name=name,
            age=age,
            gender=gender,
            previous_history=previous_history,
            doctor_id=session['doctor_id']
        )
        db.session.add(new_patient)
        db.session.commit()
        flash(f'Patient {name} added successfully!', 'success')
        return redirect(url_for('add_patient'))
    
    return render_template('add_patient.html')

@app.route('/assess-patient', methods=['GET', 'POST'])
@login_required
def assess_patient():
    """Assess existing patient"""
    doctor_id = session['doctor_id']
    patients = Patient.query.filter_by(doctor_id=doctor_id).all()
    
    if request.method == 'POST':
        patient_id = request.form.get('patient_id')
        if patient_id:
            session['current_patient_id'] = patient_id
            return redirect(url_for('questionnaire'))
        else:
            flash('Please select a patient', 'error')
    
    return render_template('assess_patient.html', patients=patients)

@app.route('/questionnaire')
@login_required
def questionnaire():
    """Serve the questionnaire page"""
    if 'current_patient_id' not in session:
        flash('Please select a patient first', 'error')
        return redirect(url_for('assess_patient'))
    
    patient = Patient.query.get(session['current_patient_id'])
    if not patient:
        flash('Patient not found', 'error')
        return redirect(url_for('assess_patient'))
    
    return render_template('questionnaire.html', patient=patient)

@app.route('/submit', methods=['POST'])
@login_required
def submit_questionnaire():
    """Handle form submission and display results"""
    try:
        # Check if patient is selected
        if 'current_patient_id' not in session:
            return jsonify({
                'success': False,
                'error': 'No patient selected'
            }), 400
        
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
        # Check for new model location first, then fallback to old location
        model_path_new = 'ml_model/ml_model/best_model_ann.pkl'
        model_path_old = 'ml_model/artifacts/best_model.pkl'
        if os.path.exists(model_path_new) or os.path.exists(model_path_old):
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
            print("⚠️  ML model not found. Run ml_model/train_models.py first to train the model.")
        
        # AI Analysis
        openai_analysis_result = None
        openai_category = None
        try:
            # Format questionnaire data for AI analysis (with questions and actual choice values)
            formatted_questionnaire = format_questionnaire_for_openai(form_data)
            
            print("\n" + "="*50)
            print("SENDING FOR AI ANALYSIS...")
            print("="*50)
            
            # Get AI analysis
            openai_analysis_result = analyze_with_openai(formatted_questionnaire)
            
            if openai_analysis_result and openai_analysis_result.get('success'):
                openai_category = openai_analysis_result.get('risk_category', 'Unknown')
                print(f"✅ AI Analysis:")
                print(f"   Risk Category: {openai_category}")
                print(f"   Solutions: {len(openai_analysis_result.get('solutions', []))} provided")
                print(f"   Suggestions: {len(openai_analysis_result.get('suggestions', []))} provided")
            else:
                error_msg = openai_analysis_result.get('error', 'Sodium Level Disorder trained model API error.') if openai_analysis_result else 'Sodium Level Disorder trained model API error.'
                # Sanitize error message - remove OpenAI URLs, references, and HTTP error codes
                if any(keyword in error_msg.lower() for keyword in ['openai.com', 'platform.openai', 'api key', 'invalid_api_key', '401', '403', '429', 'http', 'https', 'error code', 'incorrect api']):
                    error_msg = 'Sodium Level Disorder trained model API error.'
                print(f"⚠️  AI Analysis Error: {error_msg}")
                
        except Exception as openai_error:
            error_str = str(openai_error)
            # Sanitize error message - remove OpenAI URLs, references, HTTP codes, and any OpenAI-related content
            if any(keyword in error_str.lower() for keyword in ['openai.com', 'platform.openai', 'api key', 'invalid_api_key', '401', '403', '429', 'http', 'https', 'error code', 'incorrect api', 'account/api-keys']):
                error_str = 'Sodium Level Disorder trained model API error.'
            print(f"⚠️  AI Analysis Error: {error_str}")
            openai_analysis_result = {
                'success': False,
                'error': error_str
            }
        
        # Save to database
        try:
            patient_log = PatientLog(
                patient_id=session['current_patient_id'],
                doctor_id=session['doctor_id'],
                responses=json.dumps(python_output),
                openai_category=openai_category,
                total_score=total_score,
                timestamp=datetime.utcnow()
            )
            db.session.add(patient_log)
            db.session.commit()
            print(f"✅ Assessment saved to database for patient {session['current_patient_id']}")
        except Exception as db_error:
            db.session.rollback()
            print(f"⚠️  Database save error: {str(db_error)}")
        
        # Store results in session for display on results page
        session['results'] = {
            'python_output': python_output,
            'total_score': total_score,
            'max_score': 68,
            'risk_level_legacy': risk_level_legacy,
            'ml_prediction': ml_prediction_result,
            'openai_analysis': openai_analysis_result
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
@login_required
def show_results():
    """Display results page"""
    # Get results from session
    results = session.get('results')
    
    if not results:
        flash('No assessment results found', 'error')
        return redirect(url_for('dashboard'))
    
    # Get current patient ID from session
    patient_id = session.get('current_patient_id')
    if not patient_id:
        flash('No patient selected', 'error')
        return redirect(url_for('dashboard'))
    
    # Get patient information
    patient = Patient.query.get(patient_id)
    if not patient:
        flash('Patient not found', 'error')
        return redirect(url_for('dashboard'))
    
    # Get all previous assessments for this patient
    all_assessments = PatientLog.query.filter_by(
        patient_id=patient_id,
        doctor_id=session['doctor_id']
    ).order_by(PatientLog.timestamp.desc()).all()
    
    # Extract data from session
    python_output = results.get('python_output', {})
    total_score = results.get('total_score', 0)
    max_score = results.get('max_score', 68)
    risk_level_legacy = results.get('risk_level_legacy', 'Unknown')
    ml_prediction = results.get('ml_prediction', {})
    openai_analysis = results.get('openai_analysis', {})
    
    # Prepare data for charts
    assessment_dates = []
    assessment_scores = []
    assessment_categories = []
    
    for assessment in all_assessments:
        assessment_dates.append(assessment.timestamp.strftime('%Y-%m-%d'))
        assessment_scores.append(assessment.total_score if assessment.total_score else 0)
        assessment_categories.append(assessment.openai_category if assessment.openai_category else 'Unknown')
    
    # Reverse to show chronological order (oldest to newest)
    assessment_dates = list(reversed(assessment_dates))
    assessment_scores = list(reversed(assessment_scores))
    assessment_categories = list(reversed(assessment_categories))
    
    # Calculate performance trend (increasing/decreasing)
    # Positive trend = improving (score decreasing), Negative trend = worsening (score increasing)
    performance_trend = []
    if len(assessment_scores) > 1:
        for i in range(len(assessment_scores)):
            if i == 0:
                performance_trend.append(0)  # First assessment - no comparison
            else:
                # Lower score is better, so if current score < previous score, that's improvement (positive)
                # If current score > previous score, that's worsening (negative)
                trend = assessment_scores[i-1] - assessment_scores[i]
                performance_trend.append(trend)
    else:
        performance_trend = [0] if assessment_scores else []
    
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
    openai_color = "#9b59b6"  # Purple color for AI analysis
    
    # Map AI risk categories to colors
    openai_risk_colors = {
        "Low risk": "#28a745",
        "At-Risk": "#ffc107",
        "Problematic use likely": "#fd7e14",
        "High Risk/ addictive pattern": "#dc3545"
    }
    
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
    
    # Create AI analysis display
    openai_display = ""
    if openai_analysis and openai_analysis.get('success', False):
        risk_category = openai_analysis.get('risk_category', 'Unknown')
        solutions = openai_analysis.get('solutions', [])
        suggestions = openai_analysis.get('suggestions', [])
        risk_color = openai_risk_colors.get(risk_category, openai_color)
        
        solutions_html = ""
        if solutions:
            solutions_html = "<ul style='margin: 10px 0; padding-left: 20px;'>"
            for solution in solutions:
                solutions_html += f"<li style='margin: 8px 0;'>{solution}</li>"
            solutions_html += "</ul>"
        
        suggestions_html = ""
        if suggestions:
            suggestions_html = "<ul style='margin: 10px 0; padding-left: 20px;'>"
            for suggestion in suggestions:
                suggestions_html += f"<li style='margin: 8px 0;'>{suggestion}</li>"
            suggestions_html += "</ul>"
        
        openai_display = f"""
        <div class="result-card openai-analysis" style="border-left-color: {risk_color}; margin-bottom: 30px;">
            <div class="card-title">
                <span style="font-size: 1.2em;">🤖</span>
                AI Analysis
            </div>
            <div style="margin-top: 20px;">
                <p style="font-size: 1.1em; margin-bottom: 15px;"><strong>Risk Category:</strong></p>
                <div class="risk-level" style="background: {risk_color}; margin: 10px 0; padding: 15px; border-radius: 8px; color: white; font-weight: bold; text-align: center; font-size: 1.1em;">
                    {risk_category}
                </div>
                
                <div style="margin-top: 25px;">
                    <p style="font-size: 1.1em; margin-bottom: 10px;"><strong>💡 Solutions:</strong></p>
                    {solutions_html if solutions_html else "<p style='color: #666;'>No solutions provided.</p>"}
                </div>
                
                <div style="margin-top: 25px;">
                    <p style="font-size: 1.1em; margin-bottom: 10px;"><strong>✨ Suggestions:</strong></p>
                    {suggestions_html if suggestions_html else "<p style='color: #666;'>No suggestions provided.</p>"}
                </div>
            </div>
        </div>
        """
    elif openai_analysis and not openai_analysis.get('success', False):
        error_msg = openai_analysis.get('error', 'Sodium Level Disorder trained model API error.')
        # Sanitize error message - remove OpenAI URLs, references, and HTTP error codes
        if any(keyword in error_msg.lower() for keyword in ['openai.com', 'platform.openai', 'api key', 'invalid_api_key', '401', '403', '429', 'http', 'https', 'error code', 'incorrect api']):
            error_msg = 'Sodium Level Disorder trained model API error.'
        openai_display = f"""
        <div class="result-card openai-analysis" style="border-left-color: #dc3545; margin-bottom: 30px;">
            <div class="card-title">
                <span style="font-size: 1.2em;">🤖</span>
                AI Analysis
            </div>
            <div class="error-card" style="margin-top: 20px;">
                <p class="error">AI Analysis Error: {error_msg}</p>
                <p style="margin-top: 10px; color: #666; font-size: 0.9em;">
                    The trained model API is currently unavailable. Please try again later.
                </p>
            </div>
        </div>
        """
    else:
        openai_display = """
        <div class="result-card openai-analysis" style="border-left-color: #6c757d; margin-bottom: 30px;">
            <div class="card-title">
                <span style="font-size: 1.2em;">🤖</span>
                AI Analysis
            </div>
            <div class="info-card" style="margin-top: 20px;">
                <p>AI analysis not available.</p>
            </div>
        </div>
        """
    
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
                .results-grid {{ grid-template-columns: 1fr; }}
                .action-buttons {{ flex-direction: column; align-items: center; }}
                .header h1 {{ font-size: 2em; }}
                .content {{ padding: 20px; }}
            }}
        </style>
        <link rel="stylesheet" href="{{ url_for('static', filename='css/animations.css') }}">
    </head>
    <body style="position: relative;">
        <!-- Animated Background Elements -->
        <div class="animated-background">
            <div class="particle"></div>
            <div class="particle"></div>
            <div class="particle"></div>
            <div class="particle"></div>
            <div class="pulse-circle"></div>
            <div class="pulse-circle"></div>
            <div class="wave"></div>
            <div class="neural-node"></div>
            <div class="neural-node"></div>
            <div class="neural-node"></div>
            <div class="thought-bubble"></div>
            <div class="breathe-circle"></div>
            <div class="network-line"></div>
            <div class="network-line"></div>
        </div>
        <div class="container">
            <div class="header">
                <h1>🎯 Assessment Results</h1>
                <p>Digital Media & Mental Health Risk Analysis</p>
            </div>
            
            <div class="content">
                <!-- Patient Info -->
                <div class="patient-header" style="background: #f8f9fa; padding: 20px; border-radius: 10px; margin-bottom: 30px;">
                    <h2 style="color: #333; margin-bottom: 10px;">Patient: {patient.name}</h2>
                    <p style="color: #666;">Age: {patient.age} | Gender: {patient.gender}</p>
                </div>
                
                <!-- AI Analysis Section -->
                {openai_display}
                
                <!-- Patient History Table -->
                <div class="section" style="margin-top: 40px;">
                    <h2 style="color: #333; margin-bottom: 20px; font-size: 1.8em;">📋 Previous Assessment History</h2>
                    <div style="overflow-x: auto;">
                        <table style="width: 100%; border-collapse: collapse; background: white; border-radius: 10px; overflow: hidden; box-shadow: 0 2px 10px rgba(0,0,0,0.1);">
                            <thead>
                                <tr style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white;">
                                    <th style="padding: 15px; text-align: left; border-bottom: 2px solid #ddd;">Date</th>
                                    <th style="padding: 15px; text-align: left; border-bottom: 2px solid #ddd;">Score</th>
                                    <th style="padding: 15px; text-align: left; border-bottom: 2px solid #ddd;">Risk Category</th>
                                </tr>
                            </thead>
                            <tbody>
                                {' '.join([f'''
                                <tr style="border-bottom: 1px solid #eee;">
                                    <td style="padding: 12px;">{assessment.timestamp.strftime('%Y-%m-%d %H:%M')}</td>
                                    <td style="padding: 12px;"><strong>{assessment.total_score if assessment.total_score else 'N/A'}</strong> / {max_score}</td>
                                    <td style="padding: 12px;">
                                        <span style="padding: 5px 10px; border-radius: 5px; background: {openai_risk_colors.get(assessment.openai_category, '#6c757d') if assessment.openai_category else '#6c757d'}; color: white; font-weight: 600;">
                                            {assessment.openai_category if assessment.openai_category else 'N/A'}
                                        </span>
                                    </td>
                                </tr>
                                ''' for assessment in all_assessments]) if all_assessments else '<tr><td colspan="3" style="padding: 20px; text-align: center; color: #666;">No previous assessments found</td></tr>'}
                            </tbody>
                        </table>
                    </div>
                </div>
                
                <!-- Patient History Charts -->
                <div class="section" style="margin-top: 40px;">
                    <h2 style="color: #333; margin-bottom: 20px; font-size: 1.8em;">📊 Patient Assessment Trends</h2>
                    <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(400px, 1fr)); gap: 30px; margin-top: 20px;">
                        <div style="background: #f8f9fa; padding: 20px; border-radius: 10px;">
                            <h3 style="text-align: center; margin-bottom: 15px; color: #333;">Score Trend Over Time</h3>
                            <canvas id="scoreTrendChart" width="400" height="300"></canvas>
                        </div>
                        <div style="background: #f8f9fa; padding: 20px; border-radius: 10px;">
                            <h3 style="text-align: center; margin-bottom: 15px; color: #333;">Performance Improvement Trend</h3>
                            <canvas id="performanceTrendChart" width="400" height="300"></canvas>
                        </div>
                        <div style="background: #f8f9fa; padding: 20px; border-radius: 10px;">
                            <h3 style="text-align: center; margin-bottom: 15px; color: #333;">Risk Category Distribution</h3>
                            <canvas id="riskCategoryChart" width="400" height="300"></canvas>
                        </div>
                    </div>
                </div>
                
                <!-- Action Buttons -->
                <div class="action-buttons">
                    <a href="/assess-patient" class="btn btn-primary">🔄 Assess Another Patient</a>
                    <a href="/dashboard" class="btn btn-secondary">← Back to Dashboard</a>
                </div>
                
                
                <!-- Additional Info -->
                <div class="ml-info">
                    <h3>📋 Additional Information</h3>
                    <p><a href="/ml-info">View ML Model Information & Performance</a></p>
                </div>
            </div>
        </div>
        
        <script>
            // Chart data
            const assessmentDates = {json.dumps(assessment_dates)};
            const assessmentScores = {json.dumps(assessment_scores)};
            const assessmentCategories = {json.dumps(assessment_categories)};
            const performanceTrend = {json.dumps(performance_trend)};
            
            // Chart 1: Score Trend Over Time
            const scoreTrendCtx = document.getElementById('scoreTrendChart');
            if (scoreTrendCtx) {{
                new Chart(scoreTrendCtx, {{
                    type: 'line',
                    data: {{
                        labels: assessmentDates,
                        datasets: [{{
                            label: 'Assessment Score',
                            data: assessmentScores,
                            borderColor: '#667eea',
                            backgroundColor: 'rgba(102, 126, 234, 0.1)',
                            borderWidth: 3,
                            fill: true,
                            tension: 0.4,
                            pointBackgroundColor: '#667eea',
                            pointBorderColor: '#fff',
                            pointBorderWidth: 2,
                            pointRadius: 6
                        }}]
                    }},
                    options: {{
                        responsive: true,
                        plugins: {{
                            legend: {{
                                display: true,
                                position: 'top'
                            }}
                        }},
                        scales: {{
                            y: {{
                                beginAtZero: true,
                                max: {max_score},
                                title: {{
                                    display: true,
                                    text: 'Score'
                                }}
                            }},
                            x: {{
                                title: {{
                                    display: true,
                                    text: 'Assessment Date'
                                }}
                            }}
                        }}
                    }}
                }});
            }}
            
            // Chart 2: Performance Improvement Trend
            const performanceTrendCtx = document.getElementById('performanceTrendChart');
            if (performanceTrendCtx) {{
                new Chart(performanceTrendCtx, {{
                    type: 'bar',
                    data: {{
                        labels: assessmentDates,
                        datasets: [{{
                            label: 'Performance Change',
                            data: performanceTrend,
                            backgroundColor: performanceTrend.map(val => val > 0 ? '#28a745' : val < 0 ? '#dc3545' : '#6c757d'),
                            borderWidth: 2,
                            borderColor: '#fff'
                        }}]
                    }},
                    options: {{
                        responsive: true,
                        plugins: {{
                            legend: {{
                                display: true,
                                position: 'top'
                            }},
                            tooltip: {{
                                callbacks: {{
                                    label: function(context) {{
                                        const val = context.parsed.y;
                                        if (val > 0) {{
                                            return 'Improving by ' + val + ' points';
                                        }} else if (val < 0) {{
                                            return 'Declining by ' + Math.abs(val) + ' points';
                                        }} else {{
                                            return 'No change';
                                        }}
                                    }}
                                }}
                            }}
                        }},
                        scales: {{
                            y: {{
                                title: {{
                                    display: true,
                                    text: 'Score Change'
                                }},
                                grid: {{
                                    color: function(context) {{
                                        if (context.tick.value === 0) {{
                                            return '#000';
                                        }}
                                        return '#e0e0e0';
                                    }}
                                }}
                            }},
                            x: {{
                                title: {{
                                    display: true,
                                    text: 'Assessment Date'
                                }}
                            }}
                        }}
                    }}
                }});
            }}
            
            // Chart 3: Risk Category Distribution
            const riskCategoryCtx = document.getElementById('riskCategoryChart');
            if (riskCategoryCtx) {{
                // Count categories
                const categoryCounts = {{}};
                assessmentCategories.forEach(cat => {{
                    categoryCounts[cat] = (categoryCounts[cat] || 0) + 1;
                }});
                
                const categoryLabels = Object.keys(categoryCounts);
                const categoryData = Object.values(categoryCounts);
                const categoryColors = {{
                    'Low risk': '#28a745',
                    'At-Risk': '#ffc107',
                    'Problematic use likely': '#fd7e14',
                    'High Risk/ addictive pattern': '#dc3545',
                    'Unknown': '#6c757d'
                }};
                
                new Chart(riskCategoryCtx, {{
                    type: 'doughnut',
                    data: {{
                        labels: categoryLabels,
                        datasets: [{{
                            data: categoryData,
                            backgroundColor: categoryLabels.map(cat => categoryColors[cat] || '#6c757d'),
                            borderWidth: 3,
                            borderColor: '#fff'
                        }}]
                    }},
                    options: {{
                        responsive: true,
                        plugins: {{
                            legend: {{
                                position: 'bottom'
                            }}
                        }}
                    }}
                }});
            }}
            
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
        <!-- Animated Background Elements -->
        <div class="animated-background">
            <div class="particle"></div>
            <div class="particle"></div>
            <div class="particle"></div>
            <div class="particle"></div>
            <div class="pulse-circle"></div>
            <div class="pulse-circle"></div>
            <div class="wave"></div>
            <div class="neural-node"></div>
            <div class="neural-node"></div>
            <div class="neural-node"></div>
            <div class="thought-bubble"></div>
            <div class="breathe-circle"></div>
            <div class="network-line"></div>
            <div class="network-line"></div>
        </div>
    </body>
    </html>
    """

@app.route('/ml-info')
def ml_info():
    """Display ML model information from artifacts, reports, and visualizations"""
    try:
        # Check for new model location first, then fallback to old location
        model_path_new = 'ml_model/ml_model/best_model_ann.pkl'
        model_path_old = 'ml_model/artifacts/best_model.pkl'
        if os.path.exists(model_path_new) or os.path.exists(model_path_old):
            # Load model information from artifacts
            predictor = get_predictor()
            model_info = predictor.get_model_info()
            feature_importance = predictor.get_feature_importance()
            
            # Read evaluation report
            report_content = ""
            # Check for new report location first, then fallback to old location
            report_path_new = 'ml_model/ml_model/training_report.txt'
            report_path_old = 'ml_model/reports/model_evaluation_report.txt'
            if os.path.exists(report_path_new):
                with open(report_path_new, 'r', encoding='utf-8') as f:
                    report_content = f.read()
            elif os.path.exists(report_path_old):
                with open(report_path_old, 'r', encoding='utf-8') as f:
                    report_content = f.read()
            
            # Check available visualizations
            visualizations = []
            # Check both old and new visualization directories
            viz_dir_old = 'ml_model/visualizations'
            viz_dir_new = 'ml_model/ml_model/figures'
            viz_dir = None
            if os.path.exists(viz_dir_new):
                viz_dir = viz_dir_new
            elif os.path.exists(viz_dir_old):
                viz_dir = viz_dir_old
            
            if viz_dir and os.path.exists(viz_dir):
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
                    # Use the directory we found earlier
                    if viz_dir == viz_dir_new:
                        viz_path = f"ml_model/ml_model/figures/{viz}"
                    else:
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
                <link rel="stylesheet" href="{{ url_for('static', filename='css/animations.css') }}">
                <style>
                    * {{ margin: 0; padding: 0; box-sizing: border-box; }}
                    body {{ 
                        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; 
                        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                        min-height: 100vh;
                        padding: 20px;
                        position: relative;
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
                <!-- Animated Background Elements -->
                <div class="animated-background">
                    <div class="particle"></div>
                    <div class="particle"></div>
                    <div class="pulse-circle"></div>
                    <div class="wave"></div>
                    <div class="neural-node"></div>
                    <div class="neural-node"></div>
                </div>
            </body>
            </html>
            """
        else:
            return """
            <html>
            <head>
                <title>ML Model Information</title>
                <link rel="stylesheet" href="{{ url_for('static', filename='css/animations.css') }}">
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
            <body style="position: relative;">
                <div class="container">
                    <h1>🤖 Machine Learning Model Information</h1>
                    <p class="error">ML model not found!</p>
                    <p>Please run <code>python ml_model/train_models.py</code> first to train the model.</p>
                    <a href="/" class="btn">← Back to Questionnaire</a>
                </div>
                <!-- Animated Background Elements -->
                <div class="animated-background">
                    <div class="particle"></div>
                    <div class="particle"></div>
                    <div class="pulse-circle"></div>
                    <div class="wave"></div>
                    <div class="neural-node"></div>
                </div>
            </body>
            </html>
            """
    except Exception as e:
        return f"""
        <html>
        <head>
            <title>ML Model Information</title>
            <link rel="stylesheet" href="{{ url_for('static', filename='css/animations.css') }}">
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
@app.route('/ml_model/ml_model/figures/<filename>')
def serve_visualization(filename):
    """Serve visualization images from the ml_model visualization directories"""
    # Try new location first, then fallback to old location
    new_path = 'ml_model/ml_model/figures'
    old_path = 'ml_model/visualizations'
    if os.path.exists(os.path.join(new_path, filename)):
        return send_from_directory(new_path, filename)
    elif os.path.exists(os.path.join(old_path, filename)):
        return send_from_directory(old_path, filename)
    else:
        return "Visualization not found", 404

@app.route('/patient-analysis')
@login_required
def patient_analysis():
    """Patient analysis page with visualizations"""
    doctor_id = session['doctor_id']
    patients = Patient.query.filter_by(doctor_id=doctor_id).all()
    
    # Get all assessments for this doctor's patients
    all_logs = PatientLog.query.filter_by(doctor_id=doctor_id).all()
    
    # Calculate statistics
    total_patients = len(patients)
    total_assessments = len(all_logs)
    avg_assessments_per_patient = round(total_assessments / total_patients, 2) if total_patients > 0 else 0
    
    # Risk category distribution
    risk_counts = {}
    for log in all_logs:
        if log.openai_category:
            risk_counts[log.openai_category] = risk_counts.get(log.openai_category, 0) + 1
    
    risk_labels = list(risk_counts.keys()) if risk_counts else []
    risk_data = list(risk_counts.values()) if risk_counts else []
    
    # Time-based data (last 30 days)
    from collections import defaultdict
    time_counts = defaultdict(int)
    for log in all_logs:
        date_str = log.timestamp.strftime('%Y-%m-%d')
        time_counts[date_str] += 1
    
    time_labels = sorted(time_counts.keys())[-30:]  # Last 30 days
    time_data = [time_counts.get(d, 0) for d in time_labels]
    
    # Gender distribution
    gender_counts = {}
    for patient in patients:
        gender_counts[patient.gender] = gender_counts.get(patient.gender, 0) + 1
    
    gender_labels = list(gender_counts.keys())
    gender_data = list(gender_counts.values())
    
    # Age distribution (grouped)
    age_groups = {'0-18': 0, '19-30': 0, '31-45': 0, '46-60': 0, '60+': 0}
    for patient in patients:
        if patient.age <= 18:
            age_groups['0-18'] += 1
        elif patient.age <= 30:
            age_groups['19-30'] += 1
        elif patient.age <= 45:
            age_groups['31-45'] += 1
        elif patient.age <= 60:
            age_groups['46-60'] += 1
        else:
            age_groups['60+'] += 1
    
    age_labels = list(age_groups.keys())
    age_data = list(age_groups.values())
    
    return render_template('patient_analysis.html',
                         patients=patients,
                         total_patients=total_patients,
                         total_assessments=total_assessments,
                         avg_assessments_per_patient=avg_assessments_per_patient,
                         risk_labels=json.dumps(risk_labels),
                         risk_data=json.dumps(risk_data),
                         time_labels=json.dumps(time_labels),
                         time_data=json.dumps(time_data),
                         gender_labels=json.dumps(gender_labels),
                         gender_data=json.dumps(gender_data),
                         age_labels=json.dumps(age_labels),
                         age_data=json.dumps(age_data))

if __name__ == '__main__':
    print("Starting Flask application...")
    print("Open your browser and go to: http://localhost:5000")
    print("Submit the questionnaire to see the Python Flask terminal output!")
    app.run(debug=True, host='0.0.0.0', port=5000)
