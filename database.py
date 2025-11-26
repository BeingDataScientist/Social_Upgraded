"""
Database models and initialization
"""
from flask_sqlalchemy import SQLAlchemy
from datetime import datetime
import uuid

db = SQLAlchemy()

class Doctor(db.Model):
    """Doctor/User table"""
    __tablename__ = 'doctors'
    
    id = db.Column(db.Integer, primary_key=True)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password = db.Column(db.String(255), nullable=False)
    name = db.Column(db.String(100), nullable=False)
    dob = db.Column(db.Date, nullable=False)
    profession = db.Column(db.String(100), nullable=False)
    address = db.Column(db.Text, nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    
    def __repr__(self):
        return f'<Doctor {self.email}>'

class Patient(db.Model):
    """Patient table"""
    __tablename__ = 'patients'
    
    pid = db.Column(db.String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    name = db.Column(db.String(100), nullable=False)
    age = db.Column(db.Integer, nullable=False)
    gender = db.Column(db.String(20), nullable=False)
    previous_history = db.Column(db.Text, nullable=True)  # History of mental related disorders
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    doctor_id = db.Column(db.Integer, db.ForeignKey('doctors.id'), nullable=False)
    
    # Relationship
    doctor = db.relationship('Doctor', backref=db.backref('patients', lazy=True))
    assessments = db.relationship('PatientLog', backref='patient', lazy=True, cascade='all, delete-orphan')
    
    def __repr__(self):
        return f'<Patient {self.pid} - {self.name}>'

class PatientLog(db.Model):
    """Patient assessment log table"""
    __tablename__ = 'patient_logs'
    
    id = db.Column(db.Integer, primary_key=True)
    patient_id = db.Column(db.String(36), db.ForeignKey('patients.pid'), nullable=False)
    doctor_id = db.Column(db.Integer, db.ForeignKey('doctors.id'), nullable=False)
    responses = db.Column(db.Text, nullable=False)  # JSON string of questionnaire responses
    openai_category = db.Column(db.String(100), nullable=True)  # OpenAI returned risk category
    total_score = db.Column(db.Integer, nullable=True)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow, nullable=False)
    
    # Relationships
    doctor = db.relationship('Doctor', backref=db.backref('patient_logs', lazy=True))
    
    def __repr__(self):
        return f'<PatientLog {self.id} - Patient {self.patient_id}>'

def init_db(app):
    """Initialize database with app"""
    db.init_app(app)
    with app.app_context():
        db.create_all()
        print("Database initialized successfully!")

