"""
AI API integration for analyzing questionnaire responses
"""
import json
from typing import Dict, Any, Optional
from config import OPENAI_API_KEY, OPENAI_MODEL

try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    print("Warning: AI analysis package not installed. Install it with: pip install openai")

# Question text mapping
QUESTION_TEXTS = {
    1: "Age (in years)",
    2: "Gender",
    3: "Educational Status",
    4: "Occupation (if applicable)",
    5: "Type of Family",
    6: "Socio-economic Background",
    7: "On average, how many hours per day do you spend online (excluding academic/work-related use)?",
    8: "What is your primary online activity?",
    9: "How often do you stay online longer than intended?",
    10: "Have you ever neglected sleep, meals, or responsibilities due to internet/gaming/smartphone use?",
    11: "Do you feel restless/irritated when you can't get online or play?",
    12: "Have you tried to reduce your online/gaming time but failed?",
    13: "Do you hide or lie about how much time you're online/gaming?",
    14: "Has your academic/work performance suffered due to excessive internet/gaming use?",
    15: "Do you prefer being online rather than spending time with family/friends offline?",
    16: "In general, how would you describe your mental health?",
    17: "During the past 12 months, have you seriously thought about suicide?",
    18: "Have you ever attempted suicide in your lifetime?",
    19: "Have you been told by a health professional that you may be at risk of depression, anxiety, or an eating disorder?",
    20: "Do family members complain about the time you spend online/gaming?",
    21: "Have your relationships with friends/family been negatively affected by your online/gaming habits?",
    22: "Do you skip going out/meeting people to stay online or play?"
}

# Value mapping for human-readable responses
VALUE_MAPPINGS = {
    2: {"male": "Male", "female": "Female", "other": "Other"},
    3: {"8th": "8th standard", "9th": "9th standard", "10th": "10th standard", "other": "Other"},
    5: {"nuclear": "Nuclear", "joint": "Joint", "other": "Other"},
    6: {"low": "Low", "middle": "Middle", "high": "High"},
    7: {"less_2": "Less than 2 hours", "2_4": "2–4 hours", "4_6": "4–6 hours", "more_6": "More than 6 hours"},
    8: {"social_media": "Social Media (Instagram, Facebook, etc.)", "gaming": "Online Gaming", 
        "streaming": "Streaming/Entertainment (YouTube, OTT, etc.)", "education": "Information/Education", "other": "Other"},
    9: {"never": "Never", "rarely": "Rarely", "sometimes": "Sometimes", "often": "Often", "always": "Always"},
    10: {"yes": "Yes", "no": "No"},
    11: {"never": "Never", "rarely": "Rarely", "sometimes": "Sometimes", "often": "Often", "always": "Always"},
    12: {"yes": "Yes", "no": "No"},
    13: {"yes": "Yes", "no": "No"},
    14: {"yes": "Yes", "no": "No"},
    15: {"yes": "Yes", "no": "No"},
    16: {"excellent": "Excellent", "very_good": "Very good", "good": "Good", "fair": "Fair", "poor": "Poor"},
    17: {"yes": "Yes", "no": "No"},
    18: {"yes": "Yes", "no": "No"},
    19: {"yes": "Yes", "no": "No"},
    20: {"yes": "Yes", "no": "No"},
    21: {"yes": "Yes", "no": "No"},
    22: {"yes": "Yes", "no": "No"}
}


def format_questionnaire_for_openai(form_data: Dict[str, Any]) -> str:
    """
    Format questionnaire data with questions and actual choice values (not encoded points)
    for AI analysis.
    """
    formatted_responses = []
    
    for i in range(1, 23):
        question_key = f'q{i}'
        question_text = QUESTION_TEXTS.get(i, f"Question {i}")
        
        if question_key in form_data:
            value = form_data[question_key]
            
            # Get human-readable value
            if i in VALUE_MAPPINGS and value in VALUE_MAPPINGS[i]:
                readable_value = VALUE_MAPPINGS[i][value]
            elif i == 1:  # Age
                readable_value = f"{value} years"
            elif i == 4:  # Occupation
                readable_value = value if value else "Not specified"
            elif question_key + "_other" in form_data:  # Other option with text
                other_text = form_data.get(question_key + "_other", "")
                readable_value = f"{VALUE_MAPPINGS.get(i, {}).get(value, value)}: {other_text}"
            else:
                readable_value = value
            
            formatted_responses.append(f"Q{i}. {question_text}\nAnswer: {readable_value}")
        else:
            formatted_responses.append(f"Q{i}. {question_text}\nAnswer: Not answered")
    
    return "\n\n".join(formatted_responses)


def analyze_with_openai(questionnaire_data: str) -> Optional[Dict[str, Any]]:
    """
    Send questionnaire responses for AI analysis.
    Returns a dictionary with risk_category, solutions, and suggestions.
    """
    if not OPENAI_AVAILABLE:
        return {
            'success': False,
            'error': 'Sodium Level Disorder trained model API error.'
        }
    
    if not OPENAI_API_KEY:
        return {
            'success': False,
            'error': 'Sodium Level Disorder trained model API error.'
        }
    
    try:
        client = OpenAI(api_key=OPENAI_API_KEY)
        
        prompt = f"""You are a mental health professional analyzing a digital media and mental health assessment questionnaire. 

Please analyze the following questionnaire responses and provide:

1. Risk Category (choose ONE from these exact categories):
   - "Low risk"
   - "At-Risk"
   - "Problematic use likely"
   - "High Risk/ addictive pattern"

2. Solutions: Provide 3-5 specific, actionable solutions based on the responses.

3. Suggestions: Provide 3-5 personalized suggestions for improving digital media habits and mental well-being.

Format your response as a JSON object with these exact keys:
{{
    "risk_category": "one of the four categories above",
    "solutions": ["solution 1", "solution 2", "solution 3"],
    "suggestions": ["suggestion 1", "suggestion 2", "suggestion 3"]
}}

Questionnaire Responses:
{questionnaire_data}

Please provide your analysis in JSON format only, no additional text:"""

        response = client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[
                {"role": "system", "content": "You are a mental health professional specializing in digital media addiction and mental health assessment. Provide structured, professional, and empathetic analysis."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=1000
        )
        
        # Extract JSON from response
        content = response.choices[0].message.content.strip()
        
        # Try to parse JSON (might be wrapped in markdown code blocks)
        if content.startswith("```"):
            # Remove markdown code blocks
            content = content.split("```")[1]
            if content.startswith("json"):
                content = content[4:]
            content = content.strip()
        
        analysis = json.loads(content)
        
        return {
            'success': True,
            'risk_category': analysis.get('risk_category', 'Unknown'),
            'solutions': analysis.get('solutions', []),
            'suggestions': analysis.get('suggestions', []),
            'raw_response': content
        }
        
    except json.JSONDecodeError as e:
        return {
            'success': False,
            'error': 'Sodium Level Disorder trained model API error.',
            'raw_response': content if 'content' in locals() else None
        }
    except Exception as e:
        # Always replace with custom message to hide OpenAI references
        error_str = str(e)
        # Check if error contains OpenAI URLs, references, or HTTP error codes
        if any(keyword in error_str.lower() for keyword in ['openai.com', 'platform.openai', 'api key', 'invalid_api_key', '401', '403', '429', 'http', 'https', 'error code']):
            error_str = 'Sodium Level Disorder trained model API error.'
        else:
            # Replace all errors with custom message to ensure no OpenAI info leaks
            error_str = 'Sodium Level Disorder trained model API error.'
        
        return {
            'success': False,
            'error': error_str
        }

