import random
import csv
import json
from datetime import datetime
import threading
import time

class QuestionnaireDataGenerator:
    def __init__(self):
        """Initialize the data generator with question options and scoring"""
        
        # Define all possible choices for each question
        self.question_options = {
            'Q1': {'type': 'age', 'min': 13, 'max': 65},  # Age range
            'Q2': ['male', 'female', 'other'],  # Gender
            'Q3': ['8th', '9th', '10th', 'other'],  # Education
            'Q4': {'type': 'text', 'options': ['Student', 'Unemployed', 'Part-time', 'Full-time', 'Self-employed', 'Retired']},  # Occupation
            'Q5': ['nuclear', 'joint', 'other'],  # Family type
            'Q6': ['low', 'middle', 'high'],  # SES
            'Q7': ['less_2', '2_4', '4_6', 'more_6'],  # Online hours
            'Q8': ['social_media', 'gaming', 'streaming', 'education', 'other'],  # Primary activity
            'Q9': ['never', 'rarely', 'sometimes', 'often', 'always'],  # Overuse
            'Q10': ['yes', 'no'],  # Neglect responsibilities
            'Q11': ['never', 'rarely', 'sometimes', 'often', 'always'],  # Restlessness
            'Q12': ['yes', 'no'],  # Failed attempts
            'Q13': ['yes', 'no'],  # Hiding usage
            'Q14': ['yes', 'no'],  # Academic impact
            'Q15': ['yes', 'no'],  # Social preferences
            'Q16': ['excellent', 'very_good', 'good', 'fair', 'poor'],  # Mental health
            'Q17': ['yes', 'no'],  # Suicidal thoughts
            'Q18': ['yes', 'no'],  # Suicide attempts
            'Q19': ['yes', 'no'],  # Professional diagnosis
            'Q20': ['yes', 'no'],  # Family complaints
            'Q21': ['yes', 'no'],  # Relationship impact
            'Q22': ['yes', 'no']   # Social isolation
        }
        
        # Define scoring for each question
        self.scoring_map = {
            'Q1': 0,   # Age - no scoring
            'Q2': {'male': 0, 'female': 1, 'other': 2},
            'Q3': {'8th': 0, '9th': 1, '10th': 2, 'other': 3},
            'Q4': 0,   # Occupation - no scoring
            'Q5': {'nuclear': 0, 'joint': 1, 'other': 2},
            'Q6': {'low': 0, 'middle': 1, 'high': 2},
            'Q7': {'less_2': 1, '2_4': 2, '4_6': 3, 'more_6': 4},
            'Q8': {'social_media': 3, 'gaming': 4, 'streaming': 2, 'education': 1, 'other': 2},
            'Q9': {'never': 1, 'rarely': 2, 'sometimes': 3, 'often': 4, 'always': 5},
            'Q10': {'yes': 4, 'no': 1},
            'Q11': {'never': 1, 'rarely': 2, 'sometimes': 3, 'often': 4, 'always': 5},
            'Q12': {'yes': 4, 'no': 1},
            'Q13': {'yes': 4, 'no': 1},
            'Q14': {'yes': 4, 'no': 1},
            'Q15': {'yes': 4, 'no': 1},
            'Q16': {'excellent': 1, 'very_good': 2, 'good': 3, 'fair': 4, 'poor': 5},
            'Q17': {'yes': 5, 'no': 1},
            'Q18': {'yes': 5, 'no': 1},
            'Q19': {'yes': 4, 'no': 1},
            'Q20': {'yes': 3, 'no': 1},
            'Q21': {'yes': 4, 'no': 1},
            'Q22': {'yes': 4, 'no': 1}
        }
        
        # Define realistic probability distributions for more realistic data
        self.probability_weights = {
            'Q2': {'male': 0.5, 'female': 0.45, 'other': 0.05},  # Gender distribution
            'Q3': {'8th': 0.1, '9th': 0.2, '10th': 0.6, 'other': 0.1},  # Education
            'Q6': {'low': 0.3, 'middle': 0.5, 'high': 0.2},  # SES distribution
            'Q7': {'less_2': 0.2, '2_4': 0.4, '4_6': 0.3, 'more_6': 0.1},  # Online hours
            'Q8': {'social_media': 0.4, 'gaming': 0.2, 'streaming': 0.25, 'education': 0.1, 'other': 0.05},
            'Q9': {'never': 0.1, 'rarely': 0.2, 'sometimes': 0.4, 'often': 0.2, 'always': 0.1},
            'Q10': {'yes': 0.3, 'no': 0.7},  # Neglect responsibilities
            'Q11': {'never': 0.2, 'rarely': 0.3, 'sometimes': 0.3, 'often': 0.15, 'always': 0.05},
            'Q12': {'yes': 0.4, 'no': 0.6},  # Failed attempts
            'Q13': {'yes': 0.25, 'no': 0.75},  # Hiding usage
            'Q14': {'yes': 0.35, 'no': 0.65},  # Academic impact
            'Q15': {'yes': 0.2, 'no': 0.8},  # Social preferences
            'Q16': {'excellent': 0.1, 'very_good': 0.2, 'good': 0.4, 'fair': 0.2, 'poor': 0.1},
            'Q17': {'yes': 0.1, 'no': 0.9},  # Suicidal thoughts
            'Q18': {'yes': 0.02, 'no': 0.98},  # Suicide attempts
            'Q19': {'yes': 0.15, 'no': 0.85},  # Professional diagnosis
            'Q20': {'yes': 0.3, 'no': 0.7},  # Family complaints
            'Q21': {'yes': 0.25, 'no': 0.75},  # Relationship impact
            'Q22': {'yes': 0.2, 'no': 0.8}   # Social isolation
        }

    def generate_single_response(self):
        """Generate a single questionnaire response"""
        response = {}
        scores = {}
        
        for question, options in self.question_options.items():
            if question == 'Q1':  # Age
                age = random.randint(options['min'], options['max'])
                response[question] = age
                scores[question] = 0  # Age has no scoring
                
            elif question == 'Q4':  # Occupation
                occupation = random.choice(options['options'])
                response[question] = occupation
                scores[question] = 0  # Occupation has no scoring
                
            else:  # All other questions
                if question in self.probability_weights:
                    # Use weighted random selection for more realistic data
                    choices = list(options)
                    weights = [self.probability_weights[question].get(choice, 1.0) for choice in choices]
                    selected = random.choices(choices, weights=weights)[0]
                else:
                    # Use uniform random selection
                    selected = random.choice(options)
                
                response[question] = selected
                
                # Calculate score
                if question in self.scoring_map:
                    if isinstance(self.scoring_map[question], dict):
                        scores[question] = self.scoring_map[question].get(selected, 0)
                    else:
                        scores[question] = self.scoring_map[question]
                else:
                    scores[question] = 0
        
        return response, scores

    def calculate_total_score(self, scores):
        """Calculate total score from individual question scores (excluding Q1, Q4)"""
        # Exclude Q1 (age) and Q4 (occupation) from total score calculation
        excluded_questions = ['Q1', 'Q4']
        filtered_scores = {k: v for k, v in scores.items() if k not in excluded_questions}
        return sum(filtered_scores.values())

    def determine_risk_level(self, total_score):
        """Determine risk level based on total score"""
        # New risk categories as specified
        if total_score >= 30:
            return "High risk / addictive pattern (consider referral)"
        elif total_score >= 21:
            return "Problematic use likely (structured assessment)"
        elif total_score >= 11:
            return "At-risk (brief advice/monitor)"
        else:
            return "Low risk"

    def generate_dataset(self, num_records=1000):
        """Generate a dataset with specified number of records"""
        dataset = []
        
        print(f"Generating {num_records} questionnaire responses...")
        
        for i in range(num_records):
            response, scores = self.generate_single_response()
            total_score = self.calculate_total_score(scores)
            risk_level = self.determine_risk_level(total_score)
            
            # Create a complete record
            record = {
                'record_id': i + 1,
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'total_score': total_score,
                'max_score': 68,  # Updated max score (excluding Q1, Q4)
                'risk_level': risk_level,
                **response,  # All Q1-Q22 responses
                **{f'score_{k}': v for k, v in scores.items()}  # All individual scores
            }
            
            dataset.append(record)
            
            # Progress indicator
            if (i + 1) % 100 == 0:
                print(f"Generated {i + 1}/{num_records} records...")
        
        print(f"Dataset generation complete! Generated {len(dataset)} records.")
        return dataset

    def generate_balanced_dataset(self, records_per_category=25):
        """Generate a balanced dataset with equal distribution across 4 risk categories using parallel threads"""
        total_records = records_per_category * 4
        
        # Define target score ranges for each category
        categories = [
            {"name": "Low risk", "min_score": 0, "max_score": 10, "color": "🟢"},
            {"name": "At-risk (brief advice/monitor)", "min_score": 11, "max_score": 20, "color": "🟡"},
            {"name": "Problematic use likely (structured assessment)", "min_score": 21, "max_score": 29, "color": "🟠"},
            {"name": "High risk / addictive pattern (consider referral)", "min_score": 30, "max_score": 68, "color": "🔴"}
        ]
        
        print(f"🚀 Starting parallel generation of {records_per_category} records per category...")
        print("="*80)
        
        # Shared data structures for threads
        all_records = []
        record_id_counter = [1]  # Use list to make it mutable across threads
        record_id_lock = threading.Lock()
        print_lock = threading.Lock()
        
        # Thread results storage
        thread_results = {}
        thread_stats = {}
        
        def generate_category_data(category, category_index):
            """Generate data for a specific category in a separate thread"""
            category_records = []
            category_count = 0
            attempts = 0
            start_time = time.time()
            
            with print_lock:
                print(f"{category['color']} Thread {category_index + 1}: Starting {category['name']}")
            
            while category_count < records_per_category:
                attempts += 1
                
                # Generate response with target score in mind
                response, scores = self.generate_targeted_response(category['min_score'], category['max_score'])
                total_score = self.calculate_total_score(scores)
                
                # Verify it's in the right range
                if category['min_score'] <= total_score <= category['max_score']:
                    risk_level = self.determine_risk_level(total_score)
                    
                    # Get unique record ID
                    with record_id_lock:
                        current_record_id = record_id_counter[0]
                        record_id_counter[0] += 1
                    
                    # Create a complete record
                    record = {
                        'record_id': current_record_id,
                        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                        'total_score': total_score,
                        'max_score': 68,
                        'risk_level': risk_level,
                        **response,  # All Q1-Q22 responses
                        **{f'score_{k}': v for k, v in scores.items()}  # All individual scores
                    }
                    
                    category_records.append(record)
                    category_count += 1
                    
                    # Progress update
                    with print_lock:
                        elapsed = time.time() - start_time
                        print(f"{category['color']} Thread {category_index + 1}: ✓ Record {category_count}/{records_per_category} (Score: {total_score}, Attempts: {attempts}, Time: {elapsed:.1f}s)")
                
                # Show attempt progress every 10 attempts
                elif attempts % 10 == 0:
                    with print_lock:
                        elapsed = time.time() - start_time
                        print(f"{category['color']} Thread {category_index + 1}: ... Attempt {attempts}, looking for score {category['min_score']}-{category['max_score']} (Time: {elapsed:.1f}s)")
            
            # Store results
            thread_results[category_index] = category_records
            thread_stats[category_index] = {
                'category': category['name'],
                'records': category_count,
                'attempts': attempts,
                'time': time.time() - start_time
            }
            
            with print_lock:
                print(f"{category['color']} Thread {category_index + 1}: ✅ COMPLETED! Generated {category_count} records in {attempts} attempts ({time.time() - start_time:.1f}s)")
        
        # Create and start threads
        threads = []
        for i, category in enumerate(categories):
            thread = threading.Thread(target=generate_category_data, args=(category, i))
            threads.append(thread)
            thread.start()
        
        # Wait for all threads to complete
        for thread in threads:
            thread.join()
        
        # Combine all results
        for i in range(len(categories)):
            if i in thread_results:
                all_records.extend(thread_results[i])
        
        # Print final statistics
        print("\n" + "="*80)
        print("📊 FINAL STATISTICS:")
        print("="*80)
        total_attempts = 0
        total_time = 0
        
        for i, stats in thread_stats.items():
            print(f"{categories[i]['color']} {stats['category']}:")
            print(f"   Records: {stats['records']}")
            print(f"   Attempts: {stats['attempts']}")
            print(f"   Time: {stats['time']:.1f}s")
            print(f"   Efficiency: {stats['records']/stats['attempts']*100:.1f}%")
            total_attempts += stats['attempts']
            total_time = max(total_time, stats['time'])
            print()
        
        print(f"🎯 TOTAL: {len(all_records)} records generated in {total_attempts} attempts ({total_time:.1f}s)")
        print(f"⚡ Average efficiency: {len(all_records)/total_attempts*100:.1f}%")
        print("="*80)
        
        return all_records
    
    def generate_targeted_response(self, min_score, max_score):
        """Generate a response that targets a specific score range"""
        response = {}
        scores = {}
        
        # Generate Q1 (Age) - always random
        age = random.randint(18, 65)
        response['Q1'] = age
        scores['Q1'] = age
        
        # Generate Q4 (Occupation) - always random, no score
        occupations = ['Student', 'Employee', 'Business', 'Unemployed', 'Retired']
        response['Q4'] = random.choice(occupations)
        scores['Q4'] = 0
        
        # Calculate target score for remaining questions (excluding Q1 and Q4)
        target_score = random.randint(min_score, max_score)
        
        # Get all questions that contribute to score (excluding Q1 and Q4)
        score_questions = [f'Q{i}' for i in range(2, 23) if i != 4]
        
        # Distribute the target score across these questions
        remaining_score = target_score
        question_scores = {}
        
        # First pass: assign minimum scores to each question
        for q in score_questions:
            min_possible = self.get_min_score_for_question(q)
            question_scores[q] = min_possible
            remaining_score -= min_possible
        
        # Second pass: distribute remaining score
        while remaining_score > 0 and score_questions:
            q = random.choice(score_questions)
            max_possible = self.get_max_score_for_question(q)
            current = question_scores[q]
            
            if current < max_possible:
                increase = min(remaining_score, max_possible - current)
                question_scores[q] += increase
                remaining_score -= increase
            else:
                score_questions.remove(q)
        
        # Generate responses based on target scores
        for q_num in range(2, 23):
            if q_num == 4:  # Skip Q4 (occupation)
                continue
                
            q_key = f'Q{q_num}'
            target_q_score = question_scores.get(q_key, 0)
            
            # Generate response that gives this score
            response[q_key], scores[q_key] = self.generate_response_for_score(q_key, target_q_score)
        
        return response, scores
    
    def get_min_score_for_question(self, question):
        """Get minimum possible score for a question"""
        if question in ['Q2', 'Q3', 'Q5', 'Q6']:  # Section A questions
            return 0
        elif question in ['Q7', 'Q8', 'Q9', 'Q10', 'Q11', 'Q12', 'Q13', 'Q14', 'Q15', 'Q16', 'Q17', 'Q18', 'Q19', 'Q20', 'Q21', 'Q22']:  # Other sections
            return 0
        return 0
    
    def get_max_score_for_question(self, question):
        """Get maximum possible score for a question"""
        if question in ['Q2', 'Q3', 'Q5', 'Q6']:  # Section A questions
            return 2
        elif question in ['Q7', 'Q8', 'Q9', 'Q10', 'Q11', 'Q12', 'Q13', 'Q14', 'Q15', 'Q16', 'Q17', 'Q18', 'Q19', 'Q20', 'Q21', 'Q22']:  # Other sections
            return 4
        return 0
    
    def generate_response_for_score(self, question, target_score):
        """Generate a response that gives the target score for a question"""
        if question == 'Q2':  # Gender
            options = ['Male', 'Female', 'Other']
            scores_map = {'Male': 0, 'Female': 1, 'Other': 2}
            for option, score in scores_map.items():
                if score == target_score:
                    return option, score
        elif question == 'Q3':  # Education
            options = ['8', '9', '10', 'other']
            scores_map = {'8': 0, '9': 1, '10': 2, 'other': 3}
            for option, score in scores_map.items():
                if score == target_score:
                    return option, score
        elif question == 'Q5':  # Family type
            options = ['Nuclear', 'Joint', 'Other']
            scores_map = {'Nuclear': 0, 'Joint': 1, 'Other': 2}
            for option, score in scores_map.items():
                if score == target_score:
                    return option, score
        elif question == 'Q6':  # SES
            options = ['Low', 'Middle', 'High']
            scores_map = {'Low': 0, 'Middle': 1, 'High': 2}
            for option, score in scores_map.items():
                if score == target_score:
                    return option, score
        else:  # Other questions (Q7-Q22)
            # For other questions, we need to map score to response
            # Assuming 0-4 scale for most questions
            if target_score == 0:
                return 'Never', 0
            elif target_score == 1:
                return 'Rarely', 1
            elif target_score == 2:
                return 'Sometimes', 2
            elif target_score == 3:
                return 'Often', 3
            elif target_score == 4:
                return 'Always', 4
        
        # Fallback
        return 'Sometimes', 2

    def save_to_csv(self, dataset, filename='questionnaire_data.csv', core_values_only=True):
        """Save dataset to CSV file"""
        if not dataset:
            print("No data to save!")
            return
        
        if core_values_only:
            # Create simplified dataset with only question scores
            simplified_dataset = []
            for record in dataset:
                simplified_record = {
                    'record_id': record['record_id'],
                    'total_score': record['total_score'],
                    'risk_level': record['risk_level']
                }
                # Add only the core question scores (Q1-Q22)
                for i in range(1, 23):
                    question_key = f'Q{i}'
                    score_key = f'score_{question_key}'
                    simplified_record[question_key] = record[score_key]
                simplified_dataset.append(simplified_record)
            
            # Get fieldnames from simplified record
            fieldnames = list(simplified_dataset[0].keys())
            
            with open(filename, 'w', newline='', encoding='utf-8') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(simplified_dataset)
            
            print(f"Dataset saved to {filename} (core values only)")
            print(f"Total records: {len(simplified_dataset)}")
            print(f"Columns: {len(fieldnames)}")
        else:
            # Get all fieldnames from the first record (original behavior)
            fieldnames = list(dataset[0].keys())
            
            with open(filename, 'w', newline='', encoding='utf-8') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(dataset)
            
            print(f"Dataset saved to {filename}")
            print(f"Total records: {len(dataset)}")
            print(f"Columns: {len(fieldnames)}")


    def print_sample_output(self, num_samples=5):
        """Print sample Python Flask terminal style output"""
        print("\n" + "="*60)
        print("SAMPLE PYTHON FLASK TERMINAL OUTPUT")
        print("="*60)
        
        for i in range(num_samples):
            response, scores = self.generate_single_response()
            total_score = self.calculate_total_score(scores)
            risk_level = self.determine_risk_level(total_score)
            
            # Create Python Flask terminal style output
            python_output = f"{{Q1: {response['Q1']}, Q2: {scores['Q2']}, Q3: {scores['Q3']}, Q4: {scores['Q4']}, Q5: {scores['Q5']}, Q6: {scores['Q6']}, Q7: {scores['Q7']}, Q8: {scores['Q8']}, Q9: {scores['Q9']}, Q10: {scores['Q10']}, Q11: {scores['Q11']}, Q12: {scores['Q12']}, Q13: {scores['Q13']}, Q14: {scores['Q14']}, Q15: {scores['Q15']}, Q16: {scores['Q16']}, Q17: {scores['Q17']}, Q18: {scores['Q18']}, Q19: {scores['Q19']}, Q20: {scores['Q20']}, Q21: {scores['Q21']}, Q22: {scores['Q22']}}}"
            
            print(f"\nSample {i+1}:")
            print(f"Total Score: {total_score}/68")
            print(f"Risk Level: {risk_level}")
            print(f"Python Output: {python_output}")
        
        print("="*60)

def main():
    """Main function to run the data generator"""
    generator = QuestionnaireDataGenerator()
    
    print("Digital Media & Mental Health Assessment - Data Generator")
    print("="*60)
    
    # Print sample outputs
    generator.print_sample_output(3)
    
    # Generate balanced dataset with 1250 records per category (5000 total)
    print("\nGenerating balanced dataset with 5000 records (1250 per category)...")
    dataset = generator.generate_balanced_dataset(1250)
    
    # Save to CSV (core values only) - overwrites each time
    csv_filename = "questionnaire_data.csv"
    generator.save_to_csv(dataset, csv_filename, core_values_only=True)
    
    # Print summary statistics
    print("\n" + "="*60)
    print("BALANCED DATASET SUMMARY")
    print("="*60)
    
    total_scores = [record['total_score'] for record in dataset]
    risk_levels = [record['risk_level'] for record in dataset]
    
    print(f"Total Records: {len(dataset)}")
    print(f"Average Score: {sum(total_scores)/len(total_scores):.2f}/68")
    print(f"Min Score: {min(total_scores)}")
    print(f"Max Score: {max(total_scores)}")
    print(f"\nRisk Level Distribution:")
    
    # Define the 4 categories
    categories = [
        "Low risk",
        "At-risk (brief advice/monitor)", 
        "Problematic use likely (structured assessment)",
        "High risk / addictive pattern (consider referral)"
    ]
    
    for risk in categories:
        count = risk_levels.count(risk)
        percentage = (count / len(risk_levels)) * 100
        print(f"  {risk}: {count} ({percentage:.1f}%)")

if __name__ == "__main__":
    main()
