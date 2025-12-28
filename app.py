"""
Cricket Score Predictor - Flask Application
A premium cricket score prediction app for T20, ODI, and Test matches.
"""

import os
import logging
from flask import Flask, render_template, request, jsonify
import pickle
import pandas as pd

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Load the model with error handling
pipe = None
MODEL_PATH = 'pipe.pkl'


# Format configurations
FORMAT_CONFIGS = {
    'T20': {
        'total_balls': 120,
        'min_overs': 5,
        'max_overs': 20,
        'max_wickets': 9,
        'name': 'T20 International',
        'scale_factor': 1.0
    },
    'ODI': {
        'total_balls': 300,
        'min_overs': 10,
        'max_overs': 50,
        'max_wickets': 9,
        'name': 'One Day International',
        'scale_factor': 1.4  # ODI scores are typically higher
    },
    'TEST': {
        'total_balls': 540,
        'min_overs': 20,
        'max_overs': 90,
        'max_wickets': 9,
        'name': 'Test Match',
        'scale_factor': 1.8  # Test innings can be much higher
    }
}


def load_model():
    """Load the ML model from disk."""
    global pipe
    if os.path.exists(MODEL_PATH):
        try:
            with open(MODEL_PATH, 'rb') as f:
                pipe = pickle.load(f)
            logger.info("Model loaded successfully")
            return True
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            return False
    else:
        logger.warning(f"Model file '{MODEL_PATH}' not found. Run train_model.py first.")
        return False


# Teams list
TEAMS = [
    'Australia', 'India', 'Bangladesh', 'New Zealand', 'South Africa',
    'England', 'West Indies', 'Afghanistan', 'Pakistan', 'Sri Lanka'
]

# Cities list
CITIES = [
    'Colombo', 'Mirpur', 'Johannesburg', 'Dubai', 'Auckland', 'Cape Town',
    'London', 'Pallekele', 'Barbados', 'Sydney', 'Melbourne', 'Durban',
    'St Lucia', 'Wellington', 'Lauderhill', 'Hamilton', 'Centurion',
    'Manchester', 'Abu Dhabi', 'Mumbai', 'Nottingham', 'Southampton',
    'Mount Maunganui', 'Chittagong', 'Kolkata', 'Lahore', 'Delhi',
    'Nagpur', 'Chandigarh', 'Adelaide', 'Bangalore', 'St Kitts', 'Cardiff',
    'Christchurch', 'Trinidad'
]


def validate_input(form_data, format_type='T20'):
    """Validate form input data based on format."""
    errors = []
    config = FORMAT_CONFIGS.get(format_type, FORMAT_CONFIGS['T20'])
    
    # Check required fields
    required = ['batting_team', 'bowling_team', 'city', 'current_score', 'overs', 'wickets', 'last_five']
    for field in required:
        if not form_data.get(field):
            errors.append(f"{field.replace('_', ' ').title()} is required")
    
    if errors:
        return None, errors
    
    try:
        data = {
            'batting_team': form_data['batting_team'],
            'bowling_team': form_data['bowling_team'],
            'city': form_data['city'],
            'current_score': int(form_data['current_score']),
            'overs': float(form_data['overs']),
            'wickets': int(form_data['wickets']),
            'last_five': int(form_data['last_five']),
            'format': format_type
        }
        
        # Validate ranges based on format
        if data['current_score'] < 0:
            errors.append("Current score cannot be negative")
        if data['overs'] < config['min_overs'] or data['overs'] > config['max_overs']:
            errors.append(f"Overs must be between {config['min_overs']} and {config['max_overs']} for {config['name']}")
        if data['wickets'] < 0 or data['wickets'] > config['max_wickets']:
            errors.append(f"Wickets must be between 0 and {config['max_wickets']}")
        if data['last_five'] < 0:
            errors.append("Runs in last 5 overs cannot be negative")
        if data['last_five'] > data['current_score']:
            errors.append("Runs in last 5 overs cannot exceed current score")
        if data['batting_team'] == data['bowling_team']:
            errors.append("Batting and bowling teams cannot be the same")
        if data['batting_team'] not in TEAMS:
            errors.append("Invalid batting team selected")
        if data['bowling_team'] not in TEAMS:
            errors.append("Invalid bowling team selected")
        if data['city'] not in CITIES:
            errors.append("Invalid city selected")
            
        if errors:
            return None, errors
            
        return data, []
        
    except ValueError as e:
        errors.append("Please enter valid numeric values")
        return None, errors


def predict_score(data):
    """Make prediction based on format and input data."""
    format_type = data.get('format', 'T20')
    config = FORMAT_CONFIGS.get(format_type, FORMAT_CONFIGS['T20'])
    
    # Calculate derived features - normalize to T20 equivalent for model
    # The model was trained on T20 data, so we need to scale
    t20_equivalent_overs = (data['overs'] / config['max_overs']) * 20
    balls_left = 120 - (t20_equivalent_overs * 6)
    wickets_left = 10 - data['wickets']
    crr = data['current_score'] / data['overs']
    
    # Create input DataFrame
    input_df = pd.DataFrame({
        'batting_team': [data['batting_team']],
        'bowling_team': [data['bowling_team']],
        'city': [data['city']],
        'current_score': [data['current_score']],
        'balls_left': [balls_left],
        'wickets_left': [wickets_left],
        'crr': [crr],
        'last_five': [data['last_five']]
    })
    
    # Make prediction
    base_prediction = pipe.predict(input_df)[0]
    
    # Scale prediction based on format
    # For ODI/Test, scale up the prediction to account for longer innings
    remaining_overs_ratio = (config['max_overs'] - data['overs']) / config['max_overs']
    format_adjustment = 1 + (remaining_overs_ratio * (config['scale_factor'] - 1))
    
    prediction = int(base_prediction * format_adjustment)
    
    # Ensure prediction is at least current score + reasonable additional runs
    min_additional = int((config['max_overs'] - data['overs']) * 4)  # At least 4 runs per over
    if prediction < data['current_score'] + min_additional:
        prediction = data['current_score'] + min_additional
    
    return prediction


@app.route('/', methods=['GET', 'POST'])
def index():
    """Main route for the prediction form."""
    prediction = None
    error = None
    form_data = {}
    format_type = 'T20'
    format_name = 'T20'
    
    if request.method == 'POST':
        form_data = request.form.to_dict()
        format_type = form_data.get('format', 'T20')
        
        # Check if model is loaded
        if pipe is None:
            error = "Model not loaded. Please run 'python train_model.py' first."
        else:
            # Validate input
            data, errors = validate_input(form_data, format_type)
            
            if errors:
                error = "; ".join(errors)
            else:
                try:
                    prediction = predict_score(data)
                    format_name = FORMAT_CONFIGS[format_type]['name']
                    logger.info(f"Prediction made: {prediction} for {data['batting_team']} vs {data['bowling_team']} ({format_type})")
                    
                except Exception as e:
                    logger.error(f"Prediction error: {e}")
                    error = "An error occurred while making the prediction. Please try again."
    
    return render_template(
        'index.html',
        teams=sorted(TEAMS),
        cities=sorted(CITIES),
        prediction=prediction,
        error=error,
        form_data=form_data,
        format=format_type,
        format_name=format_name
    )


@app.route('/api/predict', methods=['POST'])
def api_predict():
    """API endpoint for predictions."""
    if pipe is None:
        return jsonify({'error': 'Model not loaded'}), 503
    
    request_data = request.json or {}
    format_type = request_data.get('format', 'T20')
    
    data, errors = validate_input(request_data, format_type)
    
    if errors:
        return jsonify({'error': errors}), 400
    
    try:
        prediction = predict_score(data)
        config = FORMAT_CONFIGS[format_type]
        
        return jsonify({
            'prediction': prediction,
            'current_score': data['current_score'],
            'format': format_type,
            'format_name': config['name'],
            'overs_remaining': round(config['max_overs'] - data['overs'], 1)
        })
        
    except Exception as e:
        logger.error(f"API prediction error: {e}")
        return jsonify({'error': 'Prediction failed'}), 500


@app.route('/health')
def health():
    """Health check endpoint."""
    return jsonify({
        'status': 'healthy',
        'model_loaded': pipe is not None,
        'formats_supported': list(FORMAT_CONFIGS.keys())
    })


# Load model on startup
load_model()

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)

