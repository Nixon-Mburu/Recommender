from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import logging
import time
import torch
import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from model import EnhancedRecommenderNN, predict_rating  # Remove get_movie_recommendations
import os
import requests
from dotenv import load_dotenv

# Define device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

load_dotenv()

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__, static_folder='../frontend/recommend/build', static_url_path='')
CORS(app)

# Constants
MODEL_PATH = Path(__file__).parent / "models/movie_finder.pth"
PREPROCESSOR_PATH = Path(__file__).parent / "models/preprocessor.pkl"
ALL_GENRES = [
    'Action', 'Adventure', 'Animation', 'Comedy', 'Crime',
    'Drama', 'Fantasy', 'Horror', 'Mystery', 'Romance',
    'Sci-Fi', 'Thriller'
]

# OMDB API configuration
OMDB_API_KEY = os.getenv('OMDB_API_KEY')
OMDB_BASE_URL = 'http://www.omdbapi.com/'  

def search_movies_omdb(query, year_from=None, year_to=None):
    try:
        all_results = []
        # If no query is provided, use some popular keywords
        search_terms = [query] if query else ['action', 'adventure', 'drama']
        
        for term in search_terms:
            # Search through the year range
            current_year = year_from
            while current_year <= (year_to or year_from or 2023):
                params = {
                    'apikey': OMDB_API_KEY,
                    's': term,
                    'y': str(current_year),
                    'type': 'movie'
                }
                response = requests.get(OMDB_BASE_URL, params=params)
                if response.status_code == 200:
                    data = response.json()
                    if data.get('Response') == 'True':
                        all_results.extend(data.get('Search', []))
                current_year += 1
                if len(all_results) >= 50:  # Limit total results
                    break
            if len(all_results) >= 50:
                break
        return all_results
    except Exception as e:
        logger.error(f"OMDB search error: {str(e)}")
        return []

def get_movie_details_omdb(imdb_id):
    try:
        params = {
            'apikey': OMDB_API_KEY,
            'i': imdb_id,
            'plot': 'full'
        }
        response = requests.get(OMDB_BASE_URL, params=params)
        if response.status_code == 200:
            data = response.json()
            if data.get('Response') == 'True':
                rating = data.get('imdbRating', 'N/A')
                return {
                    'id': data.get('imdbID'),
                    'title': data.get('Title'),
                    'year': int(data.get('Year', '0').split('–')[0]),
                    'genres': data.get('Genre', '').split(', '),
                    'overview': data.get('Plot'),
                    'posterUrl': data.get('Poster'),
                    'imdbRating': rating if rating != 'N/A' else 0,
                    'rating': float(rating) if rating != 'N/A' else 0,
                    'director': data.get('Director'),
                    'actors': data.get('Actors')
                }
        return None
    except Exception as e:
        logger.error(f"OMDB details error: {str(e)}")
        return None

# Load model and preprocessor
try:
    # Load preprocessor
    preprocessor = joblib.load(PREPROCESSOR_PATH)
    sample_data = pd.DataFrame([{
        'year': 2023, 'month': 1, 'day': 1,
        'age': '30', 'gender': 'M', 'occupation': 'other'
    }])
    input_dim = preprocessor.transform(sample_data).shape[1]
    
    # Load model with the deeper architecture that matches the saved checkpoint
    model = EnhancedRecommenderNN(
        input_dim=input_dim,
        hidden_dims=[512, 256, 128, 64],  # Match the architecture used during training
        dropout_rate=0.3
    )
    
    # Load the checkpoint
    checkpoint = torch.load(MODEL_PATH, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    model.to(device)
    
except Exception as e:
    logger.error(f"Error loading model: {str(e)}")
    raise

def batch_predict_ratings(users_features, model, preprocessor, device='cpu'):
    """Make predictions for multiple users at once"""
    try:
        # Convert list of dictionaries to DataFrame if necessary
        if isinstance(users_features, list):
            users_df = pd.DataFrame(users_features)
        else:
            users_df = users_features

        # Ensure categorical columns are strings
        for col in ['age', 'gender', 'occupation']:
            if col in users_df.columns:
                users_df[col] = users_df[col].astype(str)
            else:
                users_df[col] = 'unknown'

        # Transform features
        X = preprocessor.transform(users_df)
        X_tensor = torch.FloatTensor(X).to(device)

        # Get predictions
        model.eval()
        with torch.no_grad():
            predictions = model(X_tensor)
            predictions = torch.clamp(predictions, 1.0, 5.0)

        return predictions.cpu().numpy().flatten()
    except Exception as e:
        logger.error(f"Batch prediction error: {str(e)}")
        return None

@app.before_request
def before_request():
    request.start_time = time.time()

@app.after_request
def after_request(response):
    if hasattr(request, 'start_time'):
        duration = time.time() - request.start_time
        logger.info(f"{request.method} {request.path} - {response.status_code} - {duration:.2f}s")
    return response

def process_input(data):
    try:
        # Convert input data to match the format expected by the preprocessor
        input_data = {
            'year': data.get('yearRange', {}).get('from', 2023),
            'month': 1,  # default values
            'day': 1,
            'gender': 'M',
            'occupation': 'other'
        }
        return torch.FloatTensor(preprocessor.transform(pd.DataFrame([input_data]))).unsqueeze(0)
    except Exception as e:
        raise ValueError(f"Input processing error: {str(e)}")

@app.route('/api/recommendations', methods=['POST'])
def get_recommendations():
    try:
        data = request.json
        if not data:
            return jsonify({'status': 'error', 'message': 'No input data provided'}), 400
            
        # Get movies first
        movies = search_movies_omdb(
            data.get('preferredGenres', [''])[0],
            data.get('yearRange', {}).get('from'),
            data.get('yearRange', {}).get('to')
        )
        
        if not movies:
            return jsonify({'status': 'error', 'message': 'No movies found'}), 404
            
        # Prepare batch predictions
        user_features = [{
            'year': data.get('yearRange', {}).get('from', 2023),
            'month': time.localtime().tm_mon,
            'day': time.localtime().tm_mday,
            'gender': data.get('gender', 'M'),
            'occupation': data.get('occupation', 'other')
        } for _ in movies]
        
        # Get batch predictions
        predictions = batch_predict_ratings(user_features, model, preprocessor, device)
        
        # Process results
        recommendations = []
        seen = set()
        
        for movie, pred in zip(movies, predictions):
            if len(recommendations) >= 10 or movie['imdbID'] in seen:
                continue
                
            details = get_movie_details_omdb(movie['imdbID'])
            if not details:
                continue
                
            seen.add(movie['imdbID'])
            details['predicted_rating'] = float(pred)
            recommendations.append(details)
        
        if not recommendations:
            return jsonify({'status': 'error', 'message': 'No matching movies'}), 404
            
        recommendations.sort(key=lambda x: (-x['predicted_rating'], -x['rating']))
        return jsonify({'status': 'success', 'recommendations': recommendations})
        
    except Exception as e:
        logger.exception("Recommendation error")
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/api/health', methods=['GET'])
def health_check():
    return jsonify({'status': 'healthy'})

@app.route('/api/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        prediction = predict_rating(data, model, preprocessor, device)
        
        if prediction is None:
            return jsonify({'error': 'Prediction failed'}), 500
            
        return jsonify({'predicted_rating': float(prediction[0])}), 200
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/', defaults={'path': ''})
@app.route('/<path:path>')
def serve(path):
    try:
        build_dir = Path('../frontend/recommend/build').resolve()
        if path and (build_dir / path).exists():
            return send_from_directory(str(build_dir), path)
        return send_from_directory(str(build_dir), 'index.html')
    except Exception as e:
        logger.error(f"Static file serve error: {str(e)}")
        return jsonify({'status': 'error', 'message': 'Not found'}), 404

@app.route('/favicon.ico')
def favicon():
    return '', 204

@app.errorhandler(404)
def not_found(e):
    return jsonify({'status': 'error', 'message': 'Resource not found'}), 404

if __name__ == '__main__':
    app.run(debug=True, port=5000)