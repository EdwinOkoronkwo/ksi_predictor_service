import pickle
import sys
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify
from flask_cors import CORS
from pathlib import Path
import logging

# Add root directory to sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT_DIR))

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration paths
BASE_DIR = Path(__file__).parent.parent
MODEL_PATH = BASE_DIR / "saved_models/best_model.pkl"
FEATURE_NAMES_PATH = BASE_DIR / "saved_models/feature_names.pkl"
PREPROCESSOR_PATH = BASE_DIR / "saved_models/preprocessor.pkl"

# Global variables for model and metadata
model = None
preprocessor = None
all_features = []
feature_importances = None

def load_model():
    global model, preprocessor, all_features, feature_importances
    try:
        logger.info("Loading model from %s", MODEL_PATH)
        with open(MODEL_PATH, 'rb') as f:
            model = pickle.load(f)

        logger.info("Loading feature names from %s", FEATURE_NAMES_PATH)
        with open(FEATURE_NAMES_PATH, 'rb') as f:
            all_features = pickle.load(f)
        if not isinstance(all_features, list):
            all_features = list(all_features)

        logger.info("Loading preprocessor from %s", PREPROCESSOR_PATH)
        with open(PREPROCESSOR_PATH, 'rb') as f:
            preprocessor = pickle.load(f)

        # Feature importances (if model supports it)
        if hasattr(model, 'feature_importances_'):
            feature_importances = pd.Series(model.feature_importances_, index=all_features).sort_values(ascending=False)
            logger.info("Top 10 feature importances: %s", feature_importances.head(10).to_dict())

        logger.info("Model and preprocessor loaded successfully with %d features", len(all_features))

    except Exception as e:
        logger.error("Error loading model or preprocessor: %s", str(e), exc_info=True)
        raise

# Load everything on startup
load_model()

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        if not data or 'features' not in data:
            return jsonify({'error': 'Missing features'}), 400

        input_data = data['features']
        expected_columns = ['TIME', 'LATITUDE', 'LONGITUDE', 'ROAD_CLASS', 'DISTRICT',
                           'ACCLOC', 'TRAFFCTL', 'VISIBILITY', 'LIGHT', 'RDSFCOND']

        # Validate input features
        input_keys = set(input_data.keys())
        expected_keys = set(expected_columns)
        missing_features = expected_keys - input_keys
        extra_features = input_keys - expected_keys

        if missing_features:
            return jsonify({'error': f'Missing required features: {missing_features}'}), 400
        if extra_features:
            logger.warning("Ignoring extra features: %s", extra_features)

        # Convert input to DataFrame with a single row
        input_df = pd.DataFrame([input_data])[expected_columns]  # Ensure column order matches

        # Preprocess input
        processed_input = preprocessor.transform(input_df)

        # Predict
        prediction = int(model.predict(processed_input)[0])
        proba = float(model.predict_proba(processed_input)[0][prediction]) if hasattr(model, 'predict_proba') else None

        logger.info("Prediction: %d (Confidence: %.4f)", prediction, proba if proba is not None else -1)

        return jsonify({
            'prediction': prediction,
            'probability': round(proba, 4) if proba is not None else None,
            'status': 'success'
        })

    except Exception as e:
        logger.error("Prediction error: %s", str(e), exc_info=True)
        return jsonify({'error': str(e)}), 500


@app.route('/feature_info', methods=['GET'])
def feature_info():
    try:
        logger.info("Feature metadata requested")
        return jsonify({
            'total_features': len(all_features),
            'feature_names': all_features,
            'feature_importances': feature_importances.to_dict() if feature_importances is not None else None
        })
    except Exception as e:
        logger.error("Feature info error: %s", str(e), exc_info=True)
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    logger.info("🚀 Starting Flask server on port 5000")
    app.run(host='0.0.0.0', port=5000, debug=True)
