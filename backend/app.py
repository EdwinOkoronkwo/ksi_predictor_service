import pickle
import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS
from pathlib import Path
import logging
import pandas as pd

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration
BASE_DIR = Path(__file__).parent.parent
MODEL_PATH = BASE_DIR / "saved_models/best_model.pkl"

# Global variables
model = None
all_features = None
feature_importances = None
top_n_features = 15  # Default number of top features to use


def load_model():
    global model, all_features, feature_importances

    try:
        logger.info("Loading model from %s", MODEL_PATH)
        with open(MODEL_PATH, 'rb') as f:
            model = pickle.load(f)

        # Get feature names
        all_features = (model.feature_names_in_.tolist()
                        if hasattr(model, 'feature_names_in_')
                        else [f'feature_{i}' for i in range(model.n_features_in_)])

        # Get feature importances
        if hasattr(model, 'feature_importances_'):
            logger.info("Extracting feature importances")
            feature_importances = (pd.Series(model.feature_importances_, index=all_features).sort_values(ascending=False))
            logger.info("Top 10 features by importance: %s",
                        feature_importances.head(10).to_dict())

        logger.info("Successfully loaded model with %d features", len(all_features))

    except Exception as e:
        logger.error("Model loading failed: %s", str(e), exc_info=True)
        raise


# Load model at startup
load_model()

# include the important features
TOP_FEATURES = [
    'categorical__IMPACTYPE_Pedestrian Collisions',
    'numerical__LATITUDE',
    'categorical__VEHTYPE_Automobile, Station Wagon',
    'numerical__LONGITUDE',
    'categorical__INITDIR_East',
    'numerical__combined_xy',
    'categorical__INITDIR_West',
    'categorical__MANOEUVER_Going Ahead',
    'categorical__ROAD_CLASS_Major Arterial',
    'categorical__DRIVACT_Driving Properly',
    'categorical__LIGHT_Daylight',
    'categorical__LIGHT_Dark, artificial',
    'categorical__TRAFFCTL_Traffic Signal',
    'categorical__INITDIR_South',
    'categorical__INITDIR_North'
]
@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        if not data or 'features' not in data:
            logger.warning("Missing features in request")
            return jsonify({'error': 'Missing features'}), 400

        # Create full feature dict with defaults
        input_values = {f: 0 for f in all_features}

        # Handle different input formats
        if isinstance(data['features'], dict):
            logger.debug("Received dictionary input")
            # Prioritize top features if they exist in input
            for feature in TOP_FEATURES:
                if feature in data['features']:
                    input_values[feature] = data['features'][feature]
            # then any other features
            input_values.update(data['features'])

        elif isinstance(data['features'], list):
            if len(data['features']) == len(all_features):
                logger.debug("Received full feature list")
                input_values.update(zip(all_features, data['features']))
            elif len(data['features']) <= len(TOP_FEATURES):
                logger.debug("Received top feature list")
                input_values.update(zip(TOP_FEATURES[:len(data['features'])], data['features']))
            else:
                logger.warning("Invalid list length %d (expected %d or ≤%d)",len(data['features']), len(all_features), len(TOP_FEATURES))
                return jsonify({
                    'error': f'For list input, provide either all {len(all_features)} features or up to {len(TOP_FEATURES)} top features',
                    'top_features': TOP_FEATURES
                }), 400

        # Prepare final input
        X = np.array([input_values[f] for f in all_features]).reshape(1, -1)

        # Predict
        pred = int(model.predict(X)[0])
        proba = float(model.predict_proba(X)[0][pred]) if hasattr(model, "predict_proba") else None

        # which top features were used
        used_top_features = [f for f in TOP_FEATURES if input_values[f] != 0]
        logger.info("Prediction: %d (confidence: %.2f) using %d top features: %s",
                    pred, proba if proba else 0,
                    len(used_top_features),
                    used_top_features)

        return jsonify({
            'prediction': pred,
            'probability': proba,
            'top_features_used': used_top_features,
            'status': 'success'
        })

    except Exception as e:
        logger.error("Prediction failed: %s", str(e), exc_info=True)
        return jsonify({'error': str(e)}), 500


# Update feature info endpoint
@app.route('/feature_info', methods=['GET'])
def feature_info():
    """Get feature metadata"""
    logger.info("📋 Feature info requested")
    return jsonify({
        'total_features': len(all_features),
        'top_features': TOP_FEATURES,
        'feature_importances': {k: v for k, v in feature_importances.to_dict().items() if k in TOP_FEATURES}
    })

if __name__ == '__main__':
    logger.info("Starting Flask server")
    app.run(host='0.0.0.0', port=5000, debug=True)