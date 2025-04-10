from flask import Blueprint, request, jsonify, current_app
import logging
import pandas as pd
from backend.services.all_models_loader import loaded_models, get_available_models, get_model_artifacts
from backend.config import EXPECTED_FEATURES

predict_multi_bp = Blueprint('predict_multi', __name__)
logger = logging.getLogger(__name__)

def _validate_input(input_data: dict) -> tuple or None:
    """Validation Function (same as before)"""
    if not input_data:
        logger.error("No input data received")
        return jsonify({'error': 'No input data'}), 400

    if 'features' not in input_data:
        logger.error("Missing 'features' key")
        return jsonify({'error': 'Missing features'}), 400

    if 'TIME' in input_data['features']:
        try:
            hours, minutes = map(float, input_data['features']['TIME'].split(':'))
            input_data['features']['TIME'] = hours + minutes / 60
        except (ValueError, AttributeError):
            logger.error("Invalid TIME format - expected HH:MM")
            return jsonify({
                'error': 'Invalid TIME format',
                'expected_format': 'HH:MM'
            }), 400

    missing = set(EXPECTED_FEATURES) - set(input_data['features'].keys())
    if missing:
        logger.warning(f"Missing features: {missing}")
        return jsonify({'error': f'Missing features: {list(missing)}'}), 400

    if 'model_name' not in input_data:
        logger.error("Missing 'model_name' in request")
        return jsonify({'error': 'Missing model_name'}), 400

    if input_data['model_name'] not in get_available_models():
        logger.error(f"Model '{input_data['model_name']}' not found")
        return jsonify({'error': f"Model '{input_data['model_name']}' not found"}), 400

    return None

@predict_multi_bp.route('/available_models', methods=['GET'])
def available_models():
    return jsonify({'models': get_available_models()})

@predict_multi_bp.route('/predict_multi', methods=['POST'])
def predict_with_model():
    try:
        data = request.get_json()
        if error := _validate_input(data):
            return error

        model_name = data['model_name']
        model_artifacts = get_model_artifacts(model_name)

        if not model_artifacts:
            logger.error(f"Could not retrieve artifacts for model '{model_name}'")
            return jsonify({'error': f"Could not retrieve artifacts for model '{model_name}'"}), 500

        selected_model = model_artifacts.get('model')
        selected_preprocessor = model_artifacts.get('preprocessor')

        if not selected_model or not selected_preprocessor:
            logger.error(f"Missing model or preprocessor for '{model_name}'")
            return jsonify({'error': f"Missing model or preprocessor for '{model_name}'"}), 500

        # Process input
        input_df = pd.DataFrame([data['features']])
        processed_input = selected_preprocessor.transform(input_df)

        # Get feature names after transformation
        try:
            feature_names_after_transform = selected_preprocessor.get_feature_names_out().tolist()
            processed_df = pd.DataFrame(processed_input, columns=feature_names_after_transform)
        except Exception as e:
            current_app.logger.warning(f"Could not retrieve feature names after transform: {e}")
            processed_df = pd.DataFrame(processed_input) # No column names

        proba = selected_model.predict_proba(processed_df)[0][1] if hasattr(selected_model, 'predict_proba') else None
        prediction = 1 if proba > 0.5 else 0

        logger.info(f"Prediction using '{model_name}': {prediction} (probability of class 1={proba:.2f})")
        return jsonify({
            'model_name': model_name,
            'prediction': prediction,
            'probability': round(proba, 4) if proba else None
        })

    except Exception as e:
        logger.critical(f"Prediction failed: {str(e)}", exc_info=True)
        return jsonify({'error': 'Internal server error'}), 500