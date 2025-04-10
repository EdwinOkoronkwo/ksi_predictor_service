from flask import Blueprint, request, jsonify
import logging
import pandas as pd
from backend.services.model_loader import model, preprocessor
from backend.config import EXPECTED_FEATURES

predict_bp = Blueprint('predict', __name__)
logger = logging.getLogger(__name__)


def _validate_input(input_data: dict) -> tuple or None:
    """Validation Function"""
    if not input_data:
        logger.error("No input data received")
        return jsonify({'error': 'No input data'}), 400

    if 'features' not in input_data:
        logger.error("Missing 'features' key")
        return jsonify({'error': 'Missing features'}), 400

    if 'TIME' in input_data['features']:
        try:
            # Convert "HH:MM" to decimal hours (e.g., "12:30" → 12.5)
            hours, minutes = map(float, input_data['features']['TIME'].split(':'))
            input_data['features']['TIME'] = hours + minutes / 60
        except (ValueError, AttributeError):
            logger.error("Invalid TIME format - expected HH:MM")
            return jsonify({
                'error': 'Invalid TIME format',
                'expected_format': 'HH:MM'
            }), 400

    # Check features
    missing = set(EXPECTED_FEATURES) - set(input_data['features'].keys())
    if missing:
        logger.warning(f"Missing features: {missing}")
        return jsonify({'error': f'Missing features: {list(missing)}'}), 400

    return None


@predict_bp.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        if error := _validate_input(data):
            return error

        # Process input
        input_df = pd.DataFrame([data['features']])
        processed_input = preprocessor.transform(input_df)

        # Get feature names after transformation
        try:
            feature_names_after_transform = preprocessor.get_feature_names_out().tolist()
            processed_df = pd.DataFrame(processed_input, columns=feature_names_after_transform)
        except Exception as e:
            app.logger.warning(f"Could not retrieve feature names after transform: {e}")
            processed_df = pd.DataFrame(processed_input) # No column names

        proba = model.predict_proba(processed_df)[0][1] if hasattr(model, 'predict_proba') else None
        prediction = 1 if proba > 0.5 else 0

        logger.info(f"Prediction: {prediction} (probability of class 1={proba:.2f})")
        return jsonify({
            'prediction': prediction,
            'probability': round(proba, 4) if proba else None
        })

    except Exception as e:
        logger.critical(f"Prediction failed: {str(e)}", exc_info=True)
        return jsonify({'error': 'Internal server error'}), 500
# @predict_bp.route('/predict', methods=['POST'])
# def predict():
#     try:
#         data = request.get_json()
#         if error := _validate_input(data):
#             return error
#
#         # Process input
#         input_df = pd.DataFrame([data['features']])
#         processed_input = preprocessor.transform(input_df)
#         prediction = int(model.predict(processed_input)[0])
#         proba = model.predict_proba(processed_input)[0][prediction] if hasattr(model, 'predict_proba') else None
#
#         logger.info(f"Prediction: {prediction} (confidence={proba:.2f})")
#         return jsonify({
#             'prediction': prediction,
#             'probability': round(proba, 4) if proba else None
#         })
#
#     except Exception as e:
#         logger.critical(f"Prediction failed: {str(e)}", exc_info=True)
#         return jsonify({'error': 'Internal server error'}), 500