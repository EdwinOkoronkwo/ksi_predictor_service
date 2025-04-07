from flask import Blueprint, jsonify
import logging
from backend.config import EXPECTED_FEATURES

feature_bp = Blueprint('feature_info', __name__)
logger = logging.getLogger(__name__)


@feature_bp.route('/feature_info', methods=['GET'])
def get_feature_info():
    """Returns the list of raw features expected by the model"""
    try:
        return jsonify({
            "status": "success",
            "data": {
                "required_features": EXPECTED_FEATURES,
                "count": len(EXPECTED_FEATURES),
            }
        })

    except Exception as e:
        logger.critical(f"Feature info endpoint failed: {str(e)}")
        return jsonify({
            "status": "error",
            "error": "Could not retrieve feature information",
            "solution": "Please try again or contact support"
        }), 500
