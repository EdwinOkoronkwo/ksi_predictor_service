import pickle
import pandas as pd
import logging
from pathlib import Path
from backend.config import MODEL_PATH, PREPROCESSOR_PATH

logger = logging.getLogger(__name__)

model = None
preprocessor = None
feature_importances = None
all_features = []

def load_model():
    global model, preprocessor, feature_importances, all_features

    try:
        # Load model
        with open(MODEL_PATH, 'rb') as f:
            model = pickle.load(f)

        # Load preprocessor
        with open(PREPROCESSOR_PATH, 'rb') as f:
            preprocessor = pickle.load(f)

        # Get feature names from preprocessor
        if hasattr(preprocessor, 'feature_names_in_'):
            all_features = list(preprocessor.feature_names_in_)

        # Get feature names from ColumnTransformers
        elif hasattr(preprocessor, 'get_feature_names_out'):
            all_features = preprocessor.get_feature_names_out()
        else:
            raise ValueError("Preprocessor doesn't expose feature names")

        # Calculate feature importances
        if hasattr(model, 'feature_importances_'):
            if len(model.feature_importances_) == len(all_features):
                feature_importances = pd.Series(
                    model.feature_importances_,
                    index=all_features
                ).sort_values(ascending=False)
            else:
                logger.warning(
                    f"Feature mismatch - Model has {len(model.feature_importances_)} features, but expected {len(all_features)}")
                feature_importances = None

        logger.info(f"Loaded model with features: {all_features}")

    except Exception as e:
        logger.error(f"Model loading failed: {str(e)}", exc_info=True)
        raise



load_model()