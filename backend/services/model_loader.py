import pickle
import pandas as pd
import logging
from backend.config import MODEL_PATH, FEATURE_NAMES_PATH, PREPROCESSOR_PATH

logger = logging.getLogger(__name__)

# Global variables (loaded at startup)
model = None
preprocessor = None
all_features = []
feature_importances = None


def load_model():
    global model, preprocessor, all_features, feature_importances

    try:
        with open(MODEL_PATH, 'rb') as f:
            model = pickle.load(f)

        with open(FEATURE_NAMES_PATH, 'rb') as f:
            all_features = pickle.load(f)

        with open(PREPROCESSOR_PATH, 'rb') as f:
            preprocessor = pickle.load(f)

        if hasattr(model, 'feature_importances_'):
            feature_importances = pd.Series(
                model.feature_importances_,
                index=all_features
            ).sort_values(ascending=False)

    except Exception as e:
        logger.error(f"Model loading failed: {str(e)}", exc_info=True)
        raise


# Initialize on import
load_model()