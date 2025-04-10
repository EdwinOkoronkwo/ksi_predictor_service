from pathlib import Path
BASE_DIR = Path(__file__).parent.parent

# Paths
MODEL_PATH = BASE_DIR / "saved_models/best_model.pkl"
FEATURE_NAMES_PATH = BASE_DIR / "saved_models/feature_names.pkl"
PREPROCESSOR_PATH = BASE_DIR / "saved_models/preprocessor.pkl"

# Expected features
EXPECTED_FEATURES = [
            'IMPACTYPE',
            'LATITUDE',
            'VEHTYPE',
            'LONGITUDE',
            'TIME',
            'MANOEUVER',
            'DRIVACT',
            'INITDIR',
            'TRAFFCTL',
            'INVAGE',
            'DRIVCOND',
            'INVTYPE'
        ]