import os
import pickle
import logging
import pandas as pd
from backend.config import MODEL_PATH, PREPROCESSOR_PATH  # You might not need these anymore

logger = logging.getLogger(__name__)

loaded_models = {}

def load_all_models(models_folder="saved_models_all"):
    print(f"Loading models from folder: {models_folder}")
    global loaded_models
    loaded_models = {}
    try:
        if not os.path.exists(models_folder):
            logger.warning(f"Folder '{models_folder}' not found.")
            return {}

        for filename in os.listdir(models_folder):
            if filename.endswith("_model.pkl"):
                model_name = filename.replace("_model.pkl", "")
                model_path = os.path.join(models_folder, filename)
                pipeline_path = os.path.join(models_folder, f"{model_name}_preprocessor.pkl")
                feature_names_path = os.path.join(models_folder, f"{model_name}_feature_names.pkl")

                loaded_model = None
                loaded_pipeline = None
                loaded_feature_names = None

                try:
                    with open(model_path, 'rb') as f:
                        loaded_model = pickle.load(f)
                    logger.info(f"Loaded model '{model_name}' from '{model_path}'")
                except FileNotFoundError:
                    logger.warning(f"Model file not found for '{model_name}' at '{model_path}'.")
                except Exception as e:
                    logger.error(f"Failed to load model '{model_name}' from '{model_path}': {e}")

                try:
                    with open(pipeline_path, 'rb') as f:
                        loaded_pipeline = pickle.load(f)
                    logger.info(f"Loaded pipeline for '{model_name}' from '{pipeline_path}'")
                except FileNotFoundError:
                    logger.warning(f"Pipeline file not found for '{model_name}' at '{pipeline_path}'.")
                except Exception as e:
                    logger.error(f"Failed to load pipeline for '{model_name}' from '{pipeline_path}': {e}")

                try:
                    with open(feature_names_path, 'rb') as f:
                        loaded_feature_names = pickle.load(f)
                    logger.info(f"Loaded feature names for '{model_name}' from '{feature_names_path}'")
                except FileNotFoundError:
                    logger.warning(f"Feature names file not found for '{model_name}' at '{feature_names_path}'.")
                except Exception as e:
                    logger.error(f"Failed to load feature names for '{model_name}' from '{feature_names_path}': {e}")

                if loaded_model and loaded_pipeline and loaded_feature_names:
                    loaded_models[model_name] = {
                        'model': loaded_model,
                        'preprocessor': loaded_pipeline,
                        'feature_names': loaded_feature_names
                    }
                elif loaded_model:
                    loaded_models[model_name] = {'model': loaded_model}
                    if loaded_pipeline:
                        loaded_models[model_name]['preprocessor'] = loaded_pipeline
                    if loaded_feature_names:
                        loaded_models[model_name]['feature_names'] = loaded_feature_names

        logger.info(f"Loaded {len(loaded_models)} models.")
        return loaded_models

    except Exception as e:
        logger.error(f"Unexpected error during model loading: {e}")
        return {}


load_all_models()

def get_available_models():
    return list(loaded_models.keys())

def get_model_artifacts(model_name):
    return loaded_models.get(model_name)



# # Only run this if the script is executed directly, not when imported
# if __name__ == "__main__":
#     load_all_models()
