# utils.py

import logging
import pickle
import logging
import numpy as np
import pandas as pd
import os
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score



class ColoredFormatter(logging.Formatter):
    """Formatter adding color to log records."""

    COLORS = {
        'WARNING': '\033[33m',  # Yellow (You can change this if you don't want yellow)
        'INFO': '\033[34m',     # Blue
        'DEBUG': '\033[34m',    # Blue
        'CRITICAL': '\033[35m', # Magenta
        'ERROR': '\033[31m',    # Red (Changed to red)
        'RESET': '\033[0m'      # Reset color
    }

    def format(self, record):
        log_color = self.COLORS.get(record.levelname, '')
        message = super().format(record)
        return f'{log_color}{message}{self.COLORS["RESET"]}'

def setup_logger(name):
    """Sets up a colored logger."""
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)  # Or your desired level

    handler = logging.StreamHandler()
    formatter = ColoredFormatter('%(asctime)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    return logger




class SaveLoadBestModel:
    def __init__(self, model_file_path, pipeline_file_path, feature_names_file_path, logger=None, folder="saved_models"):
        """Initializes the SaveLoadBestModel class for loading."""
        self.logger = logger or logging.getLogger(__name__)
        self.folder = folder  # Folder where models are saved
        self.model_file_path = model_file_path
        self.pipeline_file_path = pipeline_file_path
        self.feature_names_file_path = feature_names_file_path

    def save_model_and_pipeline(self, model, pipeline, feature_names):
        """Saves the model and pipeline to separate pickle files."""
        os.makedirs(self.folder, exist_ok=True)  # Create folder if it doesn't exist

        try:
            with open(os.path.join(self.folder, "best_model.pkl"), 'wb') as model_file:
                pickle.dump(model, model_file)
            self.logger.info(f"Model saved to '{os.path.join(self.folder, 'best_model.pkl')}'")

            with open(os.path.join(self.folder, "preprocessor.pkl"), 'wb') as pipeline_file:
                pickle.dump(pipeline, pipeline_file)
            self.logger.info(f"Pipeline saved to '{os.path.join(self.folder, 'preprocessor.pkl')}'")

            with open(os.path.join(self.folder, "feature_names.pkl"), 'wb') as feature_names_file:
                pickle.dump(feature_names, feature_names_file)
            self.logger.info(f"Feature names saved to '{os.path.join(self.folder, 'feature_names.pkl')}'")

        except Exception as e:
            self.logger.error(f"Failed to save model, pipeline, or feature names: {e}")

    def load_model(self):
        """Loads the model from the specified pickle file."""
        try:
            with open(self.model_file_path, 'rb') as model_file:
                loaded_model = pickle.load(model_file)
            self.logger.info(f"Model loaded from '{self.model_file_path}'")
            return loaded_model
        except FileNotFoundError:
            self.logger.error(f"Model file not found at: '{self.model_file_path}'.")
            raise
        except Exception as e:
            self.logger.error(f"Failed to load model from '{self.model_file_path}': {e}")
            raise

    def load_pipeline(self):
        """Loads the pipeline from the specified pickle file."""
        try:
            with open(self.pipeline_file_path, 'rb') as pipeline_file:
                loaded_pipeline = pickle.load(pipeline_file)
            self.logger.info(f"Pipeline loaded from '{self.pipeline_file_path}'")
            return loaded_pipeline
        except FileNotFoundError:
            self.logger.error(f"Pipeline file not found at: '{self.pipeline_file_path}'.")
            raise
        except Exception as e:
            self.logger.error(f"Failed to load pipeline from '{self.pipeline_file_path}': {e}")
            raise

    def load_feature_names(self):
        """Loads the feature names from the specified pickle file."""
        try:
            with open(self.feature_names_file_path, 'rb') as feature_names_file:
                loaded_feature_names = pickle.load(feature_names_file)
            self.logger.info(f"Feature names loaded from '{self.feature_names_file_path}'")
            return loaded_feature_names
        except FileNotFoundError:
            self.logger.error(f"Feature names file not found at: '{self.feature_names_file_path}'.")
            raise
        except Exception as e:
            self.logger.error(f"Failed to load feature names from '{self.feature_names_file_path}': {e}")
            raise

    def evaluate_loaded_model(self, loaded_model, loaded_pipeline, loaded_feature_names, X_test, y_test):
        """Evaluates the loaded model on the test set."""
        if loaded_model is None or loaded_pipeline is None or loaded_feature_names is None:
            self.logger.error("Model, pipeline, or feature names not loaded.")
            return

        try:
            X_test_transformed = loaded_pipeline.transform(X_test)
            X_test_df = pd.DataFrame(X_test_transformed, columns=loaded_feature_names)
            self.logger.info(f"Shape of transformed test data: {X_test_df.shape}")
            self.logger.info(f"First 5 columns of transformed test data: {X_test_df.columns[:5].tolist()}")

            y_pred_proba = loaded_model.predict_proba(X_test_df)
            y_pred = np.argmax(y_pred_proba, axis=1) # For multi-class

            accuracy = accuracy_score(y_test, y_pred)
            report = classification_report(y_test, y_pred, zero_division=0)
            roc_auc = roc_auc_score(y_test, y_pred_proba, multi_class='ovr') # For multi-class

            self.logger.info(f"Evaluation of loaded model:")
            self.logger.info(f"Accuracy: {accuracy:.4f}")
            self.logger.info(f"ROC AUC (OVR): {roc_auc:.4f}")
            self.logger.info(f"Classification Report:\n{report}")
        except Exception as e:
            self.logger.error(f"Model evaluation failed: {e}", exc_info=True)





class SaveLoadAllModels:
    def __init__(self, folder="saved_models_all", logger=None):
        """Initializes the SaveLoadAllModels class."""
        self.logger = logger or logging.getLogger(__name__)
        self.folder = folder
        os.makedirs(self.folder, exist_ok=True)  # Create folder if it doesn't exist

    def save_model_artifacts(self, model, pipeline, feature_names, model_name):
        """Saves the model, pipeline, and feature names to pickle files."""
        model_filename = f"{model_name}_model.pkl"
        pipeline_filename = f"{model_name}_preprocessor.pkl"
        feature_names_filename = f"{model_name}_feature_names.pkl"

        try:
            with open(os.path.join(self.folder, model_filename), 'wb') as f:
                pickle.dump(model, f)
            self.logger.info(f"Model '{model_name}' saved to '{os.path.join(self.folder, model_filename)}'")

            with open(os.path.join(self.folder, pipeline_filename), 'wb') as f:
                pickle.dump(pipeline, f)
            self.logger.info(f"Pipeline for '{model_name}' saved to '{os.path.join(self.folder, pipeline_filename)}'")

            with open(os.path.join(self.folder, feature_names_filename), 'wb') as f:
                pickle.dump(feature_names, f)
            self.logger.info(f"Feature names for '{model_name}' saved to '{os.path.join(self.folder, feature_names_filename)}'")

            return True
        except Exception as e:
            self.logger.error(f"Failed to save artifacts for model '{model_name}': {e}")
            return False

    def load_model_artifacts(self, model_name):
        """Loads the model, pipeline, and feature names for a given model name."""
        model_path = os.path.join(self.folder, f"{model_name}_model.pkl")
        pipeline_path = os.path.join(self.folder, f"{model_name}_preprocessor.pkl")
        feature_names_path = os.path.join(self.folder, f"{model_name}_feature_names.pkl")

        loaded_model = None
        loaded_pipeline = None
        loaded_feature_names = None

        try:
            with open(model_path, 'rb') as f:
                loaded_model = pickle.load(f)
            self.logger.info(f"Model '{model_name}' loaded from '{model_path}'")
        except FileNotFoundError:
            self.logger.warning(f"Model file not found for '{model_name}' at '{model_path}'.")
        except Exception as e:
            self.logger.error(f"Failed to load model '{model_name}' from '{model_path}': {e}")

        try:
            with open(pipeline_path, 'rb') as f:
                loaded_pipeline = pickle.load(f)
            self.logger.info(f"Pipeline for '{model_name}' loaded from '{pipeline_path}'")
        except FileNotFoundError:
            self.logger.warning(f"Pipeline file not found for '{model_name}' at '{pipeline_path}'.")
        except Exception as e:
            self.logger.error(f"Failed to load pipeline for '{model_name}' from '{pipeline_path}': {e}")

        try:
            with open(feature_names_path, 'rb') as f:
                loaded_feature_names = pickle.load(f)
            self.logger.info(f"Feature names for '{model_name}' loaded from '{feature_names_path}'")
        except FileNotFoundError:
            self.logger.warning(f"Feature names file not found for '{model_name}' at '{feature_names_path}'.")
        except Exception as e:
            self.logger.error(f"Failed to load feature names for '{model_name}' from '{feature_names_path}': {e}")

        return loaded_model, loaded_pipeline, loaded_feature_names




class SaveLoadBestModel:
    def __init__(self, model_file_path, pipeline_file_path, feature_names_file_path, logger=None, folder="saved_models"):
        """Initializes the SaveLoadBestModel class for loading."""
        self.logger = logger or logging.getLogger(__name__)
        self.folder = folder  # Folder where models are saved
        self.model_file_path = model_file_path
        self.pipeline_file_path = pipeline_file_path
        self.feature_names_file_path = feature_names_file_path
        os.makedirs(self.folder, exist_ok=True)  # Create folder if it doesn't exist

    def save_model_and_pipeline(self, model, pipeline, feature_names):
        """Saves the model and pipeline to separate pickle files."""
        try:
            with open(self.model_file_path, 'wb') as model_file:
                pickle.dump(model, model_file)
            self.logger.info(f"Best model saved to '{self.model_file_path}'")

            with open(self.pipeline_file_path, 'wb') as pipeline_file:
                pickle.dump(pipeline, pipeline_file)
            self.logger.info(f"Best pipeline saved to '{self.pipeline_file_path}'")

            with open(self.feature_names_file_path, 'wb') as feature_names_file:
                pickle.dump(feature_names, feature_names_file)
            self.logger.info(f"Best feature names saved to '{self.feature_names_file_path}'")

        except Exception as e:
            self.logger.error(f"Failed to save best model artifacts: {e}")

    def load_model(self):
        """Loads the model from the specified pickle file."""
        try:
            with open(self.model_file_path, 'rb') as model_file:
                loaded_model = pickle.load(model_file)
            self.logger.info(f"Best model loaded from '{self.model_file_path}'")
            return loaded_model
        except FileNotFoundError:
            self.logger.error(f"Best model file not found at: '{self.model_file_path}'.")
            raise
        except Exception as e:
            self.logger.error(f"Failed to load best model from '{self.model_file_path}': {e}")
            raise

    def load_pipeline(self):
        """Loads the pipeline from the specified pickle file."""
        try:
            with open(self.pipeline_file_path, 'rb') as pipeline_file:
                loaded_pipeline = pickle.load(pipeline_file)
            self.logger.info(f"Best pipeline loaded from '{self.pipeline_file_path}'")
            return loaded_pipeline
        except FileNotFoundError:
            self.logger.error(f"Best pipeline file not found at: '{self.pipeline_file_path}'.")
            raise
        except Exception as e:
            self.logger.error(f"Failed to load best pipeline from '{self.pipeline_file_path}': {e}")
            raise

    def load_feature_names(self):
        """Loads the feature names from the specified pickle file."""
        try:
            with open(self.feature_names_file_path, 'rb') as feature_names_file:
                loaded_feature_names = pickle.load(feature_names_file)
            self.logger.info(f"Best feature names loaded from '{self.feature_names_file_path}'")
            return loaded_feature_names
        except FileNotFoundError:
            self.logger.error(f"Best feature names file not found at: '{self.feature_names_file_path}'.")
            raise
        except Exception as e:
            self.logger.error(f"Failed to load best feature names from '{self.feature_names_file_path}': {e}")
            raise

    def evaluate_loaded_model(self, loaded_model, loaded_pipeline, loaded_feature_names, X_test, y_test):
        """Evaluates the loaded model on the test set."""
        if loaded_model is None or loaded_pipeline is None or loaded_feature_names is None:
            self.logger.error("Model, pipeline, or feature names not loaded.")
            return

        try:
            X_test_transformed = loaded_pipeline.transform(X_test)
            X_test_df = pd.DataFrame(X_test_transformed, columns=loaded_feature_names)
            self.logger.info(f"Shape of transformed test data: {X_test_df.shape}")
            self.logger.info(f"First 5 columns of transformed test data: {X_test_df.columns[:5].tolist()}")

            y_pred_proba = loaded_model.predict_proba(X_test_df)
            y_pred = np.argmax(y_pred_proba, axis=1) # For multi-class

            accuracy = accuracy_score(y_test, y_pred)
            report = classification_report(y_test, y_pred, zero_division=0)
            roc_auc = roc_auc_score(y_test, y_pred_proba, multi_class='ovr') # For multi-class

            self.logger.info(f"Evaluation of loaded model:")
            self.logger.info(f"Accuracy: {accuracy:.4f}")
            self.logger.info(f"ROC AUC (OVR): {roc_auc:.4f}")
            self.logger.info(f"Classification Report:\n{report}")
        except Exception as e:
            self.logger.error(f"Model evaluation failed: {e}", exc_info=True)

