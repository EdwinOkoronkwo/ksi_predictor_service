# predictor.py

import logging
import os
import pandas as pd
from flask import Flask, request, jsonify, render_template
from utils import SaveLoadBestModel, setup_logger
import pickle
import numpy as np
from set_up import (
    drop_irrelevant_and_high_missing_columns,
    process_time_column,
    drop_highly_correlated_features,
)

class FlaskPredictor:
    def __init__(self):
        self.app = Flask(__name__)
        self.logger = setup_logger(__name__)
        self.model = None
        self.pipeline = None
        self.feature_names = None
        self._load_model_and_pipeline()
        self._register_routes()

    def _load_model_and_pipeline(self):
        model_path = os.path.abspath('saved_models/best_model.pkl')
        pipeline_path = os.path.abspath('saved_models/preprocessor.pkl')
        feature_names_path = os.path.abspath('saved_models/feature_names.pkl')

        save_load = SaveLoadBestModel(model_path, pipeline_path, feature_names_path, self.logger)

        try:
            if os.path.exists(model_path):
                self.model = save_load.load_model()
                self.logger.info("Model loaded successfully.")
                self.logger.info(f"Loaded model type: {type(self.model)}")
            else:
                self.logger.error(f"Model file not found at: {model_path}")

            if os.path.exists(pipeline_path):
                self.pipeline = save_load.load_pipeline()
                self.logger.info("Pipeline loaded successfully.")
                self.logger.info(f"Loaded pipeline type: {type(self.pipeline)}")
            else:
                self.logger.warning(f"Pipeline file not found at: {pipeline_path}. Prediction might fail.")

            if os.path.exists(feature_names_path):
                self.feature_names = save_load.load_feature_names()
                self.logger.info("Feature names loaded successfully.")
                self.logger.info(f"Loaded feature names type: {type(self.feature_names)}")
                if isinstance(self.feature_names, list):
                    self.logger.info(f"First 10 loaded feature names: {self.feature_names[:10]}")
                else:
                    self.logger.warning("Loaded feature names is not a list.")
            else:
                self.logger.warning(f"Feature names file not found at: {feature_names_path}.")

        except Exception as e:
            self.logger.error(f"Error loading model or pipeline: {e}")
            raise

    def _register_routes(self):
        self.app.add_url_rule('/predict', methods=['POST'], view_func=self.predict)
        self.app.add_url_rule('/', methods=['GET'], view_func=self.home)

    def predict(self):
        if self.model and self.pipeline and self.feature_names:
            try:
                self.logger.info("--- Starting Prediction ---")
                data = request.get_json()
                self.logger.info(f"Raw JSON data received: {data}")


                input_df = pd.DataFrame([data])
                self.logger.info(f"Shape of input DataFrame before cleaning: {input_df.shape}")
                self.logger.info(f"Columns of input DataFrame before cleaning: {input_df.columns.tolist()}")

                # Apply data cleaning steps
                input_df = drop_irrelevant_and_high_missing_columns(input_df, drop_threshold=0.5)
                input_df = process_time_column(input_df)
                input_df = drop_highly_correlated_features(input_df, correlation_threshold=0.85)

                self.logger.info(f"Shape of input DataFrame after cleaning: {input_df.shape}")
                self.logger.info(f"Columns of input DataFrame after cleaning: {input_df.columns.tolist()}")

                self.logger.info("Applying preprocessing pipeline...")
                X_new_transformed = self.pipeline.transform(input_df) # Apply to the whole DataFrame

                self.logger.info(f"Transformed data (NumPy array):\n{X_new_transformed}")
                self.logger.info(f"Shape of transformed data: {X_new_transformed.shape}")

                if hasattr(self.pipeline, 'get_feature_names_out'):
                    feature_names_out = self.pipeline.get_feature_names_out()
                    self.logger.info(f"Output feature names from pipeline: {feature_names_out.tolist()}")
                    X_new_df = pd.DataFrame(X_new_transformed, columns=feature_names_out)
                    self.logger.info(f"Transformed DataFrame:\n{X_new_df.head()}")
                    self.logger.info(f"Columns of transformed data (first 20): {X_new_df.columns[:20].tolist()}")
                    self.logger.info("Making prediction...")
                    try:
                        prediction = self.model.predict(X_new_df[self.feature_names])
                    except KeyError as ke:
                        self.logger.error(f"KeyError during prediction. Ensure transformed data has required columns: {ke}")
                        return jsonify({'error': f'Missing feature in transformed data: {ke}'}), 400
                else:
                    self.logger.warning("Pipeline does not have 'get_feature_names_out'. Assuming model accepts NumPy array.")
                    self.logger.info("Making prediction...")
                    prediction = self.model.predict(X_new_transformed)

                self.logger.info(f"Prediction: {prediction}")
                return jsonify({'prediction': int(prediction[0])})

            except Exception as e:
                self.logger.error(f"Prediction error: {e}", exc_info=True)
                return jsonify({'error': str(e)}), 500
        else:
            return jsonify({'error': 'Model or pipeline not loaded'}), 500

    def home(self):
        return render_template('index.html')

    def run(self, debug=False, host='127.0.0.1', port=5000):
        self.app.run(debug=debug, host=host, port=port)

if __name__ == '__main__':
    predictor = FlaskPredictor()
    predictor.run(debug=False)







