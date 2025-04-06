# pipeline.py
import logging
import pandas as pd
import os
import pickle

from utils import setup_logger
import engine
import set_up
from visualizer import DataVisualization
from utils import SaveLoadBestModel
from set_up import CombinedFeaturesAdder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, auc

class TrainingPipeline:
    def __init__(self, data_path, drop_threshold=0.5, sample_reduction=0.1):
        self.logger = setup_logger(__name__)
        self.data_path = data_path
        self.data = None
        self.visualizer = None
        self.drop_threshold = drop_threshold
        self.sample_reduction = sample_reduction
        self.preprocessor = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.target_variable = 'ACCLASS'
        self.model = None
        self.feature_importance = None
        self.models = {}
        self.tuned_models = {}
        self.feature_importance = None
        self.training_times = {}
        self.best_params = {}
        self.feature_names_after_preprocessing = None
        self.model_path = None
        self.pipeline_path = None
        self.feature_names_path = None
        self.model_results = pd.DataFrame(columns=[
            'Model', 'Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC AUC', 'Train Accuracy'
        ])
        self.model_path = 'saved_models/best_model.pkl'
        self.pipeline_path = 'saved_models/preprocessor.pkl'
        self.feature_names_path = 'saved_models/feature_names.pkl'
        self.model_saver = SaveLoadBestModel(self.model_path, self.pipeline_path, self.feature_names_path, self.logger)

    def load_and_explore_data(self):
        """Loads and performs initial exploration of the data."""
        self.logger.info("--- Loading and Exploring Data ---")
        self.data = set_up.load_data(self.data_path)
        if self.data is not None:
            self.logger.info(f"Shape of loaded data: {self.data.shape}")
            self.logger.info(f"First 5 rows of loaded data:\n{self.data.head()}")
            set_up.describe_data(self.data)
            self.data = set_up.convert_acclass_to_numeric(self.data)
            self.logger.info(f"Shape of data after ACCLASS conversion: {self.data.shape}")
            # engine.perform_statistical_assessments(self.data)
            # engine.evaluate_missing_data(self.data)
            self.visualizer = DataVisualization(self.data)
            # self.visualizer.plot_missing_data_heatmaps()
            # self.visualizer.plot_histograms()
            # self.visualizer.plot_bar_plots()
            # self.visualizer.plot_pie_charts()
            # self.visualizer.plot_scatter_plots()
            # self.visualizer.plot_box_plots()
            # self.visualizer.plot_grouped_bar_plots()
            # self.visualizer.plot_heatmaps()
            # self.visualizer.plot_time_series()
            # self.visualizer.plot_missing_percentages()
            # self.visualizer.plot_pairwise_scatter_plots()
        else:
            self.logger.error("Data loading failed. Aborting exploration.")


    ###########################################################################################################
    ##############   This code is important as it enables you to switch to training with ALL Data
    ##############     DO NOT DELETE          ################################################################
    #############################################################################################################

    # def preprocess_data(self):
    #     """Performs data preprocessing steps with logging of data at each stage."""
    #     self.logger.info("--- Starting Data Preprocessing ---")
    #     if self.data is None:
    #         self.logger.error("Data not loaded. Call load_and_explore_data() first.")
    #         return
    #
    #     self.logger.info(f"Shape of data before sampling: {self.data.shape}")
    #     sampled_data = set_up.sample_data(self.data, self.sample_reduction)
    #     self.logger.info(f"Shape of data after sampling: {sampled_data.shape}")
    #     self.logger.info(f"First 5 rows of sampled data:\n{sampled_data.head()}")
    #     self.data = sampled_data.copy()
    #     y = self.data[self.target_variable]
    #     X = self.data.drop(columns=[self.target_variable], errors='ignore')
    #     self.logger.info(f"Shape of features (X) before dropping irrelevant/missing: {X.shape}")
    #     self.logger.info(f"Shape of target (y): {y.shape}")
    #
    #     X = set_up.drop_irrelevant_and_high_missing_columns(X, self.drop_threshold)
    #     self.logger.info(f"Shape of features (X) after dropping irrelevant/missing: {X.shape}")
    #     self.logger.info(f"Columns of features (X) after dropping irrelevant/missing: {X.columns.tolist()}")
    #
    #     self.logger.info(f"Shape of features (X) before processing TIME: {X.shape}")
    #     X = set_up.process_time_column(X)
    #     self.logger.info(f"Shape of features (X) after processing TIME: {X.shape}")
    #     if 'TIME' in X.columns:
    #         self.logger.info(f"First 5 values of TIME column after processing:\n{X['TIME'].head()}")
    #     else:
    #         self.logger.warning("TIME column not found after processing.")
    #
    #     self.logger.info(f"Shape of features (X) before dropping correlated features: {X.shape}")
    #     X = set_up.drop_highly_correlated_features(X)
    #     self.logger.info(f"Shape of features (X) after dropping correlated features: {X.shape}")
    #     self.logger.info(f"Columns of features (X) after dropping correlated features: {X.columns.tolist()}")
    #
    #     self.logger.info("--- Creating Preprocessing Pipeline ---")
    #     self.preprocessor = set_up.create_preprocessing_pipeline(X, CombinedFeaturesAdder())
    #     self.logger.info(f"Shape of features (X) before preprocessing: {X.shape}")
    #     X_processed, self.preprocessor = set_up.preprocess_data(X, preprocessor=self.preprocessor)
    #     self.logger.info(f"Shape of features (X_processed) after preprocessing: {X_processed.shape}")
    #     if isinstance(X_processed, pd.DataFrame):
    #         self.logger.info(f"First 5 rows of processed features (X_processed):\n{X_processed.head()}")
    #         self.feature_names_after_preprocessing = X_processed.columns.tolist()
    #         self.logger.info(
    #             f"Feature names after preprocessing (first 5): {self.feature_names_after_preprocessing[:5]}...")
    #         self.logger.info(f"Number of features after preprocessing: {len(self.feature_names_after_preprocessing)}")
    #         self.logger.info(
    #             f"All feature names after preprocessing: {self.feature_names_after_preprocessing}")  # Log all feature names
    #     else:
    #         self.logger.warning("Processed data (X_processed) is not a DataFrame.")
    #         self.feature_names_after_preprocessing = None
    #
    #     self.logger.info(f"Shape of processed features (X_processed) before splitting: {X_processed.shape}")
    #     self.logger.info(f"Shape of target (y) before splitting: {y.shape}")
    #     self.X_train, self.X_test, self.y_train, self.y_test = set_up.split_data(X_processed, y)
    #     self.logger.info(f"Shape of X_train after splitting: {self.X_train.shape}")
    #     self.logger.info(f"Shape of X_test after splitting: {self.X_test.shape}")
    #     self.logger.info(f"Shape of y_train after splitting: {self.y_train.shape}")
    #     self.logger.info(f"Shape of y_test after splitting: {self.y_test.shape}")
    #
    #     self.logger.info(f"Shape of X_train before handling imbalance: {self.X_train.shape}")
    #     self.logger.info(f"Shape of y_train before handling imbalance: {self.y_train.shape}")
    #     self.X_train, self.y_train = set_up.manage_imbalanced_classes(self.X_train, self.y_train, self.visualizer)
    #     self.logger.info(f"Shape of X_train after handling imbalance: {self.X_train.shape}")
    #     self.logger.info(f"Shape of y_train after handling imbalance: {self.y_train.shape}")
    #
    #     self.logger.info(f"Data type of y_train before conversion: {self.y_train.dtype}")
    #     self.logger.info(f"Data type of y_test before conversion: {self.y_test.dtype}")
    #     self.y_train, self.y_test = set_up.convert_target_to_int32(self.y_train, self.y_test)
    #     self.logger.info(f"Data type of y_train after conversion: {self.y_train.dtype}")
    #     self.logger.info(f"Data type of y_test after conversion: {self.y_test.dtype}")
    #     set_up.log_final_data_info(self.X_train, self.X_test, self.y_train, self.y_test)
    #     # Train the Random Forest model and get feature importance
    #     self.model, self.feature_importance = set_up.train_random_forest_model(self.X_train, self.y_train)
    #
    #     # Print feature importance
    #     if self.feature_importance is not None:
    #         print("\n--- Feature Importance from Random Forest ---")
    #         print(self.feature_importance)
    #         self.visualizer.plot_feature_importance(self.feature_importance, title="Random Forest Preprocessing Feature Importance")
    #         self.logger.info("--- Top Feature Importances ---")
    #         self.logger.info(f"First 10 important features:\n{self.feature_importance.nlargest(10)}")


    #######################################################################################################

    def preprocess_data(self):
        """Performs data preprocessing steps with logging of data at each stage."""
        self.logger.info("--- Starting Data Preprocessing (with Top 10 Features Hardcoded) ---")
        if self.data is None:
            self.logger.error("Data not loaded. Call load_and_explore_data() first.")
            return

        self.logger.info(f"Shape of data before sampling: {self.data.shape}")
        sampled_data = set_up.sample_data(self.data, self.sample_reduction)
        self.logger.info(f"Shape of data after sampling: {sampled_data.shape}")
        self.logger.info(f"First 5 rows of sampled data:\n{sampled_data.head()}")
        self.data = sampled_data.copy()
        y = self.data[self.target_variable]
        # Select the top 10 original features here
        top_10_features = [
            'TIME',
            'LATITUDE',
            'LONGITUDE',
            'ROAD_CLASS',
            'DISTRICT',
            'ACCLOC',
            'TRAFFCTL',
            'VISIBILITY',
            'LIGHT',
            'RDSFCOND'
        ]
        try:
            X = self.data[top_10_features].copy()
        except KeyError as e:
            self.logger.error(f"One or more of the top 10 features not found in the data: {e}")
            return

        self.logger.info(f"Shape of features (X) after selecting top 10: {X.shape}")
        self.logger.info(f"Columns of features (X) after selecting top 10: {X.columns.tolist()}")
        self.logger.info(f"Shape of target (y): {y.shape}")

        self.logger.info("--- Creating Preprocessing Pipeline ---")
        self.preprocessor = set_up.create_preprocessing_pipeline(X, CombinedFeaturesAdder())
        self.logger.info(f"Shape of features (X) before preprocessing: {X.shape}")
        X_processed, self.preprocessor = set_up.preprocess_data(X, preprocessor=self.preprocessor)
        self.logger.info(f"Shape of features (X_processed) after preprocessing: {X_processed.shape}")
        if isinstance(X_processed, pd.DataFrame):
            self.logger.info(f"First 5 rows of processed features (X_processed):\n{X_processed.head()}")
            self.feature_names_after_preprocessing = X_processed.columns.tolist()
            self.logger.info(
                f"Feature names after preprocessing (first 5): {self.feature_names_after_preprocessing[:5]}...")
            self.logger.info(f"Number of features after preprocessing: {len(self.feature_names_after_preprocessing)}")
            self.logger.info(
                f"All feature names after preprocessing: {self.feature_names_after_preprocessing}")  # Log all feature names
        else:
            self.logger.warning("Processed data (X_processed) is not a DataFrame.")
            self.feature_names_after_preprocessing = None

        self.logger.info(f"Shape of processed features (X_processed) before splitting: {X_processed.shape}")
        self.logger.info(f"Shape of target (y) before splitting: {y.shape}")
        self.X_train, self.X_test, self.y_train, self.y_test = set_up.split_data(X_processed, y)
        self.logger.info(f"Shape of X_train after splitting: {self.X_train.shape}")
        self.logger.info(f"Shape of X_test after splitting: {self.X_test.shape}")
        self.logger.info(f"Shape of y_train after splitting: {self.y_train.shape}")
        self.logger.info(f"Shape of y_test after splitting: {self.y_test.shape}")

        self.logger.info(f"Shape of X_train before handling imbalance: {self.X_train.shape}")
        self.logger.info(f"Shape of y_train before handling imbalance: {self.y_train.shape}")
        self.X_train, self.y_train = set_up.manage_imbalanced_classes(self.X_train, self.y_train, self.visualizer)
        self.logger.info(f"Shape of X_train after handling imbalance: {self.X_train.shape}")
        self.logger.info(f"Shape of y_train after handling imbalance: {self.y_train.shape}")

        self.logger.info(f"Data type of y_train before conversion: {self.y_train.dtype}")
        self.logger.info(f"Data type of y_test before conversion: {self.y_test.dtype}")
        self.y_train, self.y_test = set_up.convert_target_to_int32(self.y_train, self.y_test)
        self.logger.info(f"Data type of y_train after conversion: {self.y_train.dtype}")
        self.logger.info(f"Data type of y_test after conversion: {self.y_test.dtype}")
        set_up.log_final_data_info(self.X_train, self.X_test, self.y_train, self.y_test)
        # Train the Random Forest model and get feature importance
        self.model, self.feature_importance = set_up.train_random_forest_model(self.X_train, self.y_train)

        # Print feature importance
        if self.feature_importance is not None:
            print("\n--- Feature Importance from Random Forest ---")
            print(self.feature_importance)
            self.visualizer.plot_feature_importance(self.feature_importance,
                                                    title="Random Forest Preprocessing Feature Importance")
            self.logger.info("--- Top Feature Importances ---")
            self.logger.info(f"First 10 important features:\n{self.feature_importance.nlargest(10)}")


    def get_feature_importance(self):
        return set_up.get_feature_importance(self.feature_importance)

    def get_processed_data(self):
        return set_up.get_processed_data(self.X_train, self.X_test, self.y_train, self.y_test)

    def initialize_models(self):
        """Initializes the predictive models."""
        self.logger.info("--- Initializing Predictive Models ---")
        self.models = engine.instantiate_models()
        self.logger.info(f"Initialized models: {list(self.models.keys())}")

    def evaluate_model(self, model, model_name):
        """
        Evaluates a single model and returns test and train accuracies.

        Args:
            model: The model to evaluate.
            model_name (str): Name of the model.

        Returns:
            tuple: Test accuracy and train accuracy.
        """
        if model is None:
            self.logger.error(f"Model {model_name} is None. Skipping evaluation.")
            return 0, 0  # Return default accuracies or handle as needed

        y_pred = model.predict(self.X_test)
        test_accuracy = accuracy_score(self.y_test, y_pred)
        report = classification_report(self.y_test, y_pred, output_dict=True)
        cm = confusion_matrix(self.y_test, y_pred)
        self.visualizer.plot_confusion_matrix(cm, model_name)
        fpr, tpr, thresholds = roc_curve(self.y_test, model.predict_proba(self.X_test)[:, 1])
        roc_auc = auc(fpr, tpr)

        y_train_pred = model.predict(self.X_train)
        train_accuracy = accuracy_score(self.y_train, y_train_pred)

        new_row = pd.DataFrame({
            'Model': [model_name],
            'Accuracy': [test_accuracy],
            'Precision': [report['weighted avg']['precision']],
            'Recall': [report['weighted avg']['recall']],
            'F1-Score': [report['weighted avg']['f1-score']],
            'ROC AUC': [roc_auc],
            'Train Accuracy': [train_accuracy],
        })

        self.model_results = pd.concat([self.model_results, new_row], ignore_index=True)
        self.visualizer.plot_roc_curve(fpr, tpr, roc_auc, model_name)

        return test_accuracy, train_accuracy

    def evaluate_models(self):
        """Evaluates the tuned models and prints the results DataFrame."""
        self.logger.info("--- Evaluating Tuned Models ---")
        if not self.tuned_models or self.X_test is None or self.y_test is None or self.X_train is None or self.y_train is None:
            self.logger.warning("No tuned models or test/train data available for evaluation.")
            return

        accuracy_scores = {}
        for model_name, model in self.tuned_models.items():
            self.logger.info(f"Evaluating model: {model_name}")
            test_accuracy, train_accuracy = self.evaluate_model(model, model_name)
            accuracy_scores[model_name] = test_accuracy

        self.logger.info("\n--- Model Evaluation Results ---")
        print("\n--- Model Evaluation Results ---")
        print(self.model_results)  # Print the DataFrame

        self.best_model_name = self.recommend_best_model()  # Use the class method
        if self.best_model_name and not self.model_results.empty:
            self.best_model_roc_auc = \
                self.model_results[self.model_results['Model'] == self.best_model_name]['ROC AUC'].iloc[0]
            formatted_roc_auc = f"{self.best_model_roc_auc:.3f}" if pd.notna(self.best_model_roc_auc) else 'N/A'
            self.logger.info(
                f"Best performing model: {self.best_model_name} (ROC AUC: {formatted_roc_auc})"
            )
            print(
                f"Best performing model: {self.best_model_name} (ROC AUC: {formatted_roc_auc})"
            )
            # Assuming these plotting functions are now methods of your visualizer
            if self.visualizer:
                self.visualizer.plot_model_comparison(accuracy_scores)
                self.visualizer.plot_roc_auc_comparison(self.model_results)



    def train_and_tune_models(self):
        """Trains and tunes the initialized models."""
        self.logger.info("--- Training and Tuning Predictive Models ---")
        if not self.models or self.X_train is None or self.y_train is None:
            self.logger.error("Models not initialized or preprocessing not complete.")
            return

        for model_name, model in self.models.items():
            tuned_model, train_time, best_params = engine.train_and_tune_model(
                model_name, model, self.X_train, self.y_train, self.visualizer
            )
            if tuned_model:
                self.tuned_models[model_name] = tuned_model
                self.training_times[model_name] = train_time
                self.best_params[model_name] = best_params

        engine.plot_model_training_times(self.training_times)
        self.logger.info("Model training and tuning complete.")
        self.logger.info(f"Tuned models: {list(self.tuned_models.keys())}")
        self.logger.info(f"Best parameters: {self.best_params}")

    def create_and_evaluate_ensembles(self):
        """Creates ensemble models (Voting and Stacking)."""
        self.logger.info("--- Creating Ensemble Models ---")
        if not self.tuned_models or self.X_train is None or self.y_train is None:
            self.logger.warning("Tuned models or data not available for ensemble creation.")
            return

        # Voting Ensemble
        voting_ensemble = engine.create_voting_model(self.tuned_models, self.X_train, self.y_train)
        if voting_ensemble:
            self.tuned_models["Voting"] = voting_ensemble
            self.logger.info("Voting Ensemble Model created.")

        # Stacking Ensemble
        stacking_ensemble = engine.create_stacking_model(self.tuned_models, self.X_train, self.y_train)
        if stacking_ensemble:
            self.tuned_models["Stacking"] = stacking_ensemble
            self.logger.info("Stacking Ensemble Model created.")

    def recommend_best_model(self):
        """Recommends the best model based on ROC AUC."""
        if self.model_results.empty:
            self.logger.warning("No model results available to recommend a best model.")
            return None
        best_model_row = self.model_results.sort_values(by='ROC AUC', ascending=False).iloc[0]
        best_model_name = best_model_row['Model']
        best_roc_auc = best_model_row['ROC AUC']
        self.logger.info(f"Recommended best model: {best_model_name} (ROC AUC: {best_roc_auc:.3f})")
        return best_model_name

    def save_best_model(self):
        """Saves the best performing model, pipeline, and feature names using SaveLoadBestModel."""
        self.logger.info("--- Saving Best Model and Artifacts ---")
        self.logger.info(f"Best model name: {self.best_model_name}")

        best_model = self.tuned_models.get(self.best_model_name)

        if best_model and self.preprocessor and self.feature_names_after_preprocessing:
            self.model_saver.save_model_and_pipeline(
                best_model, self.preprocessor, self.feature_names_after_preprocessing
            )
            self.model_path = self.model_saver.model_file_path
            self.pipeline_path = self.model_saver.pipeline_file_path
            self.feature_names_path = self.model_saver.feature_names_file_path
        else:
            self.logger.warning("Best model, preprocessor, or feature names not available for saving.")

        self.logger.info("--- Saving process completed ---")

    def run(self):
        """Runs the entire training pipeline."""
        self.logger.info("--- Starting Training Pipeline ---")
        self.load_and_explore_data()
        if self.data is not None:
            self.preprocess_data()
            if self.X_train is not None:
                self.initialize_models()
                self.train_and_tune_models()
                self.create_and_evaluate_ensembles()
                self.evaluate_models()
                self.save_best_model()  # Add the saving step here
            else:
                self.logger.warning("Preprocessing did not complete successfully.")
        else:
            self.logger.warning("Data loading failed, cannot proceed with pipeline.")
        self.logger.info("--- Training Pipeline Finished ---")




    # Add methods for model training, evaluation, etc.