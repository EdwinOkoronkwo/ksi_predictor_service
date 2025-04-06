# engine.py
import copy
import logging
import os
import time

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from imblearn.over_sampling import SMOTE
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (
    RandomForestClassifier,
    VotingClassifier,
    StackingClassifier,
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    auc,
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from utils import setup_logger  # Assuming setup_logger is in utils.py

def instantiate_models():
    """Instantiates the predictive models and returns them in a dictionary."""
    logger = setup_logger(__name__)
    models = {
        "LogReg": LogisticRegression(random_state=42, solver='liblinear'),
        # "DTC": DecisionTreeClassifier(random_state=42),
        # "SVM": SVC(random_state=42, probability=True),
        "Forest": RandomForestClassifier(random_state=42),
        # "ANN": MLPClassifier(random_state=42, max_iter=300),
        # "NBayes": GaussianNB(),
        # "KNN": KNeighborsClassifier(),
        # "Voting": VotingClassifier(estimators=[], voting='soft'),  # Estimators will be added later
        # "Stacking": StackingClassifier(estimators=[], final_estimator=RandomForestClassifier(random_state=42), cv=3)
    }
    logger.info("Predictive models instantiated.")
    return models


def select_top_features(X_train: pd.DataFrame, feature_importance: pd.Series, topN: int = None):
    """Selects the top N features based on feature importance."""
    logger = setup_logger(__name__)
    if topN is None:
        logger.info("Keeping all features.")
        return X_train.columns.tolist()
    if feature_importance is None:
        logger.warning("Feature importance not provided, keeping all features.")
        return X_train.columns.tolist()
    sorted_features = feature_importance.sort_values(ascending=False)
    top_features = sorted_features.index[:topN].tolist()
    logger.info(f"Selected top {topN} features: {top_features[:5]}...")
    return top_features

def tune_hyperparameters(model, param_grid, X: pd.DataFrame, y: pd.Series, cv=5, scoring='accuracy', verbose=0):
    """Tunes hyperparameters of a model using GridSearchCV."""
    logger = setup_logger(__name__)
    logger.info(f"Tuning hyperparameters for {model.__class__.__name__}...")
    grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=cv, scoring=scoring, verbose=verbose, n_jobs=-1)
    grid_search.fit(X, y)
    logger.info(f"Best parameters found: {grid_search.best_params_}")
    logger.info(f"Best cross-validation score: {grid_search.best_score_:.4f}")
    return grid_search.best_estimator_, grid_search.best_params_, grid_search.cv_results_

def train_and_tune_model(model_name: str, model, X_train: pd.DataFrame, y_train: pd.Series, visualizer=None):
    """Trains and tunes a single model based on its name and logs the best parameters."""
    logger = setup_logger(__name__)
    logger.info(f"Training and tuning {model_name}...")
    start_time = time.time()
    tuned_model = None
    feature_importance = None
    best_params = None
    cv_results = None

    if model_name == "LogReg":
        param_grid = {'C': [0.1, 1, 10], 'solver': ['liblinear'], 'penalty': ['l1', 'l2'], 'max_iter': [100, 200, 500, 1000]}
        tuned_model, best_params, cv_results = tune_hyperparameters(model, param_grid, X_train, y_train)
        if hasattr(tuned_model, 'coef_'):
            feature_importance = pd.Series(tuned_model.coef_[0], index=X_train.columns)
            logger.info(f"Feature Importance (Logistic Regression):\n{feature_importance.to_string()}")
            if visualizer:
                visualizer.plot_feature_importance(feature_importance, title="Feature Importance (Logistic Regression)")

    elif model_name == "DTC":
        param_grid = {'max_depth': [3, 5, 10, None], 'min_samples_split': [2, 5, 10], 'min_samples_leaf': [1, 2, 4]}
        tuned_model, best_params, cv_results = tune_hyperparameters(model, param_grid, X_train, y_train)
        if hasattr(tuned_model, 'feature_importances_'):
            feature_importance = pd.Series(tuned_model.feature_importances_, index=X_train.columns)
            logger.info(f"Feature Importance (Decision Tree):\n{feature_importance.to_string()}")
            if visualizer:
                visualizer.plot_feature_importance(feature_importance, title="Feature Importance (Decision Tree)")

    elif model_name == "SVM":
        param_grid = {'C': [0.1, 1, 10], 'kernel': ['linear', 'rbf', 'poly'], 'gamma': ['scale', 0.1, 1]}
        tuned_model, best_params, cv_results = tune_hyperparameters(model, param_grid, X_train, y_train)

    elif model_name == "Forest":
        param_grid = {'n_estimators': [50, 100, 200], 'max_depth': [3, 5, 10, None], 'min_samples_split': [2, 5, 10], 'min_samples_leaf': [1, 2, 4]}
        tuned_model, best_params, cv_results = tune_hyperparameters(model, param_grid, X_train, y_train)
        if hasattr(tuned_model, 'feature_importances_'):
            feature_importance = pd.Series(tuned_model.feature_importances_, index=X_train.columns)
            logger.info(f"Feature Importance (Random Forest):\n{feature_importance.to_string()}")
            if visualizer:
                visualizer.plot_feature_importance(feature_importance, title="Feature Importance (Random Forest)")

    elif model_name == "ANN":
        param_grid = {
            'hidden_layer_sizes': [(64, 64,), (128, 128), (256, 256)],
            'activation': ['relu', 'tanh'],
            'alpha': [0.001, 0.01],
            'learning_rate': ['constant', 'adaptive']
        }
        tuned_model, best_params, cv_results = tune_hyperparameters(model, param_grid, X_train, y_train, verbose=1)

    elif model_name == "KNN":
        param_grid = {'n_neighbors': range(1, 21)}
        tuned_model, best_params, cv_results = tune_hyperparameters(model, param_grid, X_train, y_train)
        if visualizer and cv_results:
            mean_test_scores = cv_results['mean_test_score']
            n_neighbors_range = param_grid['n_neighbors']
            visualizer.plot_knn_optimum_k(mean_test_scores, n_neighbors_range)

    elif model_name == "NBayes":
        param_grid = {'var_smoothing': [1e-6, 1e-5, 1e-4, 1e-3, 0.005, 1e-2, 1e-1]}
        tuned_model, best_params, cv_results = tune_hyperparameters(model, param_grid, X_train, y_train)

    else:
        logger.warning(f"Model name '{model_name}' not recognized.")
        tuned_model = model

    end_time = time.time()
    training_time = end_time - start_time
    logger.info(f"{model_name} training time: {training_time:.2f} seconds")

    if best_params:
        logger.info(f"Best parameters for {model_name}: {best_params}")
    if cv_results and 'mean_test_score' in cv_results:
        logger.info(f"Cross Validation Results for {model_name}: {cv_results['mean_test_score']}")

    return tuned_model, training_time, best_params

def predict_model(model, X_test: pd.DataFrame):
    """Makes predictions using the trained model."""
    logger = setup_logger(__name__)
    logger.info(f"--- Making predictions with {model.__class__.__name__} ---")
    if X_test is None:
        raise ValueError("X_test cannot be None.")
    predictions = model.predict(X_test)
    return predictions

def create_voting_model(tuned_models: dict, X_train: pd.DataFrame, y_train: pd.Series):
    """Creates the voting ensemble model using VotingClassifier."""
    logger = setup_logger(__name__)
    estimators = [(name, model) for name, model in tuned_models.items() if
                  name not in ["Voting", "Stacking", "ANN"] and model is not None]
    if not estimators:
        logger.warning("No valid base estimators found for VotingClassifier.")
        return None
    voting_model = VotingClassifier(estimators=estimators, voting='soft')
    voting_model.fit(X_train, y_train)
    logger.info("Voting Ensemble Model created and trained.")
    return voting_model

def create_stacking_model(tuned_models: dict, X_train: pd.DataFrame, y_train: pd.Series, final_estimator=None):
    """Creates the stacking model using StackingClassifier with RandomForest as blender by default."""
    logger = setup_logger(__name__)
    estimators = [(name, model) for name, model in tuned_models.items() if
                  name not in ["Voting", "Stacking", "ANN"] and model is not None]
    if not estimators:
        logger.warning("No valid base estimators found for StackingClassifier.")
        return None

    if final_estimator is None:
        final_estimator = RandomForestClassifier(random_state=42) # Default blender

    stacking_model = StackingClassifier(estimators=estimators, final_estimator=final_estimator, cv=3)
    stacking_model.fit(X_train, y_train)
    logger.info("Stacking Ensemble Model created and trained.")
    return stacking_model

def plot_model_training_times(training_times: dict):
    """Creates a bar plot of model training times."""
    logger = setup_logger(__name__)
    plt.figure(figsize=(10, 6))
    sns.barplot(x=list(training_times.keys()), y=list(training_times.values()))
    plt.title("Model Training Times")
    plt.xlabel("Model")
    plt.ylabel("Training Time (seconds)")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.show()



def evaluate_tuned_models(tuned_models, X_test, y_test, X_train, y_train, visualizer, best_params):
    """Evaluates all tuned models and returns a dictionary of test accuracies."""
    logger = setup_logger(__name__)
    logger.info("--- Evaluating Tuned Models ---")
    model_results = pd.DataFrame()
    accuracy_scores = {}

    for model_name, model in tuned_models.items():
        logger.info(f"Evaluating {model_name}...")
        if model is None:
            logger.error(f"Model {model_name} is None. Skipping evaluation.")
            continue

        y_pred = predict_model(model, X_test)
        test_accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred, output_dict=True)
        cm = confusion_matrix(y_test, y_pred)
        visualizer.plot_confusion_matrix(cm, model_name) # Removed classes argument

        # Probability prediction for ROC AUC (if available)
        if hasattr(model, "predict_proba"):
            try:
                y_proba = model.predict_proba(X_test)[:, 1]
                fpr, tpr, thresholds = roc_curve(y_test, y_proba)
                roc_auc = auc(fpr, tpr)
                visualizer.plot_roc_curve(fpr, tpr, roc_auc, model_name)
            except Exception as e:
                logger.warning(f"Could not generate ROC AUC for {model_name}: {e}")
                roc_auc = np.nan
        else:
            roc_auc = np.nan
            logger.warning(f"Model {model_name} does not have predict_proba method.")

        y_train_pred = predict_model(model, X_train)
        train_accuracy = accuracy_score(y_train, y_train_pred)

        new_row = pd.DataFrame({
            'Model': [model_name],
            'Accuracy': [test_accuracy],
            'Precision': [report['weighted avg']['precision']],
            'Recall': [report['weighted avg']['recall']],
            'F1-Score': [report['weighted avg']['f1-score']],
            'ROC AUC': [roc_auc],
            'Train Accuracy': [train_accuracy],
        })
        model_results = pd.concat([model_results, new_row], ignore_index=True)
        visualizer.plot_train_test_metrics(train_accuracy, test_accuracy, model_name)
        accuracy_scores[model_name] = test_accuracy

    return model_results, accuracy_scores

def recommend_best_model(model_results: pd.DataFrame):
    """Recommends the best model based on ROC AUC."""
    logger = setup_logger(__name__)
    if model_results.empty:
        logger.warning("No models were evaluated. Returning None.")
        return None
    if 'ROC AUC' in model_results.columns and not model_results['ROC AUC'].isnull().all():
        best_model_row = model_results.loc[model_results['ROC AUC'].idxmax()]
        best_model_name = best_model_row['Model']
        best_roc_auc = best_model_row['ROC AUC']
        logger.info(f"Recommended best model based on ROC AUC: {best_model_name} (ROC AUC: {best_roc_auc:.3f})")
        print(f"Recommended best model: {best_model_name} (ROC AUC: {best_roc_auc:.3f})")
        return best_model_name
    else:
        logger.warning("ROC AUC data is missing or all NaN. Recommending based on Accuracy.")
        best_model_row = model_results.loc[model_results['Accuracy'].idxmax()]
        best_model_name = best_model_row['Model']
        best_accuracy = best_model_row['Accuracy']
        logger.info(f"Recommended best model based on Accuracy: {best_model_name} (Accuracy: {best_accuracy:.3f})")
        print(f"Recommended best model: {best_model_name} (Accuracy: {best_accuracy:.3f})")
        return best_model_name

def plot_model_comparison(accuracy_scores: dict):
    """Creates a bar plot comparing the accuracy of different models."""
    logger = setup_logger(__name__)
    if not accuracy_scores:
        logger.warning("No accuracy scores to plot.")
        return
    models = list(accuracy_scores.keys())
    accuracies = list(accuracy_scores.values())
    plt.figure(figsize=(10, 6))
    sns.barplot(x=models, y=accuracies)
    plt.title("Model Comparison - Accuracy")
    plt.xlabel("Model")
    plt.ylabel("Accuracy")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.show()

def plot_roc_auc_comparison(model_results: pd.DataFrame):
    """Plots ROC AUC for all evaluated models."""
    logger = setup_logger(__name__)
    if model_results.empty or 'ROC AUC' not in model_results.columns:
        logger.warning("No model results or ROC AUC data to plot.")
        return
    model_results_sorted = model_results.sort_values(by='ROC AUC', ascending=False)
    plt.figure(figsize=(10, 6))
    sns.barplot(x='Model', y='ROC AUC', data=model_results_sorted)
    plt.title("Model Comparison - ROC AUC")
    plt.xlabel("Model")
    plt.ylabel("ROC AUC")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.show()
