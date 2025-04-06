# set_up.py

import logging
import os
import time

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from imblearn.over_sampling import SMOTE
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, VotingClassifier, StackingClassifier
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

from utils import setup_logger


def load_data(data_path):
    """
    Loads data from the specified CSV file.

    Args:
        data_path (str): The path to the CSV file.

    Returns:
        pandas.DataFrame: The loaded DataFrame.
    """
    logger = setup_logger(__name__)
    try:
        logger.info(f"Loading data from: {data_path}")
        data = pd.read_csv(data_path)
        logger.info(f"Data loaded successfully. Shape: {data.shape}")
        return data
    except FileNotFoundError:
        logger.error(f"Error: File not found at {data_path}")
        return None
    except Exception as e:
        logger.error(f"An error occurred while loading data: {e}")
        return None

def describe_data(data):
    """
    Provides a basic description of the input DataFrame.

    Args:
        data (pandas.DataFrame): The DataFrame to describe.
    """
    logger = setup_logger(__name__)
    if data is None:
        logger.warning("No data provided for description.")
        return

    logger.info("--- Data Description ---")
    logger.info(f"First few rows of the data:\n{data.head()}")
    logger.info(f"Data shape: {data.shape}")
    logger.info(f"Column information:\n{data.info()}")

    for col in data.columns:
        logger.info(f"\n--- Column: {col} ---")
        logger.info(f"Data type: {data[col].dtype}")
        logger.info(f"Unique values: {data[col].nunique()}")
        if pd.api.types.is_numeric_dtype(data[col]):
            logger.info(f"Range: [{data[col].min()}, {data[col].max()}]")
            logger.info(f"Mean: {data[col].mean()}")
            logger.info(f"Median: {data[col].median()}")
            logger.info(f"Standard deviation: {data[col].std()}")
        else:
            logger.info(f"Sample values: {data[col].unique()[:5]}")

def convert_acclass_to_numeric(data):
    """
    Converts the 'ACCLASS' column to numeric (Fatal = 1, otherwise 0).

    Args:
        data (pandas.DataFrame): The DataFrame containing the 'ACCLASS' column.

    Returns:
        pandas.DataFrame: The DataFrame with the updated 'ACCLASS' column.
    """
    logger = setup_logger(__name__)
    if data is None:
        logger.warning("No data provided for ACCLASS conversion.")
        return None
    if 'ACCLASS' not in data.columns:
        logger.warning("Column 'ACCLASS' not found in the DataFrame.")
        return data
    data['ACCLASS'] = data['ACCLASS'].apply(lambda x: 1 if x == 'Fatal' else 0)
    logger.info("Column 'ACCLASS' converted to numeric.")
    return data

def perform_statistical_assessments(data):
    """
    Performs statistical assessments including descriptive statistics and correlations.

    Args:
        data (pandas.DataFrame): The DataFrame to analyze.
    """
    logger = setup_logger(__name__)
    if data is None:
        logger.warning("No data provided for statistical assessments.")
        return

    logger.info("\n--- Statistical Assessments ---")
    logger.info("Descriptive statistics:")
    logger.info(data.describe())

    logger.info("\nCorrelation matrix:")
    corr_matrix = data.corr()
    logger.info(corr_matrix)

    plt.figure(figsize=(12, 10))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f")
    plt.title("Correlation Matrix")
    plt.show()

def evaluate_missing_data(data):
    """
    Evaluates missing data in the dataset.

    Args:
        data (pandas.DataFrame): The DataFrame to evaluate for missing values.
    """
    logger = setup_logger(__name__)
    if data is None:
        logger.warning("No data provided for missing data evaluation.")
        return

    logger.info("\n--- Missing Data Evaluation ---")
    logger.info("Missing values per column:")
    logger.info(data.isnull().sum())

    missing_percentage = (data.isnull().sum() / len(data)) * 100
    logger.info("\nMissing percentage per column:")
    logger.info(missing_percentage)

    plt.figure(figsize=(10, 6))
    sns.heatmap(data.isnull(), cbar=False, cmap='viridis')
    plt.title("Missing Data Heatmap")
    plt.show()



def sample_data(data: pd.DataFrame, sample_reduction: float = 0.1):
    """
    Samples the input DataFrame.

    Args:
        data (pd.DataFrame): The input DataFrame.
        sample_reduction (float): The fraction of data to sample (default: 0.1).

    Returns:
        pd.DataFrame: The sampled DataFrame.
    """
    logger = setup_logger(__name__)
    logger.info("--- Sampling Data ---")
    sampled_data = data.sample(frac=sample_reduction, random_state=42)
    logger.info(f"Sampled data shape: {sampled_data.shape}")
    return sampled_data

def drop_irrelevant_and_high_missing_columns(data: pd.DataFrame, drop_threshold: float = 0.5, prediction_mode: bool = False):
    """
    Drops irrelevant columns and columns with a high percentage of missing values.

    Args:
        data (pd.DataFrame): The input DataFrame.
        drop_threshold (float): The threshold for dropping columns based on missing values (default: 0.5).
        prediction_mode (bool): Flag indicating if it's prediction mode (default: False).

    Returns:
        pd.DataFrame: The DataFrame with dropped columns (modifies the input DataFrame in place).
    """
    logger = setup_logger(__name__)
    context = "for prediction" if prediction_mode else ""
    logger.info(f"--- Dropping columns {context} ---")

    cols_to_drop = data.columns[data.isnull().mean() > drop_threshold]
    irrelevant_cols = ['OBJECTID', 'INDEX', 'ACCNUM', 'INJURY', 'DATE', 'STREET1', 'STREET2', 'FATAL_NO']
    logger.info(f"Dropping columns (missing > {drop_threshold*100}%): {list(cols_to_drop)}")
    logger.info(f"Dropping irrelevant columns: {irrelevant_cols}")

    data.drop(columns=cols_to_drop.union(irrelevant_cols), inplace=True, errors='ignore')
    return data

def process_time_column(data: pd.DataFrame):
    """
    Converts the 'TIME' column to numeric, filling NaNs with 0.

    Args:
        data (pd.DataFrame): The input DataFrame.

    Returns:
        pd.DataFrame: The DataFrame with the processed 'TIME' column (modifies the input DataFrame in place).
    """
    logger = setup_logger(__name__)
    logger.info("--- Processing TIME column ---")
    data['TIME'] = pd.to_numeric(data.get('TIME', 0), errors='coerce').fillna(0)
    return data

def drop_highly_correlated_features(data: pd.DataFrame, correlation_threshold: float = 0.85):
    """
    Drops highly correlated numerical features.

    Args:
        data (pd.DataFrame): The input DataFrame.
        correlation_threshold (float): The correlation threshold above which features will be dropped (default: 0.85).

    Returns:
        pd.DataFrame: The DataFrame with highly correlated features dropped (modifies the input DataFrame in place).
    """
    logger = setup_logger(__name__)
    logger.info("--- Dropping highly correlated features ---")
    num_cols = data.select_dtypes(include=np.number).columns
    corr_matrix = data[num_cols].corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    to_drop = [col for col in upper.columns if any(upper[col] > correlation_threshold)]
    logger.info(f"Columns to drop (correlation > {correlation_threshold}): {to_drop}")
    data.drop(columns=to_drop, inplace=True, errors='ignore')
    return data


import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.base import BaseEstimator, TransformerMixin
from utils import setup_logger



def create_preprocessing_pipeline(data: pd.DataFrame, combined_features_adder):
    """
    Creates a preprocessing pipeline using ColumnTransformer.

    Args:
        data (pd.DataFrame): The input DataFrame to infer column types.
        combined_features_adder (BaseEstimator, TransformerMixin): An instance of the CombinedFeaturesAdder.

    Returns:
        ColumnTransformer: The preprocessing pipeline.
    """
    logger = setup_logger(__name__)
    logger.info("--- Creating Preprocessing Pipeline ---")

    numerical_cols = data.select_dtypes(include=['int64', 'float64']).columns
    categorical_cols = data.select_dtypes(include=['object', 'uint8']).columns

    logger.info(f"Identified numerical columns: {numerical_cols.tolist()}")
    logger.info(f"Identified categorical columns: {categorical_cols.tolist()}")

    num_pipeline = Pipeline([
        ('combiner', combined_features_adder),
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    logger.info("Created numerical pipeline: CombinedFeaturesAdder -> SimpleImputer -> StandardScaler")

    cat_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])
    logger.info("Created categorical pipeline: SimpleImputer -> OneHotEncoder")

    preprocessor = ColumnTransformer([
        ('numerical', num_pipeline, numerical_cols),
        ('categorical', cat_pipeline, categorical_cols)
    ])
    logger.info("Created ColumnTransformer with numerical and categorical pipelines")
    logger.info(f"Numerical transformer applied to columns: {numerical_cols.tolist()}")
    logger.info(f"Categorical transformer applied to columns: {categorical_cols.tolist()}")

    logger.info("--- Preprocessing Pipeline Creation Completed ---")
    return preprocessor



def preprocess_data(data: pd.DataFrame, preprocessor: ColumnTransformer = None, prediction_mode: bool = False):
    """
    Preprocesses the input DataFrame using a provided pipeline or fits a new one.

    Args:
        data (pd.DataFrame): The input DataFrame.
        preprocessor (ColumnTransformer, optional): A ColumnTransformer instance.
                                                  If None and not prediction_mode, a ValueError is raised.
        prediction_mode (bool): Flag indicating if it's prediction mode (default: False).

    Returns:
        pd.DataFrame: The preprocessed DataFrame with correct column names.
        ColumnTransformer: The fitted preprocessor (only returned in training mode).
    """
    logger = setup_logger(__name__)
    context = "prediction" if prediction_mode else "training"
    logger.info(f"***** Entering engine.preprocess_data in {context} mode *****")
    logger.info(f"Data shape before preprocessing: {data.shape}")
    logger.info(f"Data columns before preprocessing: {data.columns.tolist()}")

    if not prediction_mode:
        if preprocessor is None:
            raise ValueError("Preprocessor cannot be None in training mode.")
        transformed_array = preprocessor.fit_transform(data)
        feature_names_after_transformation = preprocessor.get_feature_names_out().tolist()
        logger.info(f"Feature names after transformation (training): {feature_names_after_transformation[:5]}...")
        processed_df = pd.DataFrame(transformed_array, columns=feature_names_after_transformation).astype('float32')
        return processed_df, preprocessor
    else:
        if preprocessor is None:
            raise ValueError("Preprocessor must be provided in prediction mode.")
        transformed_array = preprocessor.transform(data)
        feature_names_after_transformation = preprocessor.get_feature_names_out().tolist()
        logger.info(f"Feature names after transformation (prediction): {feature_names_after_transformation[:5]}...")
        processed_df = pd.DataFrame(transformed_array, columns=feature_names_after_transformation).astype('float32')
        return processed_df, preprocessor



def split_data(X: pd.DataFrame, y: pd.Series, test_size: float = 0.2, random_state: int = 42):
    """
    Splits the data into training and testing sets.

    Args:
        X (pd.DataFrame): The feature DataFrame.
        y (pd.Series): The target Series.
        test_size (float): The proportion of the dataset to include in the test split (default: 0.2).
        random_state (int): The random seed for reproducibility (default: 42).

    Returns:
        tuple: A tuple containing X_train, X_test, y_train, y_test.
    """
    logger = setup_logger(__name__)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    logger.info(f"Data split into training and testing sets. Train size: {len(X_train)}, Test size: {len(X_test)}")
    return X_train, X_test, y_train, y_test

def manage_imbalanced_classes(X_train: pd.DataFrame, y_train: pd.Series, visualizer=None):
    """
    Handles class imbalance using SMOTE.

    Args:
        X_train (pd.DataFrame): The training feature DataFrame.
        y_train (pd.Series): The training target Series.
        visualizer (object, optional): An instance of the DataVisualization class for plotting. Defaults to None.

    Returns:
        tuple: A tuple containing the resampled X_train and y_train.
    """
    logger = setup_logger(__name__)
    logger.info("--- Handling Class Imbalance using SMOTE ---")
    if visualizer:
        visualizer.plot_class_distribution(y_train, 'Before SMOTE')
    smote = SMOTE(random_state=42)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
    logger.info(f"Shape of X_train before SMOTE: {X_train.shape}, after SMOTE: {X_train_resampled.shape}")
    logger.info(f"Shape of y_train before SMOTE: {y_train.shape}, after SMOTE: {y_train_resampled.shape}")
    if visualizer:
        visualizer.plot_class_distribution(y_train_resampled, 'After SMOTE')
    return X_train_resampled, y_train_resampled

def convert_target_to_int32(y_train: pd.Series, y_test: pd.Series):
    """
    Converts the target variables to int32 dtype.

    Args:
        y_train (pd.Series): The training target Series.
        y_test (pd.Series): The testing target Series.

    Returns:
        tuple: A tuple containing the converted y_train and y_test.
    """
    logger = setup_logger(__name__)
    y_train_int = y_train.astype('int32')
    y_test_int = y_test.astype('int32')
    logger.info("Target variables converted to int32.")
    return y_train_int, y_test_int

def log_final_data_info(X_train: pd.DataFrame, X_test: pd.DataFrame, y_train: pd.Series, y_test: pd.Series):
    """
    Logs the final shapes and distributions of the training and testing data.

    Args:
        X_train (pd.DataFrame): The preprocessed training feature DataFrame.
        X_test (pd.DataFrame): The preprocessed testing feature DataFrame.
        y_train (pd.Series): The preprocessed training target Series.
        y_test (pd.Series): The preprocessed testing target Series.
    """
    logger = setup_logger(__name__)
    logger.info("\n--- Final Preprocessed Data Information ---")
    logger.info(f"X_train shape: {X_train.shape}, dtype: {X_train.dtypes.iloc[0] if not X_train.empty else None}")
    logger.info(f"X_test shape: {X_test.shape}, dtype: {X_test.dtypes.iloc[0] if not X_test.empty else None}")
    logger.info(f"y_train distribution:\n{y_train.value_counts()}")
    logger.info(f"y_test distribution:\n{y_test.value_counts()}")

def train_random_forest_model(X_train: pd.DataFrame, y_train: pd.Series):
    """Trains a Random Forest model on the provided training data.

    Args:
        X_train (pd.DataFrame): The training features.
        y_train (pd.Series): The training target variable.

    Returns:
        tuple: A tuple containing the trained RandomForestClassifier model and
               a pandas Series of feature importances.
    """
    logger = setup_logger(__name__)
    logger.info("--- Training Random Forest Model ---")
    if X_train is None or y_train is None:
        raise ValueError("X_train or y_train cannot be None.")
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)
    feature_importance = pd.Series(model.feature_importances_, index=X_train.columns)
    logger.info("Random Forest model trained and feature importances extracted.")
    return model, feature_importance

def visualize_feature_importance(self, n=20):
    """Visualizes the top N feature importances."""
    if self.feature_importance is None:
        self.logger.warning("Feature importance not available. Train the Random Forest model first.")
        return
    self.logger.info(f"--- Visualizing Top {n} Feature Importances ---")
    self.visualizer.plot_feature_importance(self.feature_importance, n=n)

def get_feature_importance(self):
    """Returns the feature importance scores."""
    if self.feature_importance is None:
        self.logger.warning("Feature importance not yet calculated. Train the Random Forest model first.")
        return None
    return self.feature_importance

def get_processed_data(self):
    """Returns the preprocessed training and testing data."""
    if self.X_train is None or self.X_test is None or self.y_train is None or self.y_test is None:
        self.logger.warning("Processed data not yet available. Run preprocess_data() first.")
        return None, None, None, None
    return self.X_train, self.X_test, self.y_train, self.y_test




class CombinedFeaturesAdder(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X, y=None):
        self.feature_names_in_ = list(X.columns) if isinstance(X, pd.DataFrame) else [f"feature_{i}" for i in range(X.shape[1])]
        return self

    def transform(self, X):
        if isinstance(X, pd.DataFrame):
            latitude = X['LATITUDE'].values.reshape(-1, 1) if 'LATITUDE' in X.columns else np.zeros((X.shape[0], 1))
            longitude = X['LONGITUDE'].values.reshape(-1, 1) if 'LONGITUDE' in X.columns else np.zeros((X.shape[0], 1))
            time_col = X['TIME'].values.reshape(-1, 1) if 'TIME' in X.columns else np.zeros((X.shape[0], 1), dtype=object) # Treat as object initially
            date_col = X['DATE'].values.reshape(-1, 1) if 'DATE' in X.columns else np.zeros((X.shape[0], 1), dtype=object) # Treat as object initially
        else:
            latitude_idx = self.feature_names_in_.index('LATITUDE') if 'LATITUDE' in self.feature_names_in_ else -1
            longitude_idx = self.feature_names_in_.index('LONGITUDE') if 'LONGITUDE' in self.feature_names_in_ else -1
            time_idx = self.feature_names_in_.index('TIME') if 'TIME' in self.feature_names_in_ else -1
            date_idx = self.feature_names_in_.index('DATE') if 'DATE' in self.feature_names_in_ else -1

            latitude = X[:, latitude_idx].reshape(-1, 1) if latitude_idx != -1 else np.zeros((X.shape[0], 1))
            longitude = X[:, longitude_idx].reshape(-1, 1) if longitude_idx != -1 else np.zeros((X.shape[0], 1))
            time_col = X[:, time_idx].reshape(-1, 1) if time_idx != -1 else np.zeros((X.shape[0], 1), dtype=object)
            date_col = X[:, date_idx].reshape(-1, 1) if date_idx != -1 else np.zeros((X.shape[0], 1), dtype=object)

        combined_xy = np.sqrt(latitude ** 2 + longitude ** 2)

        hour_of_day = pd.to_numeric(pd.Series(time_col.flatten()).astype(str).str.slice(0, 2).fillna('0'), errors='coerce').fillna(0).astype(int).values.reshape(-1, 1)
        day_of_week = pd.to_datetime(pd.Series(date_col.flatten()), errors='coerce').dt.dayofweek.fillna(0).astype(int).values.reshape(-1, 1)

        return np.c_[X, combined_xy, hour_of_day, day_of_week]

    def get_feature_names_out(self, input_features=None):
        if input_features is None:
            return np.array(self.feature_names_in_ + ['combined_xy', 'hour_of_day', 'day_of_week'])
        else:
            if isinstance(input_features, np.ndarray):
                input_features = input_features.tolist()
            return np.array(input_features + ['combined_xy', 'hour_of_day', 'day_of_week'])
