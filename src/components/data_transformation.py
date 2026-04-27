import sys
import os
import pandas as pd
import numpy as np
import yaml
import joblib
from src.exception import CustomException
from src.logger import logging

class DataTransformationConfig:
    """
    Configuration class for Data Transformation.
    Reads paths from config.yaml and constructs absolute path for
    saving the preprocessor object (e.g., column names used for encoding).
    """
    def __init__(self):
        """Initializes preprocessor paths from the configuration file."""
        try:
            with open("config/config.yaml", 'r') as file:
                config = yaml.safe_load(file)
            self.preprocessor_obj_file_path = os.path.join(os.getcwd(), config['model']['preprocessor_path'])
        except Exception as e:
            raise CustomException(e, sys)

class DataTransformation:
    """
    Data Transformation component.
    Handles feature engineering, datetime parsing, column dropping,
    and categorical one-hot encoding.
    """
    def __init__(self):
        """Initializes DataTransformation with its configuration."""
        self.data_transformation_config = DataTransformationConfig()

    def initiate_data_transformation(self, train_path: str, test_path: str) -> tuple:
        """
        Initiates the data transformation process on train and test datasets.
        
        Args:
            train_path (str): File path to the training dataset.
            test_path (str): File path to the testing dataset.

        Returns:
            tuple: Contains (X_train, y_train, X_test, y_test, preprocessor_path)
                   ready for model training.
        """
        try:
            logging.info("Starting data transformation")
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info("Read train and test data completed")

            def transform_data(df: pd.DataFrame) -> pd.DataFrame:
                """
                Helper function to apply identical transformations to a dataframe.
                Extracts datetime components and applies one-hot encoding.
                """
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                df['hour'] = df['timestamp'].dt.hour
                df['day'] = df['timestamp'].dt.day
                df['month'] = df['timestamp'].dt.month
                df['year'] = df['timestamp'].dt.year
                
                df.drop(['timestamp', 't2'], axis=1, inplace=True)
                
                # Apply one-hot encoding on categorical features
                df = pd.get_dummies(df, columns=['weather_code', 'is_holiday', 'is_weekend', 'season'], drop_first=True)
                return df

            logging.info("Applying transformations to train and test sets")
            
            target_column_name = "cnt"

            # Transform features
            train_df = transform_data(train_df)
            test_df = transform_data(test_df)

            # Ensure both train and test have same dummy columns by aligning them
            train_df, test_df = train_df.align(test_df, join='left', axis=1, fill_value=0)

            X_train = train_df.drop(columns=[target_column_name], axis=1)
            y_train = train_df[target_column_name]

            X_test = test_df.drop(columns=[target_column_name], axis=1)
            y_test = test_df[target_column_name]

            # Save the column names expected by the model for inference (used by the Web App)
            os.makedirs(os.path.dirname(self.data_transformation_config.preprocessor_obj_file_path), exist_ok=True)
            joblib.dump(list(X_train.columns), self.data_transformation_config.preprocessor_obj_file_path)

            logging.info(f"Saved preprocessor (column names) object.")

            return (
                X_train, y_train,
                X_test, y_test,
                self.data_transformation_config.preprocessor_obj_file_path,
            )

        except Exception as e:
            raise CustomException(e, sys)
