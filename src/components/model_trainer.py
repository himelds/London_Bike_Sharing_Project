import os
import sys
import joblib
import yaml
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from src.exception import CustomException
from src.logger import logging

class ModelTrainerConfig:
    """
    Configuration class for Model Training.
    Reads hyperparameters and output paths from the config.yaml file.
    """
    def __init__(self):
        """Initializes model paths and parameters from configuration file."""
        try:
            with open("config/config.yaml", 'r') as file:
                config = yaml.safe_load(file)
            self.trained_model_file_path = os.path.join(os.getcwd(), config['model']['saved_model_path'])
            self.rf_params = config['hyperparameters']['random_forest']
        except Exception as e:
            raise CustomException(e, sys)

class ModelTrainer:
    """
    Model Trainer component.
    Handles the instantiation, training, and evaluation of the machine
    learning model. Currently utilizes Random Forest Regressor.
    """
    def __init__(self):
        """Initializes the ModelTrainer component with its configuration."""
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self, X_train, y_train, X_test, y_test) -> tuple:
        """
        Trains and evaluates the machine learning model.
        
        Args:
            X_train (pd.DataFrame): Training features.
            y_train (pd.Series): Training target variable.
            X_test (pd.DataFrame): Testing features.
            y_test (pd.Series): Testing target variable.

        Returns:
            tuple: Contains (R2 Score, Mean Absolute Error, Root Mean Squared Error).
        """
        try:
            logging.info("Training Random Forest Regressor")
            
            # Using Random Forest as it was the recommended model in the README
            model = RandomForestRegressor(**self.model_trainer_config.rf_params)
            model.fit(X_train, y_train)

            logging.info("Evaluating model on test set")
            y_pred = model.predict(X_test)
            
            # Calculate evaluation metrics
            r2 = r2_score(y_test, y_pred)
            mae = mean_absolute_error(y_test, y_pred)
            rmse = mean_squared_error(y_test, y_pred)**0.5
            
            logging.info(f"Model Performance: R2={r2}, MAE={mae}, RMSE={rmse}")
            print(f"Model Performance: R2={r2}, MAE={mae}, RMSE={rmse}")

            # Save the trained model to disk
            os.makedirs(os.path.dirname(self.model_trainer_config.trained_model_file_path), exist_ok=True)
            joblib.dump(model, self.model_trainer_config.trained_model_file_path)
            logging.info("Saved trained model")

            return r2, mae, rmse

        except Exception as e:
            raise CustomException(e, sys)
