"""
Training Pipeline Orchestrator.
This script coordinates the entire machine learning pipeline, running
data ingestion, data transformation, and model training in sequence.
"""

from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer
from src.logger import logging

if __name__ == "__main__":
    logging.info("Training Pipeline Started")
    
    # Step 1: Data Ingestion
    obj = DataIngestion()
    train_data, test_data = obj.initiate_data_ingestion()
    
    # Step 2: Data Transformation
    data_transformation = DataTransformation()
    X_train, y_train, X_test, y_test, preprocessor_path = data_transformation.initiate_data_transformation(train_data, test_data)
    
    # Step 3: Model Training
    model_trainer = ModelTrainer()
    model_trainer.initiate_model_trainer(X_train, y_train, X_test, y_test)
    
    logging.info("Training Pipeline Completed")
