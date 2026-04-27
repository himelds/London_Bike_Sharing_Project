import os
import sys
import pandas as pd
import yaml
from sklearn.model_selection import train_test_split
from src.exception import CustomException
from src.logger import logging
from dataclasses import dataclass

@dataclass
class DataIngestionConfig:
    """
    Configuration class for Data Ingestion.
    Reads paths from the config.yaml file and constructs absolute paths
    for the raw, train, and test data files.
    """
    def __init__(self):
        """Initializes data paths from the configuration file."""
        try:
            with open("config/config.yaml", 'r') as file:
                config = yaml.safe_load(file)
            self.train_data_path: str = os.path.join(os.getcwd(), config['data']['train_data_path'])
            self.test_data_path: str = os.path.join(os.getcwd(), config['data']['test_data_path'])
            self.raw_data_path: str = os.path.join(os.getcwd(), config['data']['raw_data_path'])
        except Exception as e:
            raise CustomException(e, sys)

class DataIngestion:
    """
    Data Ingestion component.
    Responsible for reading the raw dataset, performing a train-test split,
    and saving the split datasets into designated paths.
    """
    def __init__(self):
        """Initializes the DataIngestion component with its configuration."""
        self.ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self) -> tuple:
        """
        Initiates the data ingestion process.
        
        Reads the raw data, creates the necessary directories, splits the
        data into 80% training and 20% testing sets, and saves them as CSVs.

        Returns:
            tuple: A tuple containing the paths to the train and test CSV files.
        """
        logging.info("Entered the data ingestion method or component")
        try:
            # Read raw data
            df = pd.read_csv(self.ingestion_config.raw_data_path)
            logging.info('Read the dataset as dataframe')

            # Create directory for saving the output files
            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path), exist_ok=True)

            logging.info("Train test split initiated")
            # Perform train-test split
            train_set, test_set = train_test_split(df, test_size=0.2, random_state=42)

            # Save train and test sets
            train_set.to_csv(self.ingestion_config.train_data_path, index=False, header=True)
            test_set.to_csv(self.ingestion_config.test_data_path, index=False, header=True)

            logging.info("Ingestion of the data is completed")

            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )
        except Exception as e:
            raise CustomException(e, sys)

if __name__=="__main__":
    obj=DataIngestion()
    obj.initiate_data_ingestion()
