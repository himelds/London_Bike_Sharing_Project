"""
Logging Module.
Configures the global logging behavior for the project. Logs are saved
in a 'logs' directory with a timestamped filename.
"""

import logging
import os
from datetime import datetime

# Generate a log file name based on the current timestamp
LOG_FILE = f"{datetime.now().strftime('%m_%d_%Y_%H_%M_%S')}.log"

# Define the absolute path to the logs directory and ensure it exists
logs_path = os.path.join(os.getcwd(), "logs", LOG_FILE)
os.makedirs(os.path.dirname(logs_path), exist_ok=True)

LOG_FILE_PATH = os.path.join(logs_path)

# Configure the basic settings for the logging module
logging.basicConfig(
    filename=LOG_FILE_PATH,
    format="[ %(asctime)s ] %(lineno)d %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)
