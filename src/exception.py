"""
Custom Exception Handling Module.
Provides robust exception handling mechanisms that automatically capture
the file name, line number, and exact error message for easier debugging.
"""

import sys
from src.logger import logging

def error_message_detail(error, error_detail: sys) -> str:
    """
    Extracts detailed error information including filename and line number.

    Args:
        error (Exception): The actual exception that occurred.
        error_detail (sys): The sys module to extract traceback details.

    Returns:
        str: A formatted string detailing where the error occurred and what it was.
    """
    _, _, exc_tb = error_detail.exc_info()
    file_name = exc_tb.tb_frame.f_code.co_filename
    error_message = "Error occured in python script name [{0}] line number [{1}] error message[{2}]".format(
     file_name, exc_tb.tb_lineno, str(error))

    return error_message

class CustomException(Exception):
    """
    Custom exception class that formats standard exceptions with detailed
    traceback information.
    """
    def __init__(self, error_message, error_detail: sys):
        """
        Initializes the custom exception.

        Args:
            error_message (str or Exception): The original error message.
            error_detail (sys): The sys module for traceback details.
        """
        super().__init__(error_message)
        self.error_message = error_message_detail(error_message, error_detail=error_detail)
    
    def __str__(self) -> str:
        """Returns the fully detailed error message."""
        return self.error_message
