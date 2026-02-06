
"""Data loading and validation for fraud detection pipeline."""

import pandas as pd
from sklearn.model_selection import train_test_split        

import logging
from pathlib import Path
from typing import Union

import pandas as pd


logger = logging.getLogger(__name__)

REQUIRED_COLUMNS = [
    'transaction_id', 'amount', 'merchant_category',
    'hour', 'day_of_week', 'is_fraud'
]

# Load transactions from CSV with validation
def load_transactions(file_path: Union[str, Path]) -> pd.DataFrame:
    """Load transactions from a CSV file and validate the data."""
    logger.info(f"Loading transactions from {file_path}")
    
    # Load the data
    df = pd.read_csv(file_path)
    
    # Validate required columns
    missing_cols = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")
    
    # Validate data types
    if not pd.api.types.is_numeric_dtype(df['amount']):
        raise ValueError("Column 'amount' must be numeric.")
    if not pd.api.types.is_integer_dtype(df['hour']):
        raise ValueError("Column 'hour' must be an integer.")
    if not pd.api.types.is_integer_dtype(df['day_of_week']):
        raise ValueError("Column 'day_of_week' must be an integer.")
    if not pd.api.types.is_integer_dtype(df['is_fraud']):
        raise ValueError("Column 'is_fraud' must be an integer (0 or 1).")
    
    logger.info("Data loaded and validated successfully.")
    return df   
