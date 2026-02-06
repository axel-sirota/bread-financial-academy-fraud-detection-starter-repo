"""
Transaction Data Loader and Time Feature Engineering Module

This module provides classes for loading transactional data and creating
time-based features for fraud detection models.
"""

import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path
from typing import Union, Optional, Tuple


class TransactionDataLoader:
    """
    A class for loading transactional data and creating time-based features.
    
    This loader handles:
    - Reading transaction CSV files
    - Parsing temporal information
    - Creating time-based features for fraud detection
    
    Attributes:
        data (pd.DataFrame): Loaded transaction data
        time_column (str): Name of the time column in the data
    """
    
    def __init__(self, time_column: str = 'timestamp'):
        """
        Initialize the TransactionDataLoader.
        
        Args:
            time_column (str): Name of the column containing timestamp information.
                Default is 'timestamp'.
        """
        self.data = None
        self.time_column = time_column
    
    def load_data(self, file_path: Union[str, Path]) -> pd.DataFrame:
        """
        Load transaction data from a CSV file.
        
        Args:
            file_path (Union[str, Path]): Path to the CSV file containing transaction data.
        
        Returns:
            pd.DataFrame: Loaded transaction data.
        
        Raises:
            FileNotFoundError: If the file does not exist.
            pd.errors.ParserError: If the CSV file cannot be parsed.
        """
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        self.data = pd.read_csv(file_path)
        print(f"✓ Loaded {len(self.data)} transactions from {file_path.name}")
        print(f"  Columns: {', '.join(self.data.columns.tolist())}")
        
        return self.data
    
    def create_time_features(self) -> pd.DataFrame:
        """
        Create time-based features from transaction data.
        
        Requires:
        - 'hour' column: Hour of transaction (0-23)
        - 'day_of_week' column: Day of week (0-6, where 0 is Monday)
        
        Created Features:
        - is_morning: 1 if hour is 6-12, 0 otherwise
        - is_afternoon: 1 if hour is 12-18, 0 otherwise
        - is_evening: 1 if hour is 18-23, 0 otherwise
        - is_night: 1 if hour is 23-6, 0 otherwise
        - is_weekend: 1 if day is Saturday (5) or Sunday (6), 0 otherwise
        - is_weekday: 1 if day is Monday-Friday, 0 otherwise
        - hour_sin: Sine transformation of hour for cyclical encoding
        - hour_cos: Cosine transformation of hour for cyclical encoding
        - day_sin: Sine transformation of day for cyclical encoding
        - day_cos: Cosine transformation of day for cyclical encoding
        
        Returns:
            pd.DataFrame: Original data with added time features.
        
        Raises:
            ValueError: If required time columns are missing.
            AttributeError: If data has not been loaded yet.
        """
        if self.data is None:
            raise AttributeError("No data loaded. Call load_data() first.")
        
        required_columns = ['hour', 'day_of_week']
        missing_columns = [col for col in required_columns if col not in self.data.columns]
        
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")
        
        # Create time of day features (categorical)
        self.data['is_morning'] = ((self.data['hour'] >= 6) & (self.data['hour'] < 12)).astype(int)
        self.data['is_afternoon'] = ((self.data['hour'] >= 12) & (self.data['hour'] < 18)).astype(int)
        self.data['is_evening'] = ((self.data['hour'] >= 18) & (self.data['hour'] < 23)).astype(int)
        self.data['is_night'] = ((self.data['hour'] >= 23) | (self.data['hour'] < 6)).astype(int)
        
        # Create day of week features
        self.data['is_weekend'] = self.data['day_of_week'].isin([5, 6]).astype(int)
        self.data['is_weekday'] = (~self.data['day_of_week'].isin([5, 6])).astype(int)
        
        # Create cyclical encoding for hour (24-hour cycle)
        self.data['hour_sin'] = np.sin(2 * np.pi * self.data['hour'] / 24)
        self.data['hour_cos'] = np.cos(2 * np.pi * self.data['hour'] / 24)
        
        # Create cyclical encoding for day of week (7-day cycle)
        self.data['day_sin'] = np.sin(2 * np.pi * self.data['day_of_week'] / 7)
        self.data['day_cos'] = np.cos(2 * np.pi * self.data['day_of_week'] / 7)
        
        print("✓ Time features created successfully")
        
        return self.data
    
    def get_feature_summary(self) -> dict:
        """
        Get a summary of available time features.
        
        Returns:
            dict: Dictionary containing feature names and their descriptions.
        """
        features = {
            'Time of Day': ['is_morning', 'is_afternoon', 'is_evening', 'is_night'],
            'Day of Week': ['is_weekend', 'is_weekday'],
            'Cyclical Encoding': ['hour_sin', 'hour_cos', 'day_sin', 'day_cos']
        }
        return features
    
    def get_data(self) -> pd.DataFrame:
        """
        Get the current transaction data.
        
        Returns:
            pd.DataFrame: Current transaction data with all features.
        
        Raises:
            AttributeError: If data has not been loaded yet.
        """
        if self.data is None:
            raise AttributeError("No data loaded. Call load_data() first.")
        
        return self.data.copy()
    
    def get_statistics(self) -> dict:
        """
        Get statistics about the loaded transaction data.
        
        Returns:
            dict: Dictionary containing data statistics.
        
        Raises:
            AttributeError: If data has not been loaded yet.
        """
        if self.data is None:
            raise AttributeError("No data loaded. Call load_data() first.")
        
        stats = {
            'total_transactions': len(self.data),
            'columns': list(self.data.columns),
            'shape': self.data.shape,
            'missing_values': self.data.isnull().sum().to_dict(),
            'data_types': self.data.dtypes.to_dict()
        }
        
        return stats

