"""Small utility helpers for validation used across modules.

These helpers perform input validation for DataFrame columns and value ranges
to keep downstream code robust and testable.
"""

import logging
from typing import Iterable, List

import pandas as pd

logger = logging.getLogger(__name__)


def validate_required_columns(df: pd.DataFrame, required: Iterable[str]) -> None:
    """Ensure the DataFrame contains the required columns.

    Args:
        df: DataFrame to validate.
        required: Iterable of required column names.

    Raises:
        ValueError: If any required column is missing.
    """
    missing = set(required) - set(df.columns)
    if missing:
        logger.error("Missing required columns: %s", missing)
        raise ValueError(f"Missing required columns: {missing}")


def validate_hour_and_day(df: pd.DataFrame, hour_col: str = "hour", day_col: str = "day_of_week") -> None:
    """Validate that hour and day columns are within expected ranges.

    Args:
        df: DataFrame to validate.
        hour_col: Column name for hour (0-23).
        day_col: Column name for day of week (0-6).

    Raises:
        ValueError: If any values fall outside expected ranges.
    """
    if hour_col in df.columns:
        invalid_hours = df.loc[~df[hour_col].between(0, 23), hour_col]
        if not invalid_hours.empty:
            logger.error("Found invalid hour values: %s", invalid_hours.unique())
            raise ValueError(f"Invalid hour values found in column '{hour_col}'")

    if day_col in df.columns:
        invalid_days = df.loc[~df[day_col].between(0, 6), day_col]
        if not invalid_days.empty:
            logger.error("Found invalid day_of_week values: %s", invalid_days.unique())
            raise ValueError(f"Invalid day_of_week values found in column '{day_col}'")


def validate_amounts(df: pd.DataFrame, amount_col: str = "amount") -> None:
    """Validate amount column is numeric and not all null.

    Args:
        df: DataFrame to validate.
        amount_col: Column name containing transaction amounts.

    Raises:
        ValueError: If amounts are missing or not numeric.
    """
    if amount_col not in df.columns:
        raise ValueError(f"Missing amount column: {amount_col}")

    if not pd.api.types.is_numeric_dtype(df[amount_col]):
        raise ValueError(f"Column '{amount_col}' must be numeric")

    if df[amount_col].isnull().all():
        raise ValueError("All values in amount column are null")
