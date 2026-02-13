"""Feature engineering for fraud detection pipeline."""

import logging
from typing import List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

NIGHT_START_HOUR = 22
NIGHT_END_HOUR = 5
WEEKEND_START_DAY = 5

AMOUNT_TRANSFORMATIONS = {
    'amount_log': np.log1p,
    'amount_percentile': lambda x: x.rank(pct=True),
}


def create_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create time-based features from transaction data.

    Extracts and engineers time-based features to identify patterns in
    transaction timing, including weekend flags and night hour indicators.

    Args:
        df: DataFrame with 'hour' and 'day_of_week' columns.

    Returns:
        DataFrame with added time features:
        - is_weekend: Binary flag for weekend transactions
        - is_night: Binary flag for night hour transactions
        - hour_sin/hour_cos: Cyclic encoding of hour
        - day_sin/day_cos: Cyclic encoding of day of week

    Raises:
        ValueError: If required columns are missing.

    Example:
        >>> df = pd.DataFrame({'hour': [10, 22, 5], 'day_of_week': [2, 5, 6]})
        >>> result = create_time_features(df)
        >>> 'is_weekend' in result.columns
        True
    """
    required_cols = ['hour', 'day_of_week']
    missing = set(required_cols) - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    result = df.copy()

    result['is_weekend'] = (result['day_of_week'] >= WEEKEND_START_DAY).astype(int)
    result['is_night'] = (
        (result['hour'] >= NIGHT_START_HOUR) | (result['hour'] <= NIGHT_END_HOUR)
    ).astype(int)

    result['hour_sin'] = np.sin(2 * np.pi * result['hour'] / 24)
    result['hour_cos'] = np.cos(2 * np.pi * result['hour'] / 24)
    result['day_sin'] = np.sin(2 * np.pi * result['day_of_week'] / 7)
    result['day_cos'] = np.cos(2 * np.pi * result['day_of_week'] / 7)

    logger.info("Created time features")
    return result


def create_amount_features(
    df: pd.DataFrame,
    transformations: Optional[List[str]] = None
) -> pd.DataFrame:
    """Create amount-based features.

    Engineers amount-based features including logarithmic transformations,
    percentile rankings, and z-score normalization for fraud detection.

    Args:
        df: DataFrame with 'amount' column.
        transformations: List of transformations to apply. If None, applies all
            available transformations from AMOUNT_TRANSFORMATIONS.

    Returns:
        DataFrame with added amount features:
        - amount_log: Log-transformed amount
        - amount_percentile: Percentile rank of amount
        - amount_zscore: Z-score normalized amount

    Raises:
        ValueError: If 'amount' column is missing or unknown transformation
            is specified.

    Example:
        >>> df = pd.DataFrame({'amount': [100, 500, 1000]})
        >>> result = create_amount_features(df)
        >>> 'amount_zscore' in result.columns
        True
    """
    if 'amount' not in df.columns:
        raise ValueError("Missing required column: 'amount'")

    if transformations is None:
        transformations = list(AMOUNT_TRANSFORMATIONS.keys())

    result = df.copy()

    for name in transformations:
        if name not in AMOUNT_TRANSFORMATIONS:
            raise ValueError(f"Unknown transformation: {name}")
        result[name] = AMOUNT_TRANSFORMATIONS[name](result['amount'])

    mean, std = result['amount'].mean(), result['amount'].std()
    result['amount_zscore'] = (result['amount'] - mean) / std if std > 0 else 0.0

    logger.info("Created amount features")
    return result


def create_all_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create all features for fraud detection.

    Applies all feature engineering transformations in sequence: time-based
    features followed by amount-based features.

    Args:
        df: DataFrame with required columns: 'hour', 'day_of_week', 'amount'.

    Returns:
        DataFrame with all engineered features added.

    Raises:
        ValueError: If any required columns are missing.

    Example:
        >>> df = pd.DataFrame({
        ...     'hour': [10, 22],
        ...     'day_of_week': [2, 5],
        ...     'amount': [100, 500]
        ... })
        >>> result = create_all_features(df)
        >>> len(result.columns) > len(df.columns)
        True
    """
    result = df.copy()
    result = create_time_features(result)
    result = create_amount_features(result)

    logger.info(f"Created all features. Total columns: {len(result.columns)}")
    return result