from typing import Optional, List
from pathlib import Path
import logging

import pandas as pd

logger = logging.getLogger(__name__)

REQUIRED_COLUMNS: List[str] = ["amount", "is_fraud"]


def load_transactions(csv_path: Optional[str] = None) -> pd.DataFrame:
    """Load transactions CSV into a pandas DataFrame and validate it.

    Args:
        csv_path: Optional path to the CSV file. If None, defaults to
            `data/transactions_sample.csv` relative to the repo root.

    Returns:
        A validated `pandas.DataFrame` containing the transactions.

    Raises:
        FileNotFoundError: If the CSV file does not exist.
        ValueError: If the loaded DataFrame is empty or missing required columns.
    """
    path = Path(csv_path) if csv_path else Path("data") / "transactions_sample.csv"

    if not path.exists():
        logger.error("Transactions file not found: %s", path)
        raise FileNotFoundError(f"Transactions file not found: {path}")

    df = pd.read_csv(path)

    if df.empty:
        logger.error("Loaded transactions file is empty: %s", path)
        raise ValueError("Transactions file is empty")

    validate_transactions(df)
    logger.info("Loaded transactions: %d rows from %s", len(df), path)
    return df


def validate_transactions(df: pd.DataFrame) -> None:
    """Validate transaction DataFrame for required columns and basic quality.

    Args:
        df: DataFrame to validate.

    Raises:
        ValueError: If required columns are missing or contain nulls, or if
            `is_fraud` cannot be interpreted as binary labels.
    """
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        logger.error("Missing required columns: %s", missing)
        raise ValueError(f"Missing required columns: {missing}")

    null_counts = {c: int(df[c].isnull().sum()) for c in REQUIRED_COLUMNS}
    if any(v > 0 for v in null_counts.values()):
        logger.error("Required columns contain nulls: %s", null_counts)
        raise ValueError(f"Required columns contain nulls: {null_counts}")

    # Ensure `is_fraud` is binary (0/1) or boolean; try safe conversion
    try:
        if not pd.api.types.is_bool_dtype(df["is_fraud"]):
            df["is_fraud"] = df["is_fraud"].astype(int)
    except Exception as exc:  # noqa: BLE001 - we re-raise with context
        logger.error("Could not convert 'is_fraud' to integer/boolean: %s", exc)
        raise ValueError("Column 'is_fraud' must be binary (0/1) or boolean")

    logger.debug("Transaction DataFrame validation passed (%d rows)", len(df))


__all__ = ["load_transactions", "validate_transactions"]
