import numpy as np
from typing import Tuple


def get_stats(data: np.ndarray) -> Tuple[float, float]:
    """Calculate mean and standard deviation of data.
    
    Args:
        data: Input array of numerical values.
        
    Returns:
        Tuple containing (mean, standard_deviation).
    """
    return float(np.mean(data)), float(np.std(data))