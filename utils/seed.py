"""Reproducibility helper — call set_seed() at the top of every script."""
import random

import numpy as np


def set_seed(seed: int = 42) -> None:
    """Fix random seeds for Python, NumPy (XGBoost/LightGBM use these)."""
    random.seed(seed)
    np.random.seed(seed)


set_seed(42)
