"""Abstract base class shared by XGBoost and LightGBM wrappers."""

from abc import ABC, abstractmethod

import numpy as np
import pandas as pd


class BaseModel(ABC):
    """Common interface for all classifiers in this project."""

    @abstractmethod
    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        """Train the model."""

    @abstractmethod
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Return class labels (0/1)."""

    @abstractmethod
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Return probability estimates, shape (n, 2)."""

    @abstractmethod
    def save(self, path: str) -> None:
        """Persist model to disk."""

    @abstractmethod
    def load(self, path: str) -> None:
        """Load model from disk."""
