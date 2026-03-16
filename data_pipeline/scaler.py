"""
Scaler — Step 3: Data Standardization and Scaling
==================================================

The 97 NPPAD parameters operate on vastly different numerical scales and
physical units.  Normalisation ensures that every feature contributes
equally to the model's learning process.

Two scalers are provided:

* **ZScoreScaler** (primary) — transforms each feature to zero mean and
  unit variance:  ``z = (x - mu) / sigma``.
* **MinMaxScaler** (secondary) — rescales each feature to a fixed [0, 1]
  range:  ``x' = (x - x_min) / (x_max - x_min)``.  Useful when certain
  features exhibit highly skewed rather than normal distributions.

Both follow a ``fit -> transform`` API and support serialisation to JSON.

Classes
-------
ZScoreScaler
    Z-score (standard) normalisation.
MinMaxScaler
    Min-max normalisation to [0, 1].
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import polars as pl

logger = logging.getLogger(__name__)


# ======================================================================
# Z-Score Scaler
# ======================================================================

class ZScoreScaler:
    """Z-score standardisation: ``z = (x - mu) / sigma``.

    Computes per-feature statistics across **all** samples during
    :meth:`fit`, then applies the transformation sample-by-sample via
    :meth:`transform`.

    Parameters
    ----------
    epsilon : float, optional
        Small constant added to sigma to avoid division by zero for
        constant-valued features (default ``1e-8``).
    """

    def __init__(self, epsilon: float = 1e-8) -> None:
        self.epsilon = epsilon
        self.mean_: Optional[np.ndarray] = None
        self.std_: Optional[np.ndarray] = None
        self.feature_names_: Optional[List[str]] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(self, samples: List[pl.DataFrame]) -> "ZScoreScaler":
        """Compute per-feature mean and standard deviation.

        Parameters
        ----------
        samples : list[pl.DataFrame]
            List of ``(T_i, F)`` DataFrames sharing identical column names.

        Returns
        -------
        self
        """
        concatenated = pl.concat(samples)
        self.feature_names_ = concatenated.columns
        self.mean_ = concatenated.mean().to_numpy().flatten().astype(np.float64)
        self.std_ = concatenated.std().to_numpy().flatten().astype(np.float64)
        logger.info(
            "ZScoreScaler fitted on %d samples (%d total rows, %d features)",
            len(samples),
            len(concatenated),
            len(self.feature_names_),
        )
        return self

    def transform(self, samples: List[pl.DataFrame]) -> List[pl.DataFrame]:
        """Apply Z-score standardisation to each sample.

        Parameters
        ----------
        samples : list[pl.DataFrame]
            List of ``(T_i, F)`` DataFrames.

        Returns
        -------
        list[pl.DataFrame]
            Standardised DataFrames (same column order).
        """
        self._check_fitted()
        transformed: List[pl.DataFrame] = []
        for df in samples:
            values = (df.to_numpy().astype(np.float64) - self.mean_) / (
                self.std_ + self.epsilon
            )
            transformed.append(
                pl.DataFrame(values, schema=df.columns, orient="row")
            )
        return transformed

    def fit_transform(self, samples: List[pl.DataFrame]) -> List[pl.DataFrame]:
        """Convenience method: ``fit`` then ``transform``."""
        return self.fit(samples).transform(samples)

    def inverse_transform(
        self, samples: List[pl.DataFrame]
    ) -> List[pl.DataFrame]:
        """Reverse the Z-score transformation.

        Parameters
        ----------
        samples : list[pl.DataFrame]

        Returns
        -------
        list[pl.DataFrame]
        """
        self._check_fitted()
        result: List[pl.DataFrame] = []
        for df in samples:
            values = (
                df.to_numpy().astype(np.float64)
                * (self.std_ + self.epsilon)
                + self.mean_
            )
            result.append(
                pl.DataFrame(values, schema=df.columns, orient="row")
            )
        return result

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str | Path) -> None:
        """Serialise scaler statistics to a JSON file.

        Parameters
        ----------
        path : str | Path
            Destination file path (e.g. ``"scaler_params.json"``).
        """
        self._check_fitted()
        payload: Dict = {
            "type": "ZScoreScaler",
            "epsilon": self.epsilon,
            "feature_names": self.feature_names_,
            "mean": self.mean_.tolist(),
            "std": self.std_.tolist(),
        }
        Path(path).write_text(json.dumps(payload, indent=2))
        logger.info("ZScoreScaler saved to %s", path)

    @classmethod
    def load(cls, path: str | Path) -> "ZScoreScaler":
        """Load a previously saved scaler from JSON.

        Parameters
        ----------
        path : str | Path

        Returns
        -------
        ZScoreScaler
        """
        payload = json.loads(Path(path).read_text())
        scaler = cls(epsilon=payload["epsilon"])
        scaler.feature_names_ = payload["feature_names"]
        scaler.mean_ = np.array(payload["mean"], dtype=np.float64)
        scaler.std_ = np.array(payload["std"], dtype=np.float64)
        logger.info("ZScoreScaler loaded from %s", path)
        return scaler

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _check_fitted(self) -> None:
        if self.mean_ is None:
            raise RuntimeError(
                "Scaler has not been fitted yet. Call fit() first."
            )


# ======================================================================
# Min-Max Scaler
# ======================================================================

class MinMaxScaler:
    """Min-Max normalisation to [0, 1]: ``x' = (x - min) / (max - min)``.

    Useful as a secondary scaler when certain features exhibit highly
    skewed distributions rather than normal distributions.

    Parameters
    ----------
    epsilon : float, optional
        Small constant to avoid division by zero (default ``1e-8``).
    """

    def __init__(self, epsilon: float = 1e-8) -> None:
        self.epsilon = epsilon
        self.min_: Optional[np.ndarray] = None
        self.max_: Optional[np.ndarray] = None
        self.feature_names_: Optional[List[str]] = None

    def fit(self, samples: List[pl.DataFrame]) -> "MinMaxScaler":
        """Compute per-feature min and max across all samples.

        Parameters
        ----------
        samples : list[pl.DataFrame]

        Returns
        -------
        self
        """
        concatenated = pl.concat(samples)
        self.feature_names_ = concatenated.columns
        self.min_ = concatenated.min().to_numpy().flatten().astype(np.float64)
        self.max_ = concatenated.max().to_numpy().flatten().astype(np.float64)
        logger.info(
            "MinMaxScaler fitted on %d samples (%d features)",
            len(samples),
            len(self.feature_names_),
        )
        return self

    def transform(self, samples: List[pl.DataFrame]) -> List[pl.DataFrame]:
        """Apply Min-Max scaling to each sample.

        Parameters
        ----------
        samples : list[pl.DataFrame]

        Returns
        -------
        list[pl.DataFrame]
        """
        self._check_fitted()
        range_ = (self.max_ - self.min_) + self.epsilon
        transformed: List[pl.DataFrame] = []
        for df in samples:
            values = (df.to_numpy().astype(np.float64) - self.min_) / range_
            transformed.append(
                pl.DataFrame(values, schema=df.columns, orient="row")
            )
        return transformed

    def fit_transform(self, samples: List[pl.DataFrame]) -> List[pl.DataFrame]:
        """Convenience: ``fit`` then ``transform``."""
        return self.fit(samples).transform(samples)

    def inverse_transform(
        self, samples: List[pl.DataFrame]
    ) -> List[pl.DataFrame]:
        """Reverse the Min-Max transformation."""
        self._check_fitted()
        range_ = (self.max_ - self.min_) + self.epsilon
        result: List[pl.DataFrame] = []
        for df in samples:
            values = df.to_numpy().astype(np.float64) * range_ + self.min_
            result.append(
                pl.DataFrame(values, schema=df.columns, orient="row")
            )
        return result

    def save(self, path: str | Path) -> None:
        """Serialise scaler statistics to JSON."""
        self._check_fitted()
        payload: Dict = {
            "type": "MinMaxScaler",
            "epsilon": self.epsilon,
            "feature_names": self.feature_names_,
            "min": self.min_.tolist(),
            "max": self.max_.tolist(),
        }
        Path(path).write_text(json.dumps(payload, indent=2))
        logger.info("MinMaxScaler saved to %s", path)

    @classmethod
    def load(cls, path: str | Path) -> "MinMaxScaler":
        """Load a previously saved scaler from JSON."""
        payload = json.loads(Path(path).read_text())
        scaler = cls(epsilon=payload["epsilon"])
        scaler.feature_names_ = payload["feature_names"]
        scaler.min_ = np.array(payload["min"], dtype=np.float64)
        scaler.max_ = np.array(payload["max"], dtype=np.float64)
        logger.info("MinMaxScaler loaded from %s", path)
        return scaler

    def _check_fitted(self) -> None:
        if self.min_ is None:
            raise RuntimeError(
                "Scaler has not been fitted yet. Call fit() first."
            )
