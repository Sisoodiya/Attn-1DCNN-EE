"""
Data Cleaning — Step 2: Data Cleaning and Noise Reduction
=========================================================

In real-world nuclear power generation, extreme physical environments and
high ionizing radiation can negatively impact sensor operation, frequently
resulting in incomplete data or transmission failures.  This module handles:

* **Missing values** — forward-fill, backward-fill, then linear interpolation.
* **Anomalous outliers** — values that deviate beyond a configurable Z-score
  threshold are replaced with linearly interpolated values.

Classes
-------
DataCleaner
    Stateless cleaning transformer for per-sample DataFrames.
"""

from __future__ import annotations

import logging
from typing import List

import numpy as np
import polars as pl

logger = logging.getLogger(__name__)


class DataCleaner:
    """Clean individual NPPAD time-series samples.

    Parameters
    ----------
    nan_strategy : str, optional
        Strategy for handling missing values.  One of
        ``"interpolate"`` (linear interpolation, the default) or
        ``"ffill"`` (forward-fill then backward-fill).
    z_threshold : float, optional
        Z-score threshold for anomaly detection (default ``6.0``).
        Values that deviate more than *z_threshold* standard deviations
        from their column mean are flagged as outliers.
    """

    VALID_NAN_STRATEGIES = {"interpolate", "ffill"}

    def __init__(
        self,
        nan_strategy: str = "interpolate",
        z_threshold: float = 6.0,
    ) -> None:
        if nan_strategy not in self.VALID_NAN_STRATEGIES:
            raise ValueError(
                f"nan_strategy must be one of {self.VALID_NAN_STRATEGIES}, "
                f"got '{nan_strategy}'"
            )
        self.nan_strategy = nan_strategy
        self.z_threshold = z_threshold

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def handle_missing(self, df: pl.DataFrame) -> pl.DataFrame:
        """Fill missing / NaN values in a single sample DataFrame.

        Parameters
        ----------
        df : pl.DataFrame
            A ``(T, F)`` DataFrame (one simulation run).

        Returns
        -------
        pl.DataFrame
            DataFrame with no remaining NaN / null values.
        """
        n_null = self._null_cells(df)
        n_nan = sum(
            df.select(pl.col(c).is_nan().sum()).item()
            for c in df.columns
            if df[c].dtype in (pl.Float32, pl.Float64)
        )
        n_missing = n_null + n_nan

        if n_missing == 0:
            return df

        logger.debug(
            "Filling %d missing values (strategy=%s)",
            n_missing,
            self.nan_strategy,
        )

        # Replace NaN with null so Polars can handle them uniformly
        df = df.with_columns(
            pl.col(c).fill_nan(None)
            for c in df.columns
            if df[c].dtype in (pl.Float32, pl.Float64)
        )

        if self.nan_strategy == "interpolate":
            # Interpolation alone does not fill edge-null regions.
            df = df.with_columns(
                pl.all().interpolate().forward_fill().backward_fill()
            )
        else:  # ffill
            df = df.with_columns(pl.all().forward_fill().backward_fill())

        # Safety: fill any residual nulls with 0
        residual_nulls = self._null_cells(df)
        if residual_nulls > 0:
            logger.warning(
                "Residual null values (%d) filled with 0.", residual_nulls
            )
            df = df.fill_null(0.0)

        return df

    def remove_anomalies(self, df: pl.DataFrame) -> pl.DataFrame:
        """Replace anomalous outlier values with interpolated estimates.

        For each numeric column, a value is considered anomalous if its
        absolute Z-score (computed *within this sample*) exceeds
        :attr:`z_threshold`.  Flagged values are set to null and then
        linearly interpolated.

        Parameters
        ----------
        df : pl.DataFrame
            A ``(T, F)`` DataFrame (one simulation run).

        Returns
        -------
        pl.DataFrame
            DataFrame with outlier values replaced.
        """
        numeric_cols = [
            c for c in df.columns
            if df[c].dtype in (pl.Float32, pl.Float64, pl.Int32, pl.Int64)
        ]

        exprs = []
        n_outliers = 0

        for c in numeric_cols:
            col = df[c].cast(pl.Float64)
            mean = col.mean()
            std = col.std()

            if std is None or std == 0.0:
                continue

            z = ((col - mean) / std).abs()
            mask = z > self.z_threshold
            count = int(mask.sum())
            n_outliers += count

            if count > 0:
                # Set outliers to null, then interpolate
                exprs.append(
                    pl.when(
                        ((pl.col(c).cast(pl.Float64) - mean) / std).abs()
                        > self.z_threshold
                    )
                    .then(None)
                    .otherwise(pl.col(c))
                    .alias(c)
                )

        if n_outliers > 0:
            logger.debug(
                "Found %d outlier values (z > %.1f); replacing with interpolation",
                n_outliers,
                self.z_threshold,
            )
            df = df.with_columns(exprs)
            df = df.with_columns(
                pl.all().interpolate().forward_fill().backward_fill()
            )
            residual_nulls = self._null_cells(df)
            if residual_nulls > 0:
                logger.warning(
                    "Residual null values after outlier interpolation (%d) "
                    "filled with 0.",
                    residual_nulls,
                )
                df = df.fill_null(0.0)

        return df

    def clean(self, df: pl.DataFrame) -> pl.DataFrame:
        """Run the full cleaning pipeline on a single sample.

        Sequentially applies :meth:`handle_missing` and
        :meth:`remove_anomalies`.

        Parameters
        ----------
        df : pl.DataFrame
            A ``(T, F)`` DataFrame (one simulation run).

        Returns
        -------
        pl.DataFrame
            Cleaned DataFrame.
        """
        df = self.handle_missing(df)
        df = self.remove_anomalies(df)
        return df

    def clean_batch(self, samples: List[pl.DataFrame]) -> List[pl.DataFrame]:
        """Apply :meth:`clean` to a list of sample DataFrames.

        Parameters
        ----------
        samples : list[pl.DataFrame]
            List of ``(T_i, F)`` DataFrames.

        Returns
        -------
        list[pl.DataFrame]
            Cleaned DataFrames (same order).
        """
        cleaned: List[pl.DataFrame] = []
        for i, df in enumerate(samples):
            logger.debug("Cleaning sample %d / %d", i + 1, len(samples))
            cleaned.append(self.clean(df))
        logger.info("Cleaned %d samples", len(cleaned))
        return cleaned

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    @staticmethod
    def _null_cells(df: pl.DataFrame) -> int:
        """Return the total number of null cells in a DataFrame."""
        return int(sum(df[c].null_count() for c in df.columns))
