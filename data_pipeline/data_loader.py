"""
Data Loader — Step 1: Data Ingestion and Parameter Selection
============================================================

Loads the raw NPPAD CSV files from ``Operation_csv_data/``.
Each subfolder represents an accident type (e.g., LOCA, ATWS, Normal).
Each CSV inside a subfolder corresponds to a severity level and contains
97 columns: TIME plus 96 sensor parameters.

Classes
-------
NPPADDataLoader
    Discovers, reads, and organises NPPAD CSV files into labelled DataFrames.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import polars as pl

logger = logging.getLogger(__name__)


class NPPADDataLoader:
    """Load NPPAD CSV data from the ``Operation_csv_data/`` directory tree.

    Parameters
    ----------
    data_dir : str | Path
        Root directory containing per-accident-type subfolders of CSV files.
        Typically ``data/Operation_csv_data``.
    time_col : str, optional
        Name of the time column in every CSV file (default ``"TIME"``).

    Examples
    --------
    >>> loader = NPPADDataLoader("data/Operation_csv_data")
    >>> samples, labels, mapping = loader.load_all()
    >>> print(f"Loaded {len(samples)} samples across {len(mapping)} classes")
    """

    # Default ordering of NPPAD accident types for reproducible label encoding.
    # Updated to match the 13 classes actually present in the Operation_csv_data directory
    DEFAULT_ACCIDENT_ORDER: List[str] = [
        "Normal",  # Even if missing, good to keep at idx 0 if it ever arrives
        "FLB", "LLB", "LOCA", "LOCAC", "LR",
        "MD", "RI", "RW", "SGATR", "SGBTR",
        "SLBIC", "SLBOC", "TT",
    ]

    def __init__(self, data_dir: str | Path, time_col: str = "TIME") -> None:
        self.data_dir = Path(data_dir)
        self.time_col = time_col
        self._label_map: Optional[Dict[str, int]] = None

        if not self.data_dir.is_dir():
            raise FileNotFoundError(
                f"Data directory not found: {self.data_dir}"
            )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def load_single_file(self, filepath: str | Path) -> pl.DataFrame:
        """Read a single CSV file and return its DataFrame.

        Parameters
        ----------
        filepath : str | Path
            Absolute or relative path to a ``.csv`` file.

        Returns
        -------
        pl.DataFrame
            Raw DataFrame with all 97 columns.
        """
        filepath = Path(filepath)
        logger.debug("Loading %s", filepath)
        df = pl.read_csv(filepath)
        return df

    def load_accident_type(
        self, accident_dir: str | Path
    ) -> List[pl.DataFrame]:
        """Load all severity-level CSVs for a single accident type.

        Parameters
        ----------
        accident_dir : str | Path
            Path to one accident-type subfolder (e.g., ``Operation_csv_data/LOCA``).

        Returns
        -------
        list[pl.DataFrame]
            One DataFrame per CSV, sorted by filename numerically.
        """
        accident_dir = Path(accident_dir)
        csv_files = sorted(
            accident_dir.glob("*.csv"),
            key=lambda p: int(p.stem) if p.stem.isdigit() else p.stem,
        )
        if not csv_files:
            logger.warning("No CSV files found in %s", accident_dir)
            return []

        frames: List[pl.DataFrame] = []
        for f in csv_files:
            frames.append(self.load_single_file(f))
        logger.info(
            "Loaded %d files from %s", len(frames), accident_dir.name
        )
        return frames

    def load_all(
        self,
        exclude_time: bool = True,
        accident_types: Optional[List[str]] = None,
    ) -> Tuple[List[pl.DataFrame], List[int], Dict[str, int]]:
        """Walk the data directory and load every accident-type subfolder.

        Parameters
        ----------
        exclude_time : bool, optional
            If ``True`` (default), the TIME column is dropped from the
            returned feature DataFrames.
        accident_types : list[str] | None, optional
            Subset of accident-type folder names to load.  ``None`` loads all.

        Returns
        -------
        samples : list[pl.DataFrame]
            Each element is a 2-D DataFrame for one CSV file (one simulation
            run). Shape ``(T, F)`` where *T* = timesteps, *F* = 96 features
            (or 97 if ``exclude_time=False``).
        labels : list[int]
            Integer class label for every sample, aligned with *samples*.
        label_map : dict[str, int]
            Mapping from accident-type folder name to integer label.
        """
        label_map = self._build_label_map(accident_types)
        self._label_map = label_map

        samples: List[pl.DataFrame] = []
        labels: List[int] = []

        for accident_name, label_id in label_map.items():
            accident_dir = self.data_dir / accident_name
            if not accident_dir.is_dir():
                logger.warning("Skipping missing folder: %s", accident_dir)
                continue

            frames = self.load_accident_type(accident_dir)
            for df in frames:
                if exclude_time and self.time_col in df.columns:
                    df = df.drop(self.time_col)
                samples.append(df)
                labels.append(label_id)

        # Enforce a consistent column set across all samples.
        # Some CSVs may have extra columns; keep only the intersection.
        if samples:
            common_cols = set(samples[0].columns)
            for df in samples[1:]:
                common_cols &= set(df.columns)
            common_cols_sorted = [
                c for c in samples[0].columns if c in common_cols
            ]
            if any(len(df.columns) != len(common_cols_sorted) for df in samples):
                logger.warning(
                    "Column count varies across CSVs; "
                    "keeping %d common columns out of max %d",
                    len(common_cols_sorted),
                    max(len(df.columns) for df in samples),
                )
                samples = [df.select(common_cols_sorted) for df in samples]

        logger.info(
            "Total: %d samples across %d classes", len(samples), len(label_map)
        )
        return samples, labels, label_map

    def get_label_mapping(self) -> Dict[str, int]:
        """Return the most recently computed label mapping.

        Returns
        -------
        dict[str, int]
            Accident-type name → integer label.

        Raises
        ------
        RuntimeError
            If :meth:`load_all` has not been called yet.
        """
        if self._label_map is None:
            raise RuntimeError(
                "Label mapping is not available. Call load_all() first."
            )
        return self._label_map

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_label_map(
        self, accident_types: Optional[List[str]] = None
    ) -> Dict[str, int]:
        """Create a deterministic folder-name → integer-label mapping."""
        if accident_types is not None:
            ordered = accident_types
        else:
            # Use the canonical order; any new folders not in the default list
            # are appended alphabetically.
            existing = {
                d.name for d in self.data_dir.iterdir() if d.is_dir()
            }
            ordered = [
                name
                for name in self.DEFAULT_ACCIDENT_ORDER
                if name in existing
            ]
            extras = sorted(existing - set(ordered))
            ordered.extend(extras)

        return {name: idx for idx, name in enumerate(ordered)}
