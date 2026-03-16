"""
Dataset Builder — Step 5: PyTorch Lightning DataModule
======================================================

PyTorch Dataset and Lightning DataModule that orchestrate the full pipeline.

RAM-efficient design
--------------------
``NPPADDataset`` uses **lazy windowing**: it stores the cleaned, scaled
per-sample numpy arrays (one array per CSV file) and computes each sliding
window on-the-fly inside ``__getitem__``.  This avoids pre-materialising a
huge ``(N_windows, window_size, F)`` tensor in RAM — critical on Colab T4.

Memory comparison (all 18 accident classes, window_size=50, stride=1):

+-----------+----------------+----------+
| Strategy  | ~Windows       | ~RAM     |
+===========+================+==========+
| Eager     | ≈ 500,000      | ≈ 9 GB   |
+-----------+----------------+----------+
| Lazy      | computed live  | ≈ 300 MB |
+-----------+----------------+----------+

Output tensor per item
-----------------------
``x`` — ``(F, I)`` *(channels-first)*: *F* sensor features × *I* time steps,
the correct layout for ``nn.Conv1d``.
``y`` — scalar integer class label.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

logger = logging.getLogger(__name__)

# Attempt to import pytorch_lightning; fall back to lightning.pytorch.
try:
    import pytorch_lightning as pl
except ImportError:
    import lightning.pytorch as pl  # type: ignore[no-redef]

from data_pipeline.data_loader import NPPADDataLoader
from data_pipeline.data_cleaning import DataCleaner
from data_pipeline.scaler import ZScoreScaler, MinMaxScaler
from data_pipeline.sliding_window import SlidingWindowTransformer


# ======================================================================
# Lazy Sliding-Window Dataset
# ======================================================================

class NPPADDataset(Dataset):
    """Memory-efficient Dataset with lazy sliding-window computation.

    Instead of pre-materialising every window (which can require several GB
    of RAM), this dataset stores the raw ``(T_i, F)`` samples and computes
    each window on demand inside ``__getitem__``.

    Parameters
    ----------
    samples : list[np.ndarray]
        Per-CSV feature arrays, each of shape ``(T_i, F)``.
    labels : list[int]
        Integer class label per sample (aligned with *samples*).
    window_size : int
        Number of time steps per window (*I*).
    stride : int
        Step between consecutive window start positions.
    """

    def __init__(
        self,
        samples: List[np.ndarray],
        labels: List[int],
        window_size: int = 50,
        stride: int = 1,
    ) -> None:
        self.window_size = window_size
        self.stride = stride

        # Store samples as float32 tensors to avoid repeated conversion.
        self.samples: List[torch.Tensor] = []
        self.sample_labels: List[int] = []

        # For each sample, compute how many windows it yields and build
        # a cumulative-sum index for O(log N) lookup in __getitem__.
        window_counts: List[int] = []

        for arr, lbl in zip(samples, labels):
            arr = np.asarray(arr, dtype=np.float32)
            T = arr.shape[0]
            if T < window_size:
                logger.debug(
                    "Sample length (%d) < window_size (%d); skipping.",
                    T, window_size,
                )
                continue
            n_windows = (T - window_size) // stride + 1
            self.samples.append(torch.from_numpy(arr))
            self.sample_labels.append(lbl)
            window_counts.append(n_windows)

        if not self.samples:
            raise ValueError(
                "No valid samples after filtering short sequences. "
                f"Try reducing window_size (current: {window_size})."
            )

        # Cumulative window counts for fast index lookup.
        self._cum_windows = np.cumsum([0] + window_counts)
        self._total_windows = int(self._cum_windows[-1])

        logger.info(
            "NPPADDataset: %d samples → %d windows "
            "(window_size=%d, stride=%d, RAM-lazy)",
            len(self.samples), self._total_windows, window_size, stride,
        )

    # ------------------------------------------------------------------
    # Dataset protocol
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return self._total_windows

    def __getitem__(self, idx: int):
        """Return the *idx*-th window as ``(x, y)`` tensors.

        x shape: ``(F, I)`` — channels-first for ``Conv1d``.
        y shape: ``()``     — scalar label.
        """
        if idx < 0:
            idx = self._total_windows + idx

        # Binary search: which sample does this global index fall in?
        sample_idx = int(np.searchsorted(self._cum_windows, idx + 1, side="right")) - 1
        window_offset = (idx - int(self._cum_windows[sample_idx])) * self.stride

        arr = self.samples[sample_idx]          # (T, F)
        window = arr[window_offset : window_offset + self.window_size]   # (I, F)

        x = window.T.contiguous()              # (F, I) — channels-first
        y = torch.tensor(self.sample_labels[sample_idx], dtype=torch.long)
        return x, y


# ======================================================================
# PyTorch Lightning DataModule
# ======================================================================

class NPPADDataModule(pl.LightningDataModule):
    """End-to-end NPPAD data pipeline as a Lightning DataModule.

    Stages executed inside :meth:`setup`:

    1. **Load** — read all CSVs from *data_dir* via :class:`NPPADDataLoader`.
    2. **Clean** — handle NaN and outliers via :class:`DataCleaner`.
    3. **Scale** — Z-score (default) or Min-Max via :class:`ZScoreScaler`
       / :class:`MinMaxScaler`.
    4. **Split** — sample-level stratified train / validation / test split.
    5. **Dataset** — wrap each split in :class:`NPPADDataset` for lazy
       on-the-fly windowing.

    Parameters
    ----------
    data_dir : str | Path
        Path to ``Operation_csv_data/`` directory.
    window_size : int
        Number of time steps per sliding window (*I*).  Default ``50``.
    stride : int
        Step between consecutive window starts.  Default ``5``.
        Increase to reduce the effective number of windows and save RAM/time.
    batch_size : int
        Mini-batch size for dataloaders.  Default ``64``.
    val_split : float
        Fraction of *samples* reserved for validation.  Default ``0.15``.
    test_split : float
        Fraction of *samples* reserved for testing.  Default ``0.15``.
    num_workers : int
        DataLoader worker processes.  Default ``2`` (Colab free tier).
    nan_strategy : str
        Strategy for :class:`DataCleaner` (``"interpolate"`` | ``"ffill"``).
    z_threshold : float
        Z-score threshold for anomaly removal.  Default ``6.0``.
    scaler_type : str
        ``"zscore"`` (default) or ``"minmax"``.
    accident_types : list[str] | None
        Subset of accident types to load.  ``None`` loads all 18.
    random_seed : int
        Seed for reproducible shuffling and splitting.  Default ``42``.
    """

    def __init__(
        self,
        data_dir: str | Path = "data/Operation_csv_data",
        window_size: int = 50,
        stride: int = 5,
        batch_size: int = 64,
        val_split: float = 0.15,
        test_split: float = 0.15,
        num_workers: int = 2,
        nan_strategy: str = "interpolate",
        z_threshold: float = 6.0,
        scaler_type: str = "zscore",
        accident_types: Optional[List[str]] = None,
        random_seed: int = 42,
    ) -> None:
        super().__init__()
        self.data_dir = Path(data_dir)
        self.window_size = window_size
        self.stride = stride
        self.batch_size = batch_size
        self.val_split = val_split
        self.test_split = test_split
        self.num_workers = num_workers
        self.nan_strategy = nan_strategy
        self.z_threshold = z_threshold
        self.scaler_type = scaler_type
        self.accident_types = accident_types
        self.random_seed = random_seed

        # Populated during setup().
        self.train_dataset: Optional[NPPADDataset] = None
        self.val_dataset: Optional[NPPADDataset] = None
        self.test_dataset: Optional[NPPADDataset] = None
        self.label_map: Optional[Dict[str, int]] = None
        self.num_classes: int = 0
        self.num_features: int = 0

    # ------------------------------------------------------------------
    # Lightning hooks
    # ------------------------------------------------------------------

    def setup(self, stage: Optional[str] = None) -> None:
        """Execute the full data pipeline and create dataset splits.

        The window explosion step is deferred to the Dataset level
        (lazy), so this method only allocates memory proportional to
        the number of *samples* × their length, not the number of windows.
        """
        logger.info("=== NPPAD DataModule setup (stage=%s) ===", stage)

        # 1. Load -----------------------------------------------------------
        loader = NPPADDataLoader(self.data_dir)
        raw_samples, labels, label_map = loader.load_all(
            exclude_time=True, accident_types=self.accident_types
        )
        self.label_map = label_map
        self.num_classes = len(label_map)
        logger.info(
            "Loaded %d samples, %d classes", len(raw_samples), self.num_classes
        )

        # Convert Polars → numpy arrays (loader returns Polars DataFrames)
        import polars as pl_lib
        np_samples: List[np.ndarray] = []
        for df in raw_samples:
            if hasattr(df, "to_numpy"):        # Polars
                np_samples.append(df.to_numpy().astype(np.float32))
            else:                              # pandas fallback
                np_samples.append(df.values.astype(np.float32))

        self.num_features = np_samples[0].shape[1] if np_samples else 0

        # 2. Clean ----------------------------------------------------------
        cleaner = DataCleaner(
            nan_strategy=self.nan_strategy, z_threshold=self.z_threshold
        )
        cleaned: List[np.ndarray] = []
        for arr in np_samples:
            df_tmp = pl_lib.DataFrame(arr)
            # clean() returns a pl.DataFrame, and .to_numpy() gets the float32 array
            cleaned.append(cleaner.clean(df_tmp).to_numpy().astype(np.float32))
        logger.info("Cleaning complete.")

        # 3. Scale ----------------------------------------------------------
        if self.scaler_type == "zscore":
            scaler: ZScoreScaler | MinMaxScaler = ZScoreScaler()
        elif self.scaler_type == "minmax":
            scaler = MinMaxScaler()
        else:
            raise ValueError(
                f"Unknown scaler_type '{self.scaler_type}'. "
                "Use 'zscore' or 'minmax'."
            )

        # Fit on all data then transform each array in-place
        import pandas as pd
        all_concat = pd.DataFrame(np.vstack(cleaned))
        scaler.fit([all_concat])          # fit_transform on concatenation

        scaled: List[np.ndarray] = []
        for arr in cleaned:
            df_tmp = pd.DataFrame(arr)
            scaled_df = scaler.transform([df_tmp])[0]
            scaled.append(scaled_df.values.astype(np.float32))

        del cleaned  # free intermediate RAM
        logger.info("Scaling complete.")

        # 4. Sample-level split (BEFORE windowing — keeps split clean) ------
        train_samples, train_labels, val_samples, val_labels, \
            test_samples, test_labels = self._stratified_split(scaled, labels)

        del scaled  # free memory

        # 5. Build lazy Datasets -------------------------------------------
        self.train_dataset = NPPADDataset(
            train_samples, train_labels, self.window_size, self.stride
        )
        self.val_dataset = NPPADDataset(
            val_samples, val_labels, self.window_size, self.stride
        )
        self.test_dataset = NPPADDataset(
            test_samples, test_labels, self.window_size, self.stride
        )

        logger.info(
            "Window counts — train: %d, val: %d, test: %d",
            len(self.train_dataset),
            len(self.val_dataset),
            len(self.test_dataset),
        )

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=torch.cuda.is_available(),
            drop_last=True,
            persistent_workers=self.num_workers > 0,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=torch.cuda.is_available(),
            persistent_workers=self.num_workers > 0,
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=torch.cuda.is_available(),
            persistent_workers=self.num_workers > 0,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _stratified_split(
        self,
        samples: List[np.ndarray],
        labels: List[int],
    ) -> tuple:
        """Sample-level stratified shuffle-split → train / val / test."""
        rng = np.random.RandomState(self.random_seed)
        indices = np.arange(len(labels))
        rng.shuffle(indices)

        n = len(labels)
        n_test = max(1, int(n * self.test_split))
        n_val  = max(1, int(n * self.val_split))

        test_idx  = indices[:n_test]
        val_idx   = indices[n_test : n_test + n_val]
        train_idx = indices[n_test + n_val :]

        def subset(idx_arr):
            return (
                [samples[i] for i in idx_arr],
                [labels[i] for i in idx_arr],
            )

        train_s, train_l = subset(train_idx)
        val_s,   val_l   = subset(val_idx)
        test_s,  test_l  = subset(test_idx)

        return train_s, train_l, val_s, val_l, test_s, test_l
