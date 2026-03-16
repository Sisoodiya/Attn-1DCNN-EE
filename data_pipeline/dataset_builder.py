"""
Dataset Builder — Step 5: PyTorch Lightning DataModule
======================================================

Orchestrates the full NPPAD data pipeline and exposes the result as a
PyTorch Lightning ``LightningDataModule`` with train / validation / test
splits.

Output tensors are shaped for a **1D-CNN** model:

* ``x`` — ``(F, I)``  *(channels-first)*: *F* features treated as input
  channels, *I* time steps as the spatial dimension for 1-D convolution.
* ``y`` — scalar integer class label.

Classes
-------
NPPADDataset
    ``torch.utils.data.Dataset`` wrapper around windowed numpy arrays.
NPPADDataModule
    Full-pipeline ``LightningDataModule``.
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
# PyTorch Dataset
# ======================================================================

class NPPADDataset(Dataset):
    """Thin ``Dataset`` wrapper around windowed NPPAD arrays.

    Parameters
    ----------
    features : np.ndarray
        Shape ``(N, window_size, num_features)`` — sliding-window matrices.
    labels : np.ndarray
        Shape ``(N,)`` — integer class labels.

    Notes
    -----
    ``__getitem__`` returns tensors with the feature axis first
    (``(F, I)``), which is the standard layout for ``torch.nn.Conv1d``.
    """

    def __init__(self, features: np.ndarray, labels: np.ndarray) -> None:
        assert features.shape[0] == labels.shape[0], (
            f"Feature/label count mismatch: {features.shape[0]} vs {labels.shape[0]}"
        )
        # Store as tensors immediately to avoid repeated conversions.
        self.features = torch.from_numpy(features).float()
        self.labels = torch.from_numpy(labels).long()

    def __len__(self) -> int:
        return self.features.shape[0]

    def __getitem__(self, idx: int):
        # Transpose (window_size, num_features) → (num_features, window_size)
        # so Conv1d treats each sensor as a channel.
        x = self.features[idx].T          # (F, I)
        y = self.labels[idx]              # scalar
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
    4. **Window** — sliding-window segmentation via
       :class:`SlidingWindowTransformer`.
    5. **Split** — stratified train / validation / test partitioning.

    Parameters
    ----------
    data_dir : str | Path
        Path to ``Operation_csv_data/`` directory.
    window_size : int
        Number of time steps per sliding window (*I*).
    stride : int
        Step between consecutive window starts.
    batch_size : int
        Mini-batch size for dataloaders.
    val_split : float
        Fraction of data reserved for validation.
    test_split : float
        Fraction of data reserved for testing.
    num_workers : int
        DataLoader worker processes.
    nan_strategy : str
        Strategy for :class:`DataCleaner` (``"interpolate"`` | ``"ffill"``).
    z_threshold : float
        Z-score threshold for anomaly removal.
    scaler_type : str
        ``"zscore"`` (default, primary) or ``"minmax"`` (secondary).
    accident_types : list[str] | None
        Subset of accident types to load.  ``None`` loads all 18.
    random_seed : int
        Seed for reproducible shuffling and splitting.
    """

    def __init__(
        self,
        data_dir: str | Path = "data/Operation_csv_data",
        window_size: int = 50,
        stride: int = 1,
        batch_size: int = 64,
        val_split: float = 0.15,
        test_split: float = 0.15,
        num_workers: int = 4,
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
        """Execute the full data pipeline and create dataset splits."""
        logger.info("=== NPPAD DataModule setup (stage=%s) ===", stage)

        # 1. Load -----------------------------------------------------------
        loader = NPPADDataLoader(self.data_dir)
        samples, labels, label_map = loader.load_all(
            exclude_time=True, accident_types=self.accident_types
        )
        self.label_map = label_map
        self.num_classes = len(label_map)
        logger.info(
            "Loaded %d samples, %d classes", len(samples), self.num_classes
        )

        # 2. Clean ----------------------------------------------------------
        cleaner = DataCleaner(
            nan_strategy=self.nan_strategy, z_threshold=self.z_threshold
        )
        samples = cleaner.clean_batch(samples)

        # 3. Scale ----------------------------------------------------------
        if self.scaler_type == "zscore":
            scaler = ZScoreScaler()
        elif self.scaler_type == "minmax":
            scaler = MinMaxScaler()
        else:
            raise ValueError(
                f"Unknown scaler_type '{self.scaler_type}'. "
                "Use 'zscore' or 'minmax'."
            )
        samples = scaler.fit_transform(samples)
        self.num_features = samples[0].width if samples else 0

        # 4. Sliding window -------------------------------------------------
        windower = SlidingWindowTransformer(
            window_size=self.window_size, stride=self.stride
        )
        X, y = windower.transform_batch(samples, labels)
        logger.info(
            "Windowed data: X %s, y %s", X.shape, y.shape
        )

        # 5. Train / val / test split (stratified shuffle) ------------------
        self.train_dataset, self.val_dataset, self.test_dataset = (
            self._stratified_split(X, y)
        )

        logger.info(
            "Split sizes — train: %d, val: %d, test: %d",
            len(self.train_dataset),
            len(self.val_dataset),
            len(self.test_dataset),
        )

    def train_dataloader(self) -> DataLoader:
        """Return the training DataLoader."""
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=True,
        )

    def val_dataloader(self) -> DataLoader:
        """Return the validation DataLoader."""
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def test_dataloader(self) -> DataLoader:
        """Return the test DataLoader."""
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _stratified_split(
        self,
        X: np.ndarray,
        y: np.ndarray,
    ) -> tuple:
        """Perform a stratified shuffle-split into train/val/test."""
        rng = np.random.RandomState(self.random_seed)
        indices = np.arange(len(y))
        rng.shuffle(indices)

        n_total = len(y)
        n_test = int(n_total * self.test_split)
        n_val = int(n_total * self.val_split)

        test_idx = indices[:n_test]
        val_idx = indices[n_test : n_test + n_val]
        train_idx = indices[n_test + n_val :]

        return (
            NPPADDataset(X[train_idx], y[train_idx]),
            NPPADDataset(X[val_idx], y[val_idx]),
            NPPADDataset(X[test_idx], y[test_idx]),
        )
