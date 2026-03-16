"""
Sliding Window — Step 4: Sliding Window Transformation (Sequentialization)
==========================================================================

Deep learning models for time-series analysis require continuous data to
be transformed into discrete segments.  Because nuclear faults dynamically
evolve over time, this module implements a **sliding window technique** to
capture crucial temporal dependencies.

Window Mechanism
----------------
A fixed-length window of *I* time steps is moved chronologically along the
standardised time series, advancing by *stride* records at a time.  Each
window produces a 2-D matrix of shape ``(I, F)`` where *F* is the number
of features (96 NPPAD sensor parameters after TIME is dropped).

Classes
-------
SlidingWindowTransformer
    Converts variable-length time-series samples into uniform ``(I, F)``
    sliding-window matrices.
"""

from __future__ import annotations

import logging
from typing import List, Tuple

import numpy as np
import polars as pl

logger = logging.getLogger(__name__)


class SlidingWindowTransformer:
    """Segment time-series samples into fixed-length sliding windows.

    Parameters
    ----------
    window_size : int, optional
        Number of consecutive time steps per window, *I* (default ``50``).
    stride : int, optional
        Step size between successive window starting positions (default ``1``).
        A stride of 1 maximises temporal overlap and sample count.

    Examples
    --------
    >>> swt = SlidingWindowTransformer(window_size=50, stride=1)
    >>> X, y = swt.transform_batch(samples, labels)
    >>> print(X.shape)   # (N, 50, 96)
    """

    def __init__(self, window_size: int = 50, stride: int = 1) -> None:
        if window_size < 1:
            raise ValueError(f"window_size must be >= 1, got {window_size}")
        if stride < 1:
            raise ValueError(f"stride must be >= 1, got {stride}")

        self.window_size = window_size
        self.stride = stride

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def transform(
        self,
        data: np.ndarray | pl.DataFrame,
        label: int,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Apply sliding window to a single time-series sample.

        Parameters
        ----------
        data : np.ndarray | pl.DataFrame
            2-D array of shape ``(T, F)`` — *T* time steps, *F* features.
        label : int
            Integer class label for this sample.

        Returns
        -------
        windows : np.ndarray
            Shape ``(num_windows, window_size, F)`` — the ``I x F`` matrices.
        labels : np.ndarray
            Shape ``(num_windows,)`` — repeated label for every window.

        Notes
        -----
        If the sample has fewer than :attr:`window_size` time steps, a
        warning is logged and empty arrays are returned.
        """
        if isinstance(data, pl.DataFrame):
            data = data.to_numpy().astype(np.float32)
        else:
            data = np.asarray(data, dtype=np.float32)

        T, F = data.shape

        if T < self.window_size:
            logger.warning(
                "Sample length (%d) < window_size (%d); skipping.",
                T,
                self.window_size,
            )
            return (
                np.empty((0, self.window_size, F), dtype=np.float32),
                np.empty((0,), dtype=np.int64),
            )

        # Compute window start indices.
        starts = np.arange(0, T - self.window_size + 1, self.stride)
        num_windows = len(starts)

        windows = np.empty(
            (num_windows, self.window_size, F), dtype=np.float32
        )
        for i, s in enumerate(starts):
            windows[i] = data[s : s + self.window_size]

        labels = np.full(num_windows, fill_value=label, dtype=np.int64)

        return windows, labels

    def transform_batch(
        self,
        samples: List[np.ndarray | pl.DataFrame],
        labels: List[int],
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Apply sliding window across a batch of samples and concatenate.

        Parameters
        ----------
        samples : list[np.ndarray | pl.DataFrame]
            Each element is a ``(T_i, F)`` time-series.
        labels : list[int]
            Corresponding integer label for each sample.

        Returns
        -------
        all_windows : np.ndarray
            Concatenated windows, shape ``(N, window_size, F)``.
        all_labels : np.ndarray
            Concatenated labels, shape ``(N,)``.
        """
        if len(samples) != len(labels):
            raise ValueError(
                f"samples ({len(samples)}) and labels ({len(labels)}) "
                "must have the same length."
            )

        window_chunks: List[np.ndarray] = []
        label_chunks: List[np.ndarray] = []

        for idx, (sample, lbl) in enumerate(zip(samples, labels)):
            w, l = self.transform(sample, lbl)
            if w.shape[0] > 0:
                window_chunks.append(w)
                label_chunks.append(l)
            logger.debug(
                "Sample %d/%d: %d windows generated",
                idx + 1,
                len(samples),
                w.shape[0],
            )

        if not window_chunks:
            F = samples[0].shape[1] if len(samples) > 0 else 0
            return (
                np.empty((0, self.window_size, F), dtype=np.float32),
                np.empty((0,), dtype=np.int64),
            )

        all_windows = np.concatenate(window_chunks, axis=0)
        all_labels = np.concatenate(label_chunks, axis=0)

        logger.info(
            "Sliding window complete: %d total windows of shape (%d, %d)",
            all_windows.shape[0],
            self.window_size,
            all_windows.shape[2],
        )
        return all_windows, all_labels
