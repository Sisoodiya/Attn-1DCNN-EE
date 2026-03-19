"""
Reliability-Centric XAI Visualizations
======================================

Utilities for plotting:
1. Sensor-time attribution heatmaps for reliability drops.
2. Side-by-side normal vs degraded attribution comparisons.
3. Dataset-level sensor-importance bar charts.
"""

from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

import numpy as np


def plot_reliability_heatmap(
    attribution_map: np.ndarray,
    feature_names: Optional[Sequence[str]] = None,
    title: str = "Reliability Attribution Heatmap",
    figsize: Tuple[int, int] = (14, 6),
    cmap: str = "YlOrRd",
    ax=None,
):
    """Plot one ``(I, F)`` attribution map as a 2D heatmap."""
    import matplotlib.pyplot as plt

    arr = np.asarray(attribution_map, dtype=np.float32)
    if arr.ndim != 2:
        raise ValueError(
            "attribution_map must be 2-D with shape (I, F), got "
            f"{arr.shape}"
        )

    # Display as (F, I) for readable feature axis.
    heatmap = arr.T

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    im = ax.imshow(heatmap, aspect="auto", cmap=cmap, interpolation="nearest")
    ax.set_xlabel("Time Step")
    ax.set_ylabel("Sensor / Feature")
    ax.set_title(title)

    if feature_names is not None and len(feature_names) <= 40:
        ax.set_yticks(range(len(feature_names)))
        ax.set_yticklabels(feature_names, fontsize=7)

    fig.colorbar(im, ax=ax, label="Attribution Magnitude", shrink=0.8)
    fig.tight_layout()
    return fig


def plot_reliability_contrast(
    normal_map: np.ndarray,
    degraded_map: np.ndarray,
    feature_names: Optional[Sequence[str]] = None,
    titles: Tuple[str, str] = ("Normal Interval", "Degrading Interval"),
    figsize: Tuple[int, int] = (16, 6),
    cmap: str = "YlOrRd",
):
    """Side-by-side heatmaps contrasting normal vs degraded intervals."""
    import matplotlib.pyplot as plt

    n = np.asarray(normal_map, dtype=np.float32)
    d = np.asarray(degraded_map, dtype=np.float32)
    if n.ndim != 2 or d.ndim != 2:
        raise ValueError(
            "normal_map and degraded_map must both be (I, F) arrays."
        )

    fig, axes = plt.subplots(1, 2, figsize=figsize, sharey=True)
    plot_reliability_heatmap(
        n,
        feature_names=feature_names,
        title=titles[0],
        cmap=cmap,
        ax=axes[0],
    )
    plot_reliability_heatmap(
        d,
        feature_names=feature_names,
        title=titles[1],
        cmap=cmap,
        ax=axes[1],
    )
    fig.tight_layout()
    return fig


def plot_sensor_importance(
    sensor_importance: np.ndarray,
    feature_names: Optional[Sequence[str]] = None,
    top_k: int = 15,
    title: str = "Average Sensor Importance (Reliability XAI)",
    figsize: Tuple[int, int] = (10, 6),
    ax=None,
):
    """Horizontal bar chart for mean sensor attribution importance."""
    import matplotlib.pyplot as plt

    imp = np.asarray(sensor_importance, dtype=np.float32)
    if imp.ndim != 1:
        raise ValueError(
            "sensor_importance must be 1-D, got "
            f"{imp.shape}"
        )

    top_k = max(1, min(top_k, imp.shape[0]))
    idx = np.argsort(imp)[-top_k:][::-1]
    vals = imp[idx]
    labels = (
        [str(feature_names[i]) for i in idx]
        if feature_names is not None
        else [f"Feature {i}" for i in idx]
    )

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    ax.barh(range(top_k), vals[::-1], color="#1f77b4")
    ax.set_yticks(range(top_k))
    ax.set_yticklabels(labels[::-1])
    ax.set_xlabel("Mean Attribution Magnitude")
    ax.set_title(title)
    fig.tight_layout()
    return fig

