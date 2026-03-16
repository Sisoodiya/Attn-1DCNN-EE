"""
Attention Visualisation — XAI Layer 1: Inherent Interpretability
=================================================================

Extracts the 2-D attention weight grid ``(I, C)`` produced by the Soft
Attention mechanism (Component 3) and renders it as a heatmap.  Each
cell shows how much focus the model placed on a given feature channel at
a specific time step, directly revealing *where* and *when* the network
detected anomalous sensor behaviour.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np


def plot_attention_heatmap(
    attention_weights: np.ndarray,
    feature_names: Optional[List[str]] = None,
    title: str = "Attention Heatmap",
    figsize: Tuple[int, int] = (14, 6),
    cmap: str = "YlOrRd",
    top_k: Optional[int] = None,
    ax=None,
):
    """Render a single sample's attention weights as a 2-D heatmap.

    Parameters
    ----------
    attention_weights : np.ndarray
        Shape ``(I, C)`` for one sample, or ``(B, I, C)`` (first sample
        is used).  *I* = time steps, *C* = feature channels.
    feature_names : list[str] | None
        Labels for the feature-channel axis.  Defaults to integer
        indices.
    title : str
        Plot title.
    figsize : tuple[int, int]
        Figure size (width, height) in inches.
    cmap : str
        Matplotlib colour-map name.
    top_k : int | None
        If set, only display the *top_k* channels with the highest
        mean attention (reduces clutter for large channel counts).
    ax : matplotlib.axes.Axes | None
        If provided, draw on this axes instead of creating a new figure.

    Returns
    -------
    matplotlib.figure.Figure
        The rendered figure (or the parent figure of *ax*).
    """
    import matplotlib.pyplot as plt

    weights = np.asarray(attention_weights, dtype=np.float32)
    if weights.ndim == 3:
        weights = weights[0]  # take first sample from batch

    # weights: (I, C) → transpose to (C, I) for display
    heatmap = weights.T  # (C, I)

    if top_k is not None and top_k < heatmap.shape[0]:
        mean_attn = heatmap.mean(axis=1)
        top_idx = np.argsort(mean_attn)[-top_k:][::-1]
        heatmap = heatmap[top_idx]
        if feature_names is not None:
            feature_names = [feature_names[i] for i in top_idx]

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    im = ax.imshow(heatmap, aspect="auto", cmap=cmap, interpolation="nearest")
    ax.set_xlabel("Time Step")
    ax.set_ylabel("Feature Channel")
    ax.set_title(title)

    if feature_names is not None and len(feature_names) <= 40:
        ax.set_yticks(range(len(feature_names)))
        ax.set_yticklabels(feature_names, fontsize=7)

    fig.colorbar(im, ax=ax, label="Attention Weight", shrink=0.8)
    fig.tight_layout()
    return fig


def plot_attention_top_channels(
    attention_weights: np.ndarray,
    top_k: int = 10,
    feature_names: Optional[List[str]] = None,
    title: str = "Top Attended Channels (mean over time)",
    figsize: Tuple[int, int] = (10, 5),
    ax=None,
):
    """Bar chart of the *top_k* channels ranked by mean attention.

    Parameters
    ----------
    attention_weights : np.ndarray
        Shape ``(I, C)`` or ``(B, I, C)``.
    top_k : int
        Number of channels to display.
    feature_names : list[str] | None
        Channel labels.
    title : str
        Plot title.
    figsize : tuple[int, int]
        Figure size.
    ax : matplotlib.axes.Axes | None
        Optional pre-existing axes.

    Returns
    -------
    matplotlib.figure.Figure
    """
    import matplotlib.pyplot as plt

    weights = np.asarray(attention_weights, dtype=np.float32)
    if weights.ndim == 3:
        weights = weights[0]

    # weights: (I, C) → mean over time → (C,)
    mean_attn = weights.mean(axis=0)
    top_idx = np.argsort(mean_attn)[-top_k:][::-1]
    top_vals = mean_attn[top_idx]

    labels = (
        [feature_names[i] for i in top_idx]
        if feature_names
        else [f"Ch {i}" for i in top_idx]
    )

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    bars = ax.barh(range(len(top_vals)), top_vals[::-1], color="#e74c3c")
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels[::-1])
    ax.set_xlabel("Mean Attention Weight")
    ax.set_title(title)
    fig.tight_layout()
    return fig
