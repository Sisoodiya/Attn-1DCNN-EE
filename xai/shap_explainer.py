"""
SHAP Explainer — XAI Layer 2: Post-Hoc Mathematical Attribution
================================================================

Wraps the trained Attn-1DCNN-EE prediction pipeline with SHAP
(SHapley Additive exPlanations) to compute the exact marginal
contribution of each sensor feature to the diagnostic decision.

Features are categorised as:

* **Contributors** — positive SHAP values that push the prediction
  *toward* a specific fault class.
* **Offsets** — negative SHAP values that pull the prediction *toward*
  a normal / alternative state.

Uses ``shap.KernelExplainer`` (model-agnostic) so the same code works
regardless of whether the prediction target is the EE Mahalanobis
distance, the class label, or a custom scoring function.
"""

from __future__ import annotations

from typing import Callable, Dict, List, Optional, Tuple

import numpy as np


class SHAPExplainer:
    """Model-agnostic SHAP wrapper for the Attn-1DCNN-EE pipeline.

    Parameters
    ----------
    predict_fn : callable
        ``(N, D) np.ndarray → (N,) or (N, K) np.ndarray``.
        A function that maps pooled feature vectors to continuous
        scores (e.g. negative Mahalanobis distances, class
        probabilities, or decision-function values).
    feature_names : list[str] | None
        Human-readable names for the *D* input dimensions.
    """

    def __init__(
        self,
        predict_fn: Callable[[np.ndarray], np.ndarray],
        feature_names: Optional[List[str]] = None,
    ) -> None:
        self.predict_fn = predict_fn
        self.feature_names = feature_names

    # ------------------------------------------------------------------
    # Core explanation
    # ------------------------------------------------------------------

    def explain(
        self,
        X: np.ndarray,
        background: np.ndarray,
        n_samples: int | str = "auto",
    ) -> np.ndarray:
        """Compute SHAP values for *X* against a background dataset.

        Parameters
        ----------
        X : np.ndarray
            Shape ``(N, D)`` — samples to explain.
        background : np.ndarray
            Shape ``(M, D)`` — reference distribution (often a
            k-means summary of the training set, ``M ≈ 50–100``).
        n_samples : int | str
            Number of coalition samples for the Kernel estimator.

        Returns
        -------
        np.ndarray
            SHAP values with the same shape as *X* (single-output) or
            a list of arrays (multi-output).
        """
        import shap

        explainer = shap.KernelExplainer(self.predict_fn, background)
        shap_values = explainer.shap_values(X, nsamples=n_samples)
        return shap_values

    # ------------------------------------------------------------------
    # Contributor / offset decomposition
    # ------------------------------------------------------------------

    def contributors_and_offsets(
        self,
        shap_values: np.ndarray,
        top_k: Optional[int] = None,
    ) -> Dict[str, List[Tuple[str, float]]]:
        """Split features into contributors and offsets.

        Parameters
        ----------
        shap_values : np.ndarray
            Shape ``(D,)`` for a single sample, or ``(N, D)`` (averaged
            across samples automatically).
        top_k : int | None
            If set, only return the top-*k* contributors and offsets.

        Returns
        -------
        dict
            ``{"contributors": [(name, value), ...],
              "offsets": [(name, value), ...]}``
            sorted by descending absolute magnitude.
        """
        if isinstance(shap_values, list):
            shap_values = shap_values[0]

        vals = (
            np.mean(shap_values, axis=0)
            if shap_values.ndim == 2
            else shap_values
        )

        names = self.feature_names or [
            f"Feature {i}" for i in range(len(vals))
        ]

        pos = [
            (names[i], float(vals[i]))
            for i in np.where(vals > 0)[0]
        ]
        neg = [
            (names[i], float(vals[i]))
            for i in np.where(vals < 0)[0]
        ]

        # Sort by absolute magnitude descending
        pos.sort(key=lambda x: abs(x[1]), reverse=True)
        neg.sort(key=lambda x: abs(x[1]), reverse=True)

        if top_k is not None:
            pos = pos[:top_k]
            neg = neg[:top_k]

        return {"contributors": pos, "offsets": neg}

    # ------------------------------------------------------------------
    # Plotting helpers
    # ------------------------------------------------------------------

    def plot_summary(
        self,
        shap_values: np.ndarray,
        X: np.ndarray,
        **kwargs,
    ):
        """Render a SHAP beeswarm summary plot.

        Parameters
        ----------
        shap_values : np.ndarray
            SHAP values from :meth:`explain`.
        X : np.ndarray
            Original feature matrix aligned with *shap_values*.
        **kwargs
            Forwarded to ``shap.summary_plot``.
        """
        import shap

        shap.summary_plot(
            shap_values,
            X,
            feature_names=self.feature_names,
            **kwargs,
        )

    def plot_force(
        self,
        base_value: float,
        shap_values: np.ndarray,
        sample: np.ndarray,
        **kwargs,
    ):
        """Render a SHAP force plot for a single sample.

        Parameters
        ----------
        base_value : float
            Expected model output (from ``explainer.expected_value``).
        shap_values : np.ndarray
            Shape ``(D,)`` — SHAP values for one sample.
        sample : np.ndarray
            Shape ``(D,)`` — the raw feature values.
        **kwargs
            Forwarded to ``shap.force_plot``.
        """
        import shap

        shap.force_plot(
            base_value,
            shap_values,
            sample,
            feature_names=self.feature_names,
            **kwargs,
        )
