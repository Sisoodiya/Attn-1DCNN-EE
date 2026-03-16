"""
Faithfulness Evaluator — XAI Layer 3: MDMC Perturbation Analysis
=================================================================

Validates whether the SHAP-identified critical sensors genuinely drive
the model's predictions by systematically masking them and measuring
the resulting prediction change.

Protocol
--------
1. Record the original model scores for each sample.
2. For each ``top_k_ratio``, identify the top-contributing features
   (by absolute SHAP value) per sample.
3. Replace those features with a baseline (column mean or zero).
4. Re-run the model on the perturbed inputs.
5. Compute MSE and MAE between original and perturbed scores.

A large error delta confirms that the masked features were indeed
critical — proving the explanation is **faithful**.
"""

from __future__ import annotations

from typing import Callable, Dict, Sequence, Tuple

import numpy as np


class FaithfulnessEvaluator:
    """MDMC-style perturbation faithfulness evaluator.

    Parameters
    ----------
    predict_fn : callable
        ``(N, D) np.ndarray → (N,) or (N, K) np.ndarray``.
        Same scoring function used for the SHAP explainer.
    """

    def __init__(
        self,
        predict_fn: Callable[[np.ndarray], np.ndarray],
    ) -> None:
        self.predict_fn = predict_fn

    def evaluate(
        self,
        X: np.ndarray,
        shap_values: np.ndarray,
        top_k_ratios: Sequence[float] = (0.05, 0.10, 0.20),
        baseline: str = "mean",
    ) -> Dict[str, Dict[str, float]]:
        """Run the perturbation faithfulness test.

        Parameters
        ----------
        X : np.ndarray
            Shape ``(N, D)`` — original feature vectors.
        shap_values : np.ndarray
            Shape ``(N, D)`` — SHAP values aligned with *X*.
            If a list (multi-output), the first element is used.
        top_k_ratios : sequence of float
            Fractions of features to mask (e.g. 0.05 = top 5 %).
        baseline : str
            Replacement strategy for masked features:
            ``"mean"`` (column mean of *X*) or ``"zero"``.

        Returns
        -------
        dict
            Keyed by ratio label (e.g. ``"top_5%"``), each value is
            ``{"k": int, "mse": float, "mae": float}``.
        """
        if isinstance(shap_values, list):
            shap_values = shap_values[0]

        n, d = X.shape
        orig_scores = self._flatten(self.predict_fn(X))

        fill = X.mean(axis=0) if baseline == "mean" else np.zeros(d)

        results: Dict[str, Dict[str, float]] = {}

        for ratio in top_k_ratios:
            k = max(1, int(d * ratio))
            X_masked = X.copy()

            for i in range(n):
                top_idx = np.argsort(np.abs(shap_values[i]))[-k:]
                X_masked[i, top_idx] = fill[top_idx]

            perturbed_scores = self._flatten(self.predict_fn(X_masked))

            diff = orig_scores - perturbed_scores
            mse = float(np.mean(diff ** 2))
            mae = float(np.mean(np.abs(diff)))

            label = f"top_{int(ratio * 100)}%"
            results[label] = {"k": k, "mse": mse, "mae": mae}

        return results

    @staticmethod
    def _flatten(scores: np.ndarray) -> np.ndarray:
        """Ensure scores are 1-D for comparison."""
        if scores.ndim == 2 and scores.shape[1] == 1:
            return scores.ravel()
        if scores.ndim == 2:
            return scores.mean(axis=1)
        return scores
