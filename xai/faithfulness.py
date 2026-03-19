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

from itertools import combinations
from typing import Any, Callable, Dict, List, Sequence

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
        """Backward-compatible top-k perturbation faithfulness test.

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
        full = self.evaluate_top_bottom(
            X=X,
            shap_values=shap_values,
            top_k_ratios=top_k_ratios,
            baseline=baseline,
        )
        return {k: v for k, v in full.items() if k.startswith("top_")}

    def evaluate_top_bottom(
        self,
        X: np.ndarray,
        shap_values: np.ndarray,
        top_k_ratios: Sequence[float] = (0.05, 0.10, 0.20),
        baseline: str = "mean",
    ) -> Dict[str, Dict[str, float]]:
        """Bidirectional perturbation test (top-k vs bottom-k features).

        A faithful explanation should produce a larger score disruption
        when top-k important features are perturbed compared with
        perturbing bottom-k features.
        """
        if isinstance(shap_values, list):
            shap_values = shap_values[0]

        n, d = X.shape
        orig_scores = self._flatten(self.predict_fn(X))

        fill = X.mean(axis=0) if baseline == "mean" else np.zeros(d)

        results: Dict[str, Dict[str, float]] = {}

        for ratio in top_k_ratios:
            k = max(1, int(d * ratio))
            ratio_label = int(ratio * 100)

            X_top = X.copy()
            X_bottom = X.copy()

            for i in range(n):
                order = np.argsort(np.abs(shap_values[i]))
                bottom_idx = order[:k]
                top_idx = order[-k:]
                X_top[i, top_idx] = fill[top_idx]
                X_bottom[i, bottom_idx] = fill[bottom_idx]

            top_scores = self._flatten(self.predict_fn(X_top))
            bottom_scores = self._flatten(self.predict_fn(X_bottom))

            top_diff = orig_scores - top_scores
            bottom_diff = orig_scores - bottom_scores

            results[f"top_{ratio_label}%"] = {
                "k": k,
                "mse": float(np.mean(top_diff ** 2)),
                "mae": float(np.mean(np.abs(top_diff))),
                "mean_delta": float(np.mean(np.abs(top_diff))),
            }
            results[f"bottom_{ratio_label}%"] = {
                "k": k,
                "mse": float(np.mean(bottom_diff ** 2)),
                "mae": float(np.mean(np.abs(bottom_diff))),
                "mean_delta": float(np.mean(np.abs(bottom_diff))),
            }

        return results

    def attribution_stability(
        self,
        X: np.ndarray,
        attribution_fn: Callable[[np.ndarray], np.ndarray],
        noise_std: float = 0.01,
        n_trials: int = 5,
        random_seed: int = 42,
    ) -> Dict[str, float]:
        """Evaluate attribution-map stability under small input noise.

        Returns mean/std/min cosine similarity between base attribution
        maps and noisy-input attribution maps.
        """
        if n_trials < 1:
            raise ValueError(f"n_trials must be >= 1, got {n_trials}")

        rng = np.random.RandomState(random_seed)
        base = self._as_2d(attribution_fn(X))
        sims: List[float] = []

        for _ in range(n_trials):
            noisy = X + rng.normal(0.0, noise_std, size=X.shape).astype(X.dtype)
            attr_noisy = self._as_2d(attribution_fn(noisy))
            sims.append(self._mean_cosine_similarity(base, attr_noisy))

        arr = np.asarray(sims, dtype=np.float64)
        return {
            "mean_cosine_similarity": float(arr.mean()),
            "std_cosine_similarity": float(arr.std()),
            "min_cosine_similarity": float(arr.min()),
        }

    def cross_fold_consistency(
        self,
        fold_sensor_importances: Sequence[np.ndarray],
        top_k: int = 10,
    ) -> Dict[str, Any]:
        """Measure top-sensor consistency across CV folds (Jaccard)."""
        if len(fold_sensor_importances) < 2:
            raise ValueError("Need at least 2 folds for consistency analysis.")

        top_sets: List[set[int]] = []
        for imp in fold_sensor_importances:
            arr = np.asarray(imp, dtype=np.float64)
            if arr.ndim != 1:
                raise ValueError(
                    "Each fold importance vector must be 1-D, got "
                    f"{arr.shape}"
                )
            k = min(top_k, arr.shape[0])
            idx = np.argsort(arr)[-k:]
            top_sets.append(set(int(i) for i in idx))

        pairwise: List[float] = []
        for a, b in combinations(top_sets, 2):
            inter = len(a & b)
            union = len(a | b)
            score = inter / union if union > 0 else 0.0
            pairwise.append(score)

        common = set.intersection(*top_sets)
        return {
            "top_k": int(top_k),
            "pairwise_jaccard_mean": float(np.mean(pairwise)),
            "pairwise_jaccard_std": float(np.std(pairwise)),
            "common_sensor_count": int(len(common)),
            "common_sensor_indices": sorted(int(i) for i in common),
        }

    @staticmethod
    def _flatten(scores: np.ndarray) -> np.ndarray:
        """Ensure scores are 1-D for comparison."""
        if scores.ndim == 2 and scores.shape[1] == 1:
            return scores.ravel()
        if scores.ndim == 2:
            return scores.mean(axis=1)
        return scores

    @staticmethod
    def _as_2d(attributions: np.ndarray) -> np.ndarray:
        """Flatten attribution tensors to shape (N, D_flat)."""
        arr = np.asarray(attributions, dtype=np.float64)
        if arr.ndim == 1:
            return arr[None, :]
        if arr.ndim == 2:
            return arr
        return arr.reshape(arr.shape[0], -1)

    @staticmethod
    def _mean_cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
        """Cosine similarity averaged over rows."""
        if a.shape != b.shape:
            raise ValueError(
                "Cosine-similarity inputs must match in shape: "
                f"{a.shape} != {b.shape}"
            )
        num = np.sum(a * b, axis=1)
        den = np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1)
        den = np.where(den == 0.0, 1e-12, den)
        cos = num / den
        return float(np.mean(cos))
