"""
Elliptic Envelope Head — Component 4: Open-Set Fault Classification
====================================================================

Replaces the traditional closed-set SoftMax classifier with a
distance-based **Elliptic Envelope** module that can identify both
*known* NPPAD fault types **and** flag completely novel, previously
unseen anomalies as "Unknown Fault".

Pipeline
--------
1. **Pool** — ``GlobalAvgPool1d`` reduces the attention-amplified feature
   maps from ``(B, C, I)`` to a fixed-length vector ``(B, C)``.
2. **Fit** — one ``sklearn.covariance.EllipticEnvelope`` is constructed
   per known class, estimating a robust mean (μ_i) and covariance (Σ_i)
   via the **FastMCD** algorithm with ``contamination=0.01`` (99th
   percentile threshold).
3. **Predict** — for each incoming sample the Mahalanobis distance to
   every class envelope is computed:

   * **Exactly one envelope accepts** → diagnose as that known fault.
   * **No envelope accepts** → trigger "Unknown Fault" alarm (open-set).
   * **Multiple envelopes accept** → pick the nearest (smallest
     Mahalanobis distance) as a tie-breaker.

Boundary constraint
-------------------
``Ω_i ∩ Ω_j = ∅ ∀ i ≠ j`` is validated after fitting via
:meth:`EllipticEnvelopeHead.validate_boundaries`.  If well-separated
features are learned by the CNN + Attention layers, this holds
naturally.

.. note::
   ``EllipticEnvelope`` requires ``n_samples > n_features`` per class.
   Classes that violate this are skipped with a logged warning.
"""

from __future__ import annotations

import logging
import warnings
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

# Deferred import so the module can be parsed even without sklearn
# (e.g. syntax-checking in an environment that only has torch).
try:
    from sklearn.covariance import EllipticEnvelope as _SklearnEE
except ImportError:  # pragma: no cover
    _SklearnEE = None


# ======================================================================
# Neural-network pooling bridge
# ======================================================================

class GlobalAvgPool1d(nn.Module):
    """Global average pooling across the temporal dimension.

    Reduces ``(B, C, I)`` feature maps to ``(B, C)`` feature vectors
    suitable for the Elliptic Envelope classifier.
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """``(B, C, I) → (B, C)``."""
        return x.mean(dim=-1)


# ======================================================================
# Elliptic Envelope Head (sklearn, non-differentiable)
# ======================================================================

class EllipticEnvelopeHead:
    """Per-class Elliptic Envelope classifier with open-set detection.

    This is **not** an ``nn.Module`` — it operates outside the gradient
    graph on numpy feature vectors extracted after the neural backbone
    has been trained.

    Parameters
    ----------
    contamination : float
        Expected fraction of outliers in training data per class.
        ``0.01`` corresponds to the 99th-percentile Mahalanobis
        distance threshold (default).
    random_state : int
        Seed for the FastMCD estimator (default ``42``).
    """

    UNKNOWN_LABEL: int = -1

    def __init__(
        self,
        contamination: float = 0.01,
        random_state: int = 42,
        support_fraction: Optional[float] = None,
    ) -> None:
        if _SklearnEE is None:
            raise ImportError(
                "scikit-learn is required for EllipticEnvelopeHead. "
                "Install it with: pip install scikit-learn"
            )
        self.contamination = contamination
        self.random_state = random_state
        self.support_fraction = support_fraction

        # Populated by fit()
        self.envelopes_: Dict[int, _SklearnEE] = {}
        self.label_map_: Optional[Dict[str, int]] = None
        self.fitted_: bool = False

    # ------------------------------------------------------------------
    # Fitting
    # ------------------------------------------------------------------

    def fit(
        self,
        features: np.ndarray,
        labels: np.ndarray,
        label_map: Optional[Dict[str, int]] = None,
    ) -> "EllipticEnvelopeHead":
        """Fit one Elliptic Envelope per known class.

        Parameters
        ----------
        features : np.ndarray
            Shape ``(N, D)`` — pooled feature vectors from the trained
            neural backbone.
        labels : np.ndarray
            Shape ``(N,)`` — integer class labels.
        label_map : dict[str, int] | None
            Optional name → id mapping (from the data pipeline) for
            human-readable logging.

        Returns
        -------
        self
        """
        self.label_map_ = label_map
        self.envelopes_.clear()
        unique_classes = np.unique(labels)

        for cls_id in unique_classes:
            cls_id = int(cls_id)
            mask = labels == cls_id
            cls_features = features[mask]
            n_samples, n_dims = cls_features.shape

            cls_name = self._class_name(cls_id)

            # EllipticEnvelope needs n_samples > n_features
            if n_samples <= n_dims:
                logger.warning(
                    "Class %s: only %d samples for %d features — "
                    "skipping (need n_samples > n_features for MCD).",
                    cls_name, n_samples, n_dims,
                )
                continue

            ee = _SklearnEE(
                contamination=self.contamination,
                random_state=self.random_state,
                support_fraction=self.support_fraction,
            )

            # FastMCD may emit repetitive determinant-increase warnings on
            # some high-dimensional class distributions. Retry once with a
            # higher support_fraction for numerical stability.
            retried_with_higher_support = False
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always", RuntimeWarning)
                ee.fit(cls_features)

            det_warn = any(
                isinstance(w.message, RuntimeWarning)
                and "Determinant has increased" in str(w.message)
                for w in caught
            )

            if det_warn:
                retry_support = 0.8 if self.support_fraction is None else max(
                    self.support_fraction, 0.8
                )
                logger.warning(
                    "Class %s: FastMCD determinant warning detected; "
                    "retrying EllipticEnvelope with support_fraction=%.2f",
                    cls_name,
                    retry_support,
                )
                ee = _SklearnEE(
                    contamination=self.contamination,
                    random_state=self.random_state,
                    support_fraction=retry_support,
                )
                with warnings.catch_warnings():
                    warnings.filterwarnings(
                        "ignore",
                        message="Determinant has increased; this should not happen.*",
                        category=RuntimeWarning,
                    )
                    ee.fit(cls_features)
                retried_with_higher_support = True

            self.envelopes_[cls_id] = ee
            logger.info(
                "Class %s: fitted envelope on %d samples (%d dims)%s",
                cls_name,
                n_samples,
                n_dims,
                " [retry support_fraction]" if retried_with_higher_support else "",
            )

        self.fitted_ = True
        logger.info(
            "EllipticEnvelopeHead fitted: %d / %d classes modelled",
            len(self.envelopes_), len(unique_classes),
        )
        return self

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def membership_masks(
        self,
        features: np.ndarray,
    ) -> Dict[int, np.ndarray]:
        """Return inlier membership masks for every fitted envelope.

        Parameters
        ----------
        features : np.ndarray
            Shape ``(N, D)``.

        Returns
        -------
        dict[int, np.ndarray]
            ``{class_id: bool_mask}``, each mask has shape ``(N,)``.
        """
        self._check_fitted()
        if not self.envelopes_:
            return {}
        return {
            cls_id: ee.predict(features) == 1
            for cls_id, ee in self.envelopes_.items()
        }

    def decision_scores(
        self,
        features: np.ndarray,
    ) -> Dict[int, np.ndarray]:
        """Return EE decision scores for every fitted class.

        Positive values indicate inlier confidence, negative values
        indicate outlier confidence.
        """
        self._check_fitted()
        if not self.envelopes_:
            return {}
        return {
            cls_id: ee.decision_function(features).astype(np.float64)
            for cls_id, ee in self.envelopes_.items()
        }

    def mahalanobis_distances(
        self, features: np.ndarray
    ) -> Dict[int, np.ndarray]:
        """Compute squared Mahalanobis distance to every fitted class.

        Parameters
        ----------
        features : np.ndarray
            Shape ``(N, D)``.

        Returns
        -------
        dict[int, np.ndarray]
            ``{class_id: distances}`` where each value has shape
            ``(N,)``.
        """
        self._check_fitted()
        return {
            cls_id: ee.mahalanobis(features)
            for cls_id, ee in self.envelopes_.items()
        }

    def predict_binary_failure(
        self,
        features: np.ndarray,
        require_unique_acceptance: bool = False,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Detect reliability failure windows using envelope acceptance.

        Parameters
        ----------
        features : np.ndarray
            Shape ``(N, D)`` pooled feature vectors.
        require_unique_acceptance : bool
            If ``False`` (default), a sample is a failure when no
            envelope accepts it.  If ``True``, any sample not accepted
            by exactly one envelope is considered a failure.

        Returns
        -------
        is_failure : np.ndarray
            Shape ``(N,)`` boolean mask.
        risk_score : np.ndarray
            Shape ``(N,)`` non-negative risk proxy derived from the
            best envelope decision score. Larger means riskier.
        nearest_class : np.ndarray
            Shape ``(N,)`` nearest class id by Mahalanobis distance.
            ``-1`` when no envelopes are available.
        accept_counts : np.ndarray
            Shape ``(N,)`` number of envelopes accepting each sample.
        """
        self._check_fitted()
        n = int(features.shape[0])
        if n == 0:
            return (
                np.empty((0,), dtype=bool),
                np.empty((0,), dtype=np.float64),
                np.empty((0,), dtype=np.int64),
                np.empty((0,), dtype=np.int32),
            )

        if not self.envelopes_:
            return (
                np.ones((n,), dtype=bool),
                np.full((n,), np.inf, dtype=np.float64),
                np.full((n,), self.UNKNOWN_LABEL, dtype=np.int64),
                np.zeros((n,), dtype=np.int32),
            )

        memberships = self.membership_masks(features)
        accept_counts = np.zeros((n,), dtype=np.int32)
        for mask in memberships.values():
            accept_counts += mask.astype(np.int32)

        if require_unique_acceptance:
            is_failure = accept_counts != 1
        else:
            is_failure = accept_counts == 0

        # Reliability risk proxy:
        # risk = max(0, -best decision score over class envelopes)
        decision = self.decision_scores(features)
        stacked_decision = np.stack(
            [decision[cls] for cls in sorted(decision)], axis=1
        )  # (N, C_fitted)
        best_decision = stacked_decision.max(axis=1)
        risk_score = np.maximum(0.0, -best_decision)

        # Nearest envelope by Mahalanobis distance (for logging context)
        dists = self.mahalanobis_distances(features)
        cls_ids = sorted(dists)
        stacked_dist = np.stack([dists[c] for c in cls_ids], axis=1)
        nearest_idx = stacked_dist.argmin(axis=1)
        nearest_class = np.asarray(
            [cls_ids[i] for i in nearest_idx], dtype=np.int64
        )

        return is_failure, risk_score, nearest_class, accept_counts

    def log_failure_events(
        self,
        features: np.ndarray,
        timestamps: Sequence[Any],
        require_unique_acceptance: bool = False,
    ) -> List[Dict[str, Any]]:
        """Create timestamped failure-event records.

        Parameters
        ----------
        features : np.ndarray
            Shape ``(N, D)`` pooled feature vectors.
        timestamps : sequence
            Length ``N`` operating timestamps (hours, seconds, datetime,
            or any serialisable scalar).
        require_unique_acceptance : bool
            Passed to :meth:`predict_binary_failure`.

        Returns
        -------
        list[dict]
            One dictionary per failure event.  Each record includes:
            ``index``, ``timestamp``, ``risk_score``, acceptance context,
            and nearest-envelope class.
        """
        n = int(features.shape[0])
        if len(timestamps) != n:
            raise ValueError(
                "timestamps length must match number of feature rows: "
                f"{len(timestamps)} != {n}"
            )

        is_failure, risk_score, nearest_class, accept_counts = (
            self.predict_binary_failure(
                features=features,
                require_unique_acceptance=require_unique_acceptance,
            )
        )
        memberships = self.membership_masks(features) if self.envelopes_ else {}

        events: List[Dict[str, Any]] = []
        for i in np.where(is_failure)[0]:
            accepted_class_ids = [
                int(cls_id)
                for cls_id, mask in memberships.items()
                if bool(mask[i])
            ]
            nearest_id = int(nearest_class[i])

            events.append(
                {
                    "index": int(i),
                    "timestamp": self._as_scalar(timestamps[i]),
                    "failure": True,
                    "risk_score": float(risk_score[i]),
                    "accept_count": int(accept_counts[i]),
                    "accepted_class_ids": accepted_class_ids,
                    "accepted_class_names": [
                        self._class_name(c) for c in accepted_class_ids
                    ],
                    "nearest_class_id": nearest_id,
                    "nearest_class_name": (
                        self._class_name(nearest_id)
                        if nearest_id != self.UNKNOWN_LABEL
                        else "Unknown"
                    ),
                }
            )

        return events

    def predict(
        self, features: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Classify samples via the voting mechanism.

        Parameters
        ----------
        features : np.ndarray
            Shape ``(N, D)`` — pooled feature vectors.

        Returns
        -------
        predictions : np.ndarray
            Shape ``(N,)`` — predicted class id, or ``-1`` for unknown.
        is_unknown : np.ndarray
            Shape ``(N,)`` — boolean mask where ``True`` = unknown fault.
        """
        self._check_fitted()
        n = features.shape[0]
        if not self.envelopes_:
            logger.warning(
                "predict() called with no fitted envelopes; returning all unknown."
            )
            predictions = np.full(n, self.UNKNOWN_LABEL, dtype=np.int64)
            return predictions, np.ones((n,), dtype=bool)

        # Gather per-class membership and Mahalanobis distances
        memberships = self.membership_masks(features)
        distances = self.mahalanobis_distances(features)

        predictions = np.full(n, self.UNKNOWN_LABEL, dtype=np.int64)

        for i in range(n):
            accepting: List[int] = [
                cls for cls, mask in memberships.items() if mask[i]
            ]

            if len(accepting) == 1:
                # Closed-set: exactly one envelope accepts
                predictions[i] = accepting[0]
            elif len(accepting) > 1:
                # Tie-break: pick closest by Mahalanobis distance
                predictions[i] = min(
                    accepting, key=lambda c: distances[c][i]
                )
            # else: stays UNKNOWN_LABEL (-1) → open-set trigger

        is_unknown = predictions == self.UNKNOWN_LABEL
        n_unknown = int(is_unknown.sum())
        if n_unknown > 0:
            logger.info(
                "Open-set detection: %d / %d samples flagged as unknown",
                n_unknown, n,
            )
        return predictions, is_unknown

    # ------------------------------------------------------------------
    # Boundary validation
    # ------------------------------------------------------------------

    def validate_boundaries(
        self,
        features: np.ndarray,
        labels: np.ndarray,
    ) -> Dict[str, int]:
        """Check the non-overlap constraint Ω_i ∩ Ω_j = ∅.

        Counts how many training samples are accepted by more than one
        envelope (boundary violations).

        Parameters
        ----------
        features : np.ndarray
            Shape ``(N, D)`` — same features used for fitting.
        labels : np.ndarray
            Shape ``(N,)`` — true class labels.

        Returns
        -------
        dict
            ``{"total_violations": int, "total_samples": int,
            "violation_rate": float}``.
        """
        self._check_fitted()
        memberships = {
            cls_id: ee.predict(features) == 1
            for cls_id, ee in self.envelopes_.items()
        }

        # Count samples accepted by >1 envelope
        accept_counts = np.zeros(features.shape[0], dtype=np.int32)
        for mask in memberships.values():
            accept_counts += mask.astype(np.int32)

        violations = int((accept_counts > 1).sum())
        total = features.shape[0]
        rate = violations / total if total > 0 else 0.0

        if violations > 0:
            logger.warning(
                "Boundary overlap: %d / %d samples (%.2f%%) accepted "
                "by multiple envelopes",
                violations, total, rate * 100,
            )
        else:
            logger.info("Boundary constraint satisfied: no overlaps")

        return {
            "total_violations": violations,
            "total_samples": total,
            "violation_rate": rate,
        }

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _class_name(self, cls_id: int) -> str:
        """Resolve a class id to its human-readable name."""
        if self.label_map_ is not None:
            inv = {v: k for k, v in self.label_map_.items()}
            return inv.get(cls_id, str(cls_id))
        return str(cls_id)

    @staticmethod
    def _as_scalar(value: Any) -> Any:
        """Convert numpy scalar-like values into plain Python scalars."""
        if isinstance(value, np.generic):
            return value.item()
        return value

    def _check_fitted(self) -> None:
        if not self.fitted_:
            raise RuntimeError(
                "EllipticEnvelopeHead has not been fitted. "
                "Call fit() first."
            )
