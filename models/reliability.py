"""
Reliability Mathematics Layer
=============================

Converts timestamped failure-event detections into reliability metrics:

* Failure rate: ``lambda = N_fail / T``
* Mean time to failure: ``MTTF = 1 / lambda``
* Reliability curve (exponential model): ``R(t) = exp(-t / MTTF)``

This module is intentionally model-agnostic and can consume events
produced by the Elliptic Envelope monitor or any binary detector.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence

import numpy as np


@dataclass
class ReliabilitySummary:
    """Compact reliability summary for one monitoring horizon."""

    failure_count: int
    operating_time: float
    failure_rate: float
    mttf: float
    start_time: Any
    end_time: Any
    time_unit: str = "same_as_input"


class ReliabilityAnalyzer:
    """Compute reliability metrics from binary failure detections."""

    def __init__(self, eps: float = 1e-12) -> None:
        self.eps = eps

    # ------------------------------------------------------------------
    # Core formulas
    # ------------------------------------------------------------------

    def failure_rate(
        self,
        failure_count: int,
        operating_time: float,
    ) -> float:
        """Return ``lambda = failure_count / operating_time``."""
        if operating_time <= 0.0:
            return np.inf if failure_count > 0 else 0.0
        return float(failure_count / operating_time)

    def mttf(self, failure_rate: float) -> float:
        """Return ``MTTF = 1 / lambda``."""
        if failure_rate <= 0.0:
            return np.inf
        return float(1.0 / failure_rate)

    def reliability_curve(
        self,
        elapsed_time: np.ndarray,
        mttf: float,
    ) -> np.ndarray:
        """Return ``R(t) = exp(-t / MTTF)`` for each ``t``."""
        if np.isinf(mttf):
            return np.ones_like(elapsed_time, dtype=np.float64)
        return np.exp(-np.asarray(elapsed_time, dtype=np.float64) / max(mttf, self.eps))

    # ------------------------------------------------------------------
    # End-to-end analysis
    # ------------------------------------------------------------------

    def analyze(
        self,
        timestamps: Sequence[Any],
        is_failure: Sequence[bool],
        risk_scores: Optional[Sequence[float]] = None,
    ) -> Dict[str, Any]:
        """Compute reliability summary + dynamic traces.

        Parameters
        ----------
        timestamps : sequence
            Ordered operating timestamps for monitored windows.
            Values must be numeric or datetime-like for metric
            computation (event logging can still use arbitrary labels).
        is_failure : sequence[bool]
            Binary failure flags aligned with ``timestamps``.
        risk_scores : sequence[float] | None
            Optional detector risk scores aligned with windows.

        Returns
        -------
        dict
            Keys include:
            ``summary`` (:class:`ReliabilitySummary`), ``time_axis``,
            ``elapsed_time``, ``reliability``, ``cumulative_failures``,
            ``empirical_failure_rate``.
        """
        t_raw = np.asarray(list(timestamps))
        failure = np.asarray(list(is_failure), dtype=bool)
        if t_raw.shape[0] != failure.shape[0]:
            raise ValueError(
                "timestamps and is_failure must have same length: "
                f"{t_raw.shape[0]} != {failure.shape[0]}"
            )
        if t_raw.shape[0] == 0:
            raise ValueError("reliability analysis requires at least one sample.")

        t_num = self._numeric_time_axis(t_raw)
        order = np.argsort(t_num)
        t_num = t_num[order]
        t_raw_ord = t_raw[order]
        failure = failure[order]

        start_num = float(t_num[0])
        end_num = float(t_num[-1])
        elapsed = t_num - start_num
        operating_time = max(end_num - start_num, self.eps)

        fail_count = int(failure.sum())
        lam = self.failure_rate(fail_count, operating_time)
        mttf = self.mttf(lam)
        reliability = self.reliability_curve(elapsed, mttf)

        cumulative_failures = np.cumsum(failure.astype(np.int64))
        empirical_rate = cumulative_failures / np.maximum(elapsed, self.eps)

        summary = ReliabilitySummary(
            failure_count=fail_count,
            operating_time=float(operating_time),
            failure_rate=float(lam),
            mttf=float(mttf),
            start_time=self._python_scalar(t_raw_ord[0]),
            end_time=self._python_scalar(t_raw_ord[-1]),
        )

        out: Dict[str, Any] = {
            "summary": summary,
            "time_axis": np.asarray([self._python_scalar(v) for v in t_raw_ord], dtype=object),
            "elapsed_time": elapsed.astype(np.float64),
            "reliability": reliability.astype(np.float64),
            "cumulative_failures": cumulative_failures.astype(np.int64),
            "empirical_failure_rate": empirical_rate.astype(np.float64),
        }

        if risk_scores is not None:
            risk = np.asarray(list(risk_scores), dtype=np.float64)
            if risk.shape[0] != t_raw.shape[0]:
                raise ValueError(
                    "risk_scores and timestamps must have same length: "
                    f"{risk.shape[0]} != {t_raw.shape[0]}"
                )
            out["risk_scores"] = risk[order]

        return out

    def build_failure_events(
        self,
        timestamps: Sequence[Any],
        is_failure: Sequence[bool],
        risk_scores: Optional[Sequence[float]] = None,
    ) -> List[Dict[str, Any]]:
        """Convert aligned arrays into timestamped failure records."""
        t = np.asarray(list(timestamps))
        y = np.asarray(list(is_failure), dtype=bool)
        if t.shape[0] != y.shape[0]:
            raise ValueError(
                "timestamps and is_failure must have same length: "
                f"{t.shape[0]} != {y.shape[0]}"
            )

        risk = None
        if risk_scores is not None:
            risk = np.asarray(list(risk_scores), dtype=np.float64)
            if risk.shape[0] != y.shape[0]:
                raise ValueError(
                    "risk_scores and is_failure must have same length: "
                    f"{risk.shape[0]} != {y.shape[0]}"
                )

        events: List[Dict[str, Any]] = []
        for i in np.where(y)[0]:
            event: Dict[str, Any] = {
                "index": int(i),
                "timestamp": self._python_scalar(t[i]),
                "failure": True,
            }
            if risk is not None:
                event["risk_score"] = float(risk[i])
            events.append(event)
        return events

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _numeric_time_axis(self, timestamps: np.ndarray) -> np.ndarray:
        """Convert timestamp array to numeric elapsed-compatible values."""
        if np.issubdtype(timestamps.dtype, np.datetime64):
            # Convert nanoseconds to hours for interpretable rates.
            ns = timestamps.astype("datetime64[ns]").astype(np.int64)
            return ns.astype(np.float64) / (1e9 * 3600.0)

        try:
            return timestamps.astype(np.float64)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "timestamps must be numeric or datetime64 for reliability "
                "mathematics."
            ) from exc

    @staticmethod
    def _python_scalar(value: Any) -> Any:
        if isinstance(value, np.generic):
            return value.item()
        return value

