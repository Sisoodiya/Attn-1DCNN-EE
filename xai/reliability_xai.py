"""
Reliability-Level Attribution for Attn-1DCNN-EE
================================================

Provides gradient-based attributions for reliability degradation analysis.
The default target is a differentiable reliability-risk surrogate:

``risk = 1 - max_softmax(logits)``

This lets us explain *why reliability drops* at a given window by
attributing risk to specific (sensor, time-step) input elements.
"""

from __future__ import annotations

from typing import Callable, List, Optional, Sequence, Tuple

import numpy as np
import torch


class ReliabilityAttributor:
    """Integrated-gradients style attributor for reliability risk."""

    def __init__(
        self,
        model: torch.nn.Module,
        risk_fn: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
    ) -> None:
        self.model = model
        self.risk_fn = risk_fn or self._default_risk

    # ------------------------------------------------------------------
    # Attribution methods
    # ------------------------------------------------------------------

    def integrated_gradients(
        self,
        x: torch.Tensor,
        baseline: Optional[torch.Tensor] = None,
        steps: int = 32,
    ) -> torch.Tensor:
        """Compute integrated gradients for reliability risk.

        Parameters
        ----------
        x : torch.Tensor
            Input windows, shape ``(B, F, I)``.
        baseline : torch.Tensor | None
            Baseline reference with same shape as ``x``.
            Defaults to all zeros.
        steps : int
            Number of interpolation steps.

        Returns
        -------
        torch.Tensor
            Attribution tensor with same shape as ``x``.
        """
        if steps < 2:
            raise ValueError(f"steps must be >= 2, got {steps}")

        was_training = self.model.training
        self.model.eval()

        x = x.detach()
        if baseline is None:
            baseline = torch.zeros_like(x)
        else:
            baseline = baseline.detach()
            if baseline.shape != x.shape:
                raise ValueError(
                    "baseline must match x shape: "
                    f"{tuple(baseline.shape)} != {tuple(x.shape)}"
                )

        total_grad = torch.zeros_like(x)
        alphas = torch.linspace(
            0.0,
            1.0,
            steps=steps,
            device=x.device,
            dtype=x.dtype,
        )

        for alpha in alphas[1:]:
            x_step = baseline + alpha * (x - baseline)
            x_step.requires_grad_(True)

            logits = self._extract_logits(self.model(x_step))
            risk = self.risk_fn(logits)  # (B,) expected
            if risk.ndim != 1:
                risk = risk.reshape(risk.shape[0], -1).mean(dim=1)

            score = risk.sum()
            grad = torch.autograd.grad(
                outputs=score,
                inputs=x_step,
                retain_graph=False,
                create_graph=False,
            )[0]
            total_grad += grad

        avg_grad = total_grad / float(steps - 1)
        attr = (x - baseline) * avg_grad

        if was_training:
            self.model.train()

        return attr.detach()

    def input_gradients(self, x: torch.Tensor) -> torch.Tensor:
        """Single-step gradient attribution for reliability risk."""
        was_training = self.model.training
        self.model.eval()

        x = x.detach().requires_grad_(True)
        logits = self._extract_logits(self.model(x))
        risk = self.risk_fn(logits)
        if risk.ndim != 1:
            risk = risk.reshape(risk.shape[0], -1).mean(dim=1)
        score = risk.sum()
        grad = torch.autograd.grad(score, x)[0]

        if was_training:
            self.model.train()

        return grad.detach()

    # ------------------------------------------------------------------
    # Attribution post-processing
    # ------------------------------------------------------------------

    @staticmethod
    def to_time_sensor_map(
        attributions: torch.Tensor | np.ndarray,
        sample_idx: int = 0,
        absolute: bool = True,
    ) -> np.ndarray:
        """Convert one sample attribution to ``(I, F)`` map."""
        attr = attributions
        if isinstance(attr, torch.Tensor):
            attr = attr.detach().cpu().numpy()
        if attr.ndim == 2:
            # Assume (F, I)
            arr = attr
        elif attr.ndim == 3:
            arr = attr[sample_idx]  # (F, I)
        else:
            raise ValueError(
                "Expected attribution shape (F, I) or (B, F, I), got "
                f"{attr.shape}"
            )
        if absolute:
            arr = np.abs(arr)
        return arr.T  # (I, F)

    @staticmethod
    def aggregate_sensor_importance(
        attributions: torch.Tensor | np.ndarray,
        absolute: bool = True,
    ) -> np.ndarray:
        """Aggregate attribution tensor into per-sensor importance."""
        attr = attributions
        if isinstance(attr, torch.Tensor):
            attr = attr.detach().cpu().numpy()

        if attr.ndim == 2:
            arr = attr[None, ...]  # (1, F, I)
        elif attr.ndim == 3:
            arr = attr  # (B, F, I)
        else:
            raise ValueError(
                "Expected attribution shape (F, I) or (B, F, I), got "
                f"{attr.shape}"
            )

        if absolute:
            arr = np.abs(arr)

        # Mean over batch + time -> per-sensor vector
        return arr.mean(axis=(0, 2))

    @staticmethod
    def top_sensors(
        sensor_importance: np.ndarray,
        feature_names: Optional[Sequence[str]] = None,
        top_k: int = 10,
    ) -> List[Tuple[str, float]]:
        """Return top-k sensors ranked by importance."""
        if sensor_importance.ndim != 1:
            raise ValueError(
                "sensor_importance must be 1-D, got "
                f"{sensor_importance.shape}"
            )
        idx = np.argsort(sensor_importance)[-top_k:][::-1]
        names = (
            [str(feature_names[i]) for i in idx]
            if feature_names is not None
            else [f"Feature {i}" for i in idx]
        )
        return [
            (name, float(sensor_importance[i]))
            for name, i in zip(names, idx)
        ]

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_logits(model_output: object) -> torch.Tensor:
        """Handle model outputs that are tuple/list based."""
        if isinstance(model_output, (tuple, list)):
            if not model_output:
                raise ValueError("Model output tuple/list is empty.")
            return model_output[0]
        if not isinstance(model_output, torch.Tensor):
            raise TypeError(
                "Model output must be Tensor or tuple/list of Tensors."
            )
        return model_output

    @staticmethod
    def _default_risk(logits: torch.Tensor) -> torch.Tensor:
        """Differentiable reliability-risk proxy from classifier logits."""
        probs = torch.softmax(logits, dim=-1)
        confidence, _ = probs.max(dim=-1)
        return 1.0 - confidence

