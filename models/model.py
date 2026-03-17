"""
Attn-1DCNN-EE LightningModule — Full Model Assembly
=====================================================

Composes Components 2–4 into a single trainable model managed by
PyTorch Lightning.

Training strategy (two phases)
------------------------------
**Phase 1 — Supervised feature learning** (handled by Lightning Trainer):
    CNN backbone → Soft Attention → GlobalAvgPool → Linear head
    Optimised with cross-entropy loss so the backbone + attention learn
    discriminative, class-separable feature representations.

**Phase 2 — Elliptic Envelope fitting** (call :meth:`fit_envelope`):
    Features are extracted from the trained backbone + attention,
    then one ``EllipticEnvelope`` is fitted per class for open-set
    detection.  This phase is non-differentiable and runs once after
    Phase 1 completes.

Inference modes
---------------
* ``forward()`` — returns logits (linear head), attention weights, and
  pooled features.  Used during Phase 1 training / evaluation.
* ``predict_open_set()`` — runs the EE head on pooled features for
  known / unknown fault classification.  Used after Phase 2.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# Lightning import (compatible with both package names)
try:
    import pytorch_lightning as pl
except ImportError:
    import lightning.pytorch as pl  # type: ignore[no-redef]

from models.cnn_backbone import CNN1DBackbone
from models.attention import SoftAttention
from models.ee_head import EllipticEnvelopeHead, GlobalAvgPool1d

logger = logging.getLogger(__name__)


class Attn1DCNN_EE(pl.LightningModule):
    """Full Attn-1DCNN-EE model.

    Parameters
    ----------
    in_channels : int
        Number of input sensor channels (*F*).  Default ``96``.
    num_classes : int
        Number of known NPPAD fault classes for the supervised head.
    backbone_channels : list[int] | None
        Channel widths for the CNN backbone blocks.
        Default ``[64, 128, 256]``.
    backbone_kernel_sizes : list[int] | int
        Kernel size(s) for the backbone.  Default ``3``.
    lr : float
        Learning rate for AdamW.  Default ``1e-3``.
    weight_decay : float
        AdamW weight decay.  Default ``1e-4``.
    scheduler : str
        LR scheduler: ``"cosine"`` or ``"none"``.  Default ``"cosine"``.
    ee_contamination : float
        Contamination factor for the Elliptic Envelope (Phase 2).
        Default ``0.01``.
    """

    def __init__(
        self,
        in_channels: int = 96,
        num_classes: int = 13,
        backbone_channels: Optional[List[int]] = None,
        backbone_kernel_sizes: int | List[int] = 3,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        scheduler: str = "cosine",
        ee_contamination: float = 0.01,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()

        # Component 2: 1D-CNN backbone
        self.backbone = CNN1DBackbone(
            in_channels=in_channels,
            channel_sizes=backbone_channels,
            kernel_sizes=backbone_kernel_sizes,
        )

        # Component 3: Soft Attention
        self.attention = SoftAttention(in_channels=self.backbone.out_channels)

        # Pooling bridge: (B, C, I) → (B, C)
        self.pool = GlobalAvgPool1d()

        # Supervised classification head (Phase 1 training)
        self.classifier = nn.Linear(self.backbone.out_channels, num_classes)

        # Loss
        self.criterion = nn.CrossEntropyLoss()

        # Component 4: EE head (populated by fit_envelope after Phase 1)
        self.ee_head: Optional[EllipticEnvelopeHead] = None

    # ==================================================================
    # Forward pass
    # ==================================================================

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Full forward pass through the neural components.

        Parameters
        ----------
        x : torch.Tensor
            Shape ``(B, F, I)`` — batch of sliding-window matrices.

        Returns
        -------
        logits : torch.Tensor
            Shape ``(B, num_classes)`` — classification logits from the
            linear head.
        attn_weights : torch.Tensor
            Shape ``(B, I, C)`` — attention weights for XAI extraction.
        pooled : torch.Tensor
            Shape ``(B, C)`` — pooled feature vectors (input to EE head).
        """
        features = self.backbone(x)                      # (B, C, I)
        attended, attn_weights = self.attention(features) # (B, C, I), (B, I, C)
        pooled = self.pool(attended)                      # (B, C)
        logits = self.classifier(pooled)                  # (B, num_classes)
        return logits, attn_weights, pooled

    # ==================================================================
    # Lightning training hooks
    # ==================================================================

    def training_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        x, y = batch
        logits, _, _ = self(x)
        loss = self.criterion(logits, y)
        acc = (logits.argmax(dim=-1) == y).float().mean()
        self.log("train_loss", loss, prog_bar=True)
        self.log("train_acc", acc, prog_bar=True)
        return loss

    def validation_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        x, y = batch
        logits, _, _ = self(x)
        loss = self.criterion(logits, y)
        acc = (logits.argmax(dim=-1) == y).float().mean()
        self.log("val_loss", loss, prog_bar=True)
        self.log("val_acc", acc, prog_bar=True)
        return loss

    def test_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        x, y = batch
        logits, _, _ = self(x)
        loss = self.criterion(logits, y)
        acc = (logits.argmax(dim=-1) == y).float().mean()
        self.log("test_loss", loss, prog_bar=True)
        self.log("test_acc", acc, prog_bar=True)
        return loss

    def configure_optimizers(self) -> dict:
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay,
        )
        config: Dict[str, Any] = {"optimizer": optimizer}

        if self.hparams.scheduler == "cosine":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=self.trainer.max_epochs,
            )
            config["lr_scheduler"] = {
                "scheduler": scheduler,
                "interval": "epoch",
            }

        return config

    # ==================================================================
    # Phase 2: Elliptic Envelope fitting
    # ==================================================================

    @torch.no_grad()
    def extract_features(
        self, dataloader: DataLoader
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Extract pooled feature vectors from a dataloader.

        Runs the trained backbone + attention + pool in eval mode and
        collects the resulting feature vectors and labels.

        Parameters
        ----------
        dataloader : DataLoader
            Any split (train / val / test) from the ``NPPADDataModule``.

        Returns
        -------
        features : np.ndarray
            Shape ``(N, C)`` — pooled feature vectors.
        labels : np.ndarray
            Shape ``(N,)`` — integer class labels.
        """
        was_training = self.training
        self.eval()

        all_features: list[np.ndarray] = []
        all_labels: list[np.ndarray] = []

        for batch in dataloader:
            x, y = batch
            x = x.to(self.device)
            _, _, pooled = self(x)
            all_features.append(pooled.cpu().numpy())
            all_labels.append(y.numpy())

        if was_training:
            self.train()

        return np.concatenate(all_features), np.concatenate(all_labels)

    def fit_envelope(
        self,
        dataloader: DataLoader,
        label_map: Optional[Dict[str, int]] = None,
    ) -> EllipticEnvelopeHead:
        """Phase 2: fit the Elliptic Envelope on learned features.

        Parameters
        ----------
        dataloader : DataLoader
            Training dataloader (use ``dm.train_dataloader()``).
        label_map : dict[str, int] | None
            Class name → id mapping from ``NPPADDataModule.label_map``.

        Returns
        -------
        EllipticEnvelopeHead
            The fitted EE head (also stored as ``self.ee_head``).
        """
        logger.info("Phase 2: extracting features for EE fitting ...")
        features, labels = self.extract_features(dataloader)
        logger.info(
            "Extracted features: %s, labels: %s", features.shape, labels.shape
        )

        self.ee_head = EllipticEnvelopeHead(
            contamination=self.hparams.ee_contamination,
        )
        self.ee_head.fit(features, labels, label_map=label_map)

        report = self.ee_head.validate_boundaries(features, labels)
        logger.info(
            "Boundary validation: %d violations out of %d samples (%.2f%%)",
            report["total_violations"],
            report["total_samples"],
            report["violation_rate"] * 100,
        )

        return self.ee_head

    # ==================================================================
    # Open-set inference
    # ==================================================================

    @torch.no_grad()
    def predict_open_set(
        self, x: torch.Tensor
    ) -> Tuple[np.ndarray, np.ndarray, torch.Tensor]:
        """Run full open-set inference (backbone → attention → EE).

        Parameters
        ----------
        x : torch.Tensor
            Shape ``(B, F, I)`` — batch of sliding-window matrices.

        Returns
        -------
        predictions : np.ndarray
            Shape ``(B,)`` — predicted class ids (``-1`` = unknown).
        is_unknown : np.ndarray
            Shape ``(B,)`` — boolean mask for unknown faults.
        attn_weights : torch.Tensor
            Shape ``(B, I, C)`` — attention weights for XAI.
        """
        if self.ee_head is None:
            raise RuntimeError(
                "EE head not fitted. Call fit_envelope() after training."
            )
        self.eval()
        x = x.to(self.device)
        _, attn_weights, pooled = self(x)
        predictions, is_unknown = self.ee_head.predict(pooled.cpu().numpy())
        return predictions, is_unknown, attn_weights
