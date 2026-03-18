"""
Optuna + Stratified K-Fold tuning for Attn-1DCNN-EE.

This script optimizes training hyperparameters for generalization by
evaluating each trial across multiple sample-level stratified folds.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import List, Sequence, Tuple

import numpy as np
import torch
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader

try:
    import lightning.pytorch as pl  # type: ignore[no-redef]
    from lightning.pytorch.callbacks import EarlyStopping
except ImportError:
    import pytorch_lightning as pl
    from pytorch_lightning.callbacks import EarlyStopping

import polars as pl_lib

from data_pipeline.data_cleaning import DataCleaner
from data_pipeline.data_loader import NPPADDataLoader
from data_pipeline.dataset_builder import NPPADDataset
from data_pipeline.scaler import MinMaxScaler, ZScoreScaler
from models.model import Attn1DCNN_EE

logger = logging.getLogger(__name__)


def _build_scaler(name: str) -> ZScoreScaler | MinMaxScaler:
    if name == "zscore":
        return ZScoreScaler()
    if name == "minmax":
        return MinMaxScaler()
    raise ValueError(f"Unknown scaler_type: {name}")


def _scale_samples(
    train_samples: Sequence[np.ndarray],
    val_samples: Sequence[np.ndarray],
    scaler_type: str,
) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    scaler = _build_scaler(scaler_type)
    train_df = [pl_lib.DataFrame(arr) for arr in train_samples]
    scaler.fit(train_df)

    def transform(arrays: Sequence[np.ndarray]) -> List[np.ndarray]:
        out: List[np.ndarray] = []
        for arr in arrays:
            df = pl_lib.DataFrame(arr)
            out.append(
                scaler.transform([df])[0].to_numpy().astype(np.float32)
            )
        return out

    return transform(train_samples), transform(val_samples)


def _class_weights_from_windows(
    labels: np.ndarray,
    num_classes: int,
) -> List[float]:
    counts = np.bincount(labels, minlength=num_classes)
    counts = np.maximum(counts, 1)
    weights = labels.size / (num_classes * counts)
    return weights.astype(np.float32).tolist()


def _train_one_fold(
    train_samples: Sequence[np.ndarray],
    train_labels: Sequence[int],
    val_samples: Sequence[np.ndarray],
    val_labels: Sequence[int],
    num_features: int,
    num_classes: int,
    params: dict,
    seed: int,
) -> float:
    train_scaled, val_scaled = _scale_samples(
        train_samples=train_samples,
        val_samples=val_samples,
        scaler_type=params["scaler_type"],
    )

    train_ds = NPPADDataset(
        samples=list(train_scaled),
        labels=list(train_labels),
        window_size=params["window_size"],
        stride=params["stride"],
        noise_std=params["train_noise_std"],
        gain_std=params["train_gain_std"],
    )
    val_ds = NPPADDataset(
        samples=list(val_scaled),
        labels=list(val_labels),
        window_size=params["window_size"],
        stride=params["stride"],
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=params["batch_size"],
        shuffle=True,
        num_workers=params["num_workers"],
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
        persistent_workers=params["num_workers"] > 0,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=params["batch_size"],
        shuffle=False,
        num_workers=params["num_workers"],
        pin_memory=torch.cuda.is_available(),
        persistent_workers=params["num_workers"] > 0,
    )

    class_weights = None
    if params["use_class_weights"]:
        window_labels = train_ds.labels.numpy()
        class_weights = _class_weights_from_windows(
            labels=window_labels,
            num_classes=num_classes,
        )

    model = Attn1DCNN_EE(
        in_channels=num_features,
        num_classes=num_classes,
        backbone_channels=params["backbone_channels"],
        backbone_kernel_sizes=params["backbone_kernel_sizes"],
        lr=params["lr"],
        weight_decay=params["weight_decay"],
        optimizer=params["optimizer"],
        scheduler=params["scheduler"],
        cosine_min_lr=params["cosine_min_lr"],
        scheduler_gamma=params["scheduler_gamma"],
        ee_contamination=params["ee_contamination"],
        ee_support_fraction=params["ee_support_fraction"],
        backbone_dropout=params["backbone_dropout"],
        classifier_dropout=params["classifier_dropout"],
        label_smoothing=params["label_smoothing"],
        class_weights=class_weights,
    )

    early = EarlyStopping(
        monitor="val_loss",
        patience=params["patience"],
        mode="min",
        verbose=False,
    )

    pl.seed_everything(seed, workers=True)
    trainer = pl.Trainer(
        max_epochs=params["max_epochs"],
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
        callbacks=[early],
        logger=False,
        enable_checkpointing=False,
        deterministic=True,
        gradient_clip_val=params["gradient_clip_val"],
        gradient_clip_algorithm="norm",
    )
    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)

    if early.best_score is None:
        metrics = trainer.validate(model, dataloaders=val_loader, verbose=False)
        return float(metrics[0]["val_loss"])
    return float(early.best_score.detach().cpu().item())


def _load_clean_samples(
    data_dir: str | Path,
    nan_strategy: str,
    z_threshold: float,
    accident_types: Sequence[str] | None,
) -> Tuple[List[np.ndarray], np.ndarray, dict]:
    loader = NPPADDataLoader(data_dir)
    raw_samples, labels, label_map = loader.load_all(
        exclude_time=True,
        accident_types=list(accident_types) if accident_types else None,
    )

    cleaner = DataCleaner(nan_strategy=nan_strategy, z_threshold=z_threshold)
    cleaned: List[np.ndarray] = []
    for df in raw_samples:
        cleaned.append(
            cleaner.clean(df).to_numpy().astype(np.float32)
        )
    return cleaned, np.asarray(labels, dtype=np.int64), label_map


def _parse_channels(s: str) -> List[int]:
    return [int(x.strip()) for x in s.split(",") if x.strip()]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data/Operation_csv_data")
    parser.add_argument("--trials", type=int, default=20)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--window-size", type=int, default=80)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--max-epochs", type=int, default=30)
    parser.add_argument("--patience", type=int, default=6)
    parser.add_argument("--scaler-type", choices=["zscore", "minmax"], default="zscore")
    parser.add_argument("--nan-strategy", choices=["interpolate", "ffill"], default="interpolate")
    parser.add_argument("--z-threshold", type=float, default=6.0)
    parser.add_argument("--use-class-weights", action="store_true")
    parser.add_argument("--gradient-clip-val", type=float, default=1.0)
    parser.add_argument("--ee-contamination", type=float, default=0.01)
    parser.add_argument("--ee-support-fraction", type=float, default=0.8)
    parser.add_argument(
        "--channel-candidates",
        nargs="+",
        default=["32,64,128", "64,128,256"],
        help="Space-separated candidates, each as comma-separated ints.",
    )
    args = parser.parse_args()

    try:
        import optuna
    except ImportError as exc:
        raise SystemExit(
            "optuna is required. Install with: pip install optuna"
        ) from exc

    logging.basicConfig(level=logging.INFO, format="%(name)s — %(levelname)s — %(message)s")

    samples, labels, label_map = _load_clean_samples(
        data_dir=args.data_dir,
        nan_strategy=args.nan_strategy,
        z_threshold=args.z_threshold,
        accident_types=None,
    )
    num_classes = len(label_map)
    num_features = samples[0].shape[1]

    logger.info(
        "Loaded %d cleaned samples (%d classes, %d features)",
        len(samples),
        num_classes,
        num_features,
    )

    channel_candidates = [_parse_channels(s) for s in args.channel_candidates]
    skf = StratifiedKFold(
        n_splits=args.folds,
        shuffle=True,
        random_state=args.seed,
    )

    def objective(trial: "optuna.Trial") -> float:
        params = {
            "window_size": args.window_size,
            "stride": args.stride,
            "batch_size": args.batch_size,
            "num_workers": args.num_workers,
            "max_epochs": args.max_epochs,
            "patience": args.patience,
            "scaler_type": args.scaler_type,
            "ee_contamination": args.ee_contamination,
            "ee_support_fraction": args.ee_support_fraction,
            "gradient_clip_val": args.gradient_clip_val,
            "use_class_weights": args.use_class_weights,
            "backbone_channels": trial.suggest_categorical(
                "backbone_channels", channel_candidates
            ),
            "optimizer": trial.suggest_categorical(
                "optimizer", ["adamw", "adam", "rmsprop"]
            ),
            "backbone_kernel_sizes": trial.suggest_categorical(
                "backbone_kernel_sizes", [3, 5]
            ),
            "lr": trial.suggest_float("lr", 1e-4, 2e-3, log=True),
            "weight_decay": trial.suggest_float(
                "weight_decay", 1e-5, 1e-3, log=True
            ),
            "scheduler": trial.suggest_categorical(
                "scheduler", ["cosine", "exponential"]
            ),
            "cosine_min_lr": trial.suggest_float(
                "cosine_min_lr", 1e-7, 1e-5, log=True
            ),
            "scheduler_gamma": trial.suggest_float(
                "scheduler_gamma", 0.95, 0.995
            ),
            "backbone_dropout": trial.suggest_float(
                "backbone_dropout", 0.0, 0.3
            ),
            "classifier_dropout": trial.suggest_float(
                "classifier_dropout", 0.1, 0.4
            ),
            "label_smoothing": trial.suggest_float(
                "label_smoothing", 0.0, 0.08
            ),
            "train_noise_std": trial.suggest_float(
                "train_noise_std", 0.0, 0.03
            ),
            "train_gain_std": trial.suggest_float(
                "train_gain_std", 0.0, 0.08
            ),
        }

        fold_losses: List[float] = []
        for fold_idx, (train_idx, val_idx) in enumerate(
            skf.split(np.zeros(len(labels)), labels)
        ):
            train_samples = [samples[i] for i in train_idx]
            train_labels = labels[train_idx].tolist()
            val_samples = [samples[i] for i in val_idx]
            val_labels = labels[val_idx].tolist()

            fold_loss = _train_one_fold(
                train_samples=train_samples,
                train_labels=train_labels,
                val_samples=val_samples,
                val_labels=val_labels,
                num_features=num_features,
                num_classes=num_classes,
                params=params,
                seed=args.seed + fold_idx,
            )
            fold_losses.append(fold_loss)

            trial.report(float(np.mean(fold_losses)), step=fold_idx)
            if trial.should_prune():
                raise optuna.TrialPruned()

        return float(np.mean(fold_losses))

    sampler = optuna.samplers.TPESampler(seed=args.seed)
    study = optuna.create_study(direction="minimize", sampler=sampler)
    study.optimize(objective, n_trials=args.trials, show_progress_bar=True)

    best = study.best_trial
    logger.info("Best mean val_loss: %.6f", best.value)
    logger.info("Best params:")
    for k, v in best.params.items():
        logger.info("  %s = %s", k, v)


if __name__ == "__main__":
    main()
