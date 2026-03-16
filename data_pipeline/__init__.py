"""
NPPAD Data Pipeline
===================

A modular data pipeline for the Nuclear Power Plant Accident Dataset (NPPAD).
Prepares time-series sensor data for 1D-CNN models using PyTorch Lightning.

Pipeline stages:
    1. Data Loading   — Ingest raw CSV files from Operation_csv_data/
    2. Data Cleaning   — Handle missing values and remove anomalous outliers
    3. Scaling         — Z-score standardization (primary) / Min-Max (secondary)
    4. Sliding Window  — Convert time-series into fixed-length 2D matrices (I × F)
    5. Dataset Builder — PyTorch Lightning DataModule with train/val/test splits
"""

from data_pipeline.data_loader import NPPADDataLoader
from data_pipeline.data_cleaning import DataCleaner
from data_pipeline.scaler import ZScoreScaler, MinMaxScaler
from data_pipeline.sliding_window import SlidingWindowTransformer
from data_pipeline.dataset_builder import NPPADDataset, NPPADDataModule

__all__ = [
    "NPPADDataLoader",
    "DataCleaner",
    "ZScoreScaler",
    "MinMaxScaler",
    "SlidingWindowTransformer",
    "NPPADDataset",
    "NPPADDataModule",
]
