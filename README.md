# Attn-1DCNN-EE

Attn-1DCNN-EE is a hybrid model for nuclear anomaly detection that preprocesses multivariate sensor data, extracts temporal features using a 1D-CNN, highlights critical signals with soft attention, detects known and unknown faults using an Elliptic Envelope, and explains predictions with attention heatmaps and SHAP.

## Accuracy and Generalization

The training pipeline now supports stronger regularization and stability controls:

- Kaiming initialization + BatchNorm in CNN blocks
- Optional backbone/classifier dropout
- Label smoothing and AdamW + cosine schedule
- Gradient clipping
- Optional train-time noise and gain augmentation
- Optional class weighting

## Bayesian Tuning with K-Fold CV

Use Optuna (TPE) with stratified K-fold validation to tune hyperparameters for
generalization (instead of a single split score):

```bash
python3 experiments/optuna_kfold_tuning.py \
  --data-dir data/Operation_csv_data \
  --trials 20 \
  --folds 5 \
  --max-epochs 30 \
  --patience 6 \
  --window-size 80 \
  --stride 1
```
