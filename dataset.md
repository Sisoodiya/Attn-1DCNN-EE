# Dataset Card: `Operation_csv_data` for Attn-1DCNN-EE

## 1. Scope and Purpose

This document describes the exact dataset snapshot used by the project in:

- `data/Operation_csv_data`

It is written for research reproducibility and paper-ready methodology sections.  
All statistics below were computed from the local repository snapshot on **2026-03-19**.

---

## 2. Task Definition

The dataset is used for:

1. **Closed-set multi-class fault classification** (13 known classes).
2. **Open-set detection** after neural feature learning, via per-class Elliptic Envelopes.

Each CSV file is treated as one simulation sample.  
Sliding windows extracted from a file inherit that file’s class label.

---

## 3. Storage, Size, and Structure

### 3.1 Storage

- Uncompressed dataset directory: `~611 MB`
- Compressed archive: `data/Operation_csv_data.zip` (`~119 MB`)

### 3.2 Folder layout

```
data/
  Operation_csv_data/
    FLB/
    LLB/
    LOCA/
    LOCAC/
    LR/
    MD/
    RI/
    RW/
    SGATR/
    SGBTR/
    SLBIC/
    SLBOC/
    TT/
```

### 3.3 Instance definition

- One CSV file = one run (sample)
- One row = one timestep
- One column = one process variable

### 3.4 Global counts

- Total classes: `13`
- Total CSV files: `1237`
- Total rows (all CSVs): `497,375`

---

## 4. Label Space and Encoding

The data loader uses deterministic class ordering.  
For this snapshot (no `Normal` folder present), the label map is:

| Class | Label |
|---|---:|
| FLB | 0 |
| LLB | 1 |
| LOCA | 2 |
| LOCAC | 3 |
| LR | 4 |
| MD | 5 |
| RI | 6 |
| RW | 7 |
| SGATR | 8 |
| SGBTR | 9 |
| SLBIC | 10 |
| SLBOC | 11 |
| TT | 12 |

---

## 5. Schema and Feature-Space Characterization

## 5.1 Raw schema pattern

Expected raw pattern is:

- `TIME` + sensor variables

Observed:

- All files include `TIME`
- Most files have **96 feature columns** after dropping `TIME`
- A subset has **99 feature columns** after dropping `TIME`

## 5.2 Schema heterogeneity

- Unique feature schemas observed: `3`
- Common intersection schema: `96` features
- Maximum schema size: `99` features
- Files with 99 features: `25/1237` (all under class `SLBIC`)

Extra columns present only in 99-feature files:

- `WPCS`
- `WPMU`
- `WPFW`

This exactly explains loader warning messages such as:

```text
Column schema varies across CSVs; aligned 25/1237 samples to a 96-column reference schema (added=0, dropped=75).
```

Because:

- `25 samples × 3 extra columns = 75 dropped columns`

## 5.3 Reference 96-feature set (used by model input)

The baseline 96-feature schema (after `TIME` removal) is:

```text
P,TAVG,THA,THB,TCA,TCB,WRCA,WRCB,PSGA,PSGB,WFWA,WFWB,WSTA,WSTB,VOL,LVPZ,VOID,WLR,WUP,HUP,HLW,WHPI,WECS,QMWT,LSGA,LSGB,QMGA,QMGB,NSGA,NSGB,TBLD,WTRA,WTRB,TSAT,QRHR,LVCR,SCMA,SCMB,FRCL,PRB,PRBA,TRB,LWRB,DNBR,QFCL,WBK,WSPY,WCSP,HTR,MH2,CNH2,RHBR,RHMT,RHFL,RHRD,RH,PWNT,PWR,TFSB,TFPK,TF,TPCT,WCFT,WLPI,WCHG,RM1,RM2,RM3,RM4,RC87,RC131,STRB,STSG,STTB,RBLK,SGLK,DTHY,DWB,WRLA,WRLB,WLD,MBK,EBK,TKLV,FRZR,TDBR,MDBR,MCRT,MGAS,TCRT,TSLP,PPM,RRCA,RRCB,RRCO,WFLB
```

---

## 6. Sequence-Length Statistics

## 6.1 Global length statistics (rows per CSV)

- Minimum: `11`
- 25th percentile: `283`
- Median: `433`
- Mean: `402.08`
- 75th percentile: `540`
- Maximum: `638`

Examples:

- Shortest file: `data/Operation_csv_data/RI/-100.csv` (11 rows)
- Longest file: `data/Operation_csv_data/MD/41.csv` (638 rows)

## 6.2 Per-class length and window statistics

Windows are computed with:

- `I = window_size = 50`
- `n_windows = max(0, floor((T - I)/stride) + 1)`

| Class | Files | Min | Q1 | Median | Mean | Q3 | Max | Windows (stride=1) | Windows (stride=5) | Files with T<50 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| FLB | 100 | 161 | 193.5 | 206.0 | 220.07 | 227.25 | 553 | 17,107 | 3,460 | 0 |
| LLB | 101 | 538 | 553.0 | 555.0 | 554.38 | 556.0 | 558 | 51,043 | 10,250 | 0 |
| LOCA | 100 | 302 | 360.75 | 388.5 | 390.17 | 423.5 | 475 | 34,117 | 6,867 | 0 |
| LOCAC | 100 | 367 | 418.0 | 452.5 | 451.76 | 489.5 | 544 | 40,276 | 8,091 | 0 |
| LR | 99 | 551 | 555.0 | 556.0 | 556.05 | 557.0 | 558 | 50,198 | 10,081 | 0 |
| MD | 100 | 539 | 554.0 | 556.0 | 590.20 | 636.0 | 638 | 54,120 | 10,866 | 0 |
| RI | 100 | 11 | 159.0 | 160.0 | 201.99 | 317.0 | 319 | 15,983 | 3,218 | 18 |
| RW | 100 | 155 | 164.0 | 171.0 | 183.88 | 186.0 | 555 | 13,488 | 2,737 | 0 |
| SGATR | 100 | 400 | 422.75 | 441.0 | 458.60 | 494.25 | 568 | 40,960 | 8,228 | 0 |
| SGBTR | 110 | 415 | 427.0 | 443.5 | 466.05 | 504.0 | 557 | 45,876 | 9,225 | 0 |
| SLBIC | 101 | 175 | 208.0 | 224.0 | 297.65 | 494.0 | 551 | 25,114 | 5,059 | 0 |
| SLBOC | 100 | 422 | 434.75 | 482.0 | 475.38 | 505.0 | 561 | 42,638 | 8,573 | 0 |
| TT | 26 | 300 | 300.0 | 300.0 | 300.00 | 300.0 | 300 | 6,526 | 1,326 | 0 |

Observations:

- RI has the only short-sequence files (`T<50`), which are dropped during window dataset creation.
- TT has fixed-length sequences (all 300 rows) but low sample count.
- MD has the largest total window contribution at stride 1.

---

## 7. Missing-Value and Data-Quality Profile

A raw scan over all kept columns found:

- Total scanned cells: `48,285,443`
- Missing cells (`""` or `NaN` text): `92`
- Missing-cell rate: `1.905e-06`
- Rows containing at least one missing cell: `1`
- All missing cells were in class `TT`

Implications:

- Missingness is extremely sparse.
- The cleaner still includes robust null handling and final null-safety fill, so warnings can appear even with tiny missing rates.

---

## 8. Preprocessing and Leakage Controls

## 8.1 Ingestion and schema normalization

Implemented in `data_pipeline/data_loader.py`:

1. Read CSV via Polars.
2. Normalize headers (trim, BOM cleanup).
3. Drop unnamed/index-like columns.
4. Optionally drop `TIME` (`exclude_time=True` in pipeline).
5. Align all files to one reference schema (96-feature baseline).

## 8.2 Cleaning

Implemented in `data_pipeline/data_cleaning.py`:

- Missing handling:
  - Replace NaN with null
  - `interpolate` mode: linear interpolation + forward/backward fill
  - Fallback: `fill_null(0.0)` if residual nulls remain
- Outlier handling:
  - Per-sample Z-score masking (`z_threshold=6.0` default)
  - Masked values interpolated and edge-filled

## 8.3 Split and scaling (anti-leakage design)

Implemented in `data_pipeline/dataset_builder.py`:

1. **Sample-level stratified split** (not window-level split).
2. Fit scaler on **train split only**.
3. Transform val/test using train statistics.
4. Build lazy sliding-window datasets per split.

This avoids temporal/window leakage from validation/test into train statistics.

---

## 9. Split Protocol and Effective Dataset Sizes

## 9.1 Stratified split math (per class `c`)

Given class sample count `n_c`:

- `n_test = floor(n_c * test_split)`
- `n_val  = floor(n_c * val_split)`
- Safeguards enforce at least one train sample and minimal val/test when feasible.

Default values:

- `val_split = 0.15`
- `test_split = 0.15`
- `random_seed = 42`

## 9.2 Sample counts by split

Before short-sequence filtering (`T<50`):

- Train: `871`
- Val: `183`
- Test: `183`

After filtering (`window_size=50`):

- Train: `856`
- Val: `181`
- Test: `182`

Dropped files:

- Total dropped (`T<50`): `18`
- All dropped files are from class `RI`

## 9.3 Windows by split

With `window_size=50`:

- **Stride 1**
  - Train: `308,364`
  - Val: `64,095`
  - Test: `64,987`
  - Total: `437,446`
- **Stride 5**
  - Train: `62,023`
  - Val: `12,887`
  - Test: `13,071`
  - Total: `87,981`

---

## 10. Split-Level Class Distribution Tables

## 10.1 Sample counts per class (all files vs effective files)

Effective means `T >= 50`.

| Class | Train (all/effective) | Val (all/effective) | Test (all/effective) |
|---|---:|---:|---:|
| FLB | 70 / 70 | 15 / 15 | 15 / 15 |
| LLB | 71 / 71 | 15 / 15 | 15 / 15 |
| LOCA | 70 / 70 | 15 / 15 | 15 / 15 |
| LOCAC | 70 / 70 | 15 / 15 | 15 / 15 |
| LR | 71 / 71 | 14 / 14 | 14 / 14 |
| MD | 70 / 70 | 15 / 15 | 15 / 15 |
| RI | 70 / 55 | 15 / 13 | 15 / 14 |
| RW | 70 / 70 | 15 / 15 | 15 / 15 |
| SGATR | 70 / 70 | 15 / 15 | 15 / 15 |
| SGBTR | 78 / 78 | 16 / 16 | 16 / 16 |
| SLBIC | 71 / 71 | 15 / 15 | 15 / 15 |
| SLBOC | 70 / 70 | 15 / 15 | 15 / 15 |
| TT | 20 / 20 | 3 / 3 | 3 / 3 |

## 10.2 Window counts per class (stride=1, window=50)

| Class | Train windows | Val windows | Test windows |
|---|---:|---:|---:|
| FLB | 11,845 | 2,470 | 2,792 |
| LLB | 35,863 | 7,586 | 7,594 |
| LOCA | 24,051 | 5,072 | 4,994 |
| LOCAC | 28,110 | 6,069 | 6,097 |
| LR | 35,997 | 7,100 | 7,101 |
| MD | 37,672 | 8,404 | 8,044 |
| RI | 10,962 | 2,380 | 2,641 |
| RW | 9,641 | 1,878 | 1,969 |
| SGATR | 29,097 | 6,002 | 5,861 |
| SGBTR | 33,108 | 6,366 | 6,402 |
| SLBIC | 17,557 | 3,462 | 4,095 |
| SLBOC | 29,441 | 6,553 | 6,644 |
| TT | 5,020 | 753 | 753 |

---

## 11. Input Tensorization for the Neural Model

After preprocessing and windowing:

- Per window before transpose: `(I, F) = (50, 96)`
- Model input per item: `(F, I) = (96, 50)` for `Conv1d`
- Batch tensor shape: `(B, 96, 50)`

Label:

- Scalar integer in `[0, 12]`

---

## 12. Class Imbalance Analysis

## 12.1 Sample-level imbalance

- Maximum class files: `110` (SGBTR)
- Minimum class files: `26` (TT)
- Max/min sample ratio: `4.23x`

## 12.2 Window-level imbalance (train split, stride=1)

- Maximum class windows: `37,672` (MD)
- Minimum class windows: `5,020` (TT)
- Max/min window ratio: `7.50x`

Thus, imbalance is materially stronger at window level than sample level.

## 12.3 Inverse-frequency class weights (train windows, stride=1)

Using:

- `w_c = N / (C * n_c)`
- `N = 308,364` (train windows)
- `C = 13` classes

| Class | Weight |
|---|---:|
| FLB | 2.0026 |
| LLB | 0.6614 |
| LOCA | 0.9863 |
| LOCAC | 0.8438 |
| LR | 0.6590 |
| MD | 0.6297 |
| RI | 2.1639 |
| RW | 2.4604 |
| SGATR | 0.8152 |
| SGBTR | 0.7165 |
| SLBIC | 1.3510 |
| SLBOC | 0.8057 |
| TT | 4.7252 |

---

## 13. Warning Interpretation in Context

## 13.1 Schema-alignment warning

```text
Column schema varies across CSVs; aligned 25/1237 samples to a 96-column reference schema (added=0, dropped=75).
```

Meaning:

- Expected due to known `SLBIC` 99-feature files.
- Not a runtime failure.

## 13.2 Residual-null warning

```text
Residual null values ... filled with 0.
```

Meaning:

- Cleaning fallback executed after interpolation/edge fill.
- In this snapshot, missingness is extremely low but nonzero.

## 13.3 Lightning `_pytree` deprecation warning

Meaning:

- Library deprecation warning in Lightning internals.
- Not data corruption.

---

## 14. Reproducibility and Determinism Notes

1. Splits are deterministic for fixed `random_seed=42`.
2. Split is done at **file/sample level**, then windows are generated.
3. Scaling parameters are fit only on train split.
4. RI file names include negative stems (e.g., `-100.csv`), and file ordering follows loader sorting logic.
5. Effective train/val/test sample counts depend on `window_size` due `T < window_size` filtering.

---

## 15. Known Limitations for Research Reporting

1. **Class TT has low sample count** (26 files), yielding weaker statistical support.
2. **RI contributes short runs**, and 18 RI files are excluded at `window_size=50`.
3. **Schema heterogeneity exists** (SLBIC extra 3 channels), requiring alignment to 96 common features.
4. **Very low missingness** means robustness to heavy real-world sensor dropout is not strongly stress-tested by this snapshot.

---

## 16. Methods Text (Paper-Ready Draft)

Use/adapt this paragraph directly:

> We used the `Operation_csv_data` benchmark, comprising 1,237 multivariate time-series simulation runs across 13 accident classes (FLB, LLB, LOCA, LOCAC, LR, MD, RI, RW, SGATR, SGBTR, SLBIC, SLBOC, TT). Each CSV was treated as one sample, and rows represented time steps. After removing the TIME column, the reference feature dimensionality was 96. A known schema variation was present in 25 SLBIC files containing three additional variables (WPCS, WPMU, WPFW); all samples were aligned to the 96-feature reference schema. We applied sample-level stratified splitting (70/15/15 approximately), then fitted scaling parameters on train-only data to avoid leakage. Sliding windows of length 50 were extracted lazily with stride 1, yielding 308,364/64,095/64,987 windows for train/validation/test, respectively. Samples shorter than the window size were excluded at dataset build time (18 RI files). Missing values were rare (92 cells over 48.3M scanned cells) and were handled via interpolation plus residual null fallback filling.

---

## 17. Appendix A: 99-Feature Variant Files

The following files contain 99 features (96 baseline + WPCS/WPMU/WPFW):

- `data/Operation_csv_data/SLBIC/1.csv`
- `data/Operation_csv_data/SLBIC/2.csv`
- `data/Operation_csv_data/SLBIC/3.csv`
- `data/Operation_csv_data/SLBIC/4.csv`
- `data/Operation_csv_data/SLBIC/5.csv`
- `data/Operation_csv_data/SLBIC/6.csv`
- `data/Operation_csv_data/SLBIC/7.csv`
- `data/Operation_csv_data/SLBIC/8.csv`
- `data/Operation_csv_data/SLBIC/9.csv`
- `data/Operation_csv_data/SLBIC/10.csv`
- `data/Operation_csv_data/SLBIC/11.csv`
- `data/Operation_csv_data/SLBIC/12.csv`
- `data/Operation_csv_data/SLBIC/13.csv`
- `data/Operation_csv_data/SLBIC/14.csv`
- `data/Operation_csv_data/SLBIC/15.csv`
- `data/Operation_csv_data/SLBIC/16.csv`
- `data/Operation_csv_data/SLBIC/17.csv`
- `data/Operation_csv_data/SLBIC/18.csv`
- `data/Operation_csv_data/SLBIC/19.csv`
- `data/Operation_csv_data/SLBIC/20.csv`
- `data/Operation_csv_data/SLBIC/21.csv`
- `data/Operation_csv_data/SLBIC/22.csv`
- `data/Operation_csv_data/SLBIC/23.csv`
- `data/Operation_csv_data/SLBIC/24.csv`
- `data/Operation_csv_data/SLBIC/25.csv`
