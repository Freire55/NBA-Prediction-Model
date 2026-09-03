# NBA Game Outcome Prediction & Quantitative Sports Analytics Pipeline

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![Code Style: Clean & Modular](https://img.shields.io/badge/code%20style-production%20ready-green.svg)]()
[![Validation: Pandera Contracts](https://img.shields.io/badge/data%20contracts-Pandera-yellow.svg)](https://pandera.readthedocs.io/)
[![Storage: Apache Parquet](https://img.shields.io/badge/storage-Apache%20Parquet-orange.svg)]()
[![Tests: Pytest Passing](https://img.shields.io/badge/tests-18%20passed-brightgreen.svg)]()

A production-grade, leak-free machine learning system for predicting NBA regular-season game outcomes strictly using information available prior to tip-off. 

The pipeline bridges sports domain modeling with rigorous machine learning engineering: **era-adjusted pace normalization**, **continuous Margin-of-Victory Elo simulation**, **learned player latent representations (PCA)**, **player volatility & star concentration modeling**, **heterogeneous feature routing**, **chronological cross-validation**, **probability calibration**, and **SLSQP-constrained ensemble optimization**.

---

## Executive Summary & Results

The system is evaluated on every NBA regular season game from **2021 through present (6,140 held-out test games)**. All feature scalers, PCA projections, hyperparameter tuning, probability calibrations, and ensemble weights were fitted exclusively on prior historical data (**22,943 training games from 2000–2018** and **2,139 validation games from 2019–2020**).

| Model Architecture | Feature Representation | Test Accuracy | Log Loss | Brier Score | ROC-AUC | Ensemble Weight |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression** | Differentials (`DELTA_`) | 67.2% | 0.609 | 0.211 | 0.723 | 6.0% |
| **XGBoost (Hist)** | Absolute (`HOME_`, `AWAY_`) | 66.4% | 0.616 | 0.214 | 0.721 | 39.3% |
| **Deep Neural Net (MLP)** | Differentials + Latent (`EMBED_`) | 67.6% | 0.610 | 0.210 | 0.724 | 54.6% |
| **Meta-Ensemble (SLSQP)** | **Optimal Constrained Blend** | **67.6%** | **0.606** | **0.209** | **0.732** | **100.0%** |

*Key Takeaway:* The constrained validation ensemble achieves the highest overall probability calibration and discrimination, outperforming individual base learners across **log loss (0.606), Brier score (0.209), and ROC-AUC (0.732)** while achieving **67.6% accuracy**, successfully capturing non-linear interactions between player representations and team efficiency profiles.

---

## Core Machine Learning Architecture

```
                                      [NBA Stats API]
                                             │
                                             ▼
                             Raw Historical Game Logs & Box Scores
                                 (62,500+ Team Logs | 150,000+ Player Logs)
                                             │
                       ┌─────────────────────┴─────────────────────┐
                       ▼                                           ▼
             [Era Normalization]                         [Player Representation]
         Season-relative z-scores                      14 per-min rate stats + 8 adv
         eliminating pace inflation                    EWMA leak-free smoothing (α=20)
                       │                                           │
                       ▼                                           ▼
           [Team Feature Pipeline]                         [PCA Projection]
         • Continuous MOV-Elo engine                  4D latent vector representation
         • EWMA Offensive Ratings (span 3,5,10)                    │
         • Schedule density (B2B, 3-in-4)                          ▼
                       │                             [Player Volatility & Lineup]
                       │                             • Capped Game Score (μ ± 2.5σ)
                       │                             • Robust Expected Impact (μ - 0.35σ)
                       │                             • Star Duo Share (Top 2 Concentration)
                       │                                           │
                       └─────────────────────┬─────────────────────┘
                                             ▼
                             [Data Contracts & Schema Gate]
                           Pandera validation: range checks,
                         types, nullability & leakage assertion
                                             │
                                             ▼
                             [Chronological Data Partition]
                          Train: 2000–2018 (22,943 games)
                          Val:   2019–2020 ( 2,139 games)
                          Test:  2021–Pres  ( 6,140 games)
                                             │
                                             ▼
                             [Heterogeneous Feature Routing]
                  ┌──────────────────────────┼──────────────────────────┐
                  ▼                          ▼                          ▼
        Linear Representation      Non-Linear Trees           Neural Latent Vector
            (LR: DELTA_)            (XGB: HOME/AWAY)         (MLP: DELTA_ + EMBED_)
                  │                          │                          │
                  ▼                          ▼                          ▼
         TimeSeriesSplit CV         TimeSeriesSplit CV         TimeSeriesSplit CV
         GridSearchCV (C, solver)   RandomizedSearch (hist)    RandomizedSearch (adaptive)
                  │                          │                          │
                  ▼                          ▼                          ▼
         Platt / Sigmoid Calib      Platt / Sigmoid Calib      Platt / Sigmoid Calib
                  │                          │                          │
                  └──────────────────────────┼──────────────────────────┘
                                             ▼
                             [SLSQP Constrained Ensemble]
                               min Log Loss s.t. Σw = 1, w ≥ 0
                                             │
                                             ▼
                             [Explainability & Diagnostics]
                            SHAP TreeExplainer, Permutation
                            Importance, Reliability Diagrams
```

---

## Key Technical Highlights

### 1. Era-Adjustment & Pace Standardization
NBA basketball is non-stationary: scoring average surged from ~93 PPG in 2003 to ~115 PPG in 2024 due to the three-point revolution and pace changes. A 105-point performance in 2004 was elite, whereas in 2024 it represents poor offensive output.
- Every numeric box-score metric is converted to **season-relative z-scores**:
  $$Z_{i,s} = \frac{X_{i,s} - \mu_s}{\sigma_s}$$
  where $\mu_s$ and $\sigma_s$ are computed strictly over the corresponding season $s$. This allows tree and linear models to compare performances across eras without covariate shift.

### 2. Continuous Margin-of-Victory (MOV) Elo Engine
Rather than relying on discrete win/loss tracking, team strength is continuously simulated using an upgraded FiveThirtyEight-style Elo engine:
- **Expected Win Probability:**
  $$P(\text{Home}) = \frac{1}{1 + 10^{-(\text{Elo}_{\text{home}} - \text{Elo}_{\text{away}}) / 400}}$$
- **Margin-of-Victory Blowout Multiplier:**
  $$\text{Multiplier} = \ln(|\text{MOV}| + 1) \times \frac{2.2}{((\text{Elo}_{\text{winner}} - \text{Elo}_{\text{loser}}) \times 0.001) + 2.2}$$
  Dampens ratings changes for expected blowouts by heavy favorites while amplifying underdog upsets.
- **Inter-Season Mean Reversion:**
  Teams regress 25% toward the league mean between seasons to reflect roster turnover:
  $$\text{Elo}_{\text{new\_season}} = 0.75 \times \text{Elo}_{\text{prev}} + 0.25 \times 1500$$
- **High-Performance Vectorization:** Executed via vectorized home/away pairing and native NumPy array iteration, processing 62,500 games in **0.27 seconds** (an 80x speedup over standard row-by-row iteration).

### 3. Latent Player Representation Learning (PCA)
Traditional sports models aggregate raw player averages, which conflate player role with efficiency. This pipeline learns continuous player representations:
- **14 per-minute rate statistics:** Points, FGM, FGA, 3PM, 3PA, FTM, FTA, OREB, DREB, AST, STL, BLK, TOV, PF per minute.
- **8 advanced rate metrics:** True Shooting % (TS%), Effective Field Goal % (eFG%), Turnover %, Fantasy Score, Hollinger Game Score, Usage Proxy, Assist-to-Turnover Ratio, and Player Impact Estimate (PIE) proxy.
- Profiles are smoothed with an exponentially weighted moving average (half-life of 20 games, shifted by 1 game).
- A `StandardScaler` and `PCA` (fitted strictly on training seasons $\le 2018$) project these 22 metrics into a **4-dimensional latent embedding space** capturing:
  1. *Scoring volume & primary creation*
  2. *Interior rim protection vs. perimeter spacing*
  3. *Playmaking efficiency & ball security*
  4. *Defensive activity & rebounding rate*

### 4. Player Volatility & Star Duo Concentration
Basketball is driven by top-end star talent and rotation dynamics:
- **Outlier Capping:** Individual Game Scores are capped at $\mu_{\text{rolling}} \pm 2.5 \sigma_{\text{volatility}}$ to reduce sensitivity to fluke performances.
- **Robust Expected Impact:** Penalizes high-variance performances: $\text{Impact} = \mu - 0.35 \sigma$.
- **Star Duo Concentration:** Measures the share of total expected team production accounted for by the top-2 contributors (`TOP_2_IMPACT_SHARE`). The differential `DELTA_ACTIVE_ROSTER_TOP_2_SHARE` directly signals whether a game is a "Superteam vs. Balanced Depth" matchup.

### 5. Heterogeneous Feature Selection & Routing
Rather than feeding an identical feature matrix to every model, the system leverages structural inductive biases:
- **Logistic Regression (`DELTA_`):** Receives 111 pre-computed home-minus-away differentials. Linear models lack interaction terms and benefit heavily from pre-differenced comparative metrics.
- **XGBoost (`HOME_`, `AWAY_`):** Receives 101 absolute team metrics with non-era-normalized values. Decision trees learn decision boundaries and feature ratios natively without requiring differencing.
- **Multi-Layer Perceptron (`DELTA_` + `EMBED_`):** Receives 135 features combining engineered team differentials with latent player embeddings, utilizing dense non-linear layers to model synergy between team metrics and latent personnel vectors.

### 6. Probability Calibration & Constrained Ensemble
A model predicting a 70% win probability should win exactly 70 out of 100 times. In uncalibrated models (especially gradient boosted trees), log loss is distorted by overconfident tail predictions.
- **Cross-Validated Sigmoid Calibration:** Every base estimator is calibrated using Platt scaling (`CalibratedClassifierCV`) during cross-validation.
- **SLSQP Ensemble Formulation:** Ensemble weights $\mathbf{w}$ are learned by directly minimizing cross-entropy log loss on out-of-fold validation predictions:
  $$\min_{\mathbf{w}} -\frac{1}{N} \sum_{i=1}^{N} \left[ y_i \ln\left(\sum_{m} w_m \hat{p}_{m,i}\right) + (1 - y_i)\ln\left(1 - \sum_{m} w_m \hat{p}_{m,i}\right) \right]$$
  $$\text{subject to} \quad \sum_{m=1}^{M} w_m = 1, \quad w_m \ge 0 \quad \forall m$$

---

## Strict Leakage Prevention & Data Contracts

Target and temporal leakage are catastrophic in sports modeling. This repository enforces multi-layered defensive safeguards:

1. **Strict Temporal Shifting:** All rolling aggregations (`rolling_mean`, `ewma`) enforce a strict `.shift(1)` lag. Game $T$ features have zero access to game $T$ outcomes.
2. **Automated Perturbation Invariance Tests ([`tests/leakage_test.py`](tests/leakage_test.py)):** An automated test suite artificially mutates game $T$'s post-game statistics (scoring 300 points) and asserts that game $T$'s pre-game features remain **100% bitwise identical** (`assert orig == pert, abs=1e-9`).
3. **Defensive Post-Game Blocklist:** [`training/data.py`](training/data.py) explicitly filters out any column containing post-game box-score stats (`PTS`, `FGM`, `PLUS_MINUS`, `WIN`, etc.) to prevent tree models from greedily picking up leakage.
4. **Pandera Schema Contracts ([`data/schemas.py`](data/schemas.py)):** Declarative schemas validate raw team box scores, player logs, and final ML feature matrices, asserting non-negative bounds, probability ranges, and zero unexpected NaNs.

---

## Production Engineering & Reproducibility

- **Apache Parquet Storage (`pyarrow`):** Matchup datasets are serialized to columnar Parquet with Snappy compression, cutting disk space from **210 MB to 57 MB (72% reduction)** and reducing dataset loading time from **7.9s to 1.9s (4.1x speedup)**.
- **Automated Experiment Lineage:** Every training execution automatically generates `metadata.json` capturing:
  - Git Commit SHA & active branch
  - Dirty working tree flag
  - Input dataset SHA-256 fingerprint
  - Platform architecture & CPU core counts
- **Thread Contention Optimization:** Nested parallelism in cross-validation is explicitly managed (`n_jobs=1` per search estimator with `n_jobs=-1` at the CV fold level) to eliminate CPU cache thrashing.
- **Automated Test Suite:** 18 comprehensive unit tests covering data splitting, Elo mechanics, schema validation, leakage prevention, ensemble optimization, and feature selection.

---

## Project Structure

```text
nba-prediction-model/
├── data/
│   ├── schemas.py                       # Pandera data contracts & validation schemas
│   ├── raw_historical_nba.csv           # Raw team box scores (fetch_history.py)
│   ├── raw_player_game_logs.csv         # Raw player box scores (fetch_player_game_logs.py)
│   ├── era_adjusted_nba.csv             # Season-relative z-scores (era_adjustment.py)
│   ├── player_embeddings.csv            # PCA player embeddings (generate_players_embedding.py)
│   ├── ml_ready_matchups.csv            # Team matchup features (feature_engineering.py)
│   └── ml_ready_matchups_players.parquet# Final ML dataset (feature_engineering_players.py)
│
├── training/
│   ├── config.py                        # TrainingConfig, metadata tracking & grids
│   ├── data.py                          # Parquet/CSV ingestion & heterogeneous routing
│   ├── tuning.py                        # TimeSeriesSplit CV search & calibration
│   ├── training.py                      # Retraining on combined train+val sets
│   ├── ensemble.py                      # SLSQP log-loss constrained optimization
│   ├── evaluation.py                    # Metric calculation (Brier, Log Loss, AUC)
│   ├── explainability.py                # SHAP TreeExplainer & Permutation Importance
│   ├── plots.py                         # Calibration curves, ROC curves, confusion matrices
│   └── utils.py                         # JSON serialization & model unwrapping
│
├── tests/
│   ├── data_test.py                     # Chronological split & routing tests
│   ├── ensemble_test.py                 # SLSQP weight optimization tests
│   ├── feature_engineering_test.py     # Schedule, rest & Elo direction tests
│   ├── leakage_test.py                  # Strict perturbation & shift invariance tests
│   ├── model_test.py                    # Classifier calibration tests
│   ├── optimize_features_test.py        # Candidate deduplication & fast-track tests
│   ├── schema_test.py                   # Pandera schema enforcement tests
│   └── utility_test.py                  # Metadata reproducibility & JSON tests
│
├── fetch_history.py                     # Scrapes team box scores via nba_api
├── fetch_player_game_logs.py            # Scrapes player game logs via nba_api
├── era_adjustment.py                    # Statistical era normalization
├── feature_engineering.py               # Elo engine & rolling team statistics
├── generate_players_embedding.py        # Batched EWMA & PCA latent representations
├── feature_engineering_players.py       # Player aggregation & star duo share
├── optimize_features.py                 # Multi-stage sequential backward selection
├── train_models.py                      # Primary end-to-end training pipeline
├── requirements.txt                     # Core dependencies
└── README.md
```

---

## Quickstart & Execution

### 1. Environment Setup
```bash
# Clone the repository
git clone https://github.com/Freire55/NBA-Prediction-Model.git
cd NBA-Prediction-Model

# Create and activate virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
pip install pyarrow pandera
```

### 2. Run Automated Verification Suite
```bash
pytest
```
*Executes all 18 unit and leakage tests in ~3 seconds.*

### 3. Feature Generation Pipeline
```bash
# 1. Scrape box scores (optional if raw data is present)
python fetch_history.py
python fetch_player_game_logs.py

# 2. Compute era adjustments
python era_adjustment.py

# 3. Simulate continuous Elo ratings & team rolling stats
python feature_engineering.py

# 4. Generate PCA player embeddings
python generate_players_embedding.py

# 5. Aggregate player features & produce final matchup dataset
python feature_engineering_players.py
```

### 4. Train Models & Generate Explainability Reports
```bash
python train_models.py
```
This executes the full pipeline:
- Ingests dataset via Parquet
- Executes `TimeSeriesSplit` cross-validation for MLP, XGBoost, and Logistic Regression
- Calibrates probability distributions via Platt scaling
- Solves SLSQP constrained ensemble weights
- Retrains final models on combined train+val sets
- Evaluates on 6,140 held-out test games
- Generates SHAP summary plots, calibration curves, and feature rankings in `models/run_<timestamp>/`

---

## Explainability & Diagnostic Artifacts

Every experiment run exports full diagnostic figures:
- **`01_model_comparison.csv`**: Comprehensive test metrics breakdown.
- **`02_calibration_curve.png`**: Reliability diagram showing calibrated probability vs empirical win frequency.
- **`03_roc_curve.png`**: Multi-model ROC curves with AUC scores.
- **`04_confusion_matrix.png`**: Normalized confusion matrix on unseen test seasons.
- **`05_xgb_feature_importance.png`**: XGBoost gain-based feature ranking.
- **`06_mlp_feature_importance.png`**: Neural network permutation importance.
- **`07_lr_coefficients.png`**: Linear regression coefficient impact ranking.
- **`08_xgb_shap_summary.png`**: Global SHAP beeswarm plot displaying non-linear feature attribution.

---

## Tech Stack

- **Data Engineering:** `pandas`, `numpy`, `pyarrow`, `nba_api`, `pandera`
- **Modeling & Optimization:** `scikit-learn`, `xgboost`, `scipy` (SLSQP optimization)
- **Explainability & Diagnostics:** `shap`, `matplotlib`
- **Quality Assurance:** `pytest`, `typeguard`
