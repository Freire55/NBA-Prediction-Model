# NBA Game Outcome Prediction & Quantitative Sports Analytics Pipeline

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![Code Style: Clean & Modular](https://img.shields.io/badge/code%20style-production%20ready-green.svg)]()
[![Validation: Pandera Contracts](https://img.shields.io/badge/data%20contracts-Pandera-yellow.svg)](https://pandera.readthedocs.io/)
[![Storage: Apache Parquet](https://img.shields.io/badge/storage-Apache%20Parquet-orange.svg)]()
[![Tests: Pytest Passing](https://img.shields.io/badge/tests-54%20passed-brightgreen.svg)]()

A production-grade, leak-free machine learning system for predicting NBA regular-season game outcomes strictly using information available prior to tip-off. 

The pipeline bridges sports domain modeling with rigorous machine learning engineering: **era-adjusted pace normalization**, **Dean Oliver Four Factors modeling**, **arena altitude & rest penalties**, **continuous Margin-of-Victory Elo simulation**, **learned player latent representations (PCA)**, **player volatility & star hierarchy modeling**, **heterogeneous feature routing**, **multi-stage sequential backward feature selection (SBS)**, **CatBoost symmetric oblivious trees**, **Beta & Platt probability calibration model selection**, **pace-modulated continuous margin CDF conversion**, and **5-model SLSQP-constrained meta-ensemble optimization**.

---

## Executive Summary & Results

The system is evaluated on every NBA regular season game from **2021 through present (6,140 held-out test games)**. All feature scalers, PCA projections, hyperparameter tuning, probability calibrations, and ensemble weights were fitted exclusively on prior historical data (**22,943 training games from 2000–2018** and **2,139 validation games from 2019–2020**).

| Model Architecture | Feature Representation | Test Accuracy | Log Loss | Brier Score | ROC-AUC | Ensemble Weight |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression** | Differentials (`DELTA_`) | 66.3% | 0.615 | 0.213 | 0.717 | 0.0% |
| **Pace Margin (CDF)** | Expected Margin + Normal CDF | 66.8% | 0.608 | 0.211 | 0.722 | 8.3% |
| **Deep Neural Net (MLP)** | Differentials + Latent (`EMBED_`) | 67.1% | 0.611 | 0.211 | 0.721 | 21.9% |
| **CatBoost (Beta-Calibrated)** | Symmetric Trees (`HOME_`, `AWAY_`, `EMBED_`) | **67.1%** | **0.599** | **0.207** | **0.733** | 2.0% |
| **XGBoost (Beta-Calibrated)** | Absolute (`HOME_`, `AWAY_`, `EMBED_`) | 66.9% | **0.599** | **0.207** | **0.734** | 67.7% |
| **Meta-Ensemble (SLSQP)** | **Optimal 5-Model Constrained Blend** | **67.3%** | **0.598** | **0.206** | **0.736** | **100.0%** |

*Key Takeaways:*
1. **Sub-0.600 Log Loss:** The 5-model meta-ensemble achieved **0.598 Log Loss, 0.206 Brier Score, and 0.736 ROC-AUC**, breaking the 0.600 log-loss threshold on 6,140 prospective test games.
2. **Beta Calibration Advantage:** Asymmetric Beta calibration significantly improved tree classifier probabilities, dropping XGBoost test log loss from 0.613 down to 0.599 and CatBoost to 0.599 by correcting favorite/underdog probability distortions.
3. **CatBoost Standalone Precision:** CatBoost achieved the highest individual single-model test accuracy at **67.15%**, demonstrating exceptional inductive bias on raw numerical team statistics.
4. **Pace Margin Diversity:** Continuous point differential conversion via tempo-modulated normal CDF claimed an **8.3% ensemble weight**, contributing valuable complementary signal to pure binary classification.

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
         • Continuous MOV-Elo engine                  8D latent vector representation
         • Dean Oliver Four Factors (eFG, TOV, ORB, FTR)           │
         • Dynamic Game Pace (Poss/48m)                            ▼
         • Arena Altitude & Rest Congestion          [Player Volatility & Lineup]
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
                       [Multi-Stage SBS Feature Selection]
                     Weakest-to-strongest importance ranking,
                   dual-guardrail calibration & frozen ensemble
                                             │
                                             ▼
                              [Heterogeneous Feature Routing]
         ┌───────────────┬───────────────────┬───────────────────┬───────────────┬───────────────┐
         ▼               ▼                   ▼                   ▼               ▼               ▼
      Linear           Trees          Symmetric Trees          Neural         Continuous       Tempo
    (LR: DELTA)    (XGB: H/A/EMB)      (CB: H/A/EMB)       (MLP: DELTA+EMB)  (Ridge/XGB/CB)  (Game Pace)
         │               │                   │                   │               │               │
         ▼               ▼                   ▼                   ▼               └───────┬───────┘
      TS-CV           TS-CV               TS-CV               TS-CV                      ▼
    GridSearch     RandomSearch        RandomSearch        RandomSearch           Normal CDF Bridge
         │               │                   │                   │              Phi(Margin/Sigma)
         ▼               ▼                   ▼               ▼                       │
    [Platt/Beta]    [Platt/Beta]        [Platt/Beta]        [Platt/Beta]                 │
     Calibration     Calibration         Calibration         Calibration                 │
         │               │                   │                   │                       │
         └───────────────┴───────────────────┼───────────────────┴───────────────────────┘
                                             ▼
                             [SLSQP 5-Model Meta-Ensemble]
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

### 3. Latent Player Representation Learning (8D PCA)
Traditional sports models aggregate raw player averages, which conflate player role with efficiency. This pipeline learns continuous player representations:
- **14 per-minute rate statistics:** Points, FGM, FGA, 3PM, 3PA, FTM, FTA, OREB, DREB, AST, STL, BLK, TOV, PF per minute.
- **8 advanced rate metrics:** True Shooting % (TS%), Effective Field Goal % (eFG%), Turnover %, Fantasy Score, Hollinger Game Score, Usage Proxy, Assist-to-Turnover Ratio, and Player Impact Estimate (PIE) proxy.
- Profiles are smoothed with an exponentially weighted moving average (half-life of 20 games, shifted by 1 game).
- A `StandardScaler` and `PCA` (fitted strictly on training seasons $\le 2018$) project these 22 metrics into an **8-dimensional latent embedding space** capturing ~85% of total historical playstyle variance across archetypes:
  1. *Scoring volume & primary shot creation*
  2. *Interior rim protection vs. perimeter spacing*
  3. *Playmaking efficiency & ball security*
  4. *Defensive activity & rebounding rate*
  5. *Free-throw generation & downhill pressure*
  6. *Perimeter shooting gravity & 3PT efficiency*
  7. *Turnover conservatism vs. high-risk passing*
  8. *Secondary playmaking & rotational wing hustle*
- Roster aggregation computes differential sum, mean, standard deviation, and maximums across active lineups (`EMBED_DELTA_`), quantifying tactical mismatches at tip-off.

### 4. Player Volatility, Star Hierarchy & Lineup Availability
Basketball is driven by top-end star talent, rotation depth, and lineup health:
- **Outlier Capping:** Individual Game Scores are capped at $\mu_{\text{rolling}} \pm 2.5 \sigma_{\text{volatility}}$ to reduce sensitivity to fluke performances.
- **Robust Expected Impact:** Penalizes high-variance performances: $\text{Impact} = \mu - 0.35 \sigma$.
- **Unified Star, Duo, Trio & Bench Hierarchy:** Computed using minutes-weighted positive expected impact ($\text{Form} \times \text{Minutes}$), strictly enforcing $0 \le \text{Star} \le \text{Top 2} \le \text{Top 3} \le 1.0$ and $\text{Top 3} + \text{Bench} = 1.0$. Differentials like `DELTA_ACTIVE_ROSTER_TOP_2_SHARE` and `DELTA_ACTIVE_ROSTER_BENCH_SHARE` quantify superteam concentration versus rotation depth.
- **Lineup Health & Availability Deficit:** Compares tonight's active roster production against the team's rolling 10-game roster baseline (`LINEUP_AVAILABILITY_RATIO` and `LINEUP_MISSING_PRODUCTION`), immediately alerting models when stars are resting or injured on back-to-backs.
- **Rolling Team Identity:** Tracks rolling 10-game EWMA concentration (`ROLLING_STAR_SHARE_10`, `ROLLING_TOP_2_SHARE_10`) to capture whether a team is structurally heliocentric or depth-oriented.
- **Acute Fatigue Surge:** Captures short-term minutes spikes over medium-term baselines (`FATIGUE_EWMA_MINUTES_3 - FATIGUE_EWMA_MINUTES_10`).

### 5. Dean Oliver Four Factors & Dynamic Pace Normalization
Dean Oliver's "Four Factors of Basketball Success" dictate 90%+ of NBA game outcomes:
- **Shooting (40% weight):** Effective Field Goal % ($eFG\% = \frac{\text{FGM} + 0.5 \times \text{3PM}}{\text{FGA}}$)
- **Turnovers (25% weight):** Turnover Rate ($TOV\% = \frac{\text{TOV}}{\text{FGA} + 0.44 \times \text{FTA} + \text{TOV}}$)
- **Rebounding (20% weight):** Offensive Rebound % ($OREB\% = \frac{\text{OREB}}{\text{OREB} + \text{OPP\_DREB}}$), computed via vectorized leak-free game pairing.
- **Free Throws (15% weight):** Free Throw Rate ($FTR = \frac{\text{FTA}}{\text{FGA}}$)
- **Game-Pace Normalization:** Measures possessions per 48 minutes ($Pace = \frac{\text{Possessions}}{\text{Team Mins}} \times 48$), tracking multi-horizon EWMA (spans 3, 5, 10) and expected game pace (`MATCHUP_EXPECTED_PACE`).

### 6. Arena Altitude, Acclimation & Schedule Fatigue
High-altitude environments like Denver (5,280 ft) and Salt Lake City (4,226 ft) impose severe physiological strain on unacclimated visiting teams:
- **Elevation Mapping (`data/team_altitudes.csv`):** Comprehensive arena elevation data for all 30 active franchises plus historical venues (Seattle KeyArena, Vancouver GM Place, Charlotte Coliseum, etc.).
- **Non-Linear Physiological Advantage:** Models oxygen deficit with a 1,000-foot threshold and acclimation dampening:
  $$\text{Advantage} = \max\left(0, \frac{\text{Alt}_{\text{home}} - \max(\text{Alt}_{\text{away}}, 1000)}{5280}\right)$$
  Visiting Denver from sea level (Miami, Boston) incurs maximum penalty, while visiting from altitude (Utah) incurs negligible effect.
- **Compound Fatigue Penalty (`ALTITUDE_B2B_PENALTY`):** Interacts altitude disadvantage with schedule congestion (`AWAY_B2B`), penalizing tired teams playing on back-to-backs at high elevation.

### 7. Heterogeneous Feature Selection & Routing
Rather than feeding an identical feature matrix to every model, the system leverages structural inductive biases:
- **Logistic Regression (`DELTA_`):** Receives 146 pre-computed home-minus-away differentials. Linear models lack interaction terms and benefit heavily from pre-differenced comparative metrics.
- **XGBoost (`HOME_`, `AWAY_`, `EMBED_`):** Receives 212 absolute team metrics, non-linear latent embeddings, and schedule features. Decision trees learn decision boundaries, threshold interactions, and feature ratios natively without requiring differencing.
- **Multi-Layer Perceptron (`DELTA_` + `EMBED_`):** Receives 199 features combining engineered team differentials with latent player embeddings, utilizing dense non-linear layers to model synergy between team metrics and latent personnel vectors.

### 8. Multi-Stage Sequential Backward Selection (SBS) Engine
High-dimensional sports feature sets suffer from collinearity, noise, and cross-architecture interference. Rather than naive global feature dropping, this project implements a specialized **Multi-Stage Sequential Backward Selection (SBS)** engine ([`optimize_features.py`](optimize_features.py)):
- **Cross-Validation Importance Sorting:** Ranks candidate features across `TimeSeriesSplit` cross-validation folds (using model coefficients, tree gain, and permutation importance) so candidates are tested from weakest to strongest.
- **Fast-Track Screening:** Uses frozen ensemble weights and pre-cached out-of-fold predictions to evaluate candidate drops in milliseconds without retraining uninvolved models.
- **Dual-Guardrail Calibration Verification:**
  1. *Validation Ensemble Loss:* Candidate feature pruning must improve or preserve overall ensemble cross-entropy log loss.
  2. *Isolated Standalone Safety:* Candidate pruning must not degrade the target base learner's standalone calibrated loss beyond a strict safety margin ($\Delta \le +0.0005$), preventing harmful model degradation.
- **Pruned Features:** Prunes noisy collinear features (e.g. redundant 3PT EWMA horizons in LR and redundant max embedding dims in XGB) while retaining all deep personnel synergy in the MLP.
- **Single-Boolean Configuration Toggle:** Pruned features are registered in [`training/config.py`](training/config.py) under `optimized_features_to_remove` and activated via a single flag:
  ```python
  prune_optimized_features: bool = True  # Toggle to False to instantly revert to full baseline features
  ```
  Can also be toggled via environment variable: `USE_OPTIMIZED_FEATURES=1 python train_models.py`.

### 9. Asymmetric Beta Probability Calibration & Model Selection
A model predicting a 70% win probability should win exactly 70 out of 100 times. In uncalibrated models (especially gradient boosted trees), log loss is distorted by overconfident tail predictions and asymmetric underdog/favorite variance:
- **Platt Sigmoid Calibration:** Standard logistic mapping:
  $$\text{logit}(P(Y=1|p)) = a \cdot \text{logit}(p) + c \quad (a \ge 0)$$
  Platt scaling assumes symmetric distortion around $p=0.50$ ($a = b$).
- **Beta Calibration (Kull et al., 2017) ([`training/calibration.py`](training/calibration.py)):** Parametric calibration based on Beta distributions:
  $$\text{logit}(P(Y=1|p)) = a \ln(p) - b \ln(1 - p) + c \quad (a \ge 0, b \ge 0)$$
  Relaxes the symmetry constraint, allowing independent scaling for heavy favorites ($p \to 1$) versus extreme underdogs ($p \to 0$).
- **Leak-Free Model Selection:** For each architecture, the pipeline computes out-of-fold cross-validation predictions across `TimeSeriesSplit` folds, fits both Platt and Beta calibrators, and selects whichever minimizes out-of-fold log loss. On tree models (XGBoost and CatBoost), Beta calibration reliably dominated Platt scaling ($b \approx 2.1 \times a$).

### 10. CatBoost & Symmetric Oblivious Decision Trees
Gradient boosted trees often overfit tabular sports data through greedy asymmetric split paths. CatBoost introduces **symmetric (oblivious) decision trees**:
- **Oblivious Architecture:** Every split at a given tree depth uses the exact same feature and threshold across all leaf nodes. This acts as an innate structural regularizer, dramatically reducing variance and eliminating overfitting on high-leverage box-score features.
- **Raw Level Numerical Processing:** CatBoost operates directly on absolute metrics (`HOME_`, `AWAY_`, `EMBED_`) without requiring manual feature differencing or z-score transforms, achieving the project's highest single-model accuracy (**67.15%**).
- **GPU/Multi-Core Optimization:** Optimized L2 leaf regularization and multi-threaded training (`thread_count=-1`).

### 11. Continuous Margin Modeling & Pace-Modulated Normal CDF Conversion
In binary classification ($y \in \{0, 1\}$), a 1-point buzzer-beater win and a 30-point blowout are treated identically, resulting in severe information loss. In sports analytics, **continuous point differential ($\Delta \text{PTS} = \text{HOME\_PTS} - \text{AWAY\_PTS}$)** possesses a far higher signal-to-noise ratio than raw win/loss outcomes:
- **Continuous Margin Regressors (`MarginRegressor` in [`training/margin.py`](training/margin.py)):** Fits regularized $L_2$ linear models (Ridge), XGBoost, and CatBoost on comparative differential features (`DELTA_`) using chronological `TimeSeriesSplit` cross-validation while tracking residual standard deviation ($\hat{\sigma}_0$).
- **Central Limit Possession Scaling:** Point differential variance scales directly with the number of possessions played in a game:
  $$\hat{\sigma}(\hat{\text{Pace}}) = \sigma_0 \times \sqrt{\frac{\hat{\text{Pace}}}{100.0}}$$
- **Gaussian Normal CDF Probability Bridge (`PaceModulatedMarginClassifier`):** Converts continuous margin predictions into calibrated win probabilities:
  $$P(\text{Home Win}) = \Phi\left(\frac{\hat{M}}{\hat{\sigma}(\hat{\text{Pace}})}\right)$$
- **Domain Invariance (Favorite Safety vs. Upset Volatility):**
  - In high-possession games (faster tempo), variance expands $\to$ underdog upset volatility increases.
  - In low-possession games (slow grind-it-out pace), variance contracts $\to$ favorite safety increases.
- **Full Scikit-Learn Compatibility:** The resulting converter adheres to scikit-learn's `ClassifierMixin` API, providing `.predict_proba()` and temperature parameter ($\sigma_0^*$) calibration via validation log-loss minimization.

### 12. 5-Model SLSQP Constrained Meta-Ensemble
Ensemble weights $\mathbf{w}$ are learned by directly minimizing cross-entropy log loss over the validation probability simplex across all five diverse inductive paradigms:
$$\min_{\mathbf{w}} -\frac{1}{N} \sum_{i=1}^{N} \left[ y_i \ln\left(\sum_{m=1}^{5} w_m \hat{p}_{m,i}\right) + (1 - y_i)\ln\left(1 - \sum_{m=1}^{5} w_m \hat{p}_{m,i}\right) \right]$$
$$\text{subject to} \quad \sum_{m=1}^{5} w_m = 1.0, \quad w_m \ge 0.0 \quad \forall m \in \{1, \dots, 5\}$$
- **Learned Blending Formula:**
  $$\hat{P}_{\text{Ensemble}} = 0.677 \cdot P_{\text{XGBoost}} + 0.219 \cdot P_{\text{MLP}} + 0.083 \cdot P_{\text{Margin}} + 0.020 \cdot P_{\text{CatBoost}} + 0.000 \cdot P_{\text{LR}}$$
- Combining tree splits, deep embeddings, and continuous margin distributions produces optimal probabilistic sharpness and resilience against out-of-distribution games.

---

## Strict Leakage Prevention & Data Contracts

Target and temporal leakage are catastrophic in sports modeling. This repository enforces multi-layered defensive safeguards:

1. **Strict Temporal Shifting:** All rolling aggregations (`rolling_mean`, `ewma`) enforce a strict `.shift(1)` lag. Game $T$ features have zero access to game $T$ outcomes.
2. **Automated Perturbation Invariance Tests ([`tests/leakage_test.py`](tests/leakage_test.py)):** An automated test suite artificially mutates game $T$'s post-game statistics (scoring 300 points) and asserts that game $T$'s pre-game features remain **100% bitwise identical** (`assert orig == pert, abs=1e-9`).
3. **Defensive Post-Game Blocklist:** [`training/data.py`](training/data.py) explicitly filters out any column containing post-game box-score stats (`PTS`, `FGM`, `PLUS_MINUS`, `PACE`, `MIN`, `WIN`, etc.) to prevent tree models from greedily picking up leakage.
4. **Pandera Schema Contracts ([`data/schemas.py`](data/schemas.py)):** Declarative schemas validate raw team box scores, player logs, and final ML feature matrices, asserting non-negative bounds, probability ranges, and zero unexpected NaNs.
5. **Robust Vectorized Opponent Pairing:** Opponent defensive rebounds and strength mapping are performed strictly via $O(1)$ vectorized `GAME_ID` grouping (`np.where(game_count == 2, game_strength_sum - own_strength, 0.0)`), eliminating string-matching fragility and cross-game leakage.

---

## Production Engineering & Reproducibility

- **Apache Parquet Storage (`pyarrow`):** Matchup datasets are serialized to columnar Parquet with Snappy compression, cutting disk space from **210 MB to 57 MB (72% reduction)** and reducing dataset loading time from **7.9s to 1.9s (4.1x speedup)**.
- **Automated Experiment Lineage:** Every training execution automatically generates `metadata.json` capturing:
  - Git Commit SHA & active branch
  - Dirty working tree flag
  - Input dataset SHA-256 fingerprint
  - Platform architecture & CPU core counts
- **Thread Contention Optimization:** Nested parallelism in cross-validation is explicitly managed (`n_jobs=1` per search estimator with `n_jobs=-1` at the CV fold level) to eliminate CPU cache thrashing.
- **Automated Test Suite:** 54 comprehensive unit tests covering data splitting, Elo mechanics, altitude advantage, Four Factors, pace normalization, schema validation, leakage prevention, star/duo/trio share hierarchy, ensemble optimization, continuous margin regression, pace-modulated normal CDF, base model contracts, and SBS feature selection.

---

## Project Structure

```text
nba-prediction-model/
├── data/
│   ├── schemas.py                       # Pandera data contracts & validation schemas
│   ├── team_altitudes.csv               # Arena elevation lookup table (30 teams + historical)
│   ├── raw_historical_nba.csv           # Raw team box scores (fetch_history.py)
│   ├── raw_player_game_logs.csv         # Raw player box scores (fetch_player_game_logs.py)
│   ├── era_adjusted_nba.csv             # Season-relative z-scores (era_adjustment.py)
│   ├── player_embeddings.csv            # 8D PCA player embeddings (generate_players_embedding.py)
│   ├── ml_ready_matchups.csv            # Team matchup features (feature_engineering.py)
│   └── ml_ready_matchups_players.parquet# Final ML dataset (feature_engineering_players.py)
│
├── training/
│   ├── calibration.py                   # Platt & Beta calibration routines with model selection
│   ├── config.py                        # TrainingConfig, metadata tracking, grids & SBS toggle
│   ├── data.py                          # Parquet/CSV ingestion & heterogeneous routing
│   ├── margin.py                        # Continuous margin regressor & pace-modulated CDF converter
│   ├── tuning.py                        # TimeSeriesSplit CV search & calibration
│   ├── training.py                      # Retraining on combined train+val sets
│   ├── ensemble.py                      # SLSQP log-loss constrained optimization
│   ├── evaluation.py                    # Metric calculation (Brier, Log Loss, AUC)
│   ├── explainability.py                # SHAP TreeExplainer & Permutation Importance
│   ├── plots.py                         # Calibration curves, ROC curves, confusion matrices
│   └── utils.py                         # JSON serialization & model unwrapping
│
├── tests/
│   ├── calibration_test.py              # Platt & Beta calibrators and out-of-fold tests
│   ├── data_test.py                     # Chronological split & routing tests
│   ├── ensemble_test.py                 # SLSQP simplex weight optimization tests
│   ├── feature_engineering_test.py     # Schedule, rest, Four Factors, Pace & Altitude tests
│   ├── leakage_test.py                  # Strict perturbation & shift invariance tests
│   ├── margin_test.py                   # Margin regression & pace CDF probability converter tests
│   ├── model_test.py                    # Base classification architecture tuning & estimator tests
│   ├── optimize_features_test.py        # Candidate deduplication, fast-track & SBS tests
│   ├── schema_test.py                   # Pandera schema enforcement tests
│   └── utility_test.py                  # Metadata reproducibility & feature toggle tests
│
├── models/
│   └── example_run/                     # Tracked reference run with pruned features & diagnostics
│
├── fetch_history.py                     # Scrapes team box scores via nba_api
├── fetch_player_game_logs.py            # Scrapes player game logs via nba_api
├── era_adjustment.py                    # Statistical era normalization
├── feature_engineering.py               # Four Factors, Pace, Altitude & Elo engine
├── generate_players_embedding.py        # 8D PCA latent representations
├── feature_engineering_players.py       # Player aggregation & star duo share
├── optimize_features.py                 # Multi-stage sequential backward selection engine
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
*Executes all 54 unit, leakage, margin, calibration, and integration tests in ~9 seconds.*

### 3. Feature Generation Pipeline
```bash
# 1. Scrape box scores (optional if raw data is present)
python fetch_history.py
python fetch_player_game_logs.py

# 2. Compute era adjustments
python era_adjustment.py

# 3. Simulate continuous Elo ratings, Four Factors, Pace, & Altitude
python feature_engineering.py

# 4. Generate 8D PCA player embeddings
python generate_players_embedding.py

# 5. Aggregate player features & produce final matchup dataset
python feature_engineering_players.py
```

### 4. Optional: Feature Selection Optimization (SBS)
```bash
python optimize_features.py
```
Runs the multi-stage SBS engine to identify redundant or noisy features using CV importance ranking and dual-guardrail validation checks.

### 5. Train Models & Generate Diagnostic Reports
```bash
python train_models.py
```
> [!TIP]
> By default, `train_models.py` activates the SBS-pruned feature configuration (`prune_optimized_features: bool = True` in [`training/config.py`](training/config.py)). To instantly evaluate on the unpruned full feature baseline, set `prune_optimized_features = False` in config or pass the environment override:
> ```bash
> USE_OPTIMIZED_FEATURES=0 python train_models.py
> ```

This executes the full pipeline:
- Ingests dataset via Parquet with zero-leakage data contracts
- Executes `TimeSeriesSplit` cross-validation for MLP, XGBoost, CatBoost, and Logistic Regression
- Fits continuous MarginRegressor and calibrates pace-modulated normal CDF converter
- Performs model-selected probability calibration (Platt Sigmoid vs. Asymmetric Beta Calibration)
- Solves SLSQP 5-model constrained ensemble weights over the probability simplex
- Retrains final models on combined train+val sets
- Evaluates on 6,140 held-out prospective test games
- Generates SHAP summary plots, calibration curves, and feature rankings in `models/run_<timestamp>/`

---

## Explainability & Diagnostic Artifacts

Every experiment run exports full diagnostic figures and tabular metadata:
- **`01_model_comparison.csv`**: Comprehensive test metrics breakdown across all 5 architectures and meta-ensemble.
- **`02_calibration_curve.png`**: Reliability diagram showing calibrated probability vs empirical win frequency.
- **`03_roc_curve.png`**: Multi-model ROC curves with AUC scores.
- **`04_confusion_matrix.png`**: Normalized confusion matrix on unseen test seasons.
- **`05_xgb_feature_importance.png`**: XGBoost gain-based feature ranking.
- **`06_mlp_feature_importance.png`**: Neural network permutation importance.
- **`07_lr_coefficients.png`**: Linear regression coefficient impact ranking.
- **`08_xgb_shap_summary.png`**: Global SHAP beeswarm plot displaying non-linear feature attribution.
- **`09_margin_coefficients.png`**: Point differential feature weights from continuous margin regression.
- **`10_margin_residuals.png`**: Residual error diagnostic distribution vs. theoretical Gaussian curve.
- **`11_catboost_feature_importance.png`**: Native split importance from CatBoost symmetric oblivious trees.
- **`calibration_model_selection.json`**: Out-of-fold log-loss comparison selecting between Platt and Beta calibration.
- **`ensemble_formula.json`**: Learned SLSQP blending weights across all 5 models.

---

## Tech Stack

- **Data Engineering:** `pandas`, `numpy`, `pyarrow`, `nba_api`, `pandera`
- **Modeling & Optimization:** `scikit-learn`, `xgboost`, `catboost`, `scipy` (SLSQP optimization, L-BFGS-B calibration)
- **Explainability & Diagnostics:** `shap`, `matplotlib`
- **Quality Assurance:** `pytest`, `typeguard`
