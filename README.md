# NBA Game Outcome Prediction Using Machine Learning

A fully chronological, leak-free machine learning pipeline for predicting NBA game outcomes from historical team and player statistics.

The project predicts NBA regular-season games using only information that would have been available before tip-off. It combines feature engineering, player representation learning, time-aware model selection, probability calibration, ensemble learning, and explainability into a reproducible end-to-end workflow.

Rather than chasing raw accuracy through aggressive feature selection or data leakage, the project is built around sound machine learning engineering practice, and mirrors how a model would need to behave if it were run in production.

---

## Results

Final evaluation was performed on every NBA regular-season game from 2021 onward (6,140 test games), using models trained exclusively on historical seasons (22,943 training games from 2000–2018, 2,139 validation games from 2019–2020).

| Model                | Accuracy | Log Loss | Brier Score | ROC-AUC |
| --------------------- | -------: | -------: | ----------: | ------: |
| Logistic Regression   |   67.2%  |   0.609  |    0.211    |  0.723  |
| XGBoost                |   66.4%  |   0.615  |    0.214    |  0.721  |
| MLP                    |   67.3%  |   0.609  |    0.210    |  0.724  |
| **Ensemble**           | **67.4%**| **0.605**|  **0.209**  |**0.732**|

The final ensemble combines calibrated Logistic Regression, XGBoost, and Multi-Layer Perceptron (MLP) models using validation-set weight optimization, and outperforms every individual base model on every metric.

---

## Highlights

* Fully leak-free, chronological pipeline from raw box scores to final predictions
* Team- and player-level feature engineering with shifted rolling windows and EWMA smoothing
* Learned player embeddings via PCA on per-minute performance profiles
* Era-adjusted statistical normalization (season-relative z-scores)
* Continuous, chronologically simulated Elo rating system
* Heterogeneous feature sets: Logistic Regression, XGBoost, and a Multi-Layer Perceptron each see a different representation of the same game
* Time-aware hyperparameter optimization (`RandomizedSearchCV` / `GridSearchCV` with `TimeSeriesSplit`)
* Cross-validated probability calibration (sigmoid / Platt scaling)
* Validation-optimized ensemble blending (constrained log-loss minimization)
* SHAP, permutation importance, and coefficient-based explainability
* Model-specific sequential backward feature selection for pruning weak features per model
* Automatic experiment tracking and model serialization

---

## Motivation

Predicting professional basketball games is a difficult supervised learning problem. Team strength changes continuously due to injuries, roster moves, player development, scheduling effects, fatigue, and a large amount of variance that no feature set can fully capture. Many publicly available NBA prediction projects unintentionally introduce target leakage through improper feature engineering or random train/test splits, which produces unrealistically optimistic performance estimates.

This project follows one guiding rule:

> Every prediction must be generated using only information that existed before the game began.

The goal is not only to produce competitive predictions, but to demonstrate rigorous machine learning engineering: reproducible experiments, honest evaluation, and models whose behavior can be explained.

---

## Pipeline

```text
NBA Stats API
        |
        v
Historical Team & Player Game Logs
        |
        v
Era Normalization (season-relative z-scores)
        |
        v
Leak-Free Team Features (rolling stats, Elo, schedule/fatigue)
        |
        v
Player Embeddings (PCA on per-minute performance profiles)
        |
        v
Leak-Free Player Features (merged with embeddings, aggregated to team level)
        |
        v
Chronological Train / Validation / Test Split
        |
        v
Hyperparameter Optimization (RandomizedSearchCV / GridSearchCV + TimeSeriesSplit)
        |
        v
Probability Calibration (sigmoid)
        |
        v
Validation-Based Ensemble Weight Optimization
        |
        v
[Optional] Model-Specific Feature Pruning (Sequential Backward Selection)
        |
        v
Retraining on Full Historical Data
        |
        v
Evaluation & Explainability
        |
        v
Serialized Models & Reports
```

---

## Repository Structure

```text
nba-prediction-model/
|
├── data/
│   ├── raw_historical_nba.csv          # Raw team box scores (fetch_history.py)
│   ├── raw_player_game_logs.csv        # Raw player box scores (fetch_player_game_logs.py)
│   ├── era_adjusted_nba.csv            # Season-relative z-scores (era_adjustment.py)
│   ├── player_embeddings.csv           # PCA player embeddings (generate_players_embedding.py)
│   ├── ml_ready_matchups.csv           # Team-level matchup features (feature_engineering.py)
│   └── ml_ready_matchups_players.csv   # Final training dataset (feature_engineering_players.py)
│
├── training/
│   ├── __init__.py
│   ├── config.py            # Central configuration, dataclasses, search spaces
│   ├── data.py               # Dataset loading, feature selection, chronological splits
│   ├── tuning.py              # Hyperparameter search + calibration
│   ├── training.py            # Final retraining on train + validation
│   ├── ensemble.py            # Validation-optimized ensemble weighting
│   ├── evaluation.py          # Metrics computation and reporting
│   ├── explainability.py      # SHAP, permutation importance, coefficients
│   ├── plots.py               # ROC, calibration, confusion matrix, bar charts
│   └── utils.py               # Logging, serialization, pipeline timing
│
├── fetch_history.py                  # Downloads historical team box scores
├── fetch_player_game_logs.py         # Downloads historical player box scores
├── era_adjustment.py                 # Season-relative statistical normalization
├── feature_engineering.py            # Team-level rolling stats, Elo, schedule features
├── generate_players_embedding.py     # PCA-based player embedding generation
├── feature_engineering_players.py    # Player-level features, merged into final dataset
├── optimize_features.py              # Model-specific sequential backward feature selection
├── train_models.py                   # Main pipeline entry point
├── a_explanation.md                  # Extended module-by-module technical write-up
├── requirements.txt
└── README.md
```

---

## Dataset

Historical data is collected directly from the official NBA Stats API using `nba_api`.

The dataset contains every NBA regular-season game from 2000 onward, including:

* Team box scores
* Player box scores
* Game dates and matchups
* Team and player identifiers

No betting odds, proprietary datasets, or manually curated information are used.

---

## Feature Engineering

Every feature is computed exclusively from information available before each game. Rolling and exponentially-weighted statistics are always shifted by one game so the current game is never included in its own inputs.

### Team Features (`feature_engineering.py`)

* Continuous Elo ratings, simulated chronologically game-by-game
* Rolling and EWMA-smoothed (span 3 / 5 / 10) offensive rating, derived from a possessions estimate (`FGA + 0.44 × FTA − OREB + TOV`)
* Rolling and EWMA-smoothed era-normalized (z-scored) values for every numeric box-score statistic — shooting splits, rebounds, assists, steals, blocks, turnovers, fouls, and plus/minus
* Rolling and EWMA-smoothed strength of schedule, based on the opponent's own pre-game z-scored plus/minus
* Schedule density flags: back-to-backs, 3-games-in-4-nights, 4-games-in-5-nights
* Road trip length and rest-day advantage

### Player Features (`generate_players_embedding.py` + `feature_engineering_players.py`)

* Leak-free rolling Game Score (Hollinger's formula), smoothed over 3/5/10-game windows
* EWMA-smoothed expected playing time (fatigue-aware, spans of 3/5/10 games)
* Expected player impact (recent form × expected minutes), aggregated to the team level as a sum, standard deviation, and max
* Star-player concentration (share of a team's aggregate expected impact coming from its single largest contributor)
* Playing-time-weighted, team-aggregated player embeddings (see below)

All player-level statistics are computed per player-game, then aggregated from individual game logs to the team level before being merged into the matchup dataset.

---

## Player Representation Learning

Instead of representing players only through aggregate box-score statistics, the project learns compact player representations using dimensionality reduction.

For every player-game appearance:

* Fourteen per-minute rate statistics are calculated (points, field goals made/attempted, three-pointers made/attempted, free throws made/attempted, offensive/defensive rebounds, assists, steals, blocks, turnovers, and fouls, each divided by minutes played).
* Eight additional advanced metrics are computed: True Shooting %, Effective Field Goal %, Turnover %, a fantasy scoring formula, Hollinger Game Score, a per-minute usage proxy, an Assist-to-Turnover ratio, and a per-minute Player Impact Estimate (PIE) proxy.
* All twenty-two resulting statistics are smoothed with a leak-free exponentially weighted moving average (halflife of 20 games, using only prior games) to summarize each player's recent form.
* A `StandardScaler` and `PCA`, both fit exclusively on historical training seasons, compress these smoothed profiles into a four-dimensional latent embedding space.

The embeddings are:

* generated without using any future information,
* weighted by each player's expected playing time when aggregated to the team level,
* aggregated into both a playing-time-weighted mean and simple max/std summaries per team,
* turned into home/away delta features (`EMBED_DELTA_*`, `EMBED_RAW_DELTA_*`) at the matchup level.

Only these `EMBED_`-prefixed delta features are exposed to the models; the intermediate per-team embedding columns are dropped before training so that XGBoost's raw `HOME_`/`AWAY_` feature set doesn't unintentionally absorb them. The MLP is the only model configured to use the embedding features directly, since neural networks are best suited to learning nonlinear structure from continuous latent representations.

---

## Machine Learning Models

Three classifiers are trained, each on a different representation of the same game — a form of heterogeneous feature selection defined centrally in `training/config.py`:

* **Logistic Regression** — trained on engineered `DELTA_` (home-minus-away) features. As a linear model, it benefits most from relative, pre-differenced comparisons between the two teams.
* **XGBoost** — trained on the raw (non-era-normalized) `HOME_` and `AWAY_` feature sets for each team independently, explicitly excluding era-normalized `_Z_` columns. Tree-based models can learn feature interactions and scale differences directly, so no manual differencing is required.
* **Multi-Layer Perceptron (MLP)** — trained on the `DELTA_` features plus the player embedding (`EMBED_`) features, allowing it to combine engineered statistics with learned player representations.

Hyperparameters are optimized using `RandomizedSearchCV` (MLP, XGBoost) or `GridSearchCV` (Logistic Regression), all using `TimeSeriesSplit`, so every validation fold occurs strictly after its corresponding training data.

All three models are subsequently calibrated using cross-validated sigmoid (Platt) scaling before ensemble weighting.

In the reference run, this heterogeneous selection produced 111 features for Logistic Regression, 100 for XGBoost, and 132 for the MLP (which additionally carries the embedding features), with best cross-validated log losses of 0.5936, 0.5917, and 0.5974 respectively before calibration.

---

## Ensemble Learning

Instead of averaging predictions equally, the ensemble learns model weights by minimizing validation log loss, subject to the constraints that weights are non-negative and sum to one (solved with SciPy's SLSQP optimizer).

The learned ensemble from the reference run above:

| Model                | Weight |
| --------------------- | -----: |
| MLP                   |  58.1% |
| XGBoost                |  41.9% |
| Logistic Regression    |   0.0% |

In this run the optimizer assigned no weight to Logistic Regression, meaning its validation predictions added nothing beyond what MLP and XGBoost already captured together. This weighting reflects the complementary strengths of each calibrated model rather than assuming equal importance, and will vary between runs as the feature set and hyperparameters evolve — earlier reference runs, for example, have favored the MLP much more heavily (around 70%+) with a small but non-zero Logistic Regression weight.

---

## Feature Optimization

`optimize_features.py` implements a model-specific sequential backward selection routine that prunes weak features independently for the MLP, XGBoost, and Logistic Regression feature sets. For each candidate feature, it uses a fast partial retrain (only the affected model, with cached predictions for the others) to screen candidates, then confirms genuine improvements with a full recalibration and ensemble re-optimization before accepting a removal. This produces the `features_to_remove` lists consumed by `training/config.py`, letting each model carry only the features that measurably help it.

---

## Explainability

Every training run automatically generates explainability reports, including:

* SHAP summary plot for XGBoost
* Permutation importance for the MLP
* Logistic Regression coefficients (absolute value ranking)
* XGBoost built-in feature importance
* ROC curve, calibration curve, and confusion matrix for the final ensemble
* A combined feature ranking table merging all three models' importance scores

These artifacts explain both global feature importance and individual model behavior.

---

## Evaluation Strategy

The dataset is split strictly chronologically:

| Dataset    | Seasons      |
| ---------- | ------------ |
| Training   | 2000–2018    |
| Validation | 2019–2020    |
| Testing    | 2021–Present |

The validation and test seasons are never used during feature engineering, PCA fitting, hyperparameter optimization, probability calibration, feature pruning, or ensemble learning. This protocol mirrors real-world deployment, where a model only ever sees the past.

---

## Preventing Target Leakage

Leakage prevention is a central design goal of this project. It includes:

* Shifted rolling windows and EWMA statistics (the current game is never included in its own features)
* Historical, chronologically simulated Elo ratings
* Leak-free player performance profiles
* PCA and feature scalers fit exclusively on training seasons
* Strict chronological train/validation/test splits
* `TimeSeriesSplit` cross-validation during hyperparameter search
* Validation-only ensemble weight optimization and feature pruning

Every prediction is generated using only information that would have been known before tip-off.

---

## Generated Artifacts

Each training run creates a timestamped experiment directory (`training/models/run_<timestamp>/`) containing:

* Trained models and feature scalers (`.pkl`)
* Feature names and ensemble weights (`.pkl`)
* Hyperparameter search results and best parameters (`.json` / `.csv`)
* Training configuration and experiment metadata (`.json`)
* Evaluation metrics (`01_model_comparison.csv`)
* Calibration curve, ROC curve, and confusion matrix (`.png`)
* Feature importance and SHAP plots (`.png`)
* Combined feature rankings (`.csv`)
* A full training log (`training.log`)

Every experiment is reproducible from its saved configuration and random seed.

---

## Technologies

* Python 3.10+
* pandas, NumPy
* scikit-learn
* XGBoost
* SciPy (ensemble weight optimization)
* SHAP
* Matplotlib
* Joblib
* `nba_api`

---


## Future Work

Potential future extensions include:

* Learned neural player embeddings (Autoencoders)
* Graph Neural Networks for roster interactions
* CatBoost and LightGBM comparisons
* Bayesian hyperparameter optimization
* Injury report integration
* Starting lineup projections
* Travel distance modelling
* Automated daily prediction pipeline
* Interactive Streamlit dashboard

---

## Author

Developed as a personal machine learning engineering project exploring predictive sports analytics through reproducible experimentation, rigorous evaluation, and modern ML practices. The project prioritizes chronological evaluation, leak-free feature engineering, probability calibration, explainability, and reproducibility as much as predictive performance itself.