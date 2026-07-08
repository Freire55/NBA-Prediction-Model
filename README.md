# 🏀 NBA Game Outcome Prediction using Machine Learning

A fully chronological, leak-free machine learning pipeline for predicting NBA game outcomes using historical team and player statistics.

This project predicts NBA regular-season games using only information that would have been available **before tip-off**. It combines advanced feature engineering, player representation learning, time-aware model selection, probability calibration, ensemble learning, and explainability into a reproducible end-to-end machine learning workflow.

Rather than chasing raw accuracy through aggressive feature selection or data leakage, the project focuses on building a realistic prediction system that follows sound machine learning engineering practices and mirrors how a model would operate in production.

---

# 📈 Results

Final evaluation was performed on **every NBA regular-season game from 2021 onward**, using models trained exclusively on historical seasons.

| Metric          |  Ensemble |
| --------------- | --------: |
| **Accuracy**    | **66.1%** |
| **Log Loss**    | **0.615** |
| **Brier Score** | **0.213** |
| **ROC-AUC**     | **0.716** |

The final ensemble combines calibrated Logistic Regression, XGBoost and Multi-Layer Perceptron (MLP) models using validation-set optimization.

---

# ✨ Highlights

* 🔒 Fully leak-free chronological pipeline
* 🏀 Team and player-level feature engineering
* 👤 Learned player embeddings via PCA
* 📈 Era-adjusted statistical normalization
* 🏆 Continuous Elo rating system
* 🧠 Logistic Regression, XGBoost and Neural Networks
* ⚙️ Time-aware hyperparameter optimization
* 📊 Cross-validated probability calibration
* ⚖️ Validation-optimized ensemble learning
* 🔍 SHAP, permutation importance and coefficient analysis
* 📦 Automatic experiment tracking and model serialization
* 🧪 Ablation studies validating learned player representations

---

# 🚀 Motivation

Predicting professional basketball games is an exceptionally difficult supervised learning problem.

Team strength changes continuously due to injuries, roster moves, player development, scheduling effects, fatigue, and countless stochastic factors. Many publicly available NBA prediction projects unintentionally introduce target leakage through improper feature engineering or random train/test splits, leading to unrealistic performance estimates.

This project was built around one fundamental principle:

> **Every prediction must be generated using only information that existed before the game began.**

The objective is not only to produce competitive predictions, but to demonstrate rigorous machine learning engineering practices.

---

# 🏗 Pipeline

```text
NBA Stats API
        │
        ▼
Historical Team & Player Data
        │
        ▼
Era Normalization
        │
        ▼
Leak-Free Team Features
        │
        ▼
Leak-Free Player Features
        │
        ▼
Player Representation Learning (PCA)
        │
        ▼
Chronological Train / Validation / Test Split
        │
        ▼
Hyperparameter Optimization
(RandomizedSearchCV + TimeSeriesSplit)
        │
        ▼
Probability Calibration
        │
        ▼
Validation-Based Ensemble Optimization
        │
        ▼
Retraining on Full Historical Data
        │
        ▼
Evaluation & Explainability
        │
        ▼
Serialized Models & Reports
```

---

# 📂 Repository Structure

```text
nba-prediction-model/

├── data/
│   ├── raw datasets
│   ├── processed datasets
│   └── generated player embeddings
│
├── training/
│   ├── config.py
│   ├── data.py
│   ├── tuning.py
│   ├── training.py
│   ├── ensemble.py
│   ├── evaluation.py
│   ├── explainability.py
│   ├── plots.py
│   └── utils.py
│
├── fetch_history.py
├── fetch_player_game_logs.py
├── era_adjustment.py
├── feature_engineering.py
├── generate_player_embeddings.py
├── feature_engineering_players.py
├── train_models.py
└── README.md
```

---

# 📊 Dataset

Historical data is collected directly from the official NBA Stats API using `nba_api`.

The dataset contains every NBA regular-season game from **2000 onward**, including:

* Team box scores
* Player box scores
* Game dates
* Matchups
* Team identifiers
* Player identifiers

No betting odds, proprietary datasets or manually curated information are used.

---

# 🧠 Feature Engineering

Every feature is computed exclusively from historical information available before each game.

The feature set includes:

### Team Features

* Continuous Elo ratings
* Rolling offensive and defensive ratings
* Rolling shooting efficiency
* Rolling rebounding statistics
* Rolling turnover statistics
* Rolling pace
* Schedule density
* Back-to-back indicators
* Road trip length
* Rest advantage
* Era-normalized statistics

### Player Features

* Leak-free rolling Game Score
* Rolling expected playing time
* Active roster strength
* Roster consistency
* Star-player concentration
* Expected player impact

All rolling statistics are shifted so that the current game is never included.

---

# 👤 Player Representation Learning

Instead of representing players only through aggregate box-score statistics, the project learns compact player representations using dimensionality reduction.

For every player:

* Fourteen per-minute statistical rates are calculated.
* Historical exponentially weighted moving averages (EWMA) summarize recent performance.
* Principal Component Analysis (PCA), trained exclusively on historical training seasons, compresses these profiles into two latent embedding dimensions.

The embeddings are:

* generated without future information,
* weighted by expected historical playing time,
* aggregated to the team level,
* transformed into matchup-level embedding features.

A dedicated ablation study demonstrated that these learned player representations consistently improve neural network performance while passing shuffle tests designed to detect data leakage.

---

# 🤖 Machine Learning Models

Three independent classifiers are trained:

* Logistic Regression
* XGBoost
* Multi-Layer Perceptron (MLP)

Hyperparameters are optimized using **RandomizedSearchCV** with **TimeSeriesSplit**, ensuring every validation fold occurs strictly after its corresponding training data.

All models are subsequently calibrated using cross-validated Platt Scaling before ensemble optimization.

---

# ⚖️ Ensemble Learning

Instead of averaging predictions equally, the final ensemble learns optimal model weights by minimizing validation Log Loss under the constraints that weights remain non-negative and sum to one.

The final learned ensemble is approximately:

| Model               |    Weight |
| ------------------- | --------: |
| **MLP**             | **71.5%** |
| XGBoost             |     24.9% |
| Logistic Regression |      3.6% |

This weighting reflects the complementary strengths of each calibrated model rather than assigning equal importance.

---

# 📊 Model Performance

A chronological ablation study showed that learned player embeddings provide a meaningful improvement over traditional engineered statistics alone.

The project also demonstrated that:

* Carefully engineered statistical features produce a strong linear baseline.
* Learned player representations allow neural networks to capture nonlinear roster interactions beyond conventional box-score aggregates.
* Probability calibration and validation-based ensemble optimization further improve predictive performance.

---

# 🔍 Explainability

Every training run automatically generates explainability reports including:

* SHAP summary plots
* Permutation importance
* Logistic Regression coefficients
* XGBoost feature importance
* ROC curve
* Calibration curve
* Confusion matrix
* Combined feature rankings

These artifacts help explain both global feature importance and individual model behaviour.

---

# 📈 Evaluation Strategy

The dataset is split strictly chronologically.

| Dataset    | Seasons      |
| ---------- | ------------ |
| Training   | 2000–2018    |
| Validation | 2019–2020    |
| Testing    | 2021–Present |

The validation and test seasons are never used during feature engineering, PCA fitting, hyperparameter optimization, probability calibration or ensemble learning.

This evaluation protocol closely mirrors real-world deployment.

---

# 🔒 Preventing Target Leakage

Preventing target leakage is a central design goal of this project.

Leakage prevention includes:

* Shifted rolling windows
* Historical Elo ratings
* Historical player statistics
* Leak-free EWMA player profiles
* PCA fitted exclusively on training seasons
* Chronological train/validation/test splits
* TimeSeriesSplit cross-validation
* Validation-only ensemble optimization

Every prediction is generated using only information that would have been known before tip-off.

---

# 📦 Generated Artifacts

Each training run automatically creates a timestamped experiment directory containing:

* Trained models
* Feature scaler
* PCA embeddings
* Calibration models
* Learned ensemble weights
* Hyperparameter search results
* Evaluation metrics
* Feature importance rankings
* SHAP visualizations
* ROC curve
* Calibration curve
* Confusion matrix
* Training logs

Every experiment is fully reproducible.

---

# 🛠 Technologies

* Python
* Pandas
* NumPy
* Scikit-learn
* XGBoost
* SHAP
* SciPy
* Matplotlib
* Joblib
* nba_api

---

# 🔬 Future Work

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

# 👨‍💻 Author

Developed as a personal machine learning engineering project exploring predictive sports analytics through reproducible experimentation, rigorous evaluation and modern ML practices.

The project emphasizes chronological evaluation, leak-free feature engineering, probability calibration, explainability and reproducible experimentation as much as predictive performance.
