# 🏀 NBA Game Outcome Prediction using Machine Learning

Predicting professional basketball games is a challenging machine learning problem due to changing team strength, roster availability, schedule effects, and the inherently stochastic nature of sports.

This project implements a fully chronological end-to-end machine learning pipeline that predicts NBA game outcomes using only information available **before tip-off**. The pipeline combines leak-free feature engineering, time-aware model selection, probability calibration, ensemble learning, and explainability techniques to produce realistic predictions on unseen NBA seasons.

Rather than maximizing raw accuracy alone, the project emphasizes **sound machine learning engineering practices**, including chronological validation, reproducibility, calibrated probability estimation, and model interpretability.

---

# 📈 Results

| Metric | Value |
|---------|------:|
| **Accuracy** | **66.0%** |
| **Log Loss** | **0.617** |
| **Brier Score** | **0.214** |
| **ROC-AUC** | **0.710** |

Final evaluation was performed on **every regular season game from 2021 onward**, using models trained exclusively on historical seasons.

---

# ✨ Key Features

- 🔒 Leak-free feature engineering
- 🏀 Team and player-level rolling statistics
- 🏆 Continuous Elo rating system
- 📈 Era-normalized statistics
- 🧠 Logistic Regression, Neural Network (MLP) and XGBoost
- ⚙️ Time-aware hyperparameter optimization
- 📊 Cross-validated probability calibration
- ⚖️ Validation-based weighted ensemble
- 🔍 SHAP, permutation importance and coefficient analysis
- 💾 Automatic experiment tracking and model serialization

---

# 🚀 Motivation

Sports prediction is one of the most difficult supervised learning problems because outcomes depend on numerous dynamic factors that evolve throughout a season.

Many public NBA prediction projects unintentionally introduce target leakage by allowing future information into training features or evaluation.

This project was built around one guiding principle:

> **Every prediction must be generated using only information that would have been available before the game began.**

The objective is to demonstrate robust machine learning engineering practices rather than simply maximizing predictive accuracy.

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
Leak-Free Feature Engineering
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
│   └── processed datasets
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
├── feature_engineering.py
├── feature_engineering_players.py
├── era_adjustment.py
├── fetch_history.py
├── fetch_player_game_logs.py
├── train_models.py
└── README.md
```

---

# 📊 Dataset

Historical data is collected directly from the official **NBA Stats API (`nba_api`)**.

The dataset contains every NBA regular-season game from **2000 onward**, including:

- Team box scores
- Player box scores
- Matchups
- Game dates
- Team statistics

No betting odds, proprietary datasets, or manually curated information are used.

---

# 🧠 Feature Engineering

Every feature is computed exclusively from information available before each game.

The feature set includes:

- Continuous Elo ratings
- Rolling team statistics
- Rolling offensive and defensive ratings
- Era-normalized (Z-score) statistics
- Rest advantage
- Back-to-back indicators
- Road trip length
- Player Game Score
- Active roster strength
- Expected player impact

All rolling statistics are shifted so the current game is never included, eliminating target leakage.

---

# 🤖 Machine Learning Models

Three independent classifiers are trained:

- Logistic Regression
- Multi-Layer Perceptron (MLP)
- XGBoost

Hyperparameters are optimized using **RandomizedSearchCV** with **TimeSeriesSplit**, ensuring every validation fold occurs strictly after its corresponding training data.

All models are probability-calibrated before ensemble optimization.

---

# ⚖️ Ensemble Learning

Instead of averaging model predictions equally, validation-set probabilities are combined using a constrained optimization procedure that directly minimizes **Log Loss** while enforcing non-negative weights that sum to one.

For the final model, the learned ensemble assigned approximately:

| Model | Weight |
|--------|-------:|
| Logistic Regression | **67%** |
| MLP | 29% |
| XGBoost | 4% |

---

# 📈 Model Comparison

Although Neural Networks and Gradient Boosted Trees are often expected to outperform linear models, the opposite occurred in this project.

After hyperparameter optimization, probability calibration, and ensemble learning, **Logistic Regression consistently achieved the strongest overall performance**.

This suggests that the engineered features capture most of the predictive signal in a largely linear manner, while more flexible models primarily learn additional noise.

An important takeaway from this project is:

> **Strong feature engineering can be more valuable than increasing model complexity.**

---

# 🔍 Explainability

The pipeline automatically generates multiple explainability reports:

- SHAP summary plots (XGBoost)
- Permutation importance (MLP)
- Logistic Regression coefficients
- ROC curve
- Calibration curve
- Confusion matrix
- Combined feature rankings

These reports help explain both global feature importance and individual model behaviour.

---

# 📈 Evaluation Strategy

The dataset is split chronologically.

| Dataset | Seasons |
|----------|----------|
| Training | 2000–2018 |
| Validation | 2019–2020 |
| Testing | 2021–Present |

The test set is never used during feature engineering, hyperparameter tuning, ensemble optimization, or model calibration.

This evaluation protocol closely mirrors how the model would perform in a real deployment.

---

# 🔒 Preventing Target Leakage

Every stage of the pipeline was explicitly designed to avoid future information leaking into model training.

Examples include:

- Shifted rolling windows
- Pre-game Elo ratings
- Rolling player statistics
- Chronological train/validation/test split
- TimeSeriesSplit cross-validation

As a result, every prediction reflects only information that would have been available before tip-off.

---

# 📦 Generated Artifacts

Each training run automatically produces a timestamped experiment folder containing:

- Trained models
- Feature scaler
- Learned ensemble weights
- Hyperparameter search results
- Evaluation metrics
- Feature importance rankings
- SHAP visualizations
- Calibration curve
- ROC curve
- Confusion matrix
- Training logs

Every experiment is fully reproducible.

---

# 🛠 Technologies

- Python
- Pandas
- NumPy
- Scikit-learn
- XGBoost
- SHAP
- SciPy
- Matplotlib
- Joblib
- nba_api

---

# 🚀 Future Improvements

Potential future extensions include:

- Stacking ensembles
- Bayesian hyperparameter optimization
- Injury report integration
- Starting lineup projections
- Travel distance modelling
- Daily automated prediction pipeline
- Interactive Streamlit dashboard

---

# 👨‍💻 Author

Developed as a personal machine learning project exploring predictive sports analytics while applying modern machine learning engineering practices.

The project emphasizes reproducibility, chronological evaluation, explainability, and clean software architecture as much as predictive performance.