# 🏀 NBA Game Outcome Prediction using Machine Learning

An end-to-end machine learning pipeline that predicts NBA game outcomes using historical NBA data, leak-free feature engineering, and multiple machine learning models.

The project was built around a single guiding principle:

> **Every prediction uses only information that would have been available before tip-off.**

This results in a fully chronological prediction pipeline with reproducible training, explainable models, and realistic evaluation on unseen seasons.

---

# 📌 Highlights

- 📈 **66.0% Accuracy** on unseen NBA games (2021–Present)
- 📉 **0.616 Log Loss**
- 🎯 **0.214 Brier Score**
- 📊 **0.711 ROC-AUC**
- 🔒 Leak-free feature engineering
- 👥 Player-level and team-level rolling statistics
- 🏆 Continuous Elo rating system
- 🧠 Neural Network, XGBoost and Logistic Regression models
- ⚖️ Validation-based ensemble weighting
- 🔍 SHAP and Permutation Importance
- 💾 Automatic experiment tracking and model serialization

---

# 🚀 Motivation

Sports prediction is an inherently noisy machine learning problem. Rather than chasing unrealistic accuracy, this project focuses on applying sound machine learning practices:

- leak-free feature engineering
- chronological evaluation
- reproducible experiments
- calibrated probability estimation
- interpretable models
- modular software design

The goal is not only strong predictive performance but also a robust ML engineering pipeline.

---

# 🏗️ Pipeline

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
Feature Engineering
        │
        ├── Team Features
        ├── Player Features
        └── Elo Ratings
        │
        ▼
Chronological Train / Validation / Test Split
        │
        ▼
Hyperparameter Optimization
(TimeSeriesSplit + RandomizedSearchCV)
        │
        ▼
Validation-Based Ensemble Learning
        │
        ▼
Retraining on Full Historical Data
        │
        ▼
Explainability & Evaluation
        │
        ▼
Serialized Models & Reports
```

---

# 📂 Project Structure

```text
nba-prediction-model/

├── data/
├── models/
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
├── fetch_player_logs.py
├── train_models.py
└── README.md
```

---

# 📊 Data

Historical data is collected directly from the official NBA Stats API (`nba_api`) and includes every regular season game from 2000 onwards.

The dataset contains:

- Team box scores
- Player box scores
- Matchups
- Game dates
- Team statistics

No betting odds or proprietary datasets are used.

---

# 🧠 Feature Engineering

All features are computed using only information available before each game.

Feature groups include:

- Continuous Elo ratings
- Rolling team statistics
- Rolling offensive and defensive ratings
- Strength of schedule
- Era-normalized (Z-score) box-score statistics
- Rest advantage
- Back-to-back games
- Road trip length
- Player Game Score
- Expected player impact
- Active roster strength

Rolling statistics always exclude the current game, preventing target leakage.

---

# 🤖 Models

Three independent classifiers are trained:

- **Logistic Regression**
- **Neural Network (MLP)**
- **XGBoost**

The MLP and XGBoost models are optimized using **RandomizedSearchCV** with **TimeSeriesSplit**, ensuring every validation fold occurs after its corresponding training data.

---

# 🧩 Ensemble Learning

Validation predictions from the three base models are combined using **Non-Negative Least Squares (NNLS)**.

Rather than averaging predictions equally, the ensemble automatically learns the optimal weighting based on validation performance before being evaluated on completely unseen seasons.

---

# 🔍 Explainability

The project includes several explainability techniques:

- SHAP explanations for XGBoost
- Permutation Importance for the Neural Network
- Logistic Regression coefficient analysis

Feature rankings and visualizations are automatically generated after training.

---

# 📈 Evaluation

The dataset is split chronologically.

| Dataset | Seasons |
|----------|----------|
| Training | 2000–2018 |
| Validation | 2019–2020 |
| Testing | 2021–Present |

The test set is never used during training or model selection.

## Final Performance

| Model | Accuracy | Log Loss | Brier | ROC-AUC |
|---------|---------:|---------:|------:|---------:|
| Logistic Regression | **66.0%** | **0.616** | **0.214** | **0.711** |
| Neural Network | 65.9% | 0.621 | 0.216 | 0.706 |
| XGBoost | 65.2% | 0.620 | 0.216 | 0.707 |
| Weighted Ensemble | 65.8% | 0.619 | 0.215 | 0.708 |

Metrics reported:

- Accuracy
- Log Loss
- Brier Score
- ROC-AUC
- Calibration Curve
- Confusion Matrix

---

# 🔒 Preventing Target Leakage

Every stage of the pipeline was designed to avoid future information leaking into model training.

Examples include:

- Shifted rolling windows
- Pre-game Elo ratings
- Rolling player statistics
- Chronological train/validation/test split
- TimeSeriesSplit cross-validation

This ensures that each prediction reflects only information available before tip-off.

---

# 📦 Training Outputs

Every training run automatically generates:

- Trained models
- Feature scaler
- Ensemble weights
- Hyperparameters
- Feature rankings
- SHAP visualizations
- Evaluation metrics
- ROC curve
- Calibration curve
- Confusion matrix
- Training log

Each experiment is versioned for reproducibility.

---

# 🛠️ Technologies

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

# 🚀 Future Work

Possible extensions include:

- Stacking ensemble
- Bayesian hyperparameter optimization
- Injury report integration
- Starting lineup projections
- Travel distance modeling
- Betting market comparison
- Automated daily prediction pipeline
- Streamlit dashboard

---

# 👨‍💻 Author

Developed as a personal machine learning project to explore predictive sports analytics while applying modern machine learning engineering practices.

The emphasis is on reproducibility, chronological evaluation, explainability, and software engineering as much as predictive performance.