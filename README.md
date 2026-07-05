# 🏀 NBA Game Outcome Prediction using Machine Learning

An end-to-end machine learning pipeline that predicts NBA game outcomes using historical data, advanced feature engineering, and a leak-proof time-series training methodology.

Unlike many sports prediction projects that rely on end-of-season statistics or suffer from target leakage, this project was designed around one guiding principle:

> **Only information that would have been available before tip-off is used to make each prediction.**

The result is a chronologically correct prediction system that combines team strength, player form, schedule fatigue, and historical performance into an ensemble machine learning model.

---

## 📌 Project Highlights

- 📈 **66.0% Accuracy** on completely unseen NBA games (2021–Present)
- 📉 **0.616 Log Loss**
- 🎯 **0.214 Brier Score**
- 🔒 **100% Leak-Free Feature Engineering**
- 🧠 Ensemble of Logistic Regression + XGBoost with Validation-Based Stacking
- ⏳ Strict chronological training and evaluation (2000–Present)

---

# 🚀 Motivation

Predicting professional sports is an inherently noisy machine learning problem.

Rather than chasing unrealistic accuracy, the objective of this project was to build a prediction pipeline that follows sound machine learning principles:

- prevent target leakage
- use realistic historical information only
- engineer meaningful basketball features
- evaluate using proper time-series validation
- produce calibrated probabilities rather than only binary predictions

This project focuses as much on **machine learning engineering** as it does on basketball analytics.

---

# 🏗️ Pipeline Architecture

```text
Historical NBA Data
        │
        ▼
Data Collection
        │
        ▼
Era Normalization
        │
        ▼
Feature Engineering
        │
        ▼
Train / Validation / Test Split
        │
        ▼
Hyperparameter Optimization
        │
        ▼
Meta-Learner (NNLS)
        │
        ▼
Retrain on Full Historical Data
        │
        ▼
Final Evaluation
        │
        ▼
Saved Models (.pkl)
```

---

# 📂 Project Structure

```text
nba-prediction-model/

│
├── data/
│
├── models/
│   ├── rf_model.pkl
│   ├── xgb_model.pkl
│   ├── lr_model.pkl
│   ├── scaler.pkl
│   └── ensemble_weights.pkl
│
├── fetch_history.py
├── fetch_player_logs.py
├── era_adjustment.py
├── feature_engineering.py
├── feature_engineering_players.py
├── train_models.py
│
└── README.md
```

---

# 📊 Data Collection

Historical data is collected directly from the official NBA Stats API using `nba_api`.

The dataset contains every regular season game from **2000 through the present**, including:

- Team box scores
- Individual player box scores
- Game dates
- Matchups
- Team statistics

No external betting markets or proprietary datasets are used.

---

# 🧠 Feature Engineering

One of the primary goals of this project was building meaningful predictive features while ensuring **zero information leakage**.

## 🏀 Era Normalization

The NBA evolves every season.

A 105-point offensive performance in 2003 is fundamentally different from one in 2024.

To make statistics comparable across different eras, all raw box score metrics are converted into season-specific Z-Scores.

Examples include:

- Points
- Rebounds
- Assists
- Turnovers
- Plus/Minus
- Shooting efficiency

This allows the model to learn basketball trends independently of league-wide scoring inflation.

---

## ⭐ Rolling Team Form

Instead of using season averages, every statistic is computed using only the previous five games.

Example:

```text
Game 15 prediction

Uses:

Games 10-14

Never Game 15
```

This prevents future information from influencing the prediction.

---

## 🧍 Active Roster Strength

Player impact is calculated using **John Hollinger's Game Score**.

Each player's recent form is computed using a rolling five-game average.

Those values are then aggregated into team-level features:

- Active roster impact
- Active roster dispersion

This captures both:

- overall roster strength
- whether production is concentrated in a few players or distributed across the team.

---

## 🏆 Continuous Elo Rating

Every franchise begins with an Elo rating of **1500**.

The system then simulates every NBA game chronologically from 2000 onward.

Each game stores:

- pre-game Elo
- post-game Elo

Only the **pre-game** rating is available to the model.

This creates a long-term representation of franchise strength independent of season boundaries.

---

## 😴 Fatigue Modeling

Schedule density is modeled using several biological fatigue indicators.

Features include:

- Rest Days
- Back-to-Back games
- 3 Games in 4 Nights
- 4 Games in 5 Nights
- Road Trip Length
- Rest Advantage

These variables capture scheduling effects that traditional box scores ignore.

---

## 🥊 Strength of Schedule

The model estimates opponent quality using rolling historical team strength.

Instead of assuming every recent win is equally valuable, the model differentiates between:

- defeating elite opponents
- defeating weaker teams

---

# 🤖 Machine Learning Models

Three independent classifiers are trained.

## Logistic Regression

Provides an interpretable linear baseline.

## Random Forest

Captures nonlinear interactions through bagged decision trees.

## XGBoost

Gradient boosted trees optimized using randomized hyperparameter search.

---

# ⚙️ Hyperparameter Optimization

Both Random Forest and XGBoost are optimized using

- RandomizedSearchCV
- TimeSeriesSplit

TimeSeriesSplit guarantees that validation folds always occur **after** training folds, preventing temporal leakage.

---

# 🧩 Ensemble Learning

Instead of averaging predictions equally, the project learns optimal ensemble weights using **Non-Negative Least Squares (NNLS)**.

Workflow:

```text
Train Models
        │
        ▼
Predict Validation Set
        │
        ▼
Learn Optimal Weights
        │
        ▼
Retrain Models
        │
        ▼
Predict Test Set
```

This allows stronger models to contribute more while preventing overfitting.

---

# 📈 Model Evaluation

The project uses a strict chronological split.

| Dataset | Seasons |
|----------|----------|
| Training | 2000–2018 |
| Validation | 2019–2020 |
| Testing | 2021–Present |

The test data remains completely untouched until the final evaluation.

---

# 📊 Final Performance

| Model | Accuracy | Log Loss | Brier Score |
|---------|---------:|---------:|------------:|
| Logistic Regression | **66.0%** | **0.616** | **0.214** |
| Random Forest | 65.8% | 0.623 | 0.217 |
| XGBoost | 65.4% | 0.620 | 0.216 |
| **Weighted Ensemble** | **65.8%** | **0.618** | **0.214** |

The ensemble weights are learned exclusively on the validation set.

Example learned weights:

```text
Random Forest    0.0%

XGBoost         55.5%

Logistic Reg    44.5%
```

---

# 🔒 Preventing Target Leakage

A major focus of this project was eliminating hidden sources of information leakage.

Examples include:

✅ Rolling player form

✅ Rolling team statistics

✅ Historical Elo ratings

✅ Pre-game features only

✅ Chronological validation

The model never has access to information generated after the game being predicted.

---

# 💾 Deployment

After training, the following artifacts are saved automatically:

```text
models/

rf_model.pkl

xgb_model.pkl

lr_model.pkl

scaler.pkl

ensemble_weights.pkl
```

These allow predictions to be generated instantly without retraining.

---

# 🛠️ Technologies

- Python
- Pandas
- NumPy
- Scikit-learn
- XGBoost
- SciPy
- nba_api
- Joblib

---

# 🚀 Future Improvements

Potential future enhancements include:

- Injury reports
- Projected starting lineups
- Travel distance calculations
- Play-by-play clutch metrics
- Coach Elo ratings
- SHAP feature explanations
- Streamlit prediction dashboard
- Walk-forward validation across seasons

---

# 📚 Key Machine Learning Concepts Demonstrated

This project showcases practical experience with:

- Time-Series Machine Learning
- Feature Engineering
- Ensemble Learning
- Probability Calibration
- Hyperparameter Optimization
- Rolling Window Statistics
- Model Serialization
- Data Engineering
- Sports Analytics
- Leakage Prevention
- Cross Validation
- Model Evaluation
- Software Engineering for ML Pipelines

---

# 👨‍💻 Author

Developed as a personal machine learning project to explore predictive sports analytics while applying production-oriented machine learning engineering practices.

The emphasis of this project is not only predictive performance, but also building a robust, reproducible, and statistically sound machine learning pipeline.