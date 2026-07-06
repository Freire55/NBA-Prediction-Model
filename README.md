# 🏀 NBA Game Outcome Prediction using Machine Learning

An end-to-end machine learning pipeline that predicts NBA game outcomes using historical data, advanced feature engineering, model ensembling, and a leak-free time-series training methodology.

Unlike many sports prediction projects that rely on end-of-season statistics or unknowingly introduce target leakage, this project was built around one guiding principle:

> **Only information that would have been available before tip-off is used to make each prediction.**

The result is a chronologically correct prediction system that combines team strength, player form, schedule fatigue, and historical performance into a fully reproducible machine learning pipeline.

---

# 📌 Project Highlights

- 📈 **66.0% Accuracy** on completely unseen NBA games (2021–Present)
- 📉 **0.616 Log Loss**
- 🎯 **0.214 Brier Score**
- 📊 **0.711 ROC-AUC**
- 🔒 Leak-free feature engineering
- 🧠 Ensemble of **Neural Network, XGBoost and Logistic Regression**
- ⚖️ Validation-based ensemble weighting using **Non-Negative Least Squares (NNLS)**
- 🔍 Explainable AI using **SHAP** and **Permutation Importance**
- 📈 Probability calibration analysis
- 💾 Automatic model serialization for deployment
- ⏳ Strict chronological training and evaluation (2000–Present)

---

# 🚀 Motivation

Predicting professional sports is an inherently noisy machine learning problem.

Rather than chasing unrealistic accuracy, the objective of this project was to build a prediction pipeline that follows sound machine learning principles:

- prevent target leakage
- use only information available before each game
- engineer meaningful basketball features
- evaluate using proper time-series validation
- produce calibrated probabilities instead of only binary predictions
- build a reproducible and deployable machine learning pipeline

This project focuses equally on **machine learning engineering**, **software engineering**, and **basketball analytics**.

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
Chronological Train / Validation / Test Split
        │
        ▼
Hyperparameter Optimization
(TimeSeriesSplit + RandomizedSearchCV)
        │
        ▼
Validation-Based Ensemble Weight Learning (NNLS)
        │
        ▼
Retrain on Full Historical Data
        │
        ▼
Explainability
(SHAP + Permutation Importance)
        │
        ▼
Final Evaluation
        │
        ▼
Calibration & Confusion Matrix
        │
        ▼
Deployment Artifacts
```

---

# 📂 Project Structure

```text
nba-prediction-model/

│
├── data/
│
├── models/
│   ├── mlp_model.pkl
│   ├── xgb_model.pkl
│   ├── lr_model.pkl
│   ├── scaler.pkl
│   ├── feature_order.pkl
│   ├── ensemble_weights.pkl
│   │
│   ├── mlp_best_params.json
│   ├── xgb_best_params.json
│   ├── test_metrics.json
│   │
│   ├── mlp_feature_importance.csv
│   ├── xgb_feature_importance.csv
│   ├── lr_coefficients.csv
│   │
│   ├── xgb_shap_summary.png
│   ├── calibration_curve.png
│   └── ensemble_confusion_matrix.png
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

Historical data is collected directly from the official NBA Stats API (`nba_api`).

The dataset contains every regular season game from **2000 through the present**, including:

- Team box scores
- Individual player box scores
- Matchups
- Game dates
- Team statistics

No betting odds, proprietary datasets or manually curated statistics are used.

---

# 🧠 Feature Engineering

One of the primary objectives of this project was creating predictive features while ensuring **zero information leakage**.

---

## 🏀 Era Normalization

Basketball evolves every season.

A 105-point offensive performance in 2003 is fundamentally different from one in 2024.

To make statistics comparable across eras, all raw box-score metrics are converted into season-specific **Z-Scores**.

Examples include:

- Points
- Rebounds
- Assists
- Turnovers
- Shooting percentages
- Plus/Minus
- Offensive Rating
- Defensive Rating

This allows the models to learn basketball quality independently of league-wide scoring inflation.

---

## ⭐ Rolling Team Form

Instead of using season averages, every statistic is computed using only the previous **five games**.

Example:

```text
Prediction for Game 15

Uses:

Games 10-14

Never Game 15
```

This guarantees future games never influence the prediction.

---

## 👥 Active Roster Strength

Player impact is measured using **John Hollinger's Game Score**.

Each player's recent form is computed using rolling five-game averages.

Those values are aggregated into team-level features representing:

- Active roster strength
- Active roster consistency
- Recent player form

This captures both team quality and roster momentum.

---

## 🏆 Continuous Elo Rating

Every franchise begins with an Elo rating of **1500**.

The system simulates every NBA game chronologically from 2000 onward.

Each game stores:

- Pre-game Elo
- Post-game Elo

Only the **pre-game** rating is available to the prediction model.

This creates a long-term representation of team strength that naturally evolves over time.

---

## 😴 Schedule Fatigue

Scheduling effects are modeled using several fatigue indicators.

Examples include:

- Rest Days
- Rest Advantage
- Back-to-Back Games
- Home Back-to-Back
- Away Back-to-Back

These variables capture competitive disadvantages that traditional box scores ignore.

---

# 🤖 Machine Learning Models

Three independent classifiers are trained.

## 🧠 Neural Network (MLP)

A Multi-Layer Perceptron trained on standardized features.

The architecture is optimized using randomized hyperparameter search with time-series cross-validation.

---

## 🌳 XGBoost

Gradient Boosted Decision Trees optimized using randomized hyperparameter search.

Captures nonlinear interactions between engineered basketball features.

---

## 📈 Logistic Regression

Provides a highly interpretable linear baseline.

Despite its simplicity, it achieved the strongest standalone performance, demonstrating the quality of the engineered features.

---

# ⚙️ Hyperparameter Optimization

Both the Neural Network and XGBoost are optimized using:

- RandomizedSearchCV
- TimeSeriesSplit

TimeSeriesSplit guarantees validation folds always occur **after** the training folds, preventing temporal leakage.

The optimal hyperparameters are automatically exported as JSON files for reproducibility.

---

# 🧩 Ensemble Learning

Instead of averaging model predictions equally, the project learns optimal ensemble weights using **Non-Negative Least Squares (NNLS)**.

Workflow:

```text
Train Base Models
        │
        ▼
Predict Validation Set
        │
        ▼
Learn Optimal Weights
        │
        ▼
Retrain Models on Full Historical Data
        │
        ▼
Predict Unseen Test Set
```

This allows stronger models to contribute proportionally while avoiding overfitting.

---

# 🔍 Model Explainability

Understanding why a prediction is made is as important as the prediction itself.

The project includes multiple explainability techniques.

## SHAP

Tree SHAP is used to explain the XGBoost model.

A global SHAP summary plot is automatically generated after training.

---

## Permutation Importance

Permutation Importance is used to estimate feature importance for the Neural Network, allowing insight into an otherwise black-box model.

---

## Logistic Regression Coefficients

Absolute standardized coefficients are exported to provide an interpretable ranking of feature importance.

Together, these techniques provide explanations for every model in the ensemble.

---

# 📈 Model Evaluation

The project follows a strict chronological split.

| Dataset | Seasons |
|----------|----------|
| Training | 2000–2018 |
| Validation | 2019–2020 |
| Testing | 2021–Present |

The test set remains completely untouched until the final evaluation.

---

# 📊 Final Performance

| Model | Accuracy | Log Loss | Brier Score | ROC-AUC |
|---------|---------:|---------:|------------:|---------:|
| Logistic Regression | **66.0%** | **0.616** | **0.214** | **0.711** |
| Neural Network | 65.3% | 0.623 | 0.217 | 0.705 |
| XGBoost | 65.4% | 0.620 | 0.216 | 0.706 |
| **Weighted Ensemble** | 65.4% | 0.618 | 0.215 | 0.709 |

Example learned ensemble weights:

```text
Neural Network     23.5%

XGBoost            47.3%

Logistic Regression 29.1%
```

---

# 📊 Evaluation Metrics

The project evaluates every model using multiple complementary metrics.

- Accuracy
- Log Loss
- Brier Score
- ROC-AUC
- Confusion Matrix
- Probability Calibration Curve

Since the objective is probability estimation rather than simple classification, Log Loss and Brier Score receive particular emphasis.

---

# 🔒 Preventing Target Leakage

A major design goal was eliminating hidden sources of leakage.

Examples include:

✅ Rolling player statistics

✅ Rolling team form

✅ Historical Elo ratings

✅ Pre-game features only

✅ Chronological train/validation/test split

✅ TimeSeriesSplit cross-validation

The models never have access to information generated after the game being predicted.

---

# 💾 Deployment

After training, the pipeline automatically exports:

- Trained models
- Standardization scaler
- Feature ordering
- Ensemble weights
- Optimal hyperparameters
- Evaluation metrics
- Feature importance rankings
- SHAP explanations
- Calibration curve
- Confusion matrix

These artifacts allow the complete prediction pipeline to be reused without retraining while ensuring reproducibility.

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

# 🚀 Future Improvements

Potential future enhancements include:

- Live injury reports
- Projected starting lineups
- Travel distance modeling
- Coach Elo ratings
- Betting market comparison
- Automated daily prediction pipeline
- Streamlit prediction dashboard
- Walk-forward retraining across seasons

---

# 📚 Machine Learning Concepts Demonstrated

This project demonstrates practical experience with:

- Time-Series Machine Learning
- Leak-Free Feature Engineering
- Ensemble Learning
- Neural Networks
- Gradient Boosting
- Explainable AI (SHAP & Permutation Importance)
- Probability Calibration
- Hyperparameter Optimization
- Rolling Window Statistics
- Chronological Cross Validation
- Model Serialization
- Model Interpretability
- Data Engineering
- Sports Analytics
- Software Engineering for Machine Learning Pipelines

---

# 👨‍💻 Author

Developed as a personal machine learning project to explore predictive sports analytics while applying production-oriented machine learning engineering practices.

Beyond predictive performance, the primary objective of this project was to build a **robust, interpretable, reproducible, and deployment-ready machine learning pipeline** that follows industry best practices for time-series modeling and model evaluation.