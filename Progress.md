Phase 1: The Foundation, Data Ingestion, & The Random Forest Era

The Original Goal
The inception of this project was to build an NBA game outcome prediction model (Home Win/Loss) from scratch using historical box scores. The goal was to feed historical team performances into a machine learning algorithm to find patterns in how teams match up, predict future games, and eventually integrate this into your portfolio alongside your Mobile Bet Tracking App.

What Was Done (The Process)

    Data Acquisition (fetch_history.py & fetch_advanced.py): * We utilized the nba_api to download regular-season game logs and advanced team stats dating back to the year 2000.

        We intentionally filtered out playoff games because the fatigue and minute-distribution dynamics in the playoffs fundamentally break regular-season patterns.

    Initial Feature Construction: * We built a foundational dataset that looked at raw team statistics (Points, Rebounds, Assists, etc.).

    The First Modeling Attempts (The Random Forest Era):

        In the early stages, you implemented a Random Forest classifier alongside XGBoost and Logistic Regression. Random Forest is a classic starting point because it is robust to outliers and doesn't require heavy data scaling.

        We set up a basic chronological split (Train on older seasons, Test on newer seasons) to evaluate the models.

Difficulties & Bottlenecks Encountered

    The "Era Bias" Problem: This was our first massive roadblock. The NBA has changed drastically. Scoring 110 points in 2004 meant you had an elite offense; scoring 110 points in 2024 means you had a terrible shooting night. Feeding raw points into the Random Forest confused the model because the "value" of a point had inflated over 20 years.

    Target Leakage (The "Future Sight" Bug): In early iterations of sports modeling, it is very easy to accidentally use a team's end-of-season average to predict a game in November of that same season. This artificially inflates model accuracy because the model is peaking into the future.

    Random Forest Probability Limitations: While Random Forest was great for categorical accuracy, we realized it struggles to produce calibrated probabilities. A Random Forest's "probability" is just the vote fraction of its trees, which often pushes predictions toward the middle (e.g., 0.4 to 0.6) and rarely gives confident, accurate extremes compared to Logistic Regression or XGBoost.

Things Changed & Options Chosen (The Pivots)

    Pivot 1: Dropping Random Forest: As we realized that predicting sports betting outcomes requires highly accurate probabilities (Brier Score) rather than just binary accuracy, we decided to phase out Random Forest. We narrowed our focus to XGBoost (superior for tabular data interactions), MLP (Neural Networks for complex non-linear patterns), and Logistic Regression (the gold-standard for calibrated linear baselines).

    Pivot 2: Era Adjustment (era_adjustment.py): To fix the "Era Bias," we made the crucial architectural choice to convert all raw stats into Z-Scores grouped by SEASON_ID. This meant the model was no longer looking at "Points," but rather "Standard Deviations above the League Average for that specific year," allowing it to perfectly compare the 2004 Pistons to the 2017 Warriors.

    Pivot 3: Chronologically Pure Rolling Stats: To fix target leakage, we strictly enforced .shift(1).rolling(5) in Pandas. This guaranteed that the model only ever calculated team strength based on the 5 games prior to the current matchup.


---

### Phase 2: Advanced Feature Engineering & The Performance Bottlenecks

**The Goal**
With our data successfully downloaded and "Era-Adjusted" (Phase 1), the next objective was to translate historical box scores into actionable predictive features. We needed to calculate biological fatigue, rolling team strength, historical franchise prestige (Elo), and individual player impact, ultimately subtracting the Away Team's metrics from the Home Team's metrics to create "Matchup Deltas."

**What Was Done (The Process)**

1. **Schedule Density (Fatigue):** We engineered features in `feature_engineering.py` to calculate exactly how tired a team was before tip-off. We tracked Back-to-Backs (`B2B`), grueling stretches (`3_IN_4` and `4_IN_5` nights), and cumulative road trip lengths.
2. **Strength of Schedule (SOS) & Rolling Form:** We implemented a chronologically pure 5-game rolling average of Offensive Rating and Opponent Strength, strictly using `shift(1)` to prevent target leakage.
3. **The Elo Rating System:** We built a living chronological timeline that started every franchise at 1500 Elo and updated their rating after every single game in history using a standard K-Factor of 20.
4. **Player-Level Impact (`feature_engineering_players.py`):** We moved beyond team-level stats and parsed over 600,000 individual player game logs. We calculated John Hollinger's "Game Score" for every player, applied a 5-game rolling average, and weighted it by their expected playing time (minutes). We then summed these values by `TEAM_ID` to represent the actual strength of the active roster on that specific night.

**Difficulties & Bottlenecks Encountered**

1. **The Elo "Infinite Loop" Bug:** Initially, the Elo engine was taking hours (or never finishing). We discovered a massive $O(N^2)$ bottleneck: inside our 30,000-game loop, the script was executing `df.groupby('GAME_ID')` over the entire dataset *for every single game*, resulting in nearly 900 million redundant row checks.
2. **The Pandas String Engine Overhead:** Even after fixing the infinite loop by caching the grouped data into a Python dictionary, the script still took 68 seconds. The culprit was using Pandas regex/string filtering (`.str.contains(' vs. ')`) inside the massive loop.
3. **The `KeyError: 1` Player Log Crash:** When parsing player minutes (e.g., `"34:15"`), we used a lightning-fast vectorized split on the colon `:`. However, the script hard-crashed because some data batches contained no colons (only whole numbers), so Pandas never created the "seconds" column (`time_split[1]`), causing a `KeyError`.
4. **The DataFrame Fragmentation Warning:** As we merged the massive player impact metrics back into the 100+ column matchup dataset, Pandas threw severe `PerformanceWarning`s because the computer's RAM was becoming heavily fragmented by adding columns one by one.

**Things Changed & Options Chosen (The Fixes & Refactors)**

* **The Tuple Optimization (Elo Fix):** To drop the Elo simulation from 68 seconds to just 2–5 seconds, we pivoted entirely away from Pandas inside the loop. We pre-compiled the matchups into lightweight, pure-Python tuples `(home_team, away_team, home_pts, away_pts)` *before* the loop began. This bypassed the Pandas C-engine overhead entirely.
* **Defensive Data Parsing:** We implemented a dynamic fallback for the minutes parser: `if 1 not in time_split.columns: time_split[1] = 0`. This preserved our C-code vectorization speed while safely handling weird API data anomalies.
* **Memory De-fragmentation:** We added a simple `matchups_df = matchups_df.copy()` command right before calculating the Deltas. This forced Pandas to re-allocate a clean, contiguous block of memory, eliminating the performance warnings.
* **The "Data Engineering" Refactor:** Following a rigorous code review, we completely overhauled `feature_engineering.py` and `feature_engineering_players.py`. We elevated them from "data science scripts" to "production modules" by:
* Centralizing parameters (like `ROLLING_WINDOW = 5`, `INITIAL_ELO = 1500`) at the top of the files.
* Breaking the massive procedural walls of code into strictly-typed helper functions (e.g., `simulate_elo()`, `add_schedule_features()`, `calculate_game_score()`).
* Adding professional module-level docstrings and replacing raw `print()` statements with standard Python `logging`.



---

By the end of this phase, our data pipeline was generating extremely rich, leak-free, mathematically sound predictive signals in a matter of seconds.


Here is Phase 3 of our detailed retrospective, focusing on the architectural overhaul of your machine learning pipeline, the ensemble experimentation, and the crucial realization about probability calibration.

Read through this, and whenever you are ready, simply type "next" and I will generate the final phase (Phase 4: The Strategic Choices & Future Roadmap).
Phase 3: The Monolith Split, Ensembles, and The Calibration Epiphany

The Goal
With our dataset fully era-adjusted and packed with leak-free features (Elo, fatigue, player form), we needed to feed it into our machine learning models. Initially, you had a single, massive script handling the entire ML lifecycle. The goal of this phase was to elevate the codebase from a "data science script" into a Production-Ready MLOps Pipeline, and to figure out how to combine multiple models (MLP, XGBoost, Logistic Regression) into one "super-prediction."

What Was Done (The Process)

    Shattering the Monolith: We took a script that was hundreds of lines long and broke it down into a highly modular, 9-file architecture:

        config.py: Centralized magic numbers, hyperparameter grids, and chronological boundary dates (e.g., Train ends 2018, Val ends 2020).

        data.py: Handled strict temporal splitting and scaling.

        utils.py: Replaced scattered print() statements with a global logging system and a PipelineStage timer.

        explainability.py & evaluation.py: Isolated SHAP calculations, Permutation Importance, and plotting (ROC-AUC, Calibration curves).

        train_models.py: Became a clean, 6-step orchestrator that read like pseudocode.

    The Ensemble Attempt (NNLS): We implemented a Non-Negative Least Squares (NNLS) optimizer to learn the best blend of our three models. The idea was to use the Validation set to figure out exactly how much to "trust" the MLP vs. XGBoost vs. Logistic Regression.

Difficulties & Bottlenecks Encountered

    The "Ensemble Degradation" Problem: Once the pipeline was built, we looked at the evaluation metrics and noticed something alarming: The ensemble was actively making predictions worse. The standalone Logistic Regression model was beating the complex MLP + XGBoost blend.

    The Calibration Gap: We realized why the ensemble was failing. NNLS minimizes the squared error (Brier Score) of the raw probability outputs. However, models like XGBoost are notoriously overconfident (predicting 99% when the true probability is 70%). Because NNLS assumed the probabilities were already "truthful," it was forced to assign weird weights just to correct the bias of XGBoost, rather than finding a genuinely smarter combination of models.

    The Data Leakage Risk in Calibration: The obvious fix was to calibrate the models. However, standard post-hoc calibration (cv="prefit") on the validation set meant that when we finally retrained the models on the combined Train + Validation set, we would either lose the calibration entirely or risk massive data leakage by calibrating on the same data we just trained on.

Things Changed & Options Chosen (The Fixes & Refactors)

    The Calibration Integration: Instead of slapping calibration on at the end, we moved it directly into the cross-validation training step inside models.py. We wrapped the MLP, XGBoost, and LR estimators in CalibratedClassifierCV(cv=TimeSeriesSplit, method='sigmoid').

    Why this was the "Pro" Move: By using TimeSeriesSplit inside the calibrator, the models learned to adjust their overconfidence out-of-fold. When they were finally retrained on the full dataset, they maintained their robust, leak-free probability mapping.

    The "Honest" Ensemble: Once the base models were successfully calibrated, the NNLS ensemble was finally receiving "trustworthy" inputs. Even so, we recognized a massive architectural truth: Because all models look at the exact same features, they are highly correlated. * The Narrative Shift: We chose to embrace the fact that Logistic Regression was often the best standalone performer. Rather than forcing a complex Stacking Meta-Learner, we documented that the pipeline proved the superiority of the simpler, calibrated linear model for this specific feature set—a highly professional engineering conclusion.

By the end of this phase, your ML pipeline was robust, transparent, heavily serialized (saving .json configs and .pkl models for reproducibility), and mathematically honest about its probability outputs.