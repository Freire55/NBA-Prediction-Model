"""
Runs the empirical evaluation battery corresponding to EXPERIMENTS_REPORT.md
on the latest trained models in models/run_2026-10-21_17-04-37/.
"""

import json
import logging
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.special import ndtr
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss, roc_auc_score, cohen_kappa_score
import joblib

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
MODELS_DIR = Path(__file__).resolve().parent.parent / "models"
RUN_DIR = sorted(MODELS_DIR.glob("run_*"))[-1] if list(MODELS_DIR.glob("run_*")) else MODELS_DIR / "example_run"

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger("experiments")

def load_models_and_data():
    logger.info("Loading test data and models...")
    df = pd.read_csv(DATA_DIR / "ml_ready_matchups_players.csv")
    df["HOME_SEASON_ID"] = df["HOME_SEASON_ID"].astype(str)
    test_df = df[df["HOME_SEASON_ID"] > "22021"].copy().reset_index(drop=True)
    y_test = (test_df["HOME_WL"] == "W").astype(int).values
    seasons = test_df["HOME_SEASON_ID"].values
    
    # Load model artifacts
    mlp_model = joblib.load(RUN_DIR / "mlp_model.pkl")
    mlp_scaler = joblib.load(RUN_DIR / "mlp_scaler.pkl")
    mlp_feats = joblib.load(RUN_DIR / "mlp_features.pkl")
    
    xgb_model = joblib.load(RUN_DIR / "xgb_model.pkl")
    xgb_feats = joblib.load(RUN_DIR / "xgb_features.pkl")
    
    cb_model = joblib.load(RUN_DIR / "catboost_model.pkl")
    cb_feats = joblib.load(RUN_DIR / "catboost_features.pkl")
    
    margin_model = joblib.load(RUN_DIR / "margin_model.pkl")
    margin_scaler = joblib.load(RUN_DIR / "margin_scaler.pkl")
    margin_feats = joblib.load(RUN_DIR / "margin_features.pkl")
    
    with open(RUN_DIR / "ensemble_formula.json") as f:
        ensemble_formula = json.load(f)
        
    logger.info(f"Loaded {len(test_df)} prospective test games across {sorted(np.unique(seasons))}")
    return test_df, y_test, seasons, mlp_model, mlp_scaler, mlp_feats, xgb_model, xgb_feats, cb_model, cb_feats, margin_model, margin_scaler, margin_feats, ensemble_formula

def compute_probabilities(test_df, mlp_model, mlp_scaler, mlp_feats, xgb_model, xgb_feats, cb_model, cb_feats, margin_model, margin_scaler, margin_feats, ensemble_formula):
    # MLP
    X_mlp = mlp_scaler.transform(test_df[mlp_feats])
    p_mlp = mlp_model.predict_proba(X_mlp)[:, 1]
    
    # XGB
    X_xgb = test_df[xgb_feats]
    p_xgb = xgb_model.predict_proba(X_xgb)[:, 1]
    
    # CatBoost
    X_cb = test_df[cb_feats]
    p_cb = cb_model.predict_proba(X_cb)[:, 1]
    
    # Margin
    X_margin = margin_scaler.transform(test_df[margin_feats])
    pace = test_df["MATCHUP_EXPECTED_PACE"].values if "MATCHUP_EXPECTED_PACE" in test_df.columns else None
    try:
        p_margin = margin_model.predict_proba(X_margin, pace=pace)[:, 1]
    except TypeError:
        p_margin = margin_model.predict_proba(X_margin)[:, 1]
        
    # Ensemble
    p_ens = (
        ensemble_formula.get("MLP", 0.0) * p_mlp +
        ensemble_formula.get("XGBoost", 0.0) * p_xgb +
        ensemble_formula.get("CatBoost", 0.0) * p_cb +
        ensemble_formula.get("Pace Margin (CDF)", 0.0) * p_margin
    )
    p_ens = np.clip(p_ens, 1e-15, 1 - 1e-15)
    
    probs = {
        "Ensemble": p_ens,
        "MLP": p_mlp,
        "XGBoost": p_xgb,
        "CatBoost": p_cb,
        "Margin": p_margin,
    }
    return probs

def calc_metrics(y_true, p):
    p_safe = np.clip(p, 1e-15, 1 - 1e-15)
    pred = (p_safe >= 0.5).astype(int)
    return {
        "Accuracy": float(accuracy_score(y_true, pred)),
        "LogLoss": float(log_loss(y_true, p_safe)),
        "Brier": float(brier_score_loss(y_true, p_safe)),
        "ROC_AUC": float(roc_auc_score(y_true, p_safe)),
    }

def run_experiment_1(test_df, y_test, seasons, probs):
    logger.info("\n=== EXPERIMENT 1: Core Robustness ===")
    results = {}
    
    # A. Per-Season
    per_season = {}
    season_map = {
        "22022": "2022–23",
        "22023": "2023–24",
        "22024": "2024–25",
        "22025": "2025–26",
    }
    for s_id in sorted(np.unique(seasons)):
        mask = (seasons == s_id)
        s_name = season_map.get(s_id, s_id)
        m = {model: calc_metrics(y_test[mask], probs[model][mask]) for model in probs}
        per_season[s_name] = {"games": int(mask.sum()), "metrics": m}
        logger.info(f"Season {s_name} ({mask.sum()} games): Ens Acc = {m['Ensemble']['Accuracy']*100:.2f}%, LogLoss = {m['Ensemble']['LogLoss']:.4f}, AUC = {m['Ensemble']['ROC_AUC']:.4f}")
    results["per_season"] = per_season
    
    # B. Bootstrap CIs (2,000 resamples)
    logger.info("Computing 2,000 bootstrap confidence intervals...")
    n = len(y_test)
    rng = np.random.default_rng(42)
    boot_indices = rng.integers(0, n, size=(2000, n))
    
    boot_res = {}
    for model_name, p in probs.items():
        accs = []
        losses = []
        briers = []
        aucs = []
        for idx in boot_indices:
            y_b = y_test[idx]
            p_b = p[idx]
            accs.append(accuracy_score(y_b, (p_b >= 0.5).astype(int)))
            losses.append(log_loss(y_b, np.clip(p_b, 1e-15, 1 - 1e-15)))
            briers.append(brier_score_loss(y_b, p_b))
            aucs.append(roc_auc_score(y_b, p_b))
        boot_res[model_name] = {
            "Accuracy_95CI": [float(np.percentile(accs, 2.5)), float(np.mean(accs)), float(np.percentile(accs, 97.5))],
            "LogLoss_95CI": [float(np.percentile(losses, 2.5)), float(np.mean(losses)), float(np.percentile(losses, 97.5))],
            "Brier_95CI": [float(np.percentile(briers, 2.5)), float(np.mean(briers)), float(np.percentile(briers, 97.5))],
            "AUC_95CI": [float(np.percentile(aucs, 2.5)), float(np.mean(aucs)), float(np.percentile(aucs, 97.5))],
        }
        logger.info(f"{model_name} Acc 95% CI: [{boot_res[model_name]['Accuracy_95CI'][0]*100:.2f}%, {boot_res[model_name]['Accuracy_95CI'][1]*100:.2f}%, {boot_res[model_name]['Accuracy_95CI'][2]*100:.2f}%]")
    results["bootstrap_ci"] = boot_res

    # C. Home vs Away Favored
    p_ens = probs["Ensemble"]
    home_fav = (p_ens >= 0.5)
    away_fav = ~home_fav
    results["home_favored"] = {
        "games": int(home_fav.sum()),
        "pct": float(home_fav.mean()),
        "accuracy": float(accuracy_score(y_test[home_fav], (p_ens[home_fav] >= 0.5).astype(int))),
        "log_loss": float(log_loss(y_test[home_fav], p_ens[home_fav])),
    }
    results["away_favored"] = {
        "games": int(away_fav.sum()),
        "pct": float(away_fav.mean()),
        "accuracy": float(accuracy_score(y_test[away_fav], (p_ens[away_fav] >= 0.5).astype(int))),
        "log_loss": float(log_loss(y_test[away_fav], p_ens[away_fav])),
    }

    # D. Probability Bucket Stratification
    fav_prob = np.maximum(p_ens, 1.0 - p_ens)
    fav_won = (p_ens >= 0.5) == (y_test == 1)
    bins = [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 1.00]
    bucket_res = []
    for i in range(len(bins) - 1):
        low, high = bins[i], bins[i+1]
        b_mask = (fav_prob >= low) & (fav_prob < high if high < 1.0 else fav_prob <= high)
        if b_mask.sum() > 0:
            exp_win = float(fav_prob[b_mask].mean())
            act_win = float(fav_won[b_mask].mean())
            b_loss = float(log_loss(y_test[b_mask], p_ens[b_mask]))
            b_brier = float(brier_score_loss(y_test[b_mask], p_ens[b_mask]))
            bucket_res.append({
                "bucket": f"{int(low*100)}–{int(high*100)}%",
                "games": int(b_mask.sum()),
                "pct": float(b_mask.mean()),
                "expected_win_pct": exp_win,
                "actual_win_pct": act_win,
                "gap": act_win - exp_win,
                "log_loss": b_loss,
                "brier": b_brier,
            })
    results["probability_buckets"] = bucket_res
    
    # E. Regime breakdown
    close_m = fav_prob < 0.58
    mod_m = (fav_prob >= 0.58) & (fav_prob < 0.70)
    heavy_m = fav_prob >= 0.70
    results["regimes"] = {
        "close": {"games": int(close_m.sum()), "acc": float(accuracy_score(y_test[close_m], (p_ens[close_m] >= 0.5).astype(int))), "log_loss": float(log_loss(y_test[close_m], p_ens[close_m]))},
        "moderate": {"games": int(mod_m.sum()), "acc": float(accuracy_score(y_test[mod_m], (p_ens[mod_m] >= 0.5).astype(int))), "log_loss": float(log_loss(y_test[mod_m], p_ens[mod_m]))},
        "heavy": {"games": int(heavy_m.sum()), "acc": float(accuracy_score(y_test[heavy_m], (p_ens[heavy_m] >= 0.5).astype(int))), "log_loss": float(log_loss(y_test[heavy_m], p_ens[heavy_m]))},
    }
    return results

def run_experiment_2(y_test, probs):
    logger.info("\n=== EXPERIMENT 2: Probability Quality & Calibration ===")
    p_ens = probs["Ensemble"]
    
    # ECE & MCE (10 quantile bins)
    n_bins = 10
    quantiles = np.linspace(0, 1, n_bins + 1)
    bin_edges = np.quantile(p_ens, quantiles)
    bin_edges[0] = 0.0
    bin_edges[-1] = 1.0
    
    ece = 0.0
    mce = 0.0
    for i in range(n_bins):
        b_mask = (p_ens >= bin_edges[i]) & (p_ens <= bin_edges[i+1])
        if b_mask.sum() > 0:
            conf = p_ens[b_mask].mean()
            acc = y_test[b_mask].mean()
            gap = abs(acc - conf)
            ece += (b_mask.sum() / len(y_test)) * gap
            mce = max(mce, gap)
            
    # Murphy Brier Decomposition: Brier = Reliability - Resolution + Uncertainty
    # Uncertainty = c * (1 - c) where c is base rate
    c = y_test.mean()
    uncertainty = c * (1.0 - c)
    
    # Discrete binning for empirical reliability and resolution
    rel = 0.0
    res = 0.0
    n = len(y_test)
    for i in range(n_bins):
        b_mask = (p_ens >= bin_edges[i]) & (p_ens <= bin_edges[i+1])
        n_k = b_mask.sum()
        if n_k > 0:
            o_k = y_test[b_mask].mean()
            p_k = p_ens[b_mask].mean()
            rel += (n_k / n) * ((p_k - o_k) ** 2)
            res += (n_k / n) * ((o_k - c) ** 2)
            
    # Entropy / Sharpness: mean normalized binary entropy
    entropy = - (p_ens * np.log2(p_ens) + (1 - p_ens) * np.log2(1 - p_ens)).mean()
    
    res_dict = {
        "ECE": float(ece),
        "MCE": float(mce),
        "Brier_Total": float(brier_score_loss(y_test, p_ens)),
        "Brier_Reliability": float(rel),
        "Brier_Resolution": float(res),
        "Brier_Uncertainty": float(uncertainty),
        "Mean_Entropy_Bits": float(entropy),
    }
    logger.info(f"ECE: {ece*100:.3f}% | MCE: {mce*100:.3f}%")
    logger.info(f"Brier: {res_dict['Brier_Total']:.5f} (Reliability: {rel:.5f}, Resolution: {res:.5f}, Uncertainty: {uncertainty:.5f})")
    return res_dict

def run_experiment_3(y_test, probs):
    logger.info("\n=== EXPERIMENT 3: Model Diversity & Agreement ===")
    models = ["MLP", "XGBoost", "CatBoost", "Margin"]
    
    # Correlation matrix
    corr_matrix = {}
    for m1 in models:
        corr_matrix[m1] = {}
        for m2 in models:
            corr_matrix[m1][m2] = float(np.corrcoef(probs[m1], probs[m2])[0, 1])
            
    # Cohen's Kappa & Disagreement
    preds = {m: (probs[m] >= 0.5).astype(int) for m in models}
    agree_matrix = {}
    for m1 in models:
        agree_matrix[m1] = {}
        for m2 in models:
            agree_matrix[m1][m2] = {
                "agreement_pct": float((preds[m1] == preds[m2]).mean()),
                "cohen_kappa": float(cohen_kappa_score(preds[m1], preds[m2]))
            }
            
    # Agreement splits across 4 base models
    stacked_preds = np.column_stack([preds[m] for m in models])
    sum_preds = stacked_preds.sum(axis=1) # 4 = all say 1, 0 = all say 0
    unanimous = (sum_preds == 4) | (sum_preds == 0)
    split_3_1 = (sum_preds == 3) | (sum_preds == 1)
    split_2_2 = (sum_preds == 2)
    
    p_ens = probs["Ensemble"]
    ens_pred = (p_ens >= 0.5).astype(int)
    
    splits_dict = {
        "unanimous_4_0": {
            "games": int(unanimous.sum()),
            "pct": float(unanimous.mean()),
            "acc": float(accuracy_score(y_test[unanimous], ens_pred[unanimous])),
        },
        "split_3_1": {
            "games": int(split_3_1.sum()),
            "pct": float(split_3_1.mean()),
            "acc": float(accuracy_score(y_test[split_3_1], ens_pred[split_3_1])),
        },
        "split_2_2": {
            "games": int(split_2_2.sum()),
            "pct": float(split_2_2.mean()),
            "acc": float(accuracy_score(y_test[split_2_2], ens_pred[split_2_2])),
        }
    }
    logger.info(f"Unanimous (4-0): {unanimous.sum()} games ({unanimous.mean()*100:.1f}%), Acc: {splits_dict['unanimous_4_0']['acc']*100:.2f}%")
    logger.info(f"Split (3-1): {split_3_1.sum()} games ({split_3_1.mean()*100:.1f}%), Acc: {splits_dict['split_3_1']['acc']*100:.2f}%")
    logger.info(f"Tie (2-2): {split_2_2.sum()} games ({split_2_2.mean()*100:.1f}%), Acc: {splits_dict['split_2_2']['acc']*100:.2f}%")
    
    return {"correlations": corr_matrix, "agreements": agree_matrix, "agreement_breakdown": splits_dict}

def run_experiment_5_stress(test_df, y_test, probs):
    logger.info("\n=== EXPERIMENT 5: Subpopulation Stress Testing ===")
    p_ens = probs["Ensemble"]
    ens_pred = (p_ens >= 0.5).astype(int)
    stress_results = {}
    
    # 1. Rest Advantage
    if "HOME_REST_DAYS" in test_df.columns and "AWAY_REST_DAYS" in test_df.columns:
        rest_diff = test_df["HOME_REST_DAYS"] - test_df["AWAY_REST_DAYS"]
        home_rest_adv = rest_diff > 0
        away_rest_adv = rest_diff < 0
        even_rest = rest_diff == 0
        stress_results["rest_advantage"] = {
            "home_rested": {"games": int(home_rest_adv.sum()), "acc": float(accuracy_score(y_test[home_rest_adv], ens_pred[home_rest_adv])), "log_loss": float(log_loss(y_test[home_rest_adv], p_ens[home_rest_adv]))},
            "away_rested": {"games": int(away_rest_adv.sum()), "acc": float(accuracy_score(y_test[away_rest_adv], ens_pred[away_rest_adv])), "log_loss": float(log_loss(y_test[away_rest_adv], p_ens[away_rest_adv]))},
            "equal_rest": {"games": int(even_rest.sum()), "acc": float(accuracy_score(y_test[even_rest], ens_pred[even_rest])), "log_loss": float(log_loss(y_test[even_rest], p_ens[even_rest]))},
        }

    # 2. Back to Back
    if "HOME_B2B" in test_df.columns and "AWAY_B2B" in test_df.columns:
        home_b2b = test_df["HOME_B2B"] == 1
        away_b2b = test_df["AWAY_B2B"] == 1
        stress_results["back_to_back"] = {
            "home_b2b": {"games": int(home_b2b.sum()), "acc": float(accuracy_score(y_test[home_b2b], ens_pred[home_b2b]))},
            "away_b2b": {"games": int(away_b2b.sum()), "acc": float(accuracy_score(y_test[away_b2b], ens_pred[away_b2b]))},
        }

    # 3. High vs Low Pace
    if "MATCHUP_EXPECTED_PACE" in test_df.columns:
        pace = test_df["MATCHUP_EXPECTED_PACE"]
        high_pace = pace >= pace.median()
        low_pace = pace < pace.median()
        stress_results["pace_regimes"] = {
            "high_pace": {"games": int(high_pace.sum()), "acc": float(accuracy_score(y_test[high_pace], ens_pred[high_pace])), "log_loss": float(log_loss(y_test[high_pace], p_ens[high_pace]))},
            "low_pace": {"games": int(low_pace.sum()), "acc": float(accuracy_score(y_test[low_pace], ens_pred[low_pace])), "log_loss": float(log_loss(y_test[low_pace], p_ens[low_pace]))},
        }
    return stress_results

def run_experiment_6_baselines(y_test, probs):
    logger.info("\n=== EXPERIMENT 6: Benchmarking Against Baselines ===")
    p_ens = probs["Ensemble"]
    
    # 1. Constant Home Baseline (historical mean winrate ~58%)
    home_prior = y_test.mean()
    p_home_prior = np.full_like(y_test, home_prior, dtype=float)
    base_home = calc_metrics(y_test, p_home_prior)
    
    ens_metrics = calc_metrics(y_test, p_ens)
    logger.info(f"Home Prior (Constant {home_prior*100:.1f}%): Acc = {base_home['Accuracy']*100:.2f}%, LogLoss = {base_home['LogLoss']:.4f}")
    logger.info(f"Meta-Ensemble Model: Acc = {ens_metrics['Accuracy']*100:.2f}%, LogLoss = {ens_metrics['LogLoss']:.4f}")
    logger.info(f"Ensemble Value-Add vs Home Baseline: +{ens_metrics['Accuracy'] - base_home['Accuracy']:.4f} Acc, -{base_home['LogLoss'] - ens_metrics['LogLoss']:.4f} LogLoss")
    
    return {
        "home_prior_baseline": base_home,
        "ensemble": ens_metrics,
        "advantage": {
            "accuracy_gain": ens_metrics["Accuracy"] - base_home["Accuracy"],
            "logloss_reduction": base_home["LogLoss"] - ens_metrics["LogLoss"],
        }
    }

def main():
    test_df, y_test, seasons, mlp_m, mlp_s, mlp_f, xgb_m, xgb_f, cb_m, cb_f, mar_m, mar_s, mar_f, formula = load_models_and_data()
    probs = compute_probabilities(test_df, mlp_m, mlp_s, mlp_f, xgb_m, xgb_f, cb_m, cb_f, mar_m, mar_s, mar_f, formula)
    
    exp1 = run_experiment_1(test_df, y_test, seasons, probs)
    exp2 = run_experiment_2(y_test, probs)
    exp3 = run_experiment_3(y_test, probs)
    exp5 = run_experiment_5_stress(test_df, y_test, probs)
    exp6 = run_experiment_6_baselines(y_test, probs)
    
    full_report = {
        "overall_test_metrics": {m: calc_metrics(y_test, probs[m]) for m in probs},
        "experiment_1_robustness": exp1,
        "experiment_2_probability_quality": exp2,
        "experiment_3_diversity": exp3,
        "experiment_5_stress": exp5,
        "experiment_6_baselines": exp6,
    }
    
    with open(MODELS_DIR / "new_experiments_evaluation.json", "w") as f:
        json.dump(full_report, f, indent=4)
        
    logger.info(f"\nSaved full evaluation suite to {MODELS_DIR / 'new_experiments_evaluation.json'}")

if __name__ == "__main__":
    main()
