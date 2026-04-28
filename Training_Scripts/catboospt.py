import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from catboost import CatBoostRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import r2_score, root_mean_squared_error, mean_absolute_error
import shap
import warnings

warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────
# HELPER FUNCTIONS
# ─────────────────────────────────────────────

def calculate_metrics(actual, predicted):
    """
    Calculates three accuracy metrics on real (un-logged) currency values.
    - MAE:  Average absolute error in euros
    - RMSE: Like MAE but penalises large errors more heavily
    - R²:   How much of the fee variation the model explains (1.0 = perfect)
    """
    mae  = mean_absolute_error(actual, predicted)
    rmse = root_mean_squared_error(actual, predicted)
    r2   = r2_score(actual, predicted)
    return mae, rmse, r2


def get_top_features(fitted_model, feature_names, top_n=10):
    """Returns the top N most important features from a fitted CatBoost model."""
    feat_imp = pd.DataFrame({
        'Feature'   : feature_names,
        'Importance': fitted_model.feature_importances_
    })
    feat_imp = feat_imp.sort_values('Importance', ascending=False).head(top_n)
    feat_imp['Importance'] = feat_imp['Importance'].apply(lambda x: f"{x:.4f}")
    return feat_imp


def format_currency(val):
    return f"€{val:,.0f}"


# ─────────────────────────────────────────────
# SHAP SUMMARY PLOT
# ─────────────────────────────────────────────

def run_shap_summary(model, X_test, dataset_name, phase_label):
    """
    Generates and saves a SHAP summary plot for the given model.

    SHAP (SHapley Additive exPlanations) shows which features had the biggest
    overall impact on predictions across all players:
      - Features are ranked top to bottom by importance
      - Each dot is one player
      - Red dots = high feature value, Blue dots = low feature value
      - Dots on the right = pushed the predicted fee UP
      - Dots on the left  = pushed the predicted fee DOWN

    Uses TreeExplainer which is optimised for CatBoost (fast on CPU).
    The plot is saved as a PNG you can drop straight into your dissertation.
    """
    print(f"\n[Generating SHAP Summary Plot — {phase_label}: {dataset_name}...]")

    explainer   = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)

    shap.summary_plot(shap_values, X_test, max_display=15, show=False)

    plt.title(f"SHAP Summary — {phase_label}: {dataset_name}", fontsize=13)
    plt.tight_layout()

    filename = f"shap_{phase_label.replace(' ', '_').replace('(+', '').replace(')', '')}_{dataset_name}_catboost.png"
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {filename}")


# ─────────────────────────────────────────────
# GRID SEARCH PARAMETERS
# ─────────────────────────────────────────────

# Every combination below is tested. GridSearchCV picks the one
# with the lowest cross-validated MAE.
PARAM_GRID = {
    'iterations'   : [200, 500],       # Number of boosting rounds
    'depth'        : [4, 6, 8],        # Depth of each tree
    'learning_rate': [0.05, 0.1],      # How much each tree contributes
}


# ─────────────────────────────────────────────
# MAIN PIPELINE
# ─────────────────────────────────────────────

def run_pipeline():

    datasets = {
        'Raw': ('training-testing-data/raw_train.csv',  'training-testing-data/raw_test.csv'),
        'EDA': ('training-testing-data/eda_train.csv',  'training-testing-data/eda_test.csv'),
        'PCA': ('training-testing-data/pca_train.csv',  'training-testing-data/pca_test.csv'),
    }

    results           = {}
    best_r2           = -np.inf
    best_dataset_name = None
    best_dataset_data = None

    # ══════════════════════════════════════════════════════════════
    # PHASE 1 — Train and evaluate all three datasets
    #           (club identity feature excluded)
    # ══════════════════════════════════════════════════════════════
    print("=" * 60)
    print("PHASE 1: CATBOOST + GRID SEARCH (Excl. Club Identity)")
    print("=" * 60)

    for name, (train_file, test_file) in datasets.items():

        try:
            train_df = pd.read_csv(train_file)
            test_df  = pd.read_csv(test_file)
        except FileNotFoundError:
            print(f"  [!] Files not found for '{name}' — skipping.")
            continue

        y_train_log   = train_df['transfer_fee_log']
        y_test_actual = test_df['transfer_fee']

        # Drop columns that are not features or would leak the target
        cols_to_drop = ['name', 'transfer_fee', 'transfer_fee_log', 'to_club_name_encoded']
        X_train = train_df.select_dtypes(exclude=['object', 'string'])
        X_test  = test_df.select_dtypes(exclude=['object', 'string'])
        X_train = X_train.drop(columns=[c for c in cols_to_drop if c in X_train.columns])
        X_test  = X_test.drop(columns=[c for c in cols_to_drop if c in X_test.columns])

        print(f"\n[Running Grid Search for {name} Dataset...]")

        # CatBoost is wrapped with verbose=0 so it doesn't flood the console
        grid_search = GridSearchCV(
            estimator  = CatBoostRegressor(verbose=0, random_seed=42),
            param_grid = PARAM_GRID,
            cv         = 3,
            scoring    = 'neg_mean_absolute_error',
            n_jobs     = -1,
            verbose    = 0
        )
        grid_search.fit(X_train, y_train_log)

        best_model  = grid_search.best_estimator_
        best_params = grid_search.best_params_

        y_pred_actual = np.exp(best_model.predict(X_test))
        mae, rmse, r2 = calculate_metrics(y_test_actual, y_pred_actual)
        top_features  = get_top_features(best_model, X_train.columns)

        results[name] = {
            'MAE': mae, 'RMSE': rmse, 'R2': r2,
            'Best_Params': best_params,
            'Top_Features': top_features,
            'Model': best_model,
            'X_Test': X_test,
        }

        if r2 > best_r2:
            best_r2           = r2
            best_dataset_name = name
            best_dataset_data = (train_df, test_df)

    # ── Print Phase 1 results ──────────────────────────────────────
    for name, res in results.items():
        print(f"\n{'='*60}")
        print(f"  {name} Dataset — Results")
        print(f"{'='*60}")
        print("  Best Hyperparameters:")
        for param, value in res['Best_Params'].items():
            print(f"    • {param}: {value}")
        print(f"  {'─'*40}")
        print(f"  MAE  : {format_currency(res['MAE'])}")
        print(f"  RMSE : {format_currency(res['RMSE'])}")
        print(f"  R²   : {res['R2']:.4f}")
        print(f"\n  Top 10 Features:")
        print(res['Top_Features'].to_string(index=False))

    # ── SHAP summary plot for each Phase 1 dataset ────────────────
    for name, res in results.items():
        run_shap_summary(res['Model'], res['X_Test'], name, "Phase 1")

    # ══════════════════════════════════════════════════════════════
    # PHASE 2 — Re-run best dataset WITH club identity feature
    # ══════════════════════════════════════════════════════════════
    if best_dataset_data is None:
        return

    print("\n" + "=" * 60)
    print(f"PHASE 2: ADDING CLUB IDENTITY — Best Dataset: {best_dataset_name}")
    print("=" * 60)

    train_df, test_df = best_dataset_data
    y_train_log   = train_df['transfer_fee_log']
    y_test_actual = test_df['transfer_fee']

    # Keep 'to_club_name_encoded' this time
    cols_to_drop_p2 = ['name', 'transfer_fee', 'transfer_fee_log']
    X_train_p2 = train_df.select_dtypes(exclude=['object', 'string'])
    X_test_p2  = test_df.select_dtypes(exclude=['object', 'string'])
    X_train_p2 = X_train_p2.drop(columns=[c for c in cols_to_drop_p2 if c in X_train_p2.columns])
    X_test_p2  = X_test_p2.drop(columns=[c for c in cols_to_drop_p2 if c in X_test_p2.columns])

    print("[Running Grid Search for Phase 2...]")

    grid_search_p2 = GridSearchCV(
        estimator  = CatBoostRegressor(verbose=0, random_seed=42),
        param_grid = PARAM_GRID,
        cv         = 3,
        scoring    = 'neg_mean_absolute_error',
        n_jobs     = -1,
        verbose    = 0
    )
    grid_search_p2.fit(X_train_p2, y_train_log)

    best_model_p2  = grid_search_p2.best_estimator_
    best_params_p2 = grid_search_p2.best_params_

    y_pred_actual_p2        = np.exp(best_model_p2.predict(X_test_p2))
    mae_p2, rmse_p2, r2_p2 = calculate_metrics(y_test_actual, y_pred_actual_p2)
    top_features_p2         = get_top_features(best_model_p2, X_train_p2.columns)

    p1 = results[best_dataset_name]

    print(f"\n{'='*60}")
    print(f"  {best_dataset_name} + Club Identity — Results")
    print(f"{'='*60}")
    print("  Best Hyperparameters:")
    for param, value in best_params_p2.items():
        print(f"    • {param}: {value}")
    print(f"  {'─'*40}")
    print(f"  MAE  : {format_currency(mae_p2)}   (Change: {format_currency(mae_p2 - p1['MAE'])})")
    print(f"  RMSE : {format_currency(rmse_p2)}   (Change: {format_currency(rmse_p2 - p1['RMSE'])})")
    print(f"  R²   : {r2_p2:.4f}   (Change: {r2_p2 - p1['R2']:+.4f})")
    print(f"\n  Top 10 Features:")
    print(top_features_p2.to_string(index=False))

    # ── SHAP summary plot for Phase 2 ─────────────────────────────
    run_shap_summary(best_model_p2, X_test_p2, best_dataset_name, "Phase 2 (+ Club Identity)")


# ─────────────────────────────────────────────
if __name__ == "__main__":
    run_pipeline()
