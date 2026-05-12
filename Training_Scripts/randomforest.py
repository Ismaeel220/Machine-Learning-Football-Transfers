import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import r2_score, root_mean_squared_error, mean_absolute_error
import shap


# ─────────────────────────────────────────────
# HELPER FUNCTIONS
# ─────────────────────────────────────────────

# Calculates the main regression performance metrics used to evaluate the model.
def calculate_metrics(actual, predicted):
    """
    Calculates three performance metrics on the original, unlogged transfer fee scale.

    MAE  : Average absolute prediction error in euros.
    RMSE : Similar to MAE, but penalises larger errors more heavily.
    R²   : Proportion of transfer fee variation explained by the model.
    """
    mae  = mean_absolute_error(actual, predicted)
    rmse = root_mean_squared_error(actual, predicted)
    r2   = r2_score(actual, predicted)
    return mae, rmse, r2


# Extracts the top N most important features from a fitted Random Forest model.
def get_top_features(fitted_model, feature_names, top_n=10):
    """
    Returns the top N features ranked by Random Forest's built in feature importance.
    """

    # Create a dataframe pairing each feature with its importance score.
    feat_imp = pd.DataFrame({
        'Feature'   : feature_names,
        'Importance': fitted_model.feature_importances_
    })

    # Sort features from most to least important and keep the top N.
    feat_imp = feat_imp.sort_values('Importance', ascending=False).head(top_n)

    # Format importance scores to four decimal places for cleaner output.
    feat_imp['Importance'] = feat_imp['Importance'].apply(lambda x: f"{x:.4f}")

    return feat_imp


# Formats large numeric transfer fee values as readable euro amounts.
def format_currency(val):
    return f"€{val:,.0f}"


# ─────────────────────────────────────────────
# SHAP SUMMARY PLOT
# ─────────────────────────────────────────────

def run_shap_summary(model, X_test, dataset_name, phase_label):
    """
    Generates and saves a SHAP summary plot for a trained Random Forest model.

    SHAP explains how each feature contributes to the model's predictions:
      - Features are ranked by overall impact.
      - Each dot represents one player.
      - Red dots indicate high feature values.
      - Blue dots indicate low feature values.
      - Dots to the right increase the predicted transfer fee.
      - Dots to the left decrease the predicted transfer fee.

    TreeExplainer is used because it is designed for tree based models such as Random Forest.
    """
    print(f"\n[Generating SHAP Summary Plot — {phase_label}: {dataset_name}...]")

    # Create a SHAP explainer for the trained Random Forest model.
    explainer = shap.TreeExplainer(model)

    # Calculate SHAP values for the test set.
    shap_values = explainer.shap_values(X_test)

    # Generate a summary plot showing the 15 most influential features.
    shap.summary_plot(shap_values, X_test, max_display=15, show=False)

    plt.title(f"SHAP Summary — {phase_label}: {dataset_name}", fontsize=13)
    plt.tight_layout()

    # Save the SHAP plot as a PNG file for later analysis or dissertation use.
    filename = f"shap_{phase_label.replace(' ', '_').replace('(+', '').replace(')', '')}_{dataset_name}_random_forest.png"
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"  Saved: {filename}")


# ─────────────────────────────────────────────
# GRID SEARCH PARAMETERS
# ─────────────────────────────────────────────

# Hyperparameter combinations tested by GridSearchCV.
# The best combination is selected using cross validated MAE.
PARAM_GRID = {
    'n_estimators' : [100, 200],        # Number of trees in the forest.
    'max_depth'    : [10, 20, None],    # Maximum depth of each tree; None allows unlimited depth.
    'max_features' : ['sqrt', 'log2'],  # Number of features considered when splitting each node.
}


# ─────────────────────────────────────────────
# MAIN PIPELINE
# ─────────────────────────────────────────────

# Runs the full Random Forest modelling pipeline for all dataset versions.
def run_pipeline():

    # Dictionary containing the training and testing files for each dataset version.
    datasets = {
        'Raw': ('training-testing-data/raw_train.csv',  'training-testing-data/raw_test.csv'),
        'EDA': ('training-testing-data/eda_train.csv',  'training-testing-data/eda_test.csv'),
        'PCA': ('training-testing-data/pca_train.csv',  'training-testing-data/pca_test.csv'),
    }

    # Stores the model results for each dataset.
    results = {}

    # Initial best R² is set to negative infinity so the first valid model becomes the baseline best model.
    best_r2 = -np.inf

    # Placeholders for the best performing dataset from Phase 1.
    best_dataset_name = None
    best_dataset_data = None

    # ══════════════════════════════════════════════════════════════
    # PHASE 1 — Train and evaluate all datasets
    #           Club identity feature excluded
    # ══════════════════════════════════════════════════════════════

    print("=" * 60)
    print("PHASE 1: RANDOM FOREST + GRID SEARCH — Club Identity Excluded")
    print("=" * 60)

    # Train and evaluate a Random Forest model for each dataset version.
    for name, (train_file, test_file) in datasets.items():

        # Load the training and testing data. If files are missing, skip that dataset.
        try:
            train_df = pd.read_csv(train_file)
            test_df  = pd.read_csv(test_file)
        except FileNotFoundError:
            print(f"  [!] Files not found for '{name}' — skipping.")
            continue

        # Use log transformed transfer fee for training and actual transfer fee for evaluation.
        y_train_log   = train_df['transfer_fee_log']
        y_test_actual = test_df['transfer_fee']

        # Remove identifier columns, target columns, and club identity to prevent leakage in Phase 1.
        cols_to_drop = ['name', 'transfer_fee', 'transfer_fee_log', 'to_club_name_encoded']

        # Keep only numeric features, as non numeric features are not used in this pipeline.
        X_train = train_df.select_dtypes(exclude=['object', 'string'])
        X_test  = test_df.select_dtypes(exclude=['object', 'string'])

        # Drop non feature columns, checking they exist first to avoid errors.
        X_train = X_train.drop(columns=[c for c in cols_to_drop if c in X_train.columns])
        X_test  = X_test.drop(columns=[c for c in cols_to_drop if c in X_test.columns])

        print(f"\n[Running Grid Search for {name} Dataset...]")

        # Set up Random Forest with GridSearchCV to tune hyperparameters using 3 fold cross validation.
        grid_search = GridSearchCV(
            estimator  = RandomForestRegressor(random_state=42),
            param_grid = PARAM_GRID,
            cv         = 3,
            scoring    = 'neg_mean_absolute_error',
            n_jobs     = -1,
            verbose    = 0
        )

        # Train all hyperparameter combinations and select the best model based on validation MAE.
        grid_search.fit(X_train, y_train_log)

        # Extract the best trained Random Forest model and its selected hyperparameters.
        best_model  = grid_search.best_estimator_
        best_params = grid_search.best_params_

        # Predict on unseen test data and convert predictions from log scale back to euros.
        y_pred_actual = np.exp(best_model.predict(X_test))

        # Evaluate predictive performance on the original transfer fee scale.
        mae, rmse, r2 = calculate_metrics(y_test_actual, y_pred_actual)

        # Extract the top 10 most important features from the trained model.
        top_features = get_top_features(best_model, X_train.columns)

        # Store all relevant outputs for this dataset.
        results[name] = {
            'MAE': mae,
            'RMSE': rmse,
            'R2': r2,
            'Best_Params': best_params,
            'Top_Features': top_features,
            'Model': best_model,
            'X_Test': X_test,
        }

        # Track the dataset with the highest R² score for use in Phase 2.
        if r2 > best_r2:
            best_r2           = r2
            best_dataset_name = name
            best_dataset_data = (train_df, test_df)

    # ── Print Phase 1 results ──────────────────────────────────────

    # Display performance metrics, best hyperparameters, and feature importance for each dataset.
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

    # Generate SHAP summary plots for each Phase 1 model.
    for name, res in results.items():
        run_shap_summary(res['Model'], res['X_Test'], name, "Phase 1")

    # ══════════════════════════════════════════════════════════════
    # PHASE 2 — Re run best dataset with club identity included
    # ══════════════════════════════════════════════════════════════

    # Stop the pipeline if no valid dataset was successfully trained in Phase 1.
    if best_dataset_data is None:
        return

    print("\n" + "=" * 60)
    print(f"PHASE 2: ADDING CLUB IDENTITY — Best Dataset: {best_dataset_name}")
    print("=" * 60)

    # Retrieve the best performing train and test data from Phase 1.
    train_df, test_df = best_dataset_data

    # Define the target variables again for Phase 2.
    y_train_log   = train_df['transfer_fee_log']
    y_test_actual = test_df['transfer_fee']

    # In Phase 2, only identifier and target columns are removed.
    # The club identity feature is kept to test its effect on model performance.
    cols_to_drop_p2 = ['name', 'transfer_fee', 'transfer_fee_log']

    # Keep only numeric features.
    X_train_p2 = train_df.select_dtypes(exclude=['object', 'string'])
    X_test_p2  = test_df.select_dtypes(exclude=['object', 'string'])

    # Drop identifier and target columns while retaining the club identity feature.
    X_train_p2 = X_train_p2.drop(columns=[c for c in cols_to_drop_p2 if c in X_train_p2.columns])
    X_test_p2  = X_test_p2.drop(columns=[c for c in cols_to_drop_p2 if c in X_test_p2.columns])

    print("[Running Grid Search for Phase 2...]")

    # Repeat grid search because adding a new feature can change the best hyperparameters.
    grid_search_p2 = GridSearchCV(
        estimator  = RandomForestRegressor(random_state=42),
        param_grid = PARAM_GRID,
        cv         = 3,
        scoring    = 'neg_mean_absolute_error',
        n_jobs     = -1,
        verbose    = 0
    )

    # Train the Phase 2 model using the best dataset with club identity included.
    grid_search_p2.fit(X_train_p2, y_train_log)

    # Extract the best Phase 2 model and its selected hyperparameters.
    best_model_p2  = grid_search_p2.best_estimator_
    best_params_p2 = grid_search_p2.best_params_

    # Predict test fees and convert predictions back from log scale to euros.
    y_pred_actual_p2 = np.exp(best_model_p2.predict(X_test_p2))

    # Evaluate the Phase 2 model using the same metrics as Phase 1.
    mae_p2, rmse_p2, r2_p2 = calculate_metrics(y_test_actual, y_pred_actual_p2)

    # Extract the most important features from the Phase 2 model.
    top_features_p2 = get_top_features(best_model_p2, X_train_p2.columns)

    # Retrieve the Phase 1 results for the same dataset to allow direct comparison.
    p1 = results[best_dataset_name]

    print(f"\n{'='*60}")
    print(f"  {best_dataset_name} + Club Identity — Results")
    print(f"{'='*60}")

    print("  Best Hyperparameters:")
    for param, value in best_params_p2.items():
        print(f"    • {param}: {value}")

    print(f"  {'─'*40}")

    # Print Phase 2 performance and show the change compared with Phase 1.
    print(f"  MAE  : {format_currency(mae_p2)}   (Change: {format_currency(mae_p2 - p1['MAE'])})")
    print(f"  RMSE : {format_currency(rmse_p2)}   (Change: {format_currency(rmse_p2 - p1['RMSE'])})")
    print(f"  R²   : {r2_p2:.4f}   (Change: {r2_p2 - p1['R2']:+.4f})")

    print(f"\n  Top 10 Features:")
    print(top_features_p2.to_string(index=False))

    # Generate a SHAP summary plot for the final Phase 2 model.
    run_shap_summary(best_model_p2, X_test_p2, best_dataset_name, "Phase 2 (+ Club Identity)")


# Run the full pipeline only when this script is executed directly.
if __name__ == "__main__":
    run_pipeline()