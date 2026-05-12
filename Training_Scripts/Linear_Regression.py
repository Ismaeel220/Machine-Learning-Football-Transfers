import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, root_mean_squared_error, mean_absolute_error
import shap


# ═════════════════════════════════════════════
# HELPER FUNCTIONS
# ═════════════════════════════════════════════

# Calculates the main regression performance metrics used to evaluate the model.
def calculate_metrics(actual, predicted):
    """
    Calculates three performance metrics on the original unlogged transfer fee scale.

    MAE  : Average absolute prediction error in euros.
    RMSE : Similar to MAE, but penalises larger errors more heavily.
    R²   : Proportion of transfer fee variation explained by the model.
    """
    mae  = mean_absolute_error(actual, predicted)
    rmse = root_mean_squared_error(actual, predicted)
    r2   = r2_score(actual, predicted)
    return mae, rmse, r2


# Extracts the top N most influential features from a fitted Linear Regression model.
def get_top_features(model, feature_names, top_n=10):
    """
    Returns the top N features ranked by the absolute size of their coefficients.

    A positive coefficient increases the predicted transfer fee.
    A negative coefficient decreases the predicted transfer fee.
    A larger absolute coefficient means the feature has a stronger influence.
    """

    # Create a dataframe pairing each feature with its regression coefficient.
    feat_imp = pd.DataFrame({
        'Feature'    : feature_names,
        'Coefficient': model.coef_
    })

    # Sort features by absolute coefficient size so both large positive and negative effects are captured.
    feat_imp = feat_imp.reindex(
        feat_imp['Coefficient'].abs().sort_values(ascending=False).index
    ).head(top_n)

    return feat_imp


# Formats large numeric transfer fee values as readable euro amounts.
def format_currency(val):
    return f"€{val:,.0f}"


# ═════════════════════════════════════════════
# SHAP SUMMARY PLOT
# ═════════════════════════════════════════════

def run_shap_summary(model, X_train, X_test, dataset_name, phase_label):
    """
    Generates and saves a SHAP summary plot for a trained Linear Regression model.

    SHAP explains how each feature contributes to the model's predictions:
      • Features are ranked by overall impact.
      • Each dot represents one player.
      • Red dots indicate high feature values.
      • Blue dots indicate low feature values.
      • Dots to the right increase the predicted transfer fee.
      • Dots to the left decrease the predicted transfer fee.

    LinearExplainer is used because it is appropriate for Linear Regression models.
    """
    print(f"\n[Generating SHAP Summary Plot — {phase_label}: {dataset_name}...]")

    # Create a SHAP explainer using the training data as the background reference.
    explainer = shap.LinearExplainer(model, X_train)

    # Calculate SHAP values for the test set.
    shap_values = np.array(explainer.shap_values(X_test), dtype=float)

    # Generate a summary plot showing the 15 most influential features.
    shap.summary_plot(shap_values, X_test, max_display=15, show=False)

    plt.title(f"SHAP Summary — {phase_label}: {dataset_name}", fontsize=13)
    plt.tight_layout()

    # Save the SHAP plot as a PNG file for later analysis or dissertation use.
    filename = f"shap_{phase_label.replace(' ', '_').replace('(+', '').replace(')', '')}_{dataset_name}_Linear_Regression.png"
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"  Saved: {filename}")


# ═════════════════════════════════════════════
# MAIN PIPELINE
# ═════════════════════════════════════════════

# Runs the full Linear Regression modelling pipeline for all dataset versions.
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
    # PHASE 1
    # Train and evaluate all datasets with club identity excluded
    # ══════════════════════════════════════════════════════════════

    print("=" * 60)
    print("PHASE 1: LINEAR REGRESSION EVALUATION — Club Identity Excluded")
    print("=" * 60)

    # Train and evaluate a Linear Regression model for each dataset version.
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

        # Linear Regression does not use grid search here, so the model is fitted directly.
        model = LinearRegression()

        # Train the Linear Regression model on the training features and log transfer fees.
        model.fit(X_train, y_train_log)

        # Predict on unseen test data and convert predictions from log scale back to euros.
        y_pred_actual = np.exp(model.predict(X_test))

        # Evaluate predictive performance on the original transfer fee scale.
        mae, rmse, r2 = calculate_metrics(y_test_actual, y_pred_actual)

        # Extract the top 10 most influential features using coefficient size.
        top_features = get_top_features(model, X_train.columns)

        # Store all relevant outputs for this dataset.
        results[name] = {
            'MAE': mae,
            'RMSE': rmse,
            'R2': r2,
            'Top_Features': top_features,
            'Model': model,
            'X_Train': X_train,
            'X_Test': X_test,
        }

        # Track the dataset with the highest R² score for use in Phase 2.
        if r2 > best_r2:
            best_r2           = r2
            best_dataset_name = name
            best_dataset_data = (train_df, test_df)

    # ═════════════════════════════════════════════
    # PRINT PHASE 1 RESULTS
    # ═════════════════════════════════════════════

    # Display performance metrics and coefficient based feature importance for each dataset.
    for name, res in results.items():
        print(f"\n{'='*60}")
        print(f"  {name} Dataset — Results")
        print(f"{'='*60}")

        print(f"  MAE  : {format_currency(res['MAE'])}")
        print(f"  RMSE : {format_currency(res['RMSE'])}")
        print(f"  R²   : {res['R2']:.4f}")

        print(f"\n  Top 10 Features by coefficient size:")
        print(res['Top_Features'].to_string(index=False))

    # Generate SHAP summary plots for each Phase 1 model.
    for name, res in results.items():
        run_shap_summary(res['Model'], res['X_Train'], res['X_Test'], name, "Phase 1")

    # ══════════════════════════════════════════════════════════════
    # PHASE 2
    # Re run the best dataset with club identity included
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

    # Fit a new Linear Regression model using the best dataset with club identity included.
    model_p2 = LinearRegression()
    model_p2.fit(X_train_p2, y_train_log)

    # Predict test fees and convert predictions back from log scale to euros.
    y_pred_actual_p2 = np.exp(model_p2.predict(X_test_p2))

    # Evaluate the Phase 2 model using the same metrics as Phase 1.
    mae_p2, rmse_p2, r2_p2 = calculate_metrics(y_test_actual, y_pred_actual_p2)

    # Extract the most influential features from the Phase 2 model.
    top_features_p2 = get_top_features(model_p2, X_train_p2.columns)

    # Retrieve the Phase 1 results for the same dataset to allow direct comparison.
    p1 = results[best_dataset_name]

    print(f"\n{'='*60}")
    print(f"  {best_dataset_name} + Club Identity — Results")
    print(f"{'='*60}")

    # Print Phase 2 performance and show the change compared with Phase 1.
    print(f"  MAE  : {format_currency(mae_p2)}   (Change: {format_currency(mae_p2 - p1['MAE'])})")
    print(f"  RMSE : {format_currency(rmse_p2)}   (Change: {format_currency(rmse_p2 - p1['RMSE'])})")
    print(f"  R²   : {r2_p2:.4f}   (Change: {r2_p2 - p1['R2']:+.4f})")

    print(f"\n  Top 10 Features by coefficient size:")
    print(top_features_p2.to_string(index=False))

    # Generate a SHAP summary plot for the final Phase 2 model.
    run_shap_summary(model_p2, X_train_p2, X_test_p2, best_dataset_name, "Phase 2 (+ Club Identity)")


# Run the full pipeline only when this script is executed directly.
if __name__ == "__main__":
    run_pipeline()