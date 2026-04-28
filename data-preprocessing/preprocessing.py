import pandas as pd
import numpy as np
from category_encoders import LeaveOneOutEncoder
from sklearn.preprocessing import RobustScaler
from sklearn.decomposition import PCA
import os
from pathlib import Path


# 1. Load the transfer Dataset
df = pd.read_csv('/training-testing-data/transfer_dataset.csv')

# split data based on seasons for testing and training
train_df = df[df['season'].isin([2020, 2021, 2022, 2023])].copy()
test_df = df[df['season'] == 2024].copy()

# =========================================================================
# 1. RAW DATA GENERATION
# =========================================================================

# FILTER OUT GOALKEEPERS
train_df = train_df[train_df['position'] != 'Goalkeeper'].copy()
test_df = test_df[test_df['position'] != 'Goalkeeper'].copy()

# EXPLICITLY DROP GOALKEEPER COLUMNS
gk_cols_to_drop = ['% Penalty saves', 'Crosses Stopped', 'Saves', 'Saves %', 'Clean Sheets', '% Clean sheets']
train_df = train_df.drop(columns=gk_cols_to_drop, errors='ignore')
test_df = test_df.drop(columns=gk_cols_to_drop, errors='ignore')

# method to take averages for each position for each stat
def impute_combined_averages(train_data, test_data):
    numeric_cols = train_data.select_dtypes(include=[np.number]).columns
    
    for pos in train_data['position'].unique():
        pos_means = train_data[train_data['position'] == pos][numeric_cols].mean()
        train_loc = train_data['position'] == pos
        test_loc = test_data['position'] == pos
        
        train_data.loc[train_loc, numeric_cols] = train_data.loc[train_loc, numeric_cols].fillna(pos_means)
        test_data.loc[test_loc, numeric_cols] = test_data.loc[test_loc, numeric_cols].fillna(pos_means)
            
    train_data[numeric_cols] = train_data[numeric_cols].fillna(0)
    test_data[numeric_cols] = test_data[numeric_cols].fillna(0)
    return train_data, test_data

train_df, test_df = impute_combined_averages(train_df, test_df)

# 4. One-Hot Encode Player Positions
train_df = pd.get_dummies(train_df, columns=['position'], drop_first=False)
test_df = pd.get_dummies(test_df, columns=['position'], drop_first=False)
train_df, test_df = train_df.align(test_df, join='left', axis=1, fill_value=0)

# APPLY LOG TRANSFORMATION AND FEATURE ENGINEERING TO RAW DATA
train_df['transfer_fee_log'] = np.log1p(train_df['transfer_fee'])
test_df['transfer_fee_log'] = np.log1p(test_df['transfer_fee'])

train_df['age_squared'] = train_df['age'] ** 2
test_df['age_squared'] = test_df['age'] ** 2

loo_encoder = LeaveOneOutEncoder(cols=['to_club_name'])
train_df['to_club_name_encoded'] = loo_encoder.fit_transform(train_df['to_club_name'], train_df['transfer_fee_log'])
test_df['to_club_name_encoded'] = loo_encoder.transform(test_df['to_club_name'])

# define target folder
save_dir = Path.cwd() / "training-testing-data"
os.makedirs(save_dir, exist_ok=True)

# save initial raw data
train_df.to_csv(save_dir / 'raw_train.csv', index=False)
test_df.to_csv(save_dir / 'raw_test.csv', index=False)

print("Raw train and test datasets generated (including age_squared and to_club_name_encoded)")

# Keep copies of the clean raw data to branch off for EDA and PCA
raw_train_df = train_df.copy()
raw_test_df = test_df.copy()

# =========================================================================
# 2. EDA DATASET GENERATION (Branched from Raw)
# =========================================================================
print("\nGenerating EDA Dataset...")

eda_train_df = raw_train_df.copy()
eda_test_df = raw_test_df.copy()

# Define all characteristics that must survive EDA filtering
protected_features = [
    'name', 'to_club_name', 'season', 'transfer_fee', 'transfer_fee_log', 'to_club_name_encoded',
    'age', 'age_squared', 'days_left_on_contract', 'has_advanced_stats',
    'height_in_cm', 'is_left_footed', 'is_tier1_nation',
    'position_Attack', 'position_Defender', 'position_Midfield'
]

target_corrs = eda_train_df.select_dtypes(include=[np.number]).corr()['transfer_fee_log'].abs()

# ---------------------------------------------------------
# A. Drop mostly empty columns (Safeguarding protected features)
# ---------------------------------------------------------
threshold = 0.99
numeric_cols = eda_train_df.select_dtypes(include=[np.number]).columns
cols_to_drop_empty = []

for col in numeric_cols:
    if col in protected_features:
        continue
    percentage_empty = (eda_train_df[col].isna() | (eda_train_df[col] == 0)).mean()
    if percentage_empty >= threshold:
        cols_to_drop_empty.append(col)

eda_train_df = eda_train_df.drop(columns=cols_to_drop_empty, errors='ignore')
eda_test_df = eda_test_df[eda_train_df.columns]

# ---------------------------------------------------------
# B. Drop redundant correlated columns (Safeguarding protected features)
# ---------------------------------------------------------
import pandas as pd
import numpy as np

# 1. Create the Correlation Matrix and isolate the Upper Triangle
corr_matrix = eda_train_df.select_dtypes(include=[np.number]).corr().abs()
upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))

dropped_corr = set()
drop_log = [] # Track the exact comparisons

# 2. Iterate and apply feature selection logic
for col in upper.columns:
    # Skip if the base column is protected
    if col in protected_features:
        continue 
        
    high_corr_features = upper.index[upper[col] >= 0.90].tolist()
    for f in high_corr_features:
        # Skip if the correlated counterpart is protected
        if f in protected_features:
            continue 
            
        # Skip if either feature has already been processed and dropped
        if col in dropped_corr or f in dropped_corr:
            continue
            
        # Get target correlations for the tie-breaker
        tc1, tc2 = target_corrs.get(col, 0), target_corrs.get(f, 0)
        
        # Determine winner and loser explicitly based on target correlation
        if tc2 >= tc1:
            drop_feature, kept_feature = col, f
            drop_tc, keep_tc = tc1, tc2
        else:
            drop_feature, kept_feature = f, col
            drop_tc, keep_tc = tc2, tc1
            
        dropped_corr.add(drop_feature)
        
        # Record the exact comparison for the report
        drop_log.append({
            'Dropped Feature': drop_feature,
            'Kept Feature': kept_feature,
            'Inter-Feature Corr': round(upper[col][f], 4),
            'Dropped Target Corr': round(drop_tc, 4),
            'Kept Target Corr': round(keep_tc, 4)
        })

# 3. Execute the drops
outfield_drops = list(dropped_corr)
eda_train_df = eda_train_df.drop(columns=outfield_drops, errors='ignore')
eda_test_df = eda_test_df.drop(columns=outfield_drops, errors='ignore')

# 4. Save EDA Data
eda_train_df.to_csv(save_dir / 'eda_train.csv', index=False)
eda_test_df.to_csv(save_dir / 'eda_test.csv', index=False)

# 5. Output Summary & Reports
print("EDA Datasets Generated!\n")

print(f"--- Dropped {len(cols_to_drop_empty)} Mostly Empty Stats ---")
print(cols_to_drop_empty)
print("\n")

print(f"--- Dropped {len(outfield_drops)} Highly Correlated Stats ---")
if drop_log:
    # Convert the log to a DataFrame for a clean, tabular printout
    report_df = pd.DataFrame(drop_log)
    
    # Sort by the highest inter-feature correlation to see the worst offenders first
    report_df = report_df.sort_values(by='Inter-Feature Corr', ascending=False).reset_index(drop=True)
    
    # Print the DataFrame 
    print(report_df.to_string())
else:
    print("No correlated features were dropped.")
    print(outfield_drops) # Prints the empty list to match your original output
print("\n")
# =========================================================
# 3. PRINCIPAL COMPONENT ANALYSIS (PCA) ON RAW DATA
# =========================================================
print("\nGenerating PCA Dataset from RAW Data...")

# Use the copies we saved from the raw data step
pca_train_df = raw_train_df.copy()
pca_test_df = raw_test_df.copy()

# Define columns that should not be fed into PCA
protected_cols_pca = [
    'name', 'to_club_name', 'season', 'transfer_fee', 'transfer_fee_log', 'to_club_name_encoded',
    'age', 'age_squared', 'days_left_on_contract', 'has_advanced_stats',
    'height_in_cm', 'is_left_footed', 'is_tier1_nation',
    'position_Attack', 'position_Defender', 'position_Midfield'
]
# ROBUST SCALING
def scale_features(train_data, test_data):
    non_scale_cols = protected_cols_pca
    bool_cols = train_data.select_dtypes(include=['bool']).columns.tolist()
    
    cols_to_scale = [c for c in train_data.select_dtypes(include=[np.number]).columns 
                     if c not in non_scale_cols + bool_cols and train_data[c].nunique() > 2]
    
    scaler = RobustScaler()
    train_data[cols_to_scale] = scaler.fit_transform(train_data[cols_to_scale])
    test_data[cols_to_scale] = scaler.transform(test_data[cols_to_scale])
    
    return train_data, test_data

# APPLY SCALING TO RAW DATA
pca_train_df, pca_test_df = scale_features(pca_train_df, pca_test_df)

# Isolate protected features
admin_train = pca_train_df[[c for c in protected_cols_pca if c in pca_train_df.columns]].reset_index(drop=True)
admin_test = pca_test_df[[c for c in protected_cols_pca if c in pca_test_df.columns]].reset_index(drop=True)

# Isolate features for PCA
feature_cols = [c for c in pca_train_df.columns if c not in protected_cols_pca and pca_train_df[c].dtype in [np.float64, np.float32, np.int64, np.int32]]
X_train = pca_train_df[feature_cols].reset_index(drop=True)
X_test = pca_test_df[feature_cols].reset_index(drop=True)

# Fit PCA
pca = PCA(n_components=0.95, random_state=42)
train_pcs = pca.fit_transform(X_train)
test_pcs = pca.transform(X_test)

# Reassemble Dataframes
pc_columns = [f"PC{i+1}" for i in range(train_pcs.shape[1])]
train_pca_df = pd.DataFrame(train_pcs, columns=pc_columns)
test_pca_df = pd.DataFrame(test_pcs, columns=pc_columns)

pca_train_final = pd.concat([admin_train, train_pca_df], axis=1)
pca_test_final = pd.concat([admin_test, test_pca_df], axis=1)

print(f"-> Dimensionality Reduction: {len(feature_cols)} raw stats compressed into {pca.n_components_} Components.")

outfield_loadings = pd.DataFrame(pca.components_.T, columns=pc_columns, index=feature_cols)
outfield_loadings.to_csv(save_dir / 'pca_loadings.csv')
pca_train_final.to_csv(save_dir / 'pca_train.csv', index=False)
pca_test_final.to_csv(save_dir / 'pca_test.csv', index=False)

print("PCA Datasets and Full Loading CSVs successfully saved!")