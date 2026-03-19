import pandas as pd
import numpy as np
from category_encoders import LeaveOneOutEncoder as LeaveOneOutEncoder
from sklearn.preprocessing import RobustScaler


# 1. Load the transfer Dataset
df = pd.read_csv('/../training-testing-data/transfer_dataset.csv')

# split data based on seasons for tetsing and trainign
# Train on 2020-2023, Test on 2024 as we want to predicit future transfers
train_df = df[df['season'].isin([2020, 2021, 2022, 2023])].copy()
test_df = df[df['season'] == 2024].copy()

# method to take averages for each posiiton for each stat
def impute_combined_averages(train_data, test_data):
    #get only numeric data
    numeric_cols = train_data.select_dtypes(include=[np.number]).columns
    
    # loop throug each unique position
    for pos in train_data['position'].unique():
        # Get the mean stats for this specific position from the training data
        pos_means = train_data[train_data['position'] == pos][numeric_cols].mean()
        
# Find exactly which rows in both datasets belong to this position        
        train_loc = train_data['position'] == pos
        test_loc = test_data['position'] == pos
        
# Fill in the missing values  with the position average we just calculated
        #  apply the train_data average to  the train_data and the test_data        
        train_data.loc[train_loc, numeric_cols] = train_data.loc[train_loc, numeric_cols].fillna(pos_means)
        test_data.loc[test_loc, numeric_cols] = test_data.loc[test_loc, numeric_cols].fillna(pos_means)
            
# If a stat is completely blank for a certain position in the training data 
    # (like Goalkeepers having zero data for "Crosses"), the mean will be NaN.
    # replace them with 0
    train_data[numeric_cols] = train_data[numeric_cols].fillna(0)
    test_data[numeric_cols] = test_data[numeric_cols].fillna(0)
        
    return train_data, test_data

#apply and run the method to the datatsets

train_df, test_df = impute_combined_averages(train_df, test_df)

# 4. One-Hot Encode Player Positions
# pd.get_dummies creates new binary columns (e.g., "position_Defender", "position_Midfield")
# where a 1 means player is that position and 0 means they are not 
train_df = pd.get_dummies(train_df, columns=['position'], drop_first=False)
test_df = pd.get_dummies(test_df, columns=['position'], drop_first=False)

# if a position existed in the Train data, but nobody in the Test data plays that position?
# .align() ensures both datasets have the exact same columns. 
# If a column is missing in one, it creates it and fills it with 0s.
train_df, test_df = train_df.align(test_df, join='left', axis=1, fill_value=0)

# welath tax based on club player transferred to 
loo_encoder = LeaveOneOutEncoder(cols=['to_club_name'])

# # .fit_transform looks at the training data, calculates the average 'transfer_fee' 
# each club pays, and replaces the club name with that average number.
# leave one out encoding excludes the current rows fee from the average to avoid data leakage and overfitting
train_df['to_club_name_encoded'] = loo_encoder.fit_transform(train_df['to_club_name'], train_df['transfer_fee'])

# unseen lubs get the global average
test_df['to_club_name_encoded'] = loo_encoder.transform(test_df['to_club_name'])

train_df['transfer_fee_log'] = np.log1p(train_df['transfer_fee'])
test_df['transfer_fee_log'] = np.log1p(test_df['transfer_fee'])

# save initial raw data
train_df.to_csv('raw_train.csv', index=False)
test_df.to_csv('raw_test.csv', index=False)

print(" Raw train and test datasets  generated")

# =========================================================================
# Data set 2 generation based of eda 
# =========================================================================
#normalise data
# transfer fees wer ehevaily right skewed so apply a log scale to transfer fee
train_df['transfer_fee_log'] = np.log1p(train_df['transfer_fee'])
test_df['transfer_fee_log'] = np.log1p(test_df['transfer_fee'])

#age has s a non linear realtionship with transfer fee so we will use age squared 
train_df['age_squared'] = train_df['age'] ** 2
test_df['age_squared'] = test_df['age'] ** 2

# Calculate how strongly every single feature correlates with the  log-transformed target.
#  save this to use as a comparison when comparing columns that will be dropped 
numeric_cols_global = train_df.select_dtypes(include=[np.number]).columns
target_corrs = train_df[numeric_cols_global].corr()['transfer_fee_log'].abs()

#split data based on goalkeepers and outfield players  so will train these models sperately 
gk_train = train_df[train_df['position_Goalkeeper'] == 1].copy()
gk_test = test_df[test_df['position_Goalkeeper'] == 1].copy()
position_cols = [col for col in train_df.columns if 'position_' in col]
gk_train = gk_train.drop(columns=position_cols)
gk_test = gk_test.drop(columns=position_cols)

outfield_train = train_df[train_df['position_Goalkeeper'] == 0].copy()
outfield_test = test_df[test_df['position_Goalkeeper'] == 0].copy()

#drop features with 0 correlation with the target 
#if 99 percent of a colummn for a specific feature is blank then it is dropped as correlation to the traget is 0
threshold = 0.99
#protection incase these columns are empty save them in an array so  can be sure they are not deleted 
protected_cols = ['transfer_fee', 'transfer_fee_log', 'age', 'days_left_on_contract']


#get only numeric data 
gk_numeric_cols = gk_train.select_dtypes(include=[np.number]).columns
#empty list in which all clumns that will be dropped for the goalkeeper dataset will be stored
gk_cols_to_drop = []

#loop through all the nueric goalkeepr columns 
for col in gk_numeric_cols:
    #skip columns defined earlier that should do not be dropped
    if col in protected_cols:
        continue
        
    # Calculate what percentage of the column is completely Zero or NaN
    percentage_empty = (gk_train[col].isna() | (gk_train[col] == 0)).mean()
    
    # If 99%+ of the goalkeepers have a zero/blank for that feature add it to the list of features to drop
    if percentage_empty >= threshold:
        gk_cols_to_drop.append(col)

print(f"dropping {(gk_cols_to_drop)}  irrelevant columns with 0 correlation to target for Goalkeepers...")

# Drop the columns from the training dataset
gk_train = gk_train.drop(columns=gk_cols_to_drop, errors='ignore')
#force the testing dataset to match the columns in the training dataset
gk_test = gk_test[gk_train.columns] 


#get all numeric columns for outfield players
outfield_numeric_cols = outfield_train.select_dtypes(include=[np.number]).columns
#empty list for outfield columns that we will drop
outfield_cols_to_drop = []

#loop through all ourfield columns if in the list of features that should neve rbe dropped skip
for col in outfield_numeric_cols:
    if col in protected_cols:
        continue
        
    # Calculate what percentage of the column is completely Zero or NaN
    percentage_empty = (outfield_train[col].isna() | (outfield_train[col] == 0)).mean()
    
    # If 99%+ of the outfielders have a zero/blank add it to the 
    if percentage_empty >= threshold:
        outfield_cols_to_drop.append(col)

print(f"dropping {(outfield_cols_to_drop)}  irrelevant columns for Outfielders with 0 correlation with the target...")

# Drop from Training dataset
outfield_train = outfield_train.drop(columns=outfield_cols_to_drop, errors='ignore')
#force the testing dataset to copy the features in the training dataset
outfield_test = outfield_test[outfield_train.columns]

#create an instance of leave one out encoder and specify target
loo_encoder_eda = LeaveOneOutEncoder(cols=['to_club_name'])

#.fit_transform():encoder studies the training data and works out each clubs average spend
#  exclusing that specific column and swaps clubs names for the average
outfield_train['to_club_name_encoded'] = loo_encoder_eda.fit_transform(outfield_train['to_club_name'], outfield_train['transfer_fee_log'])
#applies the encoding from the training data to the tesing data to avoid leaking data (transform)
outfield_test['to_club_name_encoded'] = loo_encoder_eda.transform(outfield_test['to_club_name'])

#repeat same process for goalkeeping data
gk_train['to_club_name_encoded'] = loo_encoder_eda.fit_transform(gk_train['to_club_name'], gk_train['transfer_fee_log'])
gk_test['to_club_name_encoded'] = loo_encoder_eda.transform(gk_test['to_club_name'])

# ---------------------------------------------------------
# 5. ROBUST SCALING
# ---------------------------------------------------------
def scale_features(train_data, test_data):
    # exclude the target and non numeric columns those will not be scaled 
    non_scale_cols = ['name', 'to_club_name', 'season', 'transfer_fee', 'transfer_fee_log']
    #exclude boolean columns also 
    bool_cols = train_data.select_dtypes(include=['bool']).columns.tolist()
    
    # Select only continuous numeric columns with more than 2 unique values.
    cols_to_scale = [c for c in train_data.select_dtypes(include=[np.number]).columns 
                     if c not in non_scale_cols + bool_cols and train_data[c].nunique() > 2]
    
    #create a robust scaler inatance
    scaler = RobustScaler()
    #train and apply scaler on training data
    train_data[cols_to_scale] = scaler.fit_transform(train_data[cols_to_scale])
    #apply on testing data 
    test_data[cols_to_scale] = scaler.transform(test_data[cols_to_scale])
    return train_data, test_data

# Apply the scaler to both datasets
outfield_train, outfield_test = scale_features(outfield_train, outfield_test)
gk_train, gk_test = scale_features(gk_train, gk_test)

# drop features that are highly correlated with one naother and hence show the same thing 
# If two features are basically identical (e.g., "Passes Attempted" and "Passes Completed"), 
#  drops redundant stats reduce curse of over dimensionality
def correlated_features_to_drop(corr_file, target_corrs_dict, threshold=0.90):
    # Read a pre generated CSV containing feature to feature correlations
    corr_df = pd.read_csv(corr_file)
    corr_df['Correlation'] = corr_df['Correlation'].abs()
    
    # Find feature pairs that have a correlation higher than 0.90 but exclude 1.0 as that is a feature compared to itself 
    high_corr = corr_df[(corr_df['Correlation'] >= threshold) & (corr_df['Correlation'] < 0.999)].sort_values(by='Correlation', ascending=False)
    
    #store features that will be dropped in a set (no duplicates)
    dropped = set()

    #loop through the correlation table and get each rows number and data
    for row_number, row_data in high_corr.iterrows():
        #extract the names of the features 
        f1, f2 = row_data['Feature 1'], row_data['Feature 2']
        
        # If we already dropped one of the features skip to the next pair
        if f1 in dropped or f2 in dropped:
            continue
            
        # Get how well each feature predicts the transfer fee if cannot find it give 0
        tc1, tc2 = target_corrs_dict.get(f1, 0), target_corrs_dict.get(f2, 0)
        
        # Drop the feature that is weaker at predicting the transfer fee
        drop_feature = f2 if tc1 >= tc2 else f1
        dropped.add(drop_feature)
        
    return list(dropped)

# Get the lists of features to drop based on the external correlation CSVs
outfield_drops = correlated_features_to_drop('./exploratory-data-analysis/outfield_pairwise_correlations.csv', target_corrs)
gk_drops = correlated_features_to_drop('./exploratory-data-analysis/gk_pairwise_correlations.csv', target_corrs)

# Drop those redundant features from the datasets
outfield_train = outfield_train.drop(columns=outfield_drops, errors='ignore')
outfield_test = outfield_test.drop(columns=outfield_drops, errors='ignore')
gk_train = gk_train.drop(columns=gk_drops, errors='ignore')
gk_test = gk_test.drop(columns=gk_drops, errors='ignore')

# 7. SAVE FINAL EDA DATASETS

gk_train.to_csv('gk_eda_train.csv', index=False)
gk_test.to_csv('gk_eda_test.csv', index=False)
outfield_train.to_csv('outfield_eda_train.csv', index=False)
outfield_test.to_csv('outfield_eda_test.csv', index=False)

print(f" EDA Datasets Generated! Dropped {(outfield_drops)} Outfield stats and {(gk_drops)} GK stats.")

# =========================================================
# 8. PRINCIPAL COMPONENT ANALYSIS (PCA) DATASETS 
# =========================================================
from sklearn.decomposition import PCA
from sklearn.preprocessing import RobustScaler

print(" Generating PCA Dataset")

# Define columns that Pshould not be fed into PCA
protected_cols = ['name', 'to_club_name', 'season', 'transfer_fee', 'transfer_fee_log', 'to_club_name_encoded']

#outfield players pca
print("\nProcessing Outfielders PCA...")

# 1. Isolate the protected features
outfield_admin_train = outfield_train[[c for c in protected_cols if c in outfield_train.columns]].reset_index(drop=True)
outfield_admin_test = outfield_test[[c for c in protected_cols if c in outfield_test.columns]].reset_index(drop=True)

# 2. Isolate the features that we will actually feed into pca
outfield_feature_cols = [c for c in outfield_train.columns if c not in protected_cols and outfield_train[c].dtype in [np.float64, np.float32, np.int64, np.int32]]
X_train_out = outfield_train[outfield_feature_cols].reset_index(drop=True)
X_test_out = outfield_test[outfield_feature_cols].reset_index(drop=True)

# apply the robust scaler to data that we will feed into the pca model
pca_scaler_out = RobustScaler()
X_train_out_scaled = pd.DataFrame(pca_scaler_out.fit_transform(X_train_out), columns=X_train_out.columns)
X_test_out_scaled = pd.DataFrame(pca_scaler_out.transform(X_test_out), columns=X_test_out.columns)

# Fit PCA Model (Keep 95% variance) on scaled data
pca_out = PCA(n_components=0.95, random_state=42)
train_pcs_out = pca_out.fit_transform(X_train_out_scaled)
test_pcs_out = pca_out.transform(X_test_out_scaled)

# Create n column names for the compressed data (e.g., PC1, PC2, PC3...)

outfield_pc_columns = [f"PC{i+1}" for i in range(train_pcs_out.shape[1])]
train_pca_df_out = pd.DataFrame(train_pcs_out, columns=outfield_pc_columns)
test_pca_df_out = pd.DataFrame(test_pcs_out, columns=outfield_pc_columns)

# 5. Reassemble the final PCA datasets
#add on the protected data
outfield_pca_train = pd.concat([outfield_admin_train, train_pca_df_out], axis=1)
outfield_pca_test = pd.concat([outfield_admin_test, test_pca_df_out], axis=1)

print(f"-> Dimensionality Reduction: {len(outfield_feature_cols)} stats compressed into {pca_out.n_components_} Components.")



outfield_loadings = pd.DataFrame(pca_out.components_.T, columns=outfield_pc_columns, index=outfield_feature_cols)



# Save ALL component recipes to CSV
outfield_loadings.to_csv('outfield_pca_loadings.csv')


#pca 

print("Processing Goalkeepers PCA...")

# 1. Isolate data which we will not apply pca too 
gk_admin_train = gk_train[[c for c in protected_cols if c in gk_train.columns]].reset_index(drop=True)
gk_admin_test = gk_test[[c for c in protected_cols if c in gk_test.columns]].reset_index(drop=True)

# Isolate the features which we will actually apply the pca too
gk_feature_cols = [c for c in gk_train.columns if c not in protected_cols and gk_train[c].dtype in [np.float64, np.float32, np.int64, np.int32]]
X_train_gk = gk_train[gk_feature_cols].reset_index(drop=True)
X_test_gk = gk_test[gk_feature_cols].reset_index(drop=True)

# apply robus scaler to the data
pca_scaler_gk = RobustScaler()
X_train_gk_scaled = pd.DataFrame(pca_scaler_gk.fit_transform(X_train_gk), columns=X_train_gk.columns)
X_test_gk_scaled = pd.DataFrame(pca_scaler_gk.transform(X_test_gk), columns=X_test_gk.columns)

#  Fit PCA Model Keep 95% variance on the SCALED data
pca_gk = PCA(n_components=0.95, random_state=42)
train_pcs_gk = pca_gk.fit_transform(X_train_gk_scaled)
test_pcs_gk = pca_gk.transform(X_test_gk_scaled)

gk_pc_columns = [f"PC{i+1}" for i in range(train_pcs_gk.shape[1])]
train_pca_df_gk = pd.DataFrame(train_pcs_gk, columns=gk_pc_columns)
test_pca_df_gk = pd.DataFrame(test_pcs_gk, columns=gk_pc_columns)

#  Reassemble the final PCA datasets
gk_pca_train = pd.concat([gk_admin_train, train_pca_df_gk], axis=1)
gk_pca_test = pd.concat([gk_admin_test, test_pca_df_gk], axis=1)

print(f"-> Dimensionality Reduction: {len(gk_feature_cols)} stats compressed into {pca_gk.n_components_} Components.")


gk_loadings = pd.DataFrame(pca_gk.components_.T, columns=gk_pc_columns, index=gk_feature_cols)



# Save ALL component recipes to CSV
gk_loadings.to_csv('gk_pca_loadings.csv')


# ---------------------------------------------------------
# PART C: SAVE FINAL PCA DATASETS
# ---------------------------------------------------------
outfield_pca_train.to_csv('outfield_pca_train.csv', index=False)
outfield_pca_test.to_csv('outfield_pca_test.csv', index=False)
gk_pca_train.to_csv('gk_pca_train.csv', index=False)
gk_pca_test.to_csv('gk_pca_test.csv', index=False)

print(" PCA Datasets and Full Loading CSVs successfully saved!")