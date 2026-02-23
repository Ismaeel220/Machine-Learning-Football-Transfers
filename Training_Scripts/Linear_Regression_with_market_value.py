import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score

print("Loading Data...")
# Load training data
train_df = pd.read_csv('training_data.csv')
# Load testing data
test_df = pd.read_csv('testing_data.csv')

# Combine datasets so can apply one hot encoding to the position column since machines cannot read numbers
combined = pd.concat([train_df, test_df])

# Apply one hot encoding to the positions column
combined = pd.get_dummies(combined, columns=['position'], drop_first=True)

# Split them back apart using the 'season' column
train_df = combined[combined['season'].isin([2021, 2022])]
test_df = combined[combined['season'] == 2023]


# Drop columns which are irrevant and the algorithm should not learn from
features_to_drop = ['name', 'to_club_name', 'season', 'transfer_fee']

X_train = train_df.drop(columns=features_to_drop)
y_train = train_df['transfer_fee']

X_test = test_df.drop(columns=features_to_drop)
y_test = test_df['transfer_fee']

#train the linear regression model
print("Training the Linear Regression Model")
model = LinearRegression()
model.fit(X_train, y_train)

# testing the model
print("Testing the Linear Regression Model ")
predictions = model.predict(X_test)
#calculate mean absoloute error 
mean_error = mean_absolute_error(y_test, predictions)
#calculate the r squared accuracy 
r_accuracy = r2_score(y_test, predictions)

print(f"\n Model Performance")
print(f"Mean Absolute Error: €{mean_error:,.2f}")
print(f"R-Squared (Accuracy Score): {r_accuracy:.3f}")

# create a data frame which shows how much the linear regression model thinks each feature contributes to a players price
feature_cost = pd.DataFrame({
    'Feature': X_train.columns,
    'Euro Value Added': model.coef_
})

# Sort how much the model thinks each feature contributes to price from high to low
feature_cost = feature_cost.sort_values(by='Euro Value Added', ascending=False)

print("\n--- HOW THE MODEL PRICED THE STATS ---")
# Format the numbers to look like currency
feature_cost['Euro Value Added'] = feature_cost['Euro Value Added'].apply(lambda x: f"€{x:,.0f}")
print(feature_cost.to_string(index=False))